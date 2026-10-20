"""
BasketBee - the Supervisor (M5): the "queen bee" that runs the hive.

A LangGraph state machine:

    route ──> forecast_agent ──> inventory_agent ──> recommend ──> compose ──> END
      │            (each agent node is optional; the plan decides which run)
      └──> direct (greetings / coming soon / out of scope - no agents needed)

  route      fast keyword rules, LLM fallback only for unclear questions
  agents     specialist agents from specialists.py (LLM + tools)
  recommend  turns CRITICAL/HIGH stock risks into PENDING recommendations (needs human approval)
  compose    builds the final answer + VERIFIES every number against the tool data
Every step is written to the audit_log table. Conversation memory comes from a
LangGraph checkpointer, so follow-up questions work within a session.

Try the interactive console (Ollama running, no uvicorn needed):
    python -m backend.agents.supervisor
"""

import json
import re
import sys
import time
import uuid
from typing import Annotated, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.agents import approvals
from backend.agents.llm import get_llm
from backend.agents.router import route
from backend.agents.specialists import build_forecast_agent, build_inventory_agent, run_agent
from backend.agents.tools import get_erp
from backend.core.config import settings
from backend.db.database import SessionLocal
from backend.db.models import AuditLog, Recommendation

AGENT_TITLES = {"forecast": "Demand forecast", "inventory": "Stock & reorders"}

DIRECT_ANSWERS = {
    "smalltalk": (
        "Hi! I'm BasketBee, your supply-chain assistant. I can:\n"
        "- forecast demand for any product (festivals like Diwali included)\n"
        "- find products that will run out at any warehouse\n"
        "- recommend reorders, which you approve before any order is placed\n\n"
        "Try: \"What will run out in the south warehouse before Diwali?\""
    ),
    "coming_soon": (
        "That needs the transport, supplier or warehouse agents. They are designed (modules M7-M10) "
        "and arrive in the next phase. Right now I can forecast demand, check stock risks and "
        "recommend reorders."
    ),
    "out_of_scope": (
        "I can only help with BasketBee's supply chain: demand forecasts, stock levels, "
        "run-out risks and reorders. Could you rephrase your question around those?"
    ),
}


class SupervisorState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # conversation memory (questions + final answers)
    session_id: str
    question: str
    intent: str
    route_method: str
    plan: list[str]
    queue: list[str]
    agent_runs: list[dict]
    recommendations: list[dict]
    answer: str
    verification: dict


# ---------------------------------------------------------------------------
# Number verification (anti-hallucination guard)
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def inr(value: float) -> str:
    """Indian digit grouping: 917412 -> ₹9,17,412"""
    s = str(int(round(value)))
    if len(s) > 3:
        head, tail, groups = s[:-3], s[-3:], []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + groups) + "," + tail
    return f"₹{s}"


def _numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUM_RE.findall(DATE_RE.sub(" ", text))]


def verify_numbers(answer: str, agent_runs: list[dict], question: str) -> dict:
    """Every number in the answer must come from a tool result (or the question).
    Small numbers (<= 31: list numbering, days, warehouse counts) are allowed."""
    source = json.dumps([step["output"] for run in agent_runs for step in run["trace"]], default=str)
    allowed = set(_numbers_in(source) + _numbers_in(question))
    allowed_dates = set(DATE_RE.findall(source + " " + question)) | {settings.business_date, "2026-11-08"}

    def ok(x: float) -> bool:
        return x <= 31 or any(abs(x - a) <= max(1.0, 0.01 * a) for a in allowed)

    nums = _numbers_in(answer)
    bad_nums = sorted({n for n in nums if not ok(n)})
    bad_dates = sorted(d for d in set(DATE_RE.findall(answer)) if d not in allowed_dates)
    return {
        "numbers_checked": len(nums) + len(DATE_RE.findall(answer)),
        "unverified": [f"{n:g}" for n in bad_nums] + bad_dates,
        "verified": not bad_nums and not bad_dates,
    }


# ---------------------------------------------------------------------------
# The Supervisor
# ---------------------------------------------------------------------------
class BasketBeeSupervisor:
    def __init__(self, llm: BaseChatModel | None = None, session_factory: sessionmaker = SessionLocal):
        self.llm = llm or get_llm()
        self.session_factory = session_factory
        self.agents = {"forecast": build_forecast_agent(self.llm),
                       "inventory": build_inventory_agent(self.llm)}
        self._emit: Callable[[str, dict], None] = lambda kind, data: None
        self.graph = self._build_graph()

    # ----- graph -------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(SupervisorState)
        g.add_node("route", self._route)
        g.add_node("forecast_agent", self._agent_node("forecast"))
        g.add_node("inventory_agent", self._agent_node("inventory"))
        g.add_node("recommend", self._recommend)
        g.add_node("compose", self._compose)
        g.add_node("direct", self._direct)

        g.add_edge(START, "route")
        for node in ("route", "forecast_agent", "inventory_agent"):
            g.add_conditional_edges(node, self._next, ["forecast_agent", "inventory_agent", "recommend", "direct"])
        g.add_edge("recommend", "compose")
        g.add_edge("compose", END)
        g.add_edge("direct", END)
        return g.compile(checkpointer=MemorySaver())

    @staticmethod
    def _next(state: SupervisorState) -> str:
        if state["intent"] in DIRECT_ANSWERS:
            return "direct"
        return f"{state['queue'][0]}_agent" if state["queue"] else "recommend"

    # ----- nodes -------------------------------------------------------------
    def _route(self, state: SupervisorState) -> dict:
        question = state["messages"][-1].content
        r = route(question, self.llm)
        self._emit("route", {"intent": r.intent, "plan": r.plan, "method": r.method})
        self._audit(state["session_id"], "supervisor", "route", {"question": question},
                    {"intent": r.intent, "plan": r.plan, "method": r.method})
        return {"question": question, "intent": r.intent, "route_method": r.method,
                "plan": r.plan, "queue": [p for p in r.plan if p in self.agents],
                "agent_runs": [], "recommendations": [], "verification": {}}

    def _agent_node(self, name: str):
        def node(state: SupervisorState) -> dict:
            self._emit("agent_start", {"agent": name})
            history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage, AIMessage))][-4:]
            started = time.perf_counter()
            out = run_agent(self.agents[name], state["question"], history=history,
                            on_event=lambda k, d: self._emit(k, {**d, "agent": name}))
            run = {"agent": name, "answer": out["answer"], "trace": out["trace"],
                   "repairs": out["repairs"], "seconds": out["seconds"]}
            for step in out["trace"]:
                self._audit(state["session_id"], f"{name}_agent", step["tool"], step["args"], step["output"])
            self._audit(state["session_id"], f"{name}_agent", None, {"question": state["question"]},
                        {"answer": out["answer"]}, int(1000 * (time.perf_counter() - started)))
            self._emit("agent_done", {"agent": name, "seconds": out["seconds"]})
            return {"agent_runs": state["agent_runs"] + [run], "queue": state["queue"][1:]}
        return node

    def _recommend(self, state: SupervisorState) -> dict:
        """CRITICAL/HIGH items with an order quantity become PENDING reorder recommendations."""
        items = {}
        for run in state["agent_runs"]:
            for step in run["trace"]:
                if step["tool"] == "check_stock_risks" and step["ok"]:
                    for it in step["output"].get("items", []):
                        if it["risk"] in ("CRITICAL", "HIGH") and it["order_qty"] > 0:
                            items[(it["sku"], it["warehouse"])] = it
        if not items:
            return {"recommendations": []}

        products = get_erp().products().set_index("sku")
        recs = []
        with self.session_factory() as db:
            for (sku, wh), it in items.items():
                p = products.loc[sku]
                payload = {
                    "sku": sku, "product": it["product"], "warehouse_id": wh,
                    "supplier_id": str(p["supplier_id"]), "supplier": it["supplier"],
                    "qty": int(it["order_qty"]),
                    "value_inr": round(float(it["order_qty"] * p["unit_cost_inr"]), 2),
                    "risk": it["risk"], "runs_out": it.get("runs_out"), "per_day": it["per_day"],
                    "available": it["available"],
                }
                # one pending recommendation per product+warehouse: refresh instead of duplicating
                pending = db.scalars(select(Recommendation).where(
                    Recommendation.status == "pending", Recommendation.type == "reorder")).all()
                existing = next((r for r in pending
                                 if (json.loads(r.payload_json)["sku"], json.loads(r.payload_json)["warehouse_id"])
                                 == (sku, wh)), None)
                if existing:
                    existing.payload_json = json.dumps(payload)
                    rec = existing
                else:
                    rec = Recommendation(type="reorder", payload_json=json.dumps(payload),
                                         status="pending", created_by_agent="inventory_agent")
                    db.add(rec)
                db.flush()
                recs.append({"id": rec.rec_id, **payload})
            db.commit()
        self._emit("recommendations", {"count": len(recs)})
        return {"recommendations": recs}

    def _compose(self, state: SupervisorState) -> dict:
        runs = state["agent_runs"]
        if len(runs) == 1:
            body = runs[0]["answer"]
        else:
            body = "\n\n".join(f"**{AGENT_TITLES[r['agent']]}**\n{r['answer']}" for r in runs)
        verification = verify_numbers(body, runs, state["question"])

        if state["recommendations"]:
            total = sum(r["value_inr"] for r in state["recommendations"])
            n = len(state["recommendations"])
            body += (f"\n\n{n} reorder recommendation{'s are' if n > 1 else ' is'} waiting for your "
                     f"approval (total {inr(total)}). Nothing is ordered until you approve.")
        if not verification["verified"]:
            body += ("\n\nNote: I couldn't verify these figures against the system data, so please "
                     f"double-check them: {', '.join(verification['unverified'])}.")

        self._audit(state["session_id"], "supervisor", "compose", {"question": state["question"]},
                    {"answer": body, "verification": verification})
        return {"answer": body, "verification": verification, "messages": [AIMessage(body)]}

    def _direct(self, state: SupervisorState) -> dict:
        body = DIRECT_ANSWERS[state["intent"]]
        return {"answer": body, "verification": {"numbers_checked": 0, "unverified": [], "verified": True},
                "messages": [AIMessage(body)]}

    # ----- helpers -----------------------------------------------------------
    def _audit(self, session_id: str, agent: str, tool: str | None, inp, out, latency_ms: int | None = None):
        with self.session_factory() as db:
            db.add(AuditLog(session_id=session_id, agent=agent, tool=tool,
                            input_json=json.dumps(inp, default=str)[:4000],
                            output_json=json.dumps(out, default=str)[:4000], latency_ms=latency_ms))
            db.commit()

    # ----- public API ----------------------------------------------------------
    def ask(self, question: str, session_id: str | None = None,
            on_event: Callable[[str, dict], None] | None = None) -> dict:
        session_id = session_id or uuid.uuid4().hex[:12]
        self._emit = on_event or (lambda kind, data: None)
        started = time.perf_counter()
        state = self.graph.invoke({"messages": [HumanMessage(question)], "session_id": session_id},
                                  config={"configurable": {"thread_id": session_id}})
        return {
            "session_id": session_id, "answer": state["answer"], "intent": state["intent"],
            "route_method": state["route_method"], "plan": state["plan"],
            "agents": [{k: v for k, v in r.items() if k != "trace"} | {"tools": r["trace"]}
                       for r in state["agent_runs"]],
            "recommendations": state["recommendations"], "verification": state["verification"],
            "seconds": round(time.perf_counter() - started, 1),
        }


# ---------------------------------------------------------------------------
# Interactive console
# ---------------------------------------------------------------------------
def _console():
    print(f"\nBasketBee supervisor [{settings.llm_provider}:{settings.llm_model}]")
    print("Ask anything. Commands: /pending  /approve <id>  /reject <id>  /quit\n")
    bee = BasketBeeSupervisor()
    session = uuid.uuid4().hex[:12]
    t0 = [0.0]

    def show(kind, d):
        t = f"{time.perf_counter() - t0[0]:5.1f}s"
        if kind == "route":
            print(f"  [{t}] supervisor: intent={d['intent']} plan={d['plan']} (via {d['method']})")
        elif kind == "agent_start":
            print(f"  [{t}] {d['agent']} agent working...")
        elif kind == "tool_call":
            print(f"  [{t}]   {d['agent']} -> {d['tool']}({json.dumps(d['args'])})")
        elif kind == "repair":
            print(f"  [{t}]   rescued a text tool call: {d['tool']}")
        elif kind == "agent_done":
            print(f"  [{t}] {d['agent']} agent done ({d['seconds']}s)")
        elif kind == "recommendations":
            print(f"  [{t}] {d['count']} recommendation(s) saved for approval")

    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/pending":
            for r in approvals.list_recommendations():
                print(f"  #{r['id']} {r['risk']:<8} {r['product']} @ {r['warehouse_id']}: "
                      f"order {r['qty']} from {r['supplier']} (INR {r['value_inr']:,.0f})")
            continue
        if q.startswith(("/approve", "/reject")):
            cmd, _, rid = q.partition(" ")
            try:
                fn = approvals.approve if cmd == "/approve" else approvals.reject
                res = fn(int(rid), decided_by="console-manager")
                po = res.get("purchase_order")
                print(f"  #{rid} {res['recommendation']['status']}" + (f" -> PO {po['po_id']} created" if po else ""))
            except (ValueError, approvals.ApprovalError) as e:
                print(f"  {getattr(e, 'message', e)}")
            continue

        t0[0] = time.perf_counter()
        out = bee.ask(q, session_id=session, on_event=show)
        print(f"\nbee> {out['answer']}\n     ({out['seconds']}s, numbers verified: {out['verification']['verified']})\n")


if __name__ == "__main__":
    _console() if len(sys.argv) == 1 else print(BasketBeeSupervisor().ask(" ".join(sys.argv[1:]))["answer"])