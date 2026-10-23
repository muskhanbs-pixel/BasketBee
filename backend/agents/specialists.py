
"""
BasketBee - specialist agents (M3 + M4 agent layer).

An "agent" = an LLM + a short job description + a few tools.
The LLM decides WHICH tool to call and WHAT to pass; the tool does the maths;
the LLM then explains the result in plain language.

    Forecast agent   -> find_products, forecast_product_demand
    Inventory agent  -> find_products, get_stock_level, check_stock_risks

Try it (Ollama must be running; no uvicorn needed):
    python -m backend.agents.specialists inventory "What will run out in the south warehouse?"
    python -m backend.agents.specialists forecast "How much ghee will we sell before Diwali?"
"""

import json
import re
import sys
import time
import uuid
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.llm import get_llm
from backend.agents.tools import ALL_TOOLS, FORECAST_TOOLS, INVENTORY_TOOLS
from backend.core.config import settings

COMMON_RULES = f"""
Company: BasketBee, a grocery distributor in Bengaluru with three warehouses:
South Hub (Electronic City), North Hub (Yelahanka) and East Hub (Hoskote).
Today's date is {settings.business_date}. Diwali is on 2026-11-08.

Rules you must follow:
- ALWAYS call a tool to get numbers. NEVER invent, estimate or calculate numbers yourself.
- Only state figures that appear in a tool result, and keep units (units/day, INR, dates).
- If a tool returns an "error", explain the problem simply and suggest what the user can do.
- If the product is unclear, use find_products first.
- Call tools through the tool-calling function. Never write a tool call as JSON text.
- Answer in short, clear sentences a busy logistics manager can act on. Use a short list
  when there are several items.
""".strip()

FORECAST_PROMPT = f"""You are the Demand Forecasting Agent of BasketBee.
You predict how much of a product will sell, and when demand will spike (e.g. festivals).
Mention the forecast period, total units, the peak day, and any festival in the period.
Mention model accuracy (backtest error vs baseline) only if the user asks how reliable it is.

{COMMON_RULES}"""

INVENTORY_PROMPT = f"""You are the Inventory Optimization Agent of BasketBee.
You find products at risk of running out and recommend reorders.
For each risky item give: product, warehouse, when it runs out, recommended order quantity and supplier.
List CRITICAL items first. You only RECOMMEND orders; a manager must approve them.
When planning for Diwali, check at least 21 days ahead.

{COMMON_RULES}"""


def build_forecast_agent(llm: BaseChatModel | None = None):
    return create_agent(llm or get_llm(), FORECAST_TOOLS, system_prompt=FORECAST_PROMPT,
                        name="forecast_agent")


def build_inventory_agent(llm: BaseChatModel | None = None):
    return create_agent(llm or get_llm(), INVENTORY_TOOLS, system_prompt=INVENTORY_PROMPT,
                        name="inventory_agent")


def parse_text_tool_call(text: str) -> dict | None:
    """Small local models sometimes WRITE a tool call as text instead of making it, e.g.
    {"name": "find_products", "parameters": {"query": "diya"}}. Detect and rescue that."""
    if not text or "{" not in text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("name") not in ALL_TOOLS:
        return None
    args = data.get("parameters") or data.get("arguments") or data.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return {"name": data["name"], "args": args, "id": f"repair_{uuid.uuid4().hex[:8]}"}


def run_agent(agent, question: str, max_steps: int = 8,
              on_event: Callable[[str, dict], None] | None = None,
              history: list | None = None) -> dict:
    """Ask an agent a question. Returns the answer plus a trace of every tool call.

    on_event(kind, data) is called live: kind = 'tool_call' | 'tool_result' | 'repair' | 'answer'.
    """
    started = time.perf_counter()
    emit = on_event or (lambda kind, data: None)
    messages = list(history or []) + [HumanMessage(question)]
    repairs = 0

    while True:
        seen = len(messages)
        state = None
        for state in agent.stream({"messages": messages}, stream_mode="values",
                                  config={"recursion_limit": 2 * max_steps + 1}):
            new = state["messages"][seen:]
            seen = len(state["messages"])
            for m in new:
                if isinstance(m, AIMessage):
                    for call in m.tool_calls:
                        emit("tool_call", {"tool": call["name"], "args": call["args"]})
                elif isinstance(m, ToolMessage):
                    emit("tool_result", {"tool": m.name, "ok": '"error"' not in str(m.content)[:40]})
        messages = state["messages"] if state else messages

        # --- rescue a tool call the model wrote as plain text ---
        last = messages[-1]
        call = parse_text_tool_call(last.content) if isinstance(last, AIMessage) and not last.tool_calls else None
        if call is None or repairs >= 2:
            break
        repairs += 1
        emit("repair", {"tool": call["name"], "args": call["args"]})
        output = ALL_TOOLS[call["name"]].invoke(call["args"])
        messages = messages[:-1] + [
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"], name=call["name"]),
        ]
        emit("tool_result", {"tool": call["name"], "ok": "error" not in output})

    trace, pending = [], {}
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                pending[call["id"]] = {"tool": call["name"], "args": call["args"]}
        elif isinstance(m, ToolMessage):
            step = pending.pop(m.tool_call_id, {"tool": m.name, "args": {}})
            try:
                output = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                output = m.content
            step["ok"] = not ((isinstance(output, dict) and "error" in output)
                              or (isinstance(output, str) and output.startswith("Error")))
            step["output"] = output
            trace.append(step)

    answer = next((m.content for m in reversed(messages)
                   if isinstance(m, AIMessage) and m.content and not m.tool_calls), "")
    emit("answer", {"text": answer})
    return {"answer": answer, "trace": trace, "repairs": repairs, "messages": messages,
            "seconds": round(time.perf_counter() - started, 1)}


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("forecast", "inventory"):
        print('Usage: python -m backend.agents.specialists [forecast|inventory] "your question"')
        sys.exit(1)

    kind, question = sys.argv[1], " ".join(sys.argv[2:])
    agent = build_forecast_agent() if kind == "forecast" else build_inventory_agent()
    print(f"\n[{kind} agent | {settings.llm_provider}:{settings.llm_model}] {question}")
    print("  thinking... (the first question loads the model; on CPU this can take a minute)")
    t0 = time.perf_counter()

    def show(kind_, data):
        t = f"{time.perf_counter() - t0:5.1f}s"
        if kind_ == "tool_call":
            print(f"  [{t}] calling {data['tool']}({json.dumps(data['args'])})", flush=True)
        elif kind_ == "repair":
            print(f"  [{t}] model wrote the tool call as text - running {data['tool']} for it", flush=True)
        elif kind_ == "tool_result":
            print(f"  [{t}] {data['tool']} -> {'ok' if data['ok'] else 'ERROR'}; writing answer...", flush=True)

    out = run_agent(agent, question, on_event=show)
    print(f"\n{out['answer']}\n\n({out['seconds']}s)")