
"""
BasketBee - tests for the Supervisor (M5), router, approvals and number verification.

A scripted fake LLM plays the model; tools, database, approvals and ERP are real
(on a temporary copy of the database).

Run from the project root:
    python -m pytest tests/test_supervisor.py -v
"""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import func, select

from backend.agents import approvals
from backend.agents.router import route, route_by_rules
from backend.agents.supervisor import BasketBeeSupervisor, verify_numbers
from backend.api.common import _outages
from backend.db.models import AuditLog, PurchaseOrder, Recommendation


class ScriptedLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def script(*messages):
    return ScriptedLLM(messages=iter(messages))


def call(tool, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": cid}])


# ----- Router: fast path (no LLM) ---------------------------------------------
@pytest.mark.parametrize("question,plan", [
    ("What will run out in the south warehouse?", ["inventory"]),
    ("Which products are at risk before Diwali?", ["inventory"]),
    ("How much ghee do we have?", ["inventory"]),
    ("Are we ready for Diwali?", ["inventory"]),
    ("Forecast demand for toor dal next month", ["forecast"]),
    ("How many diya lamps will North Hub sell before Diwali?", ["forecast"]),
    ("How much ghee will we sell next week and do we have enough stock?", ["forecast", "inventory"]),
    ("Hi!", ["smalltalk"]),
    ("What can you do?", ["smalltalk"]),
    ("Plan the delivery routes for tomorrow", ["coming_soon"]),
])
def test_rules_router(question, plan):
    r = route_by_rules(question)
    assert r is not None and r.plan == plan and r.method == "rules"


def test_unclear_question_goes_to_llm():
    assert route_by_rules("Write me a poem about mangoes") is None
    r = route("Write me a poem about mangoes", script(AIMessage(content="out_of_scope")))
    assert r.intent == "out_of_scope" and r.method == "llm"


def test_llm_router_survives_chatty_reply():
    r = route("hmm the ghee situation", script(AIMessage(content="I think: inventory.")))
    assert r.plan == ["inventory"]


# ----- Number verification ----------------------------------------------------
RUNS = [{"agent": "inventory", "trace": [{"tool": "check_stock_risks", "ok": True, "output": {
    "items": [{"product": "Ghee", "order_qty": 540, "runs_out": "2026-10-19", "per_day": 33}]}}]}]


def test_verify_accepts_tool_numbers():
    v = verify_numbers("Ghee runs out on 2026-10-19. Order 540 units (33/day).", RUNS, "ghee?")
    assert v["verified"]


def test_verify_flags_invented_numbers():
    v = verify_numbers("Order 99999 units of ghee by 2026-12-25.", RUNS, "ghee?")
    assert not v["verified"]
    assert "99999" in v["unverified"] and "2026-12-25" in v["unverified"]


def test_verify_allows_rounding_and_small_numbers():
    v = verify_numbers("1. Ghee: order 541 units within 3 days.", RUNS, "ghee?")
    assert v["verified"]


# ----- Supervisor end-to-end --------------------------------------------------
def test_inventory_question_creates_pending_recommendations(temp_db):
    llm = script(call("check_stock_risks", {"warehouse": "south"}),
                 AIMessage(content="Ghee runs out on 2026-10-19; order 540 units from Kaveri Fresh Dairy."))
    bee = BasketBeeSupervisor(llm, session_factory=temp_db)
    out = bee.ask("What will run out in the south warehouse?", session_id="t1")

    assert out["plan"] == ["inventory"] and out["route_method"] == "rules"
    assert out["verification"]["verified"]
    assert len(out["recommendations"]) >= 2                          # ghee + sunflower oil at least
    assert "waiting for your approval" in out["answer"]
    with temp_db() as db:
        assert db.scalar(select(func.count()).select_from(Recommendation)
                         .where(Recommendation.status == "pending")) == len(out["recommendations"])
        assert db.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.session_id == "t1")) >= 3


def test_asking_twice_does_not_duplicate_recommendations(temp_db):
    llm = script(call("check_stock_risks", {"warehouse": "south"}), AIMessage(content="Ghee is critical."),
                 call("check_stock_risks", {"warehouse": "south"}), AIMessage(content="Still critical."))
    bee = BasketBeeSupervisor(llm, session_factory=temp_db)
    first = bee.ask("What will run out in the south warehouse?")
    second = bee.ask("What will run out in the south warehouse?")
    assert {r["id"] for r in first["recommendations"]} == {r["id"] for r in second["recommendations"]}


def test_compound_question_runs_both_agents_in_order(temp_db):
    llm = script(
        call("forecast_product_demand", {"product": "ghee", "warehouse": "south", "days": 7}),
        AIMessage(content="Ghee demand is rising this week."),
        call("check_stock_risks", {"warehouse": "south", "products": ["ghee"]}),
        AIMessage(content="Ghee will run out; reorder now."),
    )
    out = BasketBeeSupervisor(llm, session_factory=temp_db).ask(
        "How much ghee will we sell next week and do we have enough stock?")
    assert [a["agent"] for a in out["agents"]] == ["forecast", "inventory"]
    assert "Demand forecast" in out["answer"] and "Stock & reorders" in out["answer"]


def test_greeting_needs_no_llm(temp_db):
    bee = BasketBeeSupervisor(script(), session_factory=temp_db)   # empty script: any LLM call would fail
    out = bee.ask("Hello!")
    assert out["intent"] == "smalltalk" and "BasketBee" in out["answer"] and out["agents"] == []


def test_invented_number_is_flagged_in_answer(temp_db):
    llm = script(call("check_stock_risks", {"warehouse": "south"}),
                 AIMessage(content="Order 99999 units of ghee immediately."))
    out = BasketBeeSupervisor(llm, session_factory=temp_db).ask("What will run out in the south warehouse?")
    assert not out["verification"]["verified"]
    assert "99999" in out["answer"] and "double-check" in out["answer"]


def test_conversation_memory_keeps_history(temp_db):
    bee = BasketBeeSupervisor(script(), session_factory=temp_db)
    bee.ask("Hi", session_id="mem")
    bee.ask("What can you do?", session_id="mem")
    state = bee.graph.get_state({"configurable": {"thread_id": "mem"}}).values
    assert len(state["messages"]) == 4       # 2 questions + 2 answers


# ----- Approvals: human in the loop --------------------------------------------
@pytest.fixture
def pending_rec(temp_db):
    llm = script(call("check_stock_risks", {"warehouse": "south"}), AIMessage(content="Ghee is critical."))
    out = BasketBeeSupervisor(llm, session_factory=temp_db).ask("What will run out in the south warehouse?")
    return next(r for r in out["recommendations"] if r["sku"] == "DAI-006")


def test_approve_creates_purchase_order(temp_db, pending_rec):
    res = approvals.approve(pending_rec["id"], "manager", session_factory=temp_db)
    assert res["recommendation"]["status"] == "approved"
    assert res["purchase_order"]["qty"] == pending_rec["qty"]
    with temp_db() as db:
        assert db.scalar(select(func.count()).select_from(PurchaseOrder)) == 1


def test_cannot_approve_twice(temp_db, pending_rec):
    approvals.approve(pending_rec["id"], "manager", session_factory=temp_db)
    with pytest.raises(approvals.ApprovalError, match="already approved"):
        approvals.approve(pending_rec["id"], "manager", session_factory=temp_db)


def test_reject(temp_db, pending_rec):
    res = approvals.reject(pending_rec["id"], "manager", "Supplier price too high", session_factory=temp_db)
    assert res["recommendation"]["status"] == "rejected"
    assert pending_rec["id"] not in {r["id"] for r in approvals.list_recommendations(session_factory=temp_db)}


def test_erp_down_keeps_recommendation_pending(temp_db, pending_rec):
    _outages["erp"] = True
    try:
        with pytest.raises(approvals.ApprovalError, match="still pending"):
            approvals.approve(pending_rec["id"], "manager", session_factory=temp_db)
    finally:
        _outages["erp"] = False
    pending = approvals.list_recommendations(session_factory=temp_db)
    assert pending_rec["id"] in {r["id"] for r in pending}


def test_agent_crash_does_not_crash_the_answer(temp_db):
    class ExplodingLLM(ScriptedLLM):
        def _generate(self, *a, **k):
            raise ValueError("model produced garbage")
    out = BasketBeeSupervisor(ExplodingLLM(messages=iter([])), session_factory=temp_db).ask(
        "What will run out in the south warehouse?")
    assert "hit a problem and was skipped" in out["answer"]
    assert out["recommendations"] == []