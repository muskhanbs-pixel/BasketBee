
"""
BasketBee - tests for the agent tools and the agent loop (Step 7).

No real LLM needed: a scripted fake model plays the LLM's part, so these
tests run anywhere (and fast). The TOOLS underneath are completely real.

Run from the project root:
    python -m pytest tests/test_agents.py -v
"""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.agents.specialists import (build_forecast_agent, build_inventory_agent,
                                        parse_text_tool_call, run_agent)
from backend.agents.tools import (ResolveError, check_stock_risks, find_products,
                                  forecast_product_demand, get_stock_level, resolve_product,
                                  resolve_warehouse)
from backend.api.common import _outages


# ----- Name resolution: the "forgiving input" rule ---------------------------
@pytest.mark.parametrize("text,expected", [
    ("south", "WH-SOUTH"), ("Electronic City", "WH-SOUTH"), ("WH-NORTH", "WH-NORTH"),
    ("the east hub", "WH-EAST"), ("yelahanka warehouse", "WH-NORTH"), ("all", None), (None, None),
])
def test_resolve_warehouse(text, expected):
    assert resolve_warehouse(text) == expected


def test_resolve_warehouse_unknown():
    with pytest.raises(ResolveError):
        resolve_warehouse("mumbai")


@pytest.mark.parametrize("text,expected", [
    ("ghee", "DAI-006"), ("Ghee", "DAI-006"), ("DAI-006", "DAI-006"), ("dai-006", "DAI-006"),
    ("diya oil lamps", "HHC-007"), ("soan papdi", "SNK-008"), ("gee", "DAI-006"),
    ("diya lamp", "HHC-007"), ("diya lamps", "HHC-007"), ("toor dal", "STP-004"),
    ("sunflower oil", "STP-008"), ("cashews", "SNK-005"),
])
def test_resolve_product(text, expected):
    assert resolve_product(text) == expected


def test_ambiguous_product_asks_which_one():
    with pytest.raises(ResolveError, match="several"):
        resolve_product("milk")          # Toned Milk vs Full Cream Milk


# ----- Tools (called directly) ------------------------------------------------
def test_find_products_by_category():
    out = find_products.invoke({"query": "dairy"})
    assert out["total_matches"] == 8


def test_get_stock_level_all_warehouses():
    out = get_stock_level.invoke({"product": "ghee"})
    assert out["sku"] == "DAI-006" and len(out["stock"]) == 3


def test_forecast_tool_is_compact_and_sees_diwali():
    out = forecast_product_demand.invoke({"product": "diya lamps", "warehouse": "north", "days": 28})
    assert out["sku"] == "HHC-007"
    assert len(out["weekly_totals"]) == 4                     # weekly, not 28 daily numbers
    assert any("Diwali" in f for f in out["festivals_in_period"])


def test_risk_tool_finds_ghee():
    out = check_stock_risks.invoke({"warehouse": "south"})
    assert out["risk_counts"]["CRITICAL"] >= 1
    assert out["items"][0]["risk"] == "CRITICAL"
    assert len(out["items"]) <= 8
    assert any(i["product"] == "Ghee" for i in out["items"])


def test_risk_tool_output_is_small_enough_for_local_llms():
    import json
    out = check_stock_risks.invoke({"warehouse": None, "days": 21})
    assert len(json.dumps(out)) < 2500      # roughly 700 tokens


def test_exact_case_from_the_real_llama_run():
    # llama3.1:8b sent these exact arguments - they must work
    out = forecast_product_demand.invoke({"days": "20", "product": "diya lamp", "warehouse": "North Hub"})
    assert "error" not in out
    assert out["sku"] == "HHC-007" and out["warehouse"] == "WH-NORTH"


def test_tools_return_errors_as_data():
    assert forecast_product_demand.invoke({"product": "unicorn milk"})["error"] == "NOT_FOUND"
    assert check_stock_risks.invoke({"warehouse": "mars"})["error"] == "NOT_FOUND"
    _outages["wms"] = True
    try:
        assert get_stock_level.invoke({"product": "ghee"})["error"] == "UPSTREAM_DOWN"
    finally:
        _outages["wms"] = False


# ----- The agent loop with a scripted LLM --------------------------------------
class ScriptedLLM(GenericFakeChatModel):
    """Pretends to be an LLM: replays pre-written messages, including tool calls."""
    def bind_tools(self, tools, **kwargs):
        return self


def _script(*messages):
    return ScriptedLLM(messages=iter(messages))


def test_inventory_agent_calls_tool_and_answers():
    llm = _script(
        AIMessage(content="", tool_calls=[{"name": "check_stock_risks",
                                           "args": {"warehouse": "south", "days": 14}, "id": "c1"}]),
        AIMessage(content="Ghee at South Hub runs out tomorrow. Order 540 units from Kaveri Fresh Dairy."),
    )
    out = run_agent(build_inventory_agent(llm), "What will run out in the south warehouse?")
    assert out["trace"][0]["tool"] == "check_stock_risks"
    assert out["trace"][0]["ok"] is True
    assert out["trace"][0]["output"]["risk_counts"]["CRITICAL"] >= 1
    assert "Ghee" in out["answer"]


def test_forecast_agent_resolves_then_forecasts():
    llm = _script(
        AIMessage(content="", tool_calls=[{"name": "find_products", "args": {"query": "lamps"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "forecast_product_demand",
                                           "args": {"product": "HHC-007", "warehouse": "north", "days": 28},
                                           "id": "c2"}]),
        AIMessage(content="Diya lamp demand peaks on Diwali."),
    )
    out = run_agent(build_forecast_agent(llm), "How many diyas will North Hub sell?")
    assert [s["tool"] for s in out["trace"]] == ["find_products", "forecast_product_demand"]
    assert out["trace"][1]["output"]["peak_units"] > 20


def test_agent_trace_marks_tool_errors():
    llm = _script(
        AIMessage(content="", tool_calls=[{"name": "get_stock_level",
                                           "args": {"product": "unicorn milk"}, "id": "c1"}]),
        AIMessage(content="I couldn't find that product."),
    )
    out = run_agent(build_inventory_agent(llm), "How much unicorn milk do we have?")
    assert out["trace"][0]["ok"] is False
    assert out["answer"]


# ----- Rescue: model writes the tool call as text (seen with llama3.1:8b) -----
def test_parse_text_tool_call():
    call = parse_text_tool_call('Let me search.\n\n{"name": "find_products", "parameters": {"query": "diya"}}')
    assert call["name"] == "find_products" and call["args"] == {"query": "diya"}
    assert parse_text_tool_call("Ghee runs out tomorrow.") is None
    assert parse_text_tool_call('{"name": "delete_everything", "parameters": {}}') is None   # unknown tool


def test_agent_rescues_text_tool_call():
    llm = _script(
        AIMessage(content='It seems unclear.\n\n{"name": "find_products", "parameters": {"query": "diya"}}'),
        AIMessage(content="I found Diya Oil Lamps (HHC-007)."),
    )
    events = []
    out = run_agent(build_forecast_agent(llm), "diya?", on_event=lambda k, d: events.append(k))
    assert out["repairs"] == 1
    assert out["trace"][0]["tool"] == "find_products" and out["trace"][0]["ok"]
    assert "repair" in events and events[-1] == "answer"
    assert "Diya" in out["answer"]


# ----- Forgiving arguments (seen with llama3.1:8b in the real app) -------------
@pytest.mark.parametrize("value,expected", [
    (["Ghee", "Toor Dal"], ["Ghee", "Toor Dal"]),
    ("['Toor Dal', 'Cashew Nuts', 'Diya Oil Lamps']", ["Toor Dal", "Cashew Nuts", "Diya Oil Lamps"]),
    ('["ghee", "dal"]', ["ghee", "dal"]),
    ("ghee, toor dal", ["ghee", "toor dal"]),
    ("ghee", ["ghee"]),
    ("", None), ("null", None), (None, None), ("[]", None),
])
def test_as_list(value, expected):
    from backend.agents.tools import as_list
    assert as_list(value) == expected


def test_risk_tool_accepts_list_written_as_text():
    out = check_stock_risks.invoke({"products": "['Toor Dal', 'Cashew Nuts', 'Diya Oil Lamps']", "days": "21"})
    assert "error" not in out
    assert {i["product"] for i in out["items"]} >= {"Toor Dal", "Cashew Nuts", "Diya Oil Lamps"}


def test_tool_days_as_odd_strings():
    out = forecast_product_demand.invoke({"product": "ghee", "warehouse": "south", "days": "7.0"})
    assert "error" not in out and len(out["weekly_totals"]) == 1