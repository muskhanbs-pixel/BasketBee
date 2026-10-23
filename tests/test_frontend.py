
"""
BasketBee - frontend tests (Streamlit AppTest, no browser needed).

The API is replaced by a fake, so these run without uvicorn or Ollama.

Run from the project root:
    python -m pytest tests/test_frontend.py -v
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import api, theme

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")

HEALTH = {"api": "ok", "llm": "ollama:llama3.1:8b", "business_date": "2026-10-18"}
REC = {"id": 1, "status": "pending", "sku": "DAI-006", "product": "Ghee", "warehouse_id": "WH-SOUTH",
       "supplier_id": "SUP-01", "supplier": "Kaveri Fresh Dairy", "qty": 870, "value_inr": 252300.0,
       "risk": "CRITICAL", "runs_out": "2026-10-19", "per_day": 39, "available": 31,
       "decided_by": None, "comment": None}
RESULT = {
    "session_id": "s1", "answer": "Ghee runs out on 2026-10-19. Order 870 units.", "intent": "inventory",
    "route_method": "rules", "plan": ["inventory"], "seconds": 12.3,
    "agents": [{"agent": "inventory", "answer": "...", "repairs": 0, "seconds": 11.9,
                "tools": [{"tool": "check_stock_risks", "args": {"warehouse": "south", "days": 21},
                           "ok": True, "output": {"items": []}}]}],
    "recommendations": [REC], "verification": {"numbers_checked": 3, "unverified": [], "verified": True},
}


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"approve": []}
    monkeypatch.setattr(api, "health", lambda: HEALTH)
    monkeypatch.setattr(api, "approvals", lambda status="pending": [REC])
    monkeypatch.setattr(api, "purchase_orders", lambda: [])
    monkeypatch.setattr(api, "products", lambda: [{"sku": "HHC-007", "name": "Diya Oil Lamps"}])
    monkeypatch.setattr(api, "suppliers", lambda: [{"supplier_id": "SUP-01", "name": "Kaveri Fresh Dairy"}])
    monkeypatch.setattr(api, "audit", lambda limit=300, agent=None: [])

    def chat_stream(message, session_id):
        yield {"type": "route", "intent": "inventory", "plan": ["inventory"], "method": "rules"}
        yield {"type": "agent_start", "agent": "inventory"}
        yield {"type": "final", "result": RESULT}
    monkeypatch.setattr(api, "chat_stream", chat_stream)

    def approve(rec_id, who, comment=None):
        calls["approve"].append((rec_id, who))
        return {"recommendation": {**REC, "status": "approved"}, "purchase_order": {"po_id": 1}}
    monkeypatch.setattr(api, "approve", approve)

    import streamlit as st
    st.cache_data.clear()
    yield calls
    st.cache_data.clear()


def run(at: AppTest) -> AppTest:
    at.run(timeout=30)
    assert not at.exception, at.exception
    return at


def html_text(at: AppTest) -> str:
    return " ".join(str(h.proto.body) for h in at.get("html"))


# ----- formatting --------------------------------------------------------------
@pytest.mark.parametrize("value,text", [(432, "₹432"), (4896, "₹4,896"), (917412, "₹9,17,412"),
                                        (12345678, "₹1,23,45,678")])
def test_indian_rupee_format(value, text):
    assert theme.inr(value) == text


def test_flight_path_shows_every_step():
    path = theme.flight_path(RESULT)
    for step in ("Supervisor", "Inventory agent", "Checked stock risks", "Drafted 1 reorder", "Numbers checked"):
        assert step in path


# ----- pages ---------------------------------------------------------------------
def test_ask_page_welcomes_with_suggestions(fake_api):
    at = run(AppTest.from_file(APP))
    assert "Hi, I&#x27;m BasketBee." in html_text(at) or "Hi, I'm BasketBee." in html_text(at)
    assert len([b for b in at.button if b.key and b.key.startswith("sugg-")]) == 4
    assert "Approvals (1)" in str([p.proto for p in at.get("page_link")])


def test_asking_a_question_shows_answer_and_reorder(fake_api):
    at = run(AppTest.from_file(APP))
    at.button(key="sugg-0").click()
    run(at)
    assert any("Ghee runs out" in m.value for m in at.markdown)
    assert "Drafted 1 reorder" in html_text(at)
    assert at.button(key="ap-1-1") is not None


def test_approving_from_the_chat_calls_the_api(fake_api):
    at = run(AppTest.from_file(APP))
    at.button(key="sugg-0").click()
    run(at)
    at.button(key="ap-1-1").click()
    run(at)
    assert fake_api["approve"] == [(1, "Logistics manager")]


def test_offline_api_is_explained(monkeypatch, fake_api):
    def down():
        raise api.ApiError("offline", "OFFLINE")
    monkeypatch.setattr(api, "health", down)
    at = run(AppTest.from_file(APP))
    assert "API offline" in html_text(at)


def test_approvals_page(fake_api):
    at = AppTest.from_file(APP)
    run(at)
    at.switch_page("views/approvals.py")
    run(at)
    assert "Waiting for you" in html_text(at)
    assert at.button(key="a-1") is not None


def test_activity_page_empty_state(fake_api):
    at = AppTest.from_file(APP)
    run(at)
    at.switch_page("views/activity.py")
    run(at)
    assert "Nothing here yet" in html_text(at)


def test_status_page_toggles_an_outage(fake_api, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "outages", lambda: {"erp": False, "wms": False, "tms": False})
    monkeypatch.setattr(api, "set_outage", lambda s, d: calls.append((s, d)) or {})
    monkeypatch.setattr(api, "health", lambda: {**HEALTH, "db": "ok", "systems": {"erp": "ok", "wms": "ok", "tms": "ok"}})
    at = AppTest.from_file(APP)
    run(at)
    at.switch_page("views/status.py")
    run(at)
    assert "Enterprise systems" in html_text(at)
    at.toggle(key="down-erp").set_value(True)
    run(at)
    assert calls == [("erp", True)]