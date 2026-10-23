
"""
BasketBee - tests for the app API used by the frontend (Step 9).

Run from the project root:
    python -m pytest tests/test_app_api.py -v
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.api import app_api
from backend.agents.supervisor import BasketBeeSupervisor
from backend.main import app


class ScriptedLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def client(temp_db):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def scripted_bee(temp_db):
    llm = ScriptedLLM(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "check_stock_risks", "args": {"warehouse": "south"},
                                           "id": "c1"}]),
        AIMessage(content="Ghee runs out on 2026-10-19. Order 540 units from Kaveri Fresh Dairy."),
    ]))
    app_api.set_supervisor(BasketBeeSupervisor(llm, session_factory=temp_db))
    yield
    app_api.set_supervisor(None)


def _events(response):
    return [json.loads(line) for line in response.iter_lines() if line]


def test_chat_stream_sends_progress_then_final(client, scripted_bee):
    with client.stream("POST", "/api/chat/stream",
                       json={"message": "What will run out in the south warehouse?", "session_id": "ui1"}) as r:
        events = _events(r)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "route"
    assert "tool_call" in kinds and "agent_done" in kinds
    assert kinds[-1] == "final"
    result = events[-1]["result"]
    assert result["verification"]["verified"] and result["recommendations"]


def test_chat_stream_reports_llm_errors_politely(client, temp_db):
    class BrokenLLM(ScriptedLLM):
        def _generate(self, *a, **k):
            raise ConnectionError("Connection refused")

    app_api.set_supervisor(BasketBeeSupervisor(BrokenLLM(messages=iter([])), session_factory=temp_db))
    try:
        with client.stream("POST", "/api/chat/stream", json={"message": "What will run out?"}) as r:
            events = _events(r)
    finally:
        app_api.set_supervisor(None)
    assert events[-1]["type"] == "error"
    assert "Can't reach Ollama" in events[-1]["message"]


def test_approvals_flow(client, scripted_bee):
    with client.stream("POST", "/api/chat/stream", json={"message": "What will run out in the south warehouse?"}) as r:
        _events(r)
    pending = client.get("/api/approvals").json()
    assert pending
    ghee = next(p for p in pending if p["sku"] == "DAI-006")

    r = client.post(f"/api/approvals/{ghee['id']}/approve", json={"decided_by": "Priya"})
    assert r.status_code == 200 and r.json()["purchase_order"]["qty"] == ghee["qty"]

    again = client.post(f"/api/approvals/{ghee['id']}/approve", json={"decided_by": "Priya"})
    assert again.status_code == 409

    approved = client.get("/api/approvals", params={"status": "approved"}).json()
    assert [a["id"] for a in approved] == [ghee["id"]]


def test_reject_unknown_recommendation(client):
    r = client.post("/api/approvals/9999/reject", json={"decided_by": "Priya"})
    assert r.status_code == 404


def test_dashboard(client):
    d = client.get("/api/dashboard", params={"warehouse": "south", "days": 14}).json()
    assert d["scope"] == "WH-SOUTH" and d["risk_counts"]["CRITICAL"] >= 2
    assert d["festival_in_days"] == 21
    assert len(d["warehouses"]) == 3
    assert all(i["risk"] != "OK" for i in d["items"])


def test_forecast_series(client):
    f = client.get("/api/forecast", params={"product": "diya lamps", "warehouse": "north", "days": 28}).json()
    assert f["sku"] == "HHC-007" and len(f["forecast"]) == 28 and len(f["history"]) >= 50
    assert f["peak_date"] == "2026-11-08"


def test_audit_log_records_the_chat(client, scripted_bee):
    with client.stream("POST", "/api/chat/stream", json={"message": "What will run out in the south warehouse?",
                                                          "session_id": "audit-me"}) as r:
        _events(r)
    rows = client.get("/api/audit", params={"limit": 50}).json()
    assert any(r["session_id"] == "audit-me" and r["tool"] == "check_stock_risks" for r in rows)


def test_reset_demo_restores_seed_data(client, scripted_bee, temp_db):
    with client.stream("POST", "/api/chat/stream", json={"message": "What will run out in the south warehouse?"}) as r:
        _events(r)
    assert client.get("/api/approvals").json()
    client.post("/admin/outages", json={"system": "erp", "down": True})

    r = client.post("/api/demo/reset")
    assert r.status_code == 200 and r.json()["ok"]
    assert client.get("/api/approvals", params={"status": "all"}).json() == []
    assert client.get("/admin/outages").json() == {"erp": False, "wms": False, "tms": False}
    assert len(client.get("/erp/products").json()) == 60


def test_llm_check_reports_problems_politely(client, monkeypatch):
    import backend.agents.llm as llm_mod

    class Down:
        def invoke(self, *_):
            raise ConnectionError("Connection refused")
    monkeypatch.setattr(llm_mod, "get_llm", lambda **k: Down())
    r = client.post("/api/llm-check").json()
    assert r["ok"] is False and "Can't reach" in r["message"]


@pytest.mark.parametrize("error,expected", [
    (Exception("Error code: 401 - invalid_api_key"), "rejected the API key"),
    (Exception("Error code: 429 - rate_limit_exceeded"), "rate limit"),
    (ConnectionError("Connection refused"), "Can't reach"),
    (Exception("llama-server: std::bad_alloc"), "Not enough free memory"),
])
def test_friendly_errors(error, expected):
    assert expected in app_api._friendly_error(error)


def test_retired_model_message():
    err = Exception("Error code: 404 - The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.")
    assert "doesn't offer the model" in app_api._friendly_error(err)