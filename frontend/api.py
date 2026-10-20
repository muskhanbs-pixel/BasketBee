"""
BasketBee frontend - client for the BasketBee API.

The frontend never touches the database or the agents directly: everything goes
through the FastAPI server (as in the architecture diagram).
Set BASKETBEE_API to point somewhere else (default http://127.0.0.1:8000).
"""

import json
import os
from collections.abc import Iterator

import requests

BASE = os.getenv("BASKETBEE_API", "http://127.0.0.1:8000")


class ApiError(Exception):
    def __init__(self, message: str, code: str = "ERROR"):
        super().__init__(message)
        self.message, self.code = message, code


def _raise_for(r: requests.Response):
    if r.status_code < 400:
        return
    try:
        detail = r.json().get("detail", {})
    except ValueError:
        detail = {}
    if isinstance(detail, dict):
        raise ApiError(detail.get("message", r.text), detail.get("code", "ERROR"))
    raise ApiError(str(detail))


def _get(path: str, timeout: float = 120, **params):
    try:
        r = requests.get(f"{BASE}{path}", params={k: v for k, v in params.items() if v is not None},
                         timeout=timeout)
    except requests.RequestException:
        raise ApiError(f"Can't reach the BasketBee API at {BASE}. Start it with: python run.py", "OFFLINE")
    _raise_for(r)
    return r.json()


def _post(path: str, body: dict, timeout: float = 60):
    try:
        r = requests.post(f"{BASE}{path}", json=body, timeout=timeout)
    except requests.RequestException:
        raise ApiError(f"Can't reach the BasketBee API at {BASE}. Start it with: python run.py", "OFFLINE")
    _raise_for(r)
    return r.json()


# ----- endpoints ----------------------------------------------------------------
def health() -> dict:
    return _get("/api/health", timeout=5)


def products() -> list[dict]:
    return _get("/erp/products")


def suppliers() -> list[dict]:
    return _get("/erp/suppliers")


def dashboard(warehouse: str | None, days: int) -> dict:
    return _get("/api/dashboard", warehouse=warehouse, days=days, timeout=300)


def forecast(product: str, warehouse: str, days: int) -> dict:
    return _get("/api/forecast", product=product, warehouse=warehouse, days=days, timeout=120)


def approvals(status: str = "pending") -> list[dict]:
    return _get("/api/approvals", status=status)


def approve(rec_id: int, decided_by: str, comment: str | None = None) -> dict:
    return _post(f"/api/approvals/{rec_id}/approve", {"decided_by": decided_by, "comment": comment})


def reject(rec_id: int, decided_by: str, comment: str | None = None) -> dict:
    return _post(f"/api/approvals/{rec_id}/reject", {"decided_by": decided_by, "comment": comment})


def purchase_orders() -> list[dict]:
    return _get("/erp/purchase-orders")


def audit(limit: int = 300, agent: str | None = None) -> list[dict]:
    return _get("/api/audit", limit=limit, agent=agent)


def chat_stream(message: str, session_id: str) -> Iterator[dict]:
    """Yields progress events as the supervisor works, ending with a 'final' or 'error' event."""
    try:
        with requests.post(f"{BASE}/api/chat/stream", json={"message": message, "session_id": session_id},
                           stream=True, timeout=(5, 900)) as r:
            _raise_for(r)
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield json.loads(line)
    except requests.RequestException:
        yield {"type": "error",
               "message": f"Can't reach the BasketBee API at {BASE}. Start it with: python run.py"}