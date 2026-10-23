
"""
BasketBee - public API used by the frontend (M6).

    POST /api/chat/stream              ask the supervisor; streams live progress (NDJSON)
    GET  /api/approvals                list recommendations (pending / approved / rejected / all)
    POST /api/approvals/{id}/approve   approve -> purchase order created in the ERP
    POST /api/approvals/{id}/reject
    GET  /api/dashboard                stock health: risks, warehouses, pending approvals
    GET  /api/forecast                 history + forecast series for one product (for charts)
    GET  /api/audit                    the activity log (agent decisions, tool calls, approvals)
"""

import json
import queue
import threading
import time
from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.agents import approvals
from backend.agents.tools import ResolveError, get_erp, get_wms, resolve_product, resolve_warehouse
from backend.api.common import api_error
from backend.core.config import settings
from backend.db.database import get_db
from backend.db.models import AuditLog, PurchaseOrder
from backend.tools.adapters import AdapterError
from backend.tools.forecasting import ForecastRequest, ToolError, forecast_demand
from backend.tools.inventory import InventoryCheckRequest, check_inventory

router = APIRouter(prefix="/api", tags=["BasketBee app"])

_supervisor = None
_chat_lock = threading.Lock()


def get_supervisor():
    """Created on first use (building the LLM client takes a moment)."""
    global _supervisor
    if _supervisor is None:
        from backend.agents.supervisor import BasketBeeSupervisor
        _supervisor = BasketBeeSupervisor()
    return _supervisor


def set_supervisor(sup) -> None:
    """Tests use this to plug in a supervisor with a scripted LLM."""
    global _supervisor
    _supervisor = sup


def _friendly_error(exc: Exception) -> str:
    """Turn LLM/provider failures into one clear sentence that says what to do."""
    text = f"{type(exc).__name__}: {exc}"
    provider, model = settings.llm_provider, settings.llm_model
    name = {"groq": "Groq", "openai": "OpenAI", "gemini": "Gemini", "ollama": "Ollama"}.get(provider, provider)
    if type(exc).__name__ == "LLMConfigError":
        return str(exc)
    if any(w in text for w in ("AuthenticationError", "invalid_api_key", "Invalid API Key", "401")):
        return f"{name} rejected the API key. Check the key in .env, then restart BasketBee."
    if any(w in text for w in ("RateLimitError", "rate_limit", "429", "Too Many Requests")):
        return f"{name}'s free rate limit was reached. Wait about a minute, then ask again."
    if any(w in text for w in ("ConnectError", "Connection refused", "ConnectionError", "APIConnectionError",
                               "11434", "getaddrinfo", "Timeout")):
        if provider == "ollama":
            return f"Can't reach Ollama ({model}). Open the Ollama app, then ask again."
        return f"Can't reach {name}. Check the internet connection, then ask again."
    if "bad_alloc" in text or "out of memory" in text.lower():
        return "Not enough free memory to load the model. Close other apps or use a smaller model."
    if any(w in text.lower() for w in ("not found", "notfound", "does not exist", "model_not_found",
                                       "decommissioned")) and "model" in text.lower():
        hint = " Try LLM_MODEL=openai/gpt-oss-120b." if provider == "groq" else ""
        return f"{name} doesn't offer the model '{model}' (it may have been retired). Check LLM_MODEL in .env.{hint}"
    return f"Something went wrong while answering: {exc}"


LLM_FATAL = ("Connect", "Connection refused", "11434", "AuthenticationError", "RateLimitError",
             "LLMConfigError", "invalid_api_key", "bad_alloc", "Timeout")


def is_llm_fatal(exc: Exception) -> bool:
    """Errors that mean the model itself is unavailable (show them clearly, don't hide them)."""
    return any(w in f"{type(exc).__name__}: {exc}" for w in LLM_FATAL)


# ----- Chat -------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


@router.post("/chat/stream")
def chat_stream(body: ChatIn):
    """Streams one JSON object per line: progress events, then {"type": "final", "result": {...}}."""
    events: queue.Queue = queue.Queue()

    def work():
        try:
            with _chat_lock:   # one LLM, one question at a time
                result = get_supervisor().ask(
                    body.message, session_id=body.session_id,
                    on_event=lambda kind, data: events.put({"type": kind, **data}))
            events.put({"type": "final", "result": result})
        except Exception as exc:   # noqa: BLE001 - report every failure to the UI
            events.put({"type": "error", "message": _friendly_error(exc)})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield json.dumps(item, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ----- Approvals --------------------------------------------------------------
class Decision(BaseModel):
    decided_by: str = Field("manager", min_length=1, max_length=100)
    comment: str | None = Field(None, max_length=500)


def _factory(db: Session) -> sessionmaker:
    """Approvals open their own sessions on the SAME database the request uses."""
    return sessionmaker(bind=db.get_bind(), autoflush=False)


@router.get("/approvals")
def list_approvals(status: str = Query("pending", pattern="^(pending|approved|rejected|all)$"),
                   db: Session = Depends(get_db)):
    return approvals.list_recommendations(None if status == "all" else status, session_factory=_factory(db))


def _decide(fn, rec_id: int, body: Decision, **kw):
    try:
        return fn(rec_id, body.decided_by, body.comment, **kw)
    except approvals.ApprovalError as e:
        status = {"NOT_FOUND": 404, "INVALID": 409, "UPSTREAM_DOWN": 503}.get(e.code, 400)
        raise api_error(status, e.code, e.message)


@router.post("/approvals/{rec_id}/approve")
def approve(rec_id: int, body: Decision, db: Session = Depends(get_db)):
    return _decide(approvals.approve, rec_id, body, erp=get_erp(), session_factory=_factory(db))


@router.post("/approvals/{rec_id}/reject")
def reject(rec_id: int, body: Decision, db: Session = Depends(get_db)):
    return _decide(approvals.reject, rec_id, body, session_factory=_factory(db))


# ----- Dashboard --------------------------------------------------------------
@router.get("/dashboard")
def dashboard(warehouse: str | None = None, days: int = Query(14, ge=3, le=60),
              db: Session = Depends(get_db)):
    try:
        wh = resolve_warehouse(warehouse)
    except ResolveError as e:
        raise api_error(404, "NOT_FOUND", str(e))

    report = check_inventory(InventoryCheckRequest(warehouse_id=wh, horizon_days=days), get_erp(), get_wms())
    if isinstance(report, ToolError):
        raise api_error(503 if report.code == "UPSTREAM_DOWN" else 400, report.code, report.message)
    try:
        warehouses = get_wms().warehouses().to_dict("records")
    except AdapterError as e:
        raise api_error(503, e.code, e.message)

    pending = approvals.list_recommendations("pending", session_factory=_factory(db))
    today = date.fromisoformat(settings.business_date)
    return {
        "business_date": settings.business_date,
        "festival_date": settings.festival_date,
        "festival_in_days": (date.fromisoformat(settings.festival_date) - today).days,
        "scope": wh or "all",
        "horizon_days": days,
        "risk_counts": report.risk_counts,
        "items_checked": report.items_checked,
        "items": [i.model_dump(mode="json") for i in report.items],
        "total_order_value_inr": report.total_order_value_inr,
        "warehouses": warehouses,
        "pending_approvals": len(pending),
        "pending_value_inr": round(sum(r["value_inr"] for r in pending), 2),
        "purchase_orders": db.scalar(select(func.count()).select_from(PurchaseOrder)),
        "elapsed_seconds": report.elapsed_seconds,
    }


# ----- Forecast chart data ---------------------------------------------------------
@router.get("/forecast")
def forecast_series(product: str, warehouse: str = "south",
                    days: int = Query(28, ge=7, le=90), history_days: int = Query(56, ge=14, le=365)):
    try:
        sku, wh = resolve_product(product), resolve_warehouse(warehouse) or "WH-SOUTH"
    except ResolveError as e:
        raise api_error(404, "NOT_FOUND", str(e))
    r = forecast_demand(ForecastRequest(sku=sku, warehouse_id=wh, horizon_days=days, evaluate=True), get_erp())
    if isinstance(r, ToolError):
        raise api_error(503 if r.code == "UPSTREAM_DOWN" else 400, r.code, r.message)

    hist = get_erp().sales_history(sku, wh)
    hist = hist[hist["date"] > pd.Timestamp(settings.business_date) - timedelta(days=history_days)]
    product_row = get_erp().product(sku)
    return {
        "sku": sku, "product": product_row["name"], "warehouse": wh,
        "history": [{"date": d.date().isoformat(), "units": int(u)} for d, u in zip(hist["date"], hist["units_sold"])],
        "forecast": [{"date": d.date.isoformat(), "units": d.units} for d in r.daily_forecast],
        "total_units": r.total_units, "avg_daily_units": r.avg_daily_units,
        "recent_avg_daily_units": r.recent_avg_daily_units,
        "peak_date": r.peak_date.isoformat(), "peak_units": r.peak_units,
        "festivals": r.festivals_in_horizon, "model": r.model,
        "wape_pct": r.wape_pct, "baseline_wape_pct": r.baseline_wape_pct,
    }


# ----- Activity log -----------------------------------------------------------
@router.get("/audit")
def audit_log(limit: int = Query(200, ge=1, le=2000), agent: str | None = None,
              db: Session = Depends(get_db)):
    q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    if agent:
        q = q.where(AuditLog.agent == agent)
    return [{"id": a.id, "ts": a.ts.isoformat() if a.ts else None, "session_id": a.session_id,
             "agent": a.agent, "tool": a.tool, "input": a.input_json, "output": a.output_json,
             "latency_ms": a.latency_ms} for a in db.scalars(q)]


# ----- Demo controls ---------------------------------------------------------------
@router.post("/llm-check")
def llm_check():
    """Ask the model one tiny question: is it reachable, and how fast is it?"""
    started = time.perf_counter()
    try:
        from backend.agents.llm import get_llm
        reply = get_llm(temperature=0).invoke("Reply with just the word OK.")
        return {"ok": True, "seconds": round(time.perf_counter() - started, 1),
                "reply": str(reply.content)[:60], "model": f"{settings.llm_provider}:{settings.llm_model}"}
    except Exception as exc:   # noqa: BLE001
        return {"ok": False, "seconds": round(time.perf_counter() - started, 1),
                "message": _friendly_error(exc), "model": f"{settings.llm_provider}:{settings.llm_model}"}


@router.post("/demo/reset")
def reset_demo(db: Session = Depends(get_db)):
    """Restore the original seed data: clears reorders, approvals, purchase orders and the log."""
    from backend.api.common import _outages
    from backend.db.seed import seed
    target = db.get_bind()
    db.close()
    try:
        seed(target, verbose=False)
    except Exception as exc:   # noqa: BLE001
        raise api_error(500, "RESET_FAILED", f"Couldn't reset the data: {exc}")
    for system in _outages:
        _outages[system] = False
    return {"ok": True, "message": "Demo data restored. All systems are back online."}


# ----- Warm-up --------------------------------------------------------------------
def warm_up() -> None:
    """Pre-compute forecasts for every product, so the dashboard opens quickly."""
    try:
        check_inventory(InventoryCheckRequest(horizon_days=14), get_erp(), get_wms())
    except Exception:   # noqa: BLE001 - warm-up is best effort
        pass