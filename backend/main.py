"""
BasketBee API - entry point.

Run from the project root:
    python -m uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from sqlalchemy import text

from backend.api import erp, tms, wms
from backend.api.common import _outages, admin_router
from backend.core.config import settings
from backend.db.database import engine

app = FastAPI(
    title="BasketBee API",
    version="0.2.0",
    description="Agentic AI supply chain system for grocery distribution - "
                "mock ERP, WMS and TMS for the agents to use.",
)

app.include_router(erp.router)
app.include_router(wms.router)
app.include_router(tms.router)
app.include_router(admin_router)


@app.get("/api/health", tags=["Health"])
def health():
    """Is the API up, can it reach the database, and are the mock systems on?"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"
    return {
        "project": f"{settings.app_name} 🐝",
        "api": "ok",
        "db": db_status,
        "systems": {s: ("down" if down else "ok") for s, down in _outages.items()},
    }