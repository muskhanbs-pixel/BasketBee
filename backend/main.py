"""
BasketBee API - entry point.

Run from the project root:
    python -m uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/docs

(Or start the API and the app together with:  python run.py)
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from backend.api import app_api, erp, tms, wms
from backend.api.common import _outages, admin_router
from backend.core.config import settings
from backend.db.database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warmup_on_startup:
        threading.Thread(target=app_api.warm_up, daemon=True).start()
    yield


app = FastAPI(
    title="BasketBee API",
    version="0.3.0",
    description="Agentic AI supply chain system for grocery distribution - "
                "mock ERP, WMS and TMS, plus the app API used by the BasketBee frontend.",
    lifespan=lifespan,
)

app.include_router(app_api.router)
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
        "llm": f"{settings.llm_provider}:{settings.llm_model}",
        "business_date": settings.business_date,
        "systems": {s: ("down" if down else "ok") for s, down in _outages.items()},
    }