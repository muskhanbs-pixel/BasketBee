"""
BasketBee - shared helpers for the mock enterprise APIs.

1. api_error()      - one consistent error format for every endpoint
2. Outage simulator - lets you "switch off" ERP, WMS or TMS during a demo
                      to show how the agents handle a system being down.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

SYSTEMS = ("erp", "wms", "tms")
_outages: dict[str, bool] = {s: False for s in SYSTEMS}


def api_error(status: int, code: str, message: str) -> HTTPException:
    """Every error looks like: {"detail": {"code": "...", "message": "..."}}"""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def check_available(system: str):
    """FastAPI dependency: return 503 if this system has been switched off."""
    def _check():
        if _outages[system]:
            raise api_error(503, "UPSTREAM_DOWN", f"{system.upper()} is temporarily unavailable")
    return _check


# ----- Admin router to toggle outages (demo / testing only) -----------------
admin_router = APIRouter(prefix="/admin", tags=["Admin (demo)"])


class OutageToggle(BaseModel):
    system: str
    down: bool


@admin_router.get("/outages")
def get_outages():
    return _outages


@admin_router.post("/outages")
def set_outage(body: OutageToggle):
    if body.system not in SYSTEMS:
        raise api_error(400, "INVALID", f"system must be one of {SYSTEMS}")
    _outages[body.system] = body.down
    return _outages