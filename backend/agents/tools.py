
"""
BasketBee - the tools the AI agents are allowed to use.

Design rules (they matter a lot with small local LLMs):
  1. FORGIVING INPUT  - accept "ghee", "Ghee 500ml", "DAI-006"; "south", "Electronic City",
                        "WH-SOUTH". The tool resolves names; the LLM never needs codes.
  2. COMPACT OUTPUT   - small, flat dictionaries. No giant tables to confuse the model.
  3. ERRORS AS DATA   - problems come back as {"error": ..., "message": ...} so the agent
                        can explain them to the user instead of crashing.
  4. NUMBERS COME FROM HERE - the LLM explains these numbers; it never calculates its own.
"""

import ast
import difflib
import json
from functools import lru_cache

from langchain_core.tools import tool

from backend.core.config import settings
from backend.tools.adapters import AdapterError, ERPAdapter, WMSAdapter
from backend.tools.forecasting import ForecastRequest, ToolError, forecast_demand
from backend.tools.inventory import InventoryCheckRequest, check_inventory

MAX_ITEMS = 8    # max rows a tool hands back to the LLM (small models + CPU = keep it short)


# ---------------------------------------------------------------------------
# Adapters (shared by all tools)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_erp() -> ERPAdapter:
    return ERPAdapter() if settings.tools_transport == "http" else ERPAdapter.in_process()


def get_wms() -> WMSAdapter:
    return WMSAdapter(get_erp().client)


# ---------------------------------------------------------------------------
# Name resolution - turns human words into codes
# ---------------------------------------------------------------------------
WAREHOUSE_ALIASES = {
    "WH-SOUTH": ["south", "south hub", "electronic city", "ecity", "e-city", "wh-south", "wh south"],
    "WH-NORTH": ["north", "north hub", "yelahanka", "wh-north", "wh north"],
    "WH-EAST": ["east", "east hub", "hoskote", "wh-east", "wh east"],
}
ALL_WAREHOUSE_WORDS = {"all", "every", "everywhere", "all warehouses", "any", "", "none", "null"}


class ResolveError(Exception):
    pass


def resolve_warehouse(text: str | None) -> str | None:
    """'south' -> 'WH-SOUTH'. Returns None for 'all warehouses'."""
    if text is None or text.strip().lower() in ALL_WAREHOUSE_WORDS:
        return None
    t = text.strip().lower()
    for wh_id, aliases in WAREHOUSE_ALIASES.items():
        if t == wh_id.lower() or t in aliases or any(a in t for a in aliases if len(a) > 4):
            return wh_id
    raise ResolveError(f"Unknown warehouse '{text}'. BasketBee has: South Hub (Electronic City), "
                       f"North Hub (Yelahanka), East Hub (Hoskote).")


@lru_cache(maxsize=1)
def _product_index() -> dict:
    products = get_erp().products()
    return {
        "by_sku": {s.upper(): n for s, n in zip(products["sku"], products["name"])},
        "by_name": {n.lower(): s for s, n in zip(products["sku"], products["name"])},
        "categories": sorted(products["category"].unique()),
    }


def resolve_product(text: str) -> str:
    """'ghee' / 'Ghee' / 'DAI-006' / 'gee' -> 'DAI-006'."""
    idx = _product_index()
    t = text.strip()
    if t.upper() in idx["by_sku"]:
        return t.upper()
    low = t.lower()
    if low in idx["by_name"]:
        return idx["by_name"][low]
    partial = [s for n, s in idx["by_name"].items() if low in n or n in low]
    if not partial:
        # word match: "diya lamp" -> "Diya Oil Lamps" (every word the user typed is in the name)
        def words(s):
            return {w.rstrip("s") for w in s.replace("(", " ").replace(")", " ").split() if len(w) > 2}
        wanted = words(low)
        partial = [s for n, s in idx["by_name"].items() if wanted and wanted <= words(n)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(idx["by_sku"][s] for s in partial)
        raise ResolveError(f"'{text}' matches several products: {names}. Which one?")
    # Only fix small typos ("gee" -> "ghee"). Never silently swap in a different product.
    close = difflib.get_close_matches(low, idx["by_name"].keys(), n=1, cutoff=0.8)
    if close:
        return idx["by_name"][close[0]]
    maybe = difflib.get_close_matches(low, idx["by_name"].keys(), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(idx['by_sku'][idx['by_name'][m]] for m in maybe)}?" if maybe else ""
    raise ResolveError(f"No product called '{text}'.{hint}")


def resolve_category(text: str | None) -> str | None:
    if not text:
        return None
    cats = _product_index()["categories"]
    for c in cats:
        if text.strip().lower() in c.lower():
            return c
    raise ResolveError(f"Unknown category '{text}'. Categories: {', '.join(cats)}")


def _error(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def as_list(value) -> list[str] | None:
    """Small models send lists in many shapes. Accept them all:
    ['ghee', 'dal']  |  "['ghee', 'dal']"  |  '["ghee", "dal"]'  |  "ghee, dal"  |  "ghee" """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        text = str(value).strip()
        if text.lower() in ("", "none", "null", "all", "[]"):
            return None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = text.strip("[]").split(",")
            items = [str(v) for v in (parsed if isinstance(parsed, (list, tuple)) else [parsed])]
        else:
            items = text.split(",")
    items = [i.strip().strip("'\"") for i in items]
    return [i for i in items if i] or None


def as_int(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def find_products(query: str) -> dict:
    """Search BasketBee's product catalogue by name or category (e.g. 'oil', 'dairy', 'sweets').
    Use this when you are not sure which product the user means."""
    try:
        products = get_erp().products()
    except AdapterError as e:
        return _error(e.code, e.message)
    q = query.strip().lower()
    hits = products[products["name"].str.lower().str.contains(q) |
                    products["category"].str.lower().str.contains(q)]
    if hits.empty:
        close = difflib.get_close_matches(q, products["name"].str.lower(), n=5, cutoff=0.5)
        hits = products[products["name"].str.lower().isin(close)]
    return {
        "matches": [{"sku": r.sku, "name": r.name, "category": r.category, "pack": r.pack_size}
                    for r in hits.head(MAX_ITEMS).itertuples()],
        "total_matches": len(hits),
    }


@tool
def get_stock_level(product: str, warehouse: str | None = None) -> dict:
    """Current stock of ONE product (name or SKU) at one warehouse, or all warehouses if none given.
    Returns on-hand, reserved, in-transit and available units. Use for 'how much X do we have?'."""
    try:
        sku = resolve_product(product)
        wh = resolve_warehouse(warehouse)
        rows = get_wms().stock(sku=sku, warehouse_id=wh)
    except ResolveError as e:
        return _error("NOT_FOUND", str(e))
    except AdapterError as e:
        return _error(e.code, e.message)
    name = _product_index()["by_sku"][sku]
    return {
        "product": name, "sku": sku,
        "stock": [{"warehouse": r["warehouse_id"], "available": r["available"], "on_hand": r["on_hand"],
                   "reserved": r["reserved"], "in_transit": r["in_transit"]} for r in rows.to_dict("records")],
        "as_of": settings.business_date,
    }


@tool
def forecast_product_demand(product: str, warehouse: str | None = "south", days: int | str = 14) -> dict:
    """Forecast daily demand for ONE product (name or SKU) at ONE warehouse for the next `days` days
    (1-90). Detects festivals like Diwali. Returns totals, average, the peak day and model accuracy."""
    try:
        sku = resolve_product(product)
        wh = resolve_warehouse(warehouse) or "WH-SOUTH"
    except ResolveError as e:
        return _error("NOT_FOUND", str(e))
    days = max(1, min(as_int(days, 14), 90))
    r = forecast_demand(ForecastRequest(sku=sku, warehouse_id=wh, horizon_days=days), get_erp())
    if isinstance(r, ToolError):
        return _error(r.code, r.message)

    weeks = [round(sum(d.units for d in r.daily_forecast[i:i + 7])) for i in range(0, days, 7)]
    return {
        "product": _product_index()["by_sku"][sku], "sku": sku, "warehouse": wh,
        "period": f"{r.start_date} to {r.daily_forecast[-1].date}",
        "total_units": round(r.total_units), "avg_per_day": r.avg_daily_units,
        "recent_actual_avg_per_day": r.recent_avg_daily_units,
        "peak_day": str(r.peak_date), "peak_units": round(r.peak_units),
        "weekly_totals": weeks,
        "festivals_in_period": r.festivals_in_horizon,
        "model": r.model,
        "backtest_error_pct": r.wape_pct, "baseline_error_pct": r.baseline_wape_pct,
        "notes": r.notes,
    }


@tool
def check_stock_risks(warehouse: str | None = None, products: list[str] | str | None = None,
                      category: str | None = None, days: int | str = 14) -> dict:
    """Find products that will run out. Checks stock vs. forecast demand for the next `days` days
    (3-60) and recommends reorder quantities. Optional filters: warehouse ('south', 'north', 'east'
    or None for all), a list of product names, or a category ('dairy', 'snacks', ...).
    Risk levels: CRITICAL (runs out before a new order can arrive), HIGH (order today),
    WATCH (order within 3 days)."""
    try:
        wh = resolve_warehouse(warehouse)
        names = as_list(products)
        skus = list(dict.fromkeys(resolve_product(p) for p in names)) if names else None
        cat = resolve_category(category if category not in ("", "none", "null", "all") else None)
    except ResolveError as e:
        return _error("NOT_FOUND", str(e))
    days = max(3, min(as_int(days, 14), 60))
    r = check_inventory(InventoryCheckRequest(warehouse_id=wh, skus=skus, category=cat, horizon_days=days),
                        get_erp(), get_wms())
    if isinstance(r, ToolError):
        return _error(r.code, r.message)

    # Compact on purpose: every token costs time on a laptop CPU.
    items = []
    for i in r.items[:MAX_ITEMS]:
        item = {
            "risk": i.risk, "product": i.name, "sku": i.sku, "warehouse": i.warehouse_id,
            "available": i.available, "per_day": round(i.avg_daily_forecast),
            "order_qty": i.recommended_order_qty, "supplier": i.supplier_name,
        }
        if i.stockout_date:
            item["runs_out"] = str(i.stockout_date)
        if i.risk == "WATCH" and i.reorder_by:
            item["order_by"] = str(i.reorder_by)
        if "spike" in i.reason:
            item["spike"] = i.reason.split("Demand spike ahead: ")[1].rstrip(".")
        items.append(item)
    return {
        "as_of": str(r.business_date), "days_ahead": r.horizon_days,
        "scope": wh or "all warehouses",
        "risk_counts": {k: v for k, v in r.risk_counts.items() if k != "OK"},
        "items": items,
        "more_not_shown": max(len(r.items) - MAX_ITEMS, 0),
        "total_order_value_inr": round(r.total_order_value_inr),
    }


FORECAST_TOOLS = [find_products, forecast_product_demand]
INVENTORY_TOOLS = [find_products, get_stock_level, check_stock_risks]
ALL_TOOLS = {t.name: t for t in (find_products, get_stock_level, forecast_product_demand, check_stock_risks)}