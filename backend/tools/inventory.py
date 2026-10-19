"""
BasketBee - Inventory Optimization tool (M4).

Answers: "What will run out, when, and how much should we reorder?"

For every product at every warehouse in scope:
  1. Forecast demand (M3) over the supplier lead time + planning horizon.
  2. Simulate stock day by day  ->  stock-out date and days of cover.
  3. Classic inventory formulas:
        safety stock   SS  = z * sigma_daily * sqrt(lead_time + review)
        reorder point  ROP = forecast demand during (lead_time + review) + SS
        EOQ                = sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)
  4. Classify risk and recommend an order quantity (rounded to case packs,
     capped by shelf life so perishables don't expire on the shelf).

Risk levels
  CRITICAL  stock runs out BEFORE an order placed today can arrive -> expedite/transfer
  HIGH      stock position is below the reorder point             -> order today
  WATCH     will drop below the reorder point within 3 days      -> order soon
  OK        nothing to do

Speed trick - two-stage "triage":
  Stage 1: fast Ridge forecast for EVERY item, with a 25% safety buffer.
  Stage 2: accurate ensemble forecast only for items stage 1 flags as risky.
  Results are cached per business day, so repeat questions are instant.

The tool only RECOMMENDS. Placing orders needs human approval (see ERP API).

Try it from the project root (no server needed):
    python -m backend.tools.inventory WH-SOUTH
    python -m backend.tools.inventory            (all warehouses)
"""

import math
import sys
import time
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from backend.core.config import settings
from backend.tools.adapters import AdapterError, ERPAdapter, WMSAdapter
from backend.tools.forecasting import ToolError, _fit_predict

REVIEW_DAYS = 1              # we can place orders every day
IN_TRANSIT_ARRIVAL_DAY = 2   # assumption: stock already in transit arrives in 2 days
SCREEN_BUFFER = 1.25         # stage-1 inflates the fast forecast by 25% to be safe
SIGMA_WINDOW = 28            # days of history used to measure demand variability
WATCH_DAYS = 3               # "reorder soon" if we hit the reorder point within this many days
RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "OK": 3}

_forecast_cache: dict[tuple, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
class InventoryCheckRequest(BaseModel):
    warehouse_id: str | None = Field(None, description="None = all warehouses")
    skus: list[str] | None = Field(None, description="None = all products")
    category: str | None = None
    horizon_days: int = Field(14, ge=3, le=60, description="How far ahead to plan")
    service_level: float = Field(0.95, ge=0.5, le=0.999, description="Target chance of NOT running out")
    include_ok: bool = Field(False, description="Also list items with no risk")


class InventoryItem(BaseModel):
    warehouse_id: str
    sku: str
    name: str
    category: str
    risk: str                          # CRITICAL | HIGH | WATCH | OK
    reason: str
    available: int                     # on hand - reserved
    in_transit: int
    inventory_position: int            # available + in transit
    avg_daily_forecast: float
    days_of_cover: float               # how long current stock lasts
    stockout_date: date | None
    reorder_by: date | None            # last day to order before dropping below the reorder point
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    eoq: int
    recommended_order_qty: int
    order_value_inr: float
    supplier_id: str
    supplier_name: str
    expected_arrival: date | None
    forecast_model: str                # ridge_screen | ensemble_gbm_ridge


class InventoryReport(BaseModel):
    business_date: date
    horizon_days: int
    service_level: float
    items_checked: int
    items_deep_checked: int
    risk_counts: dict[str, int]
    total_order_value_inr: float
    items: list[InventoryItem]
    elapsed_seconds: float
    assumptions: list[str]


# ---------------------------------------------------------------------------
# Pure inventory maths (easy to unit-test)
# ---------------------------------------------------------------------------
def z_score(service_level: float) -> float:
    return NormalDist().inv_cdf(service_level)


def safety_stock(sigma_daily: float, protection_days: int, service_level: float) -> float:
    return z_score(service_level) * sigma_daily * math.sqrt(protection_days)


def eoq(annual_demand: float, order_cost: float, unit_cost: float, holding_pct: float) -> float:
    holding = unit_cost * holding_pct
    if annual_demand <= 0 or holding <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * order_cost / holding)


def simulate_stock(available: float, in_transit: float, daily_forecast: np.ndarray,
                   arrival_day: int = IN_TRANSIT_ARRIVAL_DAY) -> tuple[int | None, float]:
    """Walk forward day by day. Returns (day index of stock-out or None, days of cover)."""
    stock = float(available)
    for day, demand in enumerate(daily_forecast, start=1):
        if day == arrival_day:
            stock += in_transit
        if demand > 0 and stock < demand:
            return day, round(day - 1 + stock / demand, 1)
        stock -= demand
    return None, float(len(daily_forecast))


def days_until_reorder(position: float, daily_forecast: np.ndarray, rop: float) -> int | None:
    """0 = already below the reorder point; None = not within the forecast window."""
    if position < rop:
        return 0
    remaining = position - np.cumsum(daily_forecast)
    below = np.nonzero(remaining < rop)[0]
    return int(below[0]) + 1 if below.size else None


def round_up(qty: float, pack: int) -> int:
    return 0 if qty <= 0 else int(math.ceil(qty / pack) * pack)


# ---------------------------------------------------------------------------
# One item
# ---------------------------------------------------------------------------
def _assess(stock_row, product, supplier, forecast: np.ndarray, sigma: float,
            req: InventoryCheckRequest, today: date, model: str) -> InventoryItem:
    lead = int(supplier["lead_time_days"])
    protection = lead + REVIEW_DAYS
    available = int(stock_row["on_hand"] - stock_row["reserved"])
    in_transit = int(stock_row["in_transit"])
    position = available + in_transit

    ss = safety_stock(sigma, protection, req.service_level)
    demand_protection = float(forecast[:protection].sum())
    rop = demand_protection + ss
    avg_daily = float(forecast[: req.horizon_days].mean())
    q_eoq = eoq(avg_daily * 365, product["order_cost_inr"], product["unit_cost_inr"],
                product["holding_cost_pct_per_year"])

    stockout_day, cover = simulate_stock(available, in_transit, forecast[: req.horizon_days + protection])
    stockout_date = today + timedelta(days=stockout_day) if stockout_day else None

    # Reorder point is checked against demand AFTER today's protection window
    reorder_in = days_until_reorder(position, forecast[: req.horizon_days], rop)

    # --- risk ---  (an order placed today is usable from day lead+1)
    if stockout_day is not None and stockout_day <= lead:
        risk = "CRITICAL"   # runs out before a new order can arrive
    elif reorder_in == 0:
        risk = "HIGH"       # below reorder point: order today
    elif reorder_in is not None and reorder_in <= WATCH_DAYS:
        risk = "WATCH"      # will hit the reorder point in the next few days
    else:
        risk = "OK"

    # --- order quantity: cover lead time + horizon, but never more than shelf life allows ---
    # The order is placed today (CRITICAL/HIGH) or on the reorder day (WATCH).
    qty = 0
    if risk != "OK":
        order_day = 0 if risk in ("CRITICAL", "HIGH") else reorder_in
        position_then = position - float(forecast[:order_day].sum())
        window = min(protection + req.horizon_days, max(int(product["shelf_life_days"]) - 1, protection))
        order_up_to = float(forecast[order_day: order_day + window].sum()) + ss
        qty = order_up_to - position_then
        if product["shelf_life_days"] > 30:              # EOQ only makes sense for long-life goods
            qty = max(qty, q_eoq)
        qty = round_up(qty, int(product["case_pack"]))

    # --- human-readable reason ---
    window_fc = forecast[: req.horizon_days]
    peak_i = int(np.argmax(window_fc))
    normal = float(np.median(window_fc))
    peak_note = ""
    if normal > 0 and window_fc[peak_i] > 1.5 * normal:
        peak_note = (f" Demand spike ahead: ~{window_fc[peak_i]:.0f}/day on "
                     f"{today + timedelta(days=peak_i + 1)} vs ~{normal:.0f} normally.")
    if risk == "CRITICAL":
        reason = (f"Only {available} available + {in_transit} in transit (~{cover} days of cover at "
                  f"{avg_daily:.0f}/day). Runs out {stockout_date}, but a new order takes {lead} day(s): "
                  f"order now and consider an emergency transfer.")
    elif risk == "HIGH":
        reason = (f"Stock position {position} is below the reorder point of {rop:.0f} "
                  f"({lead}-day lead time + safety stock). Order today.")
    elif risk == "WATCH":
        reason = f"Hits the reorder point of {rop:.0f} in {reorder_in} day(s). Plan an order soon."
    else:
        reason = f"About {cover:.0f}+ days of cover at {avg_daily:.0f}/day. No action needed."
    reason += peak_note

    return InventoryItem(
        warehouse_id=stock_row["warehouse_id"], sku=product["sku"], name=product["name"],
        category=product["category"], risk=risk, reason=reason,
        available=available, in_transit=in_transit, inventory_position=position,
        avg_daily_forecast=round(avg_daily, 1), days_of_cover=cover, stockout_date=stockout_date,
        reorder_by=(today + timedelta(days=reorder_in)) if reorder_in is not None else None,
        lead_time_days=lead, safety_stock=round(ss), reorder_point=round(rop), eoq=round(q_eoq),
        recommended_order_qty=qty, order_value_inr=round(qty * product["unit_cost_inr"], 2),
        supplier_id=supplier["supplier_id"], supplier_name=supplier["name"],
        expected_arrival=(today + timedelta(days=lead)) if qty else None,
        forecast_model=model,
    )


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
def check_inventory(req: InventoryCheckRequest, erp: ERPAdapter | None = None,
                    wms: WMSAdapter | None = None) -> InventoryReport | ToolError:
    started = time.perf_counter()
    erp = erp or ERPAdapter()
    wms = wms or WMSAdapter(erp.client)
    today = pd.Timestamp(settings.business_date)

    # ---- load everything in a few bulk calls ----
    try:
        warehouses = wms.warehouses()
        if req.warehouse_id and req.warehouse_id not in set(warehouses["warehouse_id"]):
            return ToolError(code="NOT_FOUND", message=f"Warehouse {req.warehouse_id} does not exist")
        wh_ids = [req.warehouse_id] if req.warehouse_id else list(warehouses["warehouse_id"])
        products = erp.products(req.category).set_index("sku", drop=False)
        suppliers = erp.suppliers().set_index("supplier_id", drop=False)
        festivals = erp.festivals()
        stock = wms.stock(warehouse_id=req.warehouse_id)
        sales = pd.concat([erp.warehouse_sales(w).assign(warehouse_id=w) for w in wh_ids])
    except AdapterError as e:
        return ToolError(code=e.code, message=e.message)

    if req.skus:
        unknown = sorted(set(req.skus) - set(products.index))
        if unknown:
            return ToolError(code="NOT_FOUND", message=f"Unknown product(s): {', '.join(unknown)}")
        products = products.loc[req.skus]
    if products.empty:
        return ToolError(code="NOT_FOUND", message=f"No products found for category '{req.category}'")

    stock = stock[stock["sku"].isin(products.index) & stock["warehouse_id"].isin(wh_ids)]
    sales = sales[sales["date"] <= today]
    max_lead = int(suppliers["lead_time_days"].max())
    n_days = req.horizon_days + max_lead + REVIEW_DAYS
    future = pd.date_range(today + timedelta(days=1), periods=n_days, freq="D")
    grouped = {k: g.sort_values("date") for k, g in sales.groupby(["warehouse_id", "sku"])}

    def forecast(key, mode):
        cache_key = (key, mode, n_days, settings.business_date)
        if cache_key not in _forecast_cache:
            _forecast_cache[cache_key] = _fit_predict(grouped[key], future, festivals, mode=mode)
        return _forecast_cache[cache_key]

    items, deep = [], 0
    for row in stock.to_dict("records"):
        key = (row["warehouse_id"], row["sku"])
        if key not in grouped:
            continue
        product = products.loc[row["sku"]]
        supplier = suppliers.loc[product["supplier_id"]]
        sigma = float(grouped[key]["units_sold"].tail(SIGMA_WINDOW).std())

        # Stage 1: fast screen with a safety buffer
        screen = _assess(row, product, supplier, forecast(key, "ridge") * SCREEN_BUFFER,
                         sigma, req, today.date(), "ridge_screen")
        if screen.risk in ("OK", "WATCH"):
            item = _assess(row, product, supplier, forecast(key, "ridge"), sigma, req, today.date(), "ridge_screen")
        else:
            # Stage 2: accurate forecast for anything that looked risky
            deep += 1
            item = _assess(row, product, supplier, forecast(key, "ensemble"), sigma, req,
                           today.date(), "ensemble_gbm_ridge")
        items.append(item)

    counts = {r: sum(i.risk == r for i in items) for r in RISK_ORDER}
    shown = [i for i in items if req.include_ok or i.risk != "OK"]
    shown.sort(key=lambda i: (RISK_ORDER[i.risk], i.days_of_cover))

    return InventoryReport(
        business_date=today.date(), horizon_days=req.horizon_days, service_level=req.service_level,
        items_checked=len(items), items_deep_checked=deep, risk_counts=counts,
        total_order_value_inr=round(sum(i.order_value_inr for i in shown), 2),
        items=shown, elapsed_seconds=round(time.perf_counter() - started, 1),
        assumptions=[
            f"Stock already in transit arrives in {IN_TRANSIT_ARRIVAL_DAY} days.",
            f"Orders can be placed every day (review period {REVIEW_DAYS} day).",
            f"Demand variability measured over the last {SIGMA_WINDOW} days.",
            "Order quantities are capped by shelf life and rounded up to full case packs.",
        ],
    )


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wh = sys.argv[1] if len(sys.argv) > 1 else None
    result = check_inventory(InventoryCheckRequest(warehouse_id=wh), ERPAdapter.in_process())
    if isinstance(result, ToolError):
        print(f"ERROR {result.code}: {result.message}")
        sys.exit(1)

    print(f"\nInventory check {wh or 'ALL warehouses'} as of {result.business_date} "
          f"(next {result.horizon_days} days, {result.service_level:.0%} service level)")
    print(f"Checked {result.items_checked} items, deep-checked {result.items_deep_checked}, "
          f"in {result.elapsed_seconds}s")
    print("Risk:", ", ".join(f"{k} {v}" for k, v in result.risk_counts.items()))
    print(f"Recommended orders: INR {result.total_order_value_inr:,.0f}\n")
    print(f"{'RISK':<9}{'WH':<10}{'SKU':<9}{'PRODUCT':<22}{'AVAIL':>6}{'/DAY':>7}"
          f"{'COVER':>7}  {'RUNS OUT':<11}{'ORDER':>7}  SUPPLIER")
    for i in result.items:
        print(f"{i.risk:<9}{i.warehouse_id:<10}{i.sku:<9}{i.name[:21]:<22}{i.available:>6}"
              f"{i.avg_daily_forecast:>7.0f}{i.days_of_cover:>6.1f}d  {str(i.stockout_date or '-'):<11}"
              f"{i.recommended_order_qty:>7}  {i.supplier_name}")
    if result.items:
        print(f"\nWhy is {result.items[0].name} at the top?\n  {result.items[0].reason}")