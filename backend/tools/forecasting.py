"""
BasketBee - Demand Forecasting tool (M3).

Predicts daily demand for one product at one warehouse.

How it works
------------
1. Fetch sales history + festival calendar from the mock ERP (via the adapter).
2. Turn every date into features the model can learn from:
     day of week, salary week, month, trend, and
     "how close is this day to Diwali / Ganesh Chaturthi / ..." for each festival.
3. Train TWO models on log(demand) and average them (an "ensemble"):
     - Gradient Boosting: catches sharp festival spikes (diya lamps, sweets)
     - Ridge regression:  steady and accurate on everyday products (milk, dal)
   In our evaluation over all 180 product-warehouse pairs (Diwali 2025 window)
   the ensemble cut error from 25.9% (baseline) to 21.4%.
   See eval/evaluate_forecast.py.
4. Backtest honestly: hide the most recent N days, train on the rest, compare
   the model's error with a simple baseline (28-day moving average).
5. Retrain on ALL history and forecast the future.

Error metric: WAPE = sum(|actual - forecast|) / sum(actual).
(Like MAPE, but it doesn't blow up on days with zero sales.)

Try it from the project root (no server needed):
    python -m backend.tools.forecasting DAI-006 WH-SOUTH 28
"""

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from backend.core.config import settings
from backend.tools.adapters import AdapterError, ERPAdapter

MIN_HISTORY_DAYS = 14        # below this we refuse to forecast
ML_HISTORY_DAYS = 90         # below this we fall back to the moving average
BASELINE_WINDOW = 28


# ---------------------------------------------------------------------------
# Contracts (what goes in, what comes out)
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    sku: str
    warehouse_id: str
    horizon_days: int = Field(7, ge=1, le=90)
    evaluate: bool = Field(True, description="Run a backtest and report the error")


class DailyForecast(BaseModel):
    date: date
    units: float


class ForecastResult(BaseModel):
    sku: str
    warehouse_id: str
    horizon_days: int
    start_date: date
    daily_forecast: list[DailyForecast]
    total_units: float
    avg_daily_units: float
    peak_date: date
    peak_units: float
    recent_avg_daily_units: float      # last 28 days of actual sales, for comparison
    festivals_in_horizon: list[str]
    model: str                          # ensemble_gbm_ridge | moving_average_28d
    wape_pct: float | None = None       # model error in backtest
    baseline_wape_pct: float | None = None
    notes: list[str] = []


class ToolError(BaseModel):
    code: str                           # NOT_FOUND | INSUFFICIENT_DATA | UPSTREAM_DOWN | INVALID
    message: str


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def make_features(dates: pd.DatetimeIndex, festivals: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    X = pd.DataFrame(index=dates)
    X["trend"] = (dates - t0).days
    X["dow"] = dates.dayofweek
    X["dom"] = dates.day
    X["salary_week"] = (dates.day <= 5).astype(int)
    X["month"] = dates.month
    for name in sorted(festivals["festival"].unique()):
        col = name.lower().replace(" ", "_")
        ramp = np.zeros(len(dates))
        after = np.zeros(len(dates))
        for f in festivals[festivals["festival"] == name].itertuples():
            days_before = (f.date - dates).days.values
            window = f.window_days
            in_window = (days_before >= 0) & (days_before <= window)
            ramp = np.maximum(ramp, np.where(in_window, 1 - days_before / (window + 1), 0))
            after = np.maximum(after, ((days_before < 0) & (days_before >= -2)).astype(float))
        X[f"fest_{col}"] = ramp
        X[f"post_{col}"] = after
    return X


def linear_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ridge needs day-of-week and month as on/off columns instead of numbers."""
    L = X.drop(columns=["dow", "month", "dom"]).copy()
    L["trend"] = L["trend"] / 365.0
    for d in range(7):
        L[f"dow_{d}"] = (X["dow"] == d).astype(float)
    for m in range(1, 13):
        L[f"month_{m}"] = (X["month"] == m).astype(float)
    return L


def wape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    total = np.sum(actual)
    return float(100 * np.sum(np.abs(actual - predicted)) / total) if total > 0 else float("nan")


_feature_cache: dict[tuple, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _features_for(history: pd.DataFrame, future_dates: pd.DatetimeIndex, festivals: pd.DataFrame):
    """Most products share the same dates, so build each feature table only once."""
    dates = pd.DatetimeIndex(history["date"])
    key = (dates[0], dates[-1], len(dates), future_dates[0], len(future_dates),
           tuple(festivals["date"].astype(str)))
    if key not in _feature_cache:
        t0 = dates[0]
        _feature_cache[key] = (make_features(dates, festivals, t0), make_features(future_dates, festivals, t0))
    return _feature_cache[key]


def _fit_predict(history: pd.DataFrame, future_dates: pd.DatetimeIndex, festivals: pd.DataFrame,
                 mode: str = "ensemble") -> np.ndarray:
    """mode='ensemble' (accurate, ~0.5s) or 'ridge' (fast, ~0.01s - used for quick screening)."""
    X_train, X_future = _features_for(history, future_dates, festivals)
    y_train = np.log1p(history["units_sold"].values)

    ridge = Ridge(alpha=1.0).fit(linear_features(X_train), y_train)
    ridge_pred = np.expm1(ridge.predict(linear_features(X_future)))
    if mode == "ridge":
        return np.clip(ridge_pred, 0, None)

    # min_samples_leaf is small on purpose: each festival only gives ~8 days a year
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        min_samples_leaf=3, random_state=0)
    gbm.fit(X_train, y_train)

    pred = 0.5 * np.expm1(gbm.predict(X_future)) + 0.5 * ridge_pred
    return np.clip(pred, 0, None)


def _moving_average(history: pd.DataFrame, n_days: int) -> np.ndarray:
    return np.full(n_days, history["units_sold"].tail(BASELINE_WINDOW).mean())


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
def forecast_demand(req: ForecastRequest, erp: ERPAdapter | None = None) -> ForecastResult | ToolError:
    erp = erp or ERPAdapter()
    try:
        history = erp.sales_history(req.sku, req.warehouse_id)
        festivals = erp.festivals()
    except AdapterError as e:
        return ToolError(code=e.code, message=e.message)

    today = pd.Timestamp(settings.business_date)
    history = history[history["date"] <= today]
    notes: list[str] = []

    if len(history) < MIN_HISTORY_DAYS:
        return ToolError(code="INSUFFICIENT_DATA",
                         message=f"Only {len(history)} days of sales for {req.sku} at {req.warehouse_id}; "
                                 f"need at least {MIN_HISTORY_DAYS}.")

    use_ml = len(history) >= ML_HISTORY_DAYS
    if not use_ml:
        notes.append(f"Only {len(history)} days of history, so a moving average was used instead of ML.")

    # --- 1. honest backtest on the most recent days ---
    wape_model = wape_base = None
    if req.evaluate and use_ml:
        test_days = max(req.horizon_days, 14)
        train, test = history.iloc[:-test_days], history.iloc[-test_days:]
        pred = _fit_predict(train, pd.DatetimeIndex(test["date"]), festivals)
        wape_model = round(wape(test["units_sold"].values, pred), 1)
        wape_base = round(wape(test["units_sold"].values, _moving_average(train, test_days)), 1)

    # --- 2. final forecast using all history ---
    future = pd.date_range(today + timedelta(days=1), periods=req.horizon_days, freq="D")
    if use_ml:
        units = _fit_predict(history, future, festivals)
        model_name = "ensemble_gbm_ridge"
    else:
        units = _moving_average(history, req.horizon_days)
        model_name = "moving_average_28d"

    in_horizon = festivals[
        (festivals["date"] >= future[0]) &
        (festivals["date"] - pd.to_timedelta(festivals["window_days"], unit="D") <= future[-1])
    ]
    fest_names = [f"{r.festival} ({r.date.date()})" for r in in_horizon.itertuples()]
    peak_i = int(np.argmax(units))

    return ForecastResult(
        sku=req.sku,
        warehouse_id=req.warehouse_id,
        horizon_days=req.horizon_days,
        start_date=future[0].date(),
        daily_forecast=[DailyForecast(date=d.date(), units=round(float(u), 1)) for d, u in zip(future, units)],
        total_units=round(float(units.sum()), 1),
        avg_daily_units=round(float(units.mean()), 1),
        peak_date=future[peak_i].date(),
        peak_units=round(float(units[peak_i]), 1),
        recent_avg_daily_units=round(float(history["units_sold"].tail(BASELINE_WINDOW).mean()), 1),
        festivals_in_horizon=fest_names,
        model=model_name,
        wape_pct=wape_model,
        baseline_wape_pct=wape_base,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sku = sys.argv[1] if len(sys.argv) > 1 else "DAI-006"
    wh = sys.argv[2] if len(sys.argv) > 2 else "WH-SOUTH"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 28

    result = forecast_demand(ForecastRequest(sku=sku, warehouse_id=wh, horizon_days=horizon),
                             ERPAdapter.in_process())
    if isinstance(result, ToolError):
        print(f"ERROR {result.code}: {result.message}")
        sys.exit(1)

    print(f"\nForecast for {sku} at {wh}, next {horizon} days ({result.model})")
    print(f"  Backtest error (WAPE): model {result.wape_pct}%  vs  baseline {result.baseline_wape_pct}%")
    print(f"  Recent actual avg: {result.recent_avg_daily_units}/day   Forecast avg: {result.avg_daily_units}/day")
    print(f"  Total forecast: {result.total_units:.0f} units   Peak: {result.peak_units:.0f} on {result.peak_date}")
    print(f"  Festivals in horizon: {', '.join(result.festivals_in_horizon) or 'none'}\n")
    top = max(d.units for d in result.daily_forecast)
    for d in result.daily_forecast:
        bar = "#" * int(40 * d.units / top) if top else ""
        print(f"  {d.date}  {d.units:7.1f}  {bar}")