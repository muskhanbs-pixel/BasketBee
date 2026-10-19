"""
BasketBee - tests for the Demand Forecasting tool (M3).

Run from the project root:
    python -m pytest tests/test_forecasting.py -v
"""

import pandas as pd
import pytest

from backend.api.common import _outages
from backend.tools.adapters import ERPAdapter
from backend.tools.forecasting import (ForecastRequest, ForecastResult, ToolError,
                                       forecast_demand, wape)


@pytest.fixture(scope="module")
def erp():
    return ERPAdapter.in_process()


def test_forecast_shape(erp):
    r = forecast_demand(ForecastRequest(sku="DAI-001", warehouse_id="WH-EAST", horizon_days=7), erp)
    assert isinstance(r, ForecastResult)
    assert len(r.daily_forecast) == 7
    assert r.total_units > 0
    assert r.model == "ensemble_gbm_ridge"
    assert r.wape_pct is not None and r.baseline_wape_pct is not None


def test_detects_upcoming_diwali(erp):
    r = forecast_demand(ForecastRequest(sku="HHC-007", warehouse_id="WH-NORTH", horizon_days=28), erp)
    assert any("Diwali" in f for f in r.festivals_in_horizon)
    # diya lamps should peak around Diwali (2026-11-08) and far above normal
    assert abs((r.peak_date - pd.Timestamp("2026-11-08").date()).days) <= 3
    assert r.peak_units > 3 * r.recent_avg_daily_units


def test_no_festival_in_short_horizon(erp):
    r = forecast_demand(ForecastRequest(sku="DAI-001", warehouse_id="WH-EAST", horizon_days=5,
                                        evaluate=False), erp)
    assert r.festivals_in_horizon == []
    assert r.wape_pct is None          # evaluate=False skips the backtest


def test_unknown_product(erp):
    r = forecast_demand(ForecastRequest(sku="NOPE-1", warehouse_id="WH-EAST"), erp)
    assert isinstance(r, ToolError) and r.code == "NOT_FOUND"


def test_erp_outage(erp):
    _outages["erp"] = True
    try:
        r = forecast_demand(ForecastRequest(sku="DAI-001", warehouse_id="WH-EAST"), erp)
        assert isinstance(r, ToolError) and r.code == "UPSTREAM_DOWN"
    finally:
        _outages["erp"] = False


class TinyHistoryERP:
    """A fake ERP that only has 10 days of sales - to test the safety check."""
    def sales_history(self, sku, wh):
        return pd.DataFrame({"date": pd.date_range("2026-10-09", periods=10), "units_sold": [5] * 10})

    def festivals(self):
        return pd.DataFrame({"festival": ["Diwali"], "date": [pd.Timestamp("2026-11-08")], "window_days": [7]})


def test_insufficient_data():
    r = forecast_demand(ForecastRequest(sku="NEW-001", warehouse_id="WH-EAST"), TinyHistoryERP())
    assert isinstance(r, ToolError) and r.code == "INSUFFICIENT_DATA"


def test_horizon_limits():
    with pytest.raises(ValueError):
        ForecastRequest(sku="DAI-001", warehouse_id="WH-EAST", horizon_days=0)
    with pytest.raises(ValueError):
        ForecastRequest(sku="DAI-001", warehouse_id="WH-EAST", horizon_days=365)


def test_wape():
    assert wape([10, 10], [10, 10]) == 0
    assert wape([10, 10], [5, 15]) == 50