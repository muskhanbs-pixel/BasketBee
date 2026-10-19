"""
BasketBee - tests for the Inventory Optimization tool (M4).

Run from the project root:
    python -m pytest tests/test_inventory.py -v
"""

import numpy as np
import pytest

from backend.api.common import _outages
from backend.tools.adapters import ERPAdapter
from backend.tools.forecasting import ToolError
from backend.tools.inventory import (InventoryCheckRequest, InventoryReport, check_inventory,
                                     days_until_reorder, eoq, round_up, safety_stock,
                                     simulate_stock, z_score)


# ----- Pure maths ------------------------------------------------------------
def test_z_score():
    assert z_score(0.5) == pytest.approx(0, abs=1e-9)
    assert z_score(0.95) == pytest.approx(1.645, abs=0.001)


def test_safety_stock_grows_with_lead_time():
    assert safety_stock(10, 4, 0.95) == pytest.approx(2 * safety_stock(10, 1, 0.95))


def test_eoq_textbook():
    # D=1000/yr, S=50, unit cost 10, holding 20% -> EOQ = sqrt(2*1000*50/2) = 223.6
    assert eoq(1000, 50, 10, 0.20) == pytest.approx(223.6, abs=0.1)
    assert eoq(0, 50, 10, 0.2) == 0


def test_simulate_stock_runs_out():
    day, cover = simulate_stock(available=25, in_transit=0, daily_forecast=np.array([10, 10, 10, 10]))
    assert day == 3 and cover == 2.5


def test_simulate_stock_in_transit_saves_the_day():
    day, _ = simulate_stock(available=15, in_transit=100, daily_forecast=np.array([10, 10, 10, 10]),
                            arrival_day=2)
    assert day is None


def test_days_until_reorder():
    assert days_until_reorder(50, np.array([10] * 10), rop=60) == 0
    assert days_until_reorder(100, np.array([10] * 10), rop=75) == 3
    assert days_until_reorder(1000, np.array([10] * 10), rop=10) is None


def test_round_up_to_case_pack():
    assert round_up(25, 12) == 36
    assert round_up(0, 12) == 0


# ----- The tool on real seed data --------------------------------------------
@pytest.fixture(scope="module")
def south_report():
    r = check_inventory(InventoryCheckRequest(warehouse_id="WH-SOUTH", include_ok=True),
                        ERPAdapter.in_process())
    assert isinstance(r, InventoryReport)
    return r


def _item(report, sku):
    return next(i for i in report.items if i.sku == sku)


def test_planted_diwali_risks_are_critical(south_report):
    # the data generator deliberately left ghee and sunflower oil almost empty
    for sku in ("DAI-006", "STP-008"):
        item = _item(south_report, sku)
        assert item.risk == "CRITICAL"
        assert item.recommended_order_qty > 0


def test_critical_items_listed_first(south_report):
    order = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "OK": 3}
    ranks = [order[i.risk] for i in south_report.items]
    assert ranks == sorted(ranks)


def test_orders_are_full_case_packs(south_report):
    products = ERPAdapter.in_process().products().set_index("sku")
    for i in south_report.items:
        assert i.recommended_order_qty % products.loc[i.sku, "case_pack"] == 0


def test_ok_items_order_nothing(south_report):
    assert all(i.recommended_order_qty == 0 for i in south_report.items if i.risk == "OK")


def test_watch_items_get_a_planned_order(south_report):
    watch = [i for i in south_report.items if i.risk == "WATCH"]
    assert watch and all(i.recommended_order_qty > 0 and i.reorder_by for i in watch)


def test_perishables_capped_by_shelf_life(south_report):
    milk = _item(south_report, "DAI-001")   # 3-day shelf life
    assert milk.recommended_order_qty < milk.avg_daily_forecast * 4


def test_ok_hidden_by_default():
    r = check_inventory(InventoryCheckRequest(warehouse_id="WH-SOUTH", skus=["DAI-006", "PRC-001"]),
                        ERPAdapter.in_process())
    assert all(i.risk != "OK" for i in r.items)


# ----- Failure handling ------------------------------------------------------
def test_unknown_warehouse():
    r = check_inventory(InventoryCheckRequest(warehouse_id="WH-MARS"), ERPAdapter.in_process())
    assert isinstance(r, ToolError) and r.code == "NOT_FOUND"


def test_unknown_sku():
    r = check_inventory(InventoryCheckRequest(skus=["NOPE-1"]), ERPAdapter.in_process())
    assert isinstance(r, ToolError) and r.code == "NOT_FOUND"


def test_wms_outage():
    _outages["wms"] = True
    try:
        r = check_inventory(InventoryCheckRequest(warehouse_id="WH-SOUTH"), ERPAdapter.in_process())
        assert isinstance(r, ToolError) and r.code == "UPSTREAM_DOWN"
    finally:
        _outages["wms"] = False