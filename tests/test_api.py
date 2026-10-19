"""
BasketBee - tests for the mock ERP / WMS / TMS APIs.

Run from the project root:
    python -m pytest tests/test_api.py -v

The tests work on a temporary COPY of basketbee.db, so your real
database is never changed. Run `python -m backend.db.seed` first.
"""

import shutil

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.common import _outages
from backend.core.config import PROJECT_ROOT
from backend.db.database import get_db
from backend.db.models import Recommendation
from backend.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    real_db = PROJECT_ROOT / "basketbee.db"
    if not real_db.exists():
        pytest.skip("Run `python -m backend.db.seed` first")
    test_db = tmp_path_factory.mktemp("db") / "test.db"
    shutil.copy(real_db, test_db)

    engine = create_engine(f"sqlite:///{test_db.as_posix()}", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine, autoflush=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        c.TestSession = TestSession      # handy for inserting test rows
        yield c
    app.dependency_overrides.clear()
    engine.dispose()


# ----- Health ----------------------------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["api"] == "ok"


# ----- ERP -------------------------------------------------------------------
def test_list_products(client):
    r = client.get("/erp/products")
    assert r.status_code == 200
    assert len(r.json()) == 60


def test_filter_products_by_category(client):
    r = client.get("/erp/products", params={"category": "Dairy"})
    assert {p["category"] for p in r.json()} == {"Dairy"}


def test_unknown_product_returns_404(client):
    r = client.get("/erp/products/XYZ-999")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "NOT_FOUND"


def test_sales_history_for_one_product(client):
    r = client.get("/erp/orders", params={"sku": "DAI-006", "warehouse_id": "WH-SOUTH",
                                          "from": "2026-10-01", "to": "2026-10-18"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 18
    assert all(row["sku"] == "DAI-006" for row in rows)


def test_bad_date_range(client):
    r = client.get("/erp/orders", params={"from": "2026-10-18", "to": "2026-10-01"})
    assert r.status_code == 400


# ----- WMS -------------------------------------------------------------------
def test_warehouse_utilization(client):
    r = client.get("/wms/warehouses")
    assert len(r.json()) == 3
    assert all(0 <= w["utilization_pct"] for w in r.json())


def test_stock_for_one_item(client):
    r = client.get("/wms/stock", params={"sku": "DAI-006", "warehouse_id": "WH-SOUTH"})
    item = r.json()[0]
    assert item["available"] == item["on_hand"] - item["reserved"]


def test_dark_stores_by_warehouse(client):
    r = client.get("/wms/dark-stores", params={"warehouse_id": "WH-EAST"})
    assert len(r.json()) == 5


# ----- Purchase orders: human-in-the-loop rule -------------------------------
def _add_recommendation(client, status):
    with client.TestSession() as db:
        rec = Recommendation(type="reorder", payload_json="{}", status=status,
                             created_by_agent="inventory_agent")
        db.add(rec)
        db.commit()
        return rec.rec_id


def test_po_rejected_without_approval(client):
    rec_id = _add_recommendation(client, "pending")
    r = client.post("/erp/purchase-orders", json={
        "recommendation_id": rec_id, "supplier_id": "SUP-01", "warehouse_id": "WH-SOUTH",
        "sku": "DAI-006", "qty": 100, "approved_by": "nobody"})
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "NOT_APPROVED"


def test_po_created_after_approval_and_stock_in_transit(client):
    before = client.get("/wms/stock", params={"sku": "DAI-006", "warehouse_id": "WH-SOUTH"}).json()[0]
    rec_id = _add_recommendation(client, "approved")
    r = client.post("/erp/purchase-orders", json={
        "recommendation_id": rec_id, "supplier_id": "SUP-01", "warehouse_id": "WH-SOUTH",
        "sku": "DAI-006", "qty": 100, "approved_by": "manager"})
    assert r.status_code == 201
    after = client.get("/wms/stock", params={"sku": "DAI-006", "warehouse_id": "WH-SOUTH"}).json()[0]
    assert after["in_transit"] == before["in_transit"] + 100


# ----- TMS -------------------------------------------------------------------
def test_create_shipment_with_vehicle(client):
    vehicle = client.get("/tms/vehicles", params={"warehouse_id": "WH-SOUTH",
                                                  "status": "available"}).json()[0]
    r = client.post("/tms/shipments", json={"origin_warehouse": "WH-SOUTH",
                                             "destination_store": "DS-01",
                                             "vehicle_id": vehicle["vehicle_id"]})
    assert r.status_code == 201
    # the same vehicle can't be booked twice
    r2 = client.post("/tms/shipments", json={"origin_warehouse": "WH-SOUTH",
                                              "destination_store": "DS-02",
                                              "vehicle_id": vehicle["vehicle_id"]})
    assert r2.status_code == 409


# ----- Outage simulator ------------------------------------------------------
def test_outage_returns_503(client):
    client.post("/admin/outages", json={"system": "erp", "down": True})
    try:
        r = client.get("/erp/products")
        assert r.status_code == 503
        assert r.json()["detail"]["code"] == "UPSTREAM_DOWN"
        assert client.get("/wms/stock").status_code == 200   # other systems still work
    finally:
        client.post("/admin/outages", json={"system": "erp", "down": False})
    assert _outages["erp"] is False