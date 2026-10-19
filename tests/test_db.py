"""
BasketBee - database tests.

Run from the project root:
    python -m pytest tests/test_db.py -v

These check that the seed loaded correctly and that the relationships work.
They read the real database, so run `python -m backend.db.seed` first.
"""

from sqlalchemy import func, select

from backend.db.database import SessionLocal
from backend.db.models import (DarkStore, Inventory, Product, SalesHistory,
                               Supplier, Warehouse)


def test_row_counts():
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Warehouse)) == 3
        assert db.scalar(select(func.count()).select_from(DarkStore)) == 15
        assert db.scalar(select(func.count()).select_from(Product)) == 60
        assert db.scalar(select(func.count()).select_from(Inventory)) == 180  # 60 SKUs x 3 warehouses
        assert db.scalar(select(func.count()).select_from(SalesHistory)) > 100_000


def test_relationships():
    with SessionLocal() as db:
        south = db.get(Warehouse, "WH-SOUTH")
        assert len(south.dark_stores) == 6
        milk = db.get(Product, "DAI-001")
        assert milk.supplier.name == "Kaveri Fresh Dairy"


def test_every_product_has_a_supplier():
    with SessionLocal() as db:
        orphans = db.scalars(select(Product).where(
            ~Product.supplier_id.in_(select(Supplier.supplier_id)))).all()
        assert orphans == []


def test_stock_is_never_negative():
    with SessionLocal() as db:
        bad = db.scalars(select(Inventory).where(Inventory.on_hand < 0)).all()
        assert bad == []


def test_demo_stockout_risk_exists():
    """The Diwali demo needs some products with very low stock."""
    with SessionLocal() as db:
        ghee = db.get(Inventory, ("WH-SOUTH", "DAI-006"))
        avg = db.scalar(select(func.avg(SalesHistory.units_sold)).where(
            SalesHistory.sku == "DAI-006", SalesHistory.warehouse_id == "WH-SOUTH"))
        assert ghee.on_hand < avg * 3   # less than ~3 days of cover