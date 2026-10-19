"""
BasketBee - Mock WMS (M2).

Imitates a warehouse management system:
    warehouses (with utilization), dark stores and stock levels.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.api.common import api_error, check_available
from backend.api.schemas import DarkStoreOut, StockOut, WarehouseOut, WarehouseUtilization
from backend.db.database import get_db
from backend.db.models import DarkStore, Inventory, Product, Warehouse

router = APIRouter(prefix="/wms", tags=["WMS (mock)"], dependencies=[Depends(check_available("wms"))])


@router.get("/warehouses", response_model=list[WarehouseUtilization])
def list_warehouses(db: Session = Depends(get_db)):
    stored = dict(db.execute(
        select(Inventory.warehouse_id, func.sum(Inventory.on_hand)).group_by(Inventory.warehouse_id)
    ).all())
    result = []
    for wh in db.scalars(select(Warehouse).order_by(Warehouse.warehouse_id)):
        units = int(stored.get(wh.warehouse_id, 0))
        result.append(WarehouseUtilization(
            **WarehouseOut.model_validate(wh).model_dump(),
            units_stored=units,
            utilization_pct=round(100 * units / wh.capacity_units, 1),
        ))
    return result


@router.get("/dark-stores", response_model=list[DarkStoreOut])
def list_dark_stores(warehouse_id: str | None = None, db: Session = Depends(get_db)):
    q = select(DarkStore).order_by(DarkStore.store_id)
    if warehouse_id:
        q = q.where(DarkStore.served_by_warehouse == warehouse_id)
    return db.scalars(q).all()


@router.get("/stock", response_model=list[StockOut])
def get_stock(sku: str | None = None, warehouse_id: str | None = None, db: Session = Depends(get_db)):
    """Stock levels. Filter by sku, warehouse, both, or neither (everything)."""
    if sku and not db.get(Product, sku):
        raise api_error(404, "NOT_FOUND", f"Product {sku} does not exist")
    if warehouse_id and not db.get(Warehouse, warehouse_id):
        raise api_error(404, "NOT_FOUND", f"Warehouse {warehouse_id} does not exist")

    q = select(Inventory).order_by(Inventory.warehouse_id, Inventory.sku)
    if sku:
        q = q.where(Inventory.sku == sku)
    if warehouse_id:
        q = q.where(Inventory.warehouse_id == warehouse_id)
    return db.scalars(q).all()