"""
BasketBee - Mock ERP (M2).

Imitates an enterprise resource planning system:
    products, suppliers, sales orders (history) and purchase orders.

Safety rule: a purchase order can ONLY be created from a recommendation
that a human has approved (human-in-the-loop).
"""

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.common import api_error, check_available
from backend.api.schemas import (ProductOut, PurchaseOrderIn, PurchaseOrderOut,
                                 SalesRecord, SupplierOut)
from backend.db.database import get_db
from backend.db.models import (Inventory, Product, PurchaseOrder, Recommendation,
                               SalesHistory, Supplier, Warehouse)

router = APIRouter(prefix="/erp", tags=["ERP (mock)"], dependencies=[Depends(check_available("erp"))])


# ----- Products --------------------------------------------------------------
@router.get("/products", response_model=list[ProductOut])
def list_products(category: str | None = None, db: Session = Depends(get_db)):
    q = select(Product).order_by(Product.sku)
    if category:
        q = q.where(Product.category == category)
    return db.scalars(q).all()


@router.get("/products/{sku}", response_model=ProductOut)
def get_product(sku: str, db: Session = Depends(get_db)):
    product = db.get(Product, sku)
    if not product:
        raise api_error(404, "NOT_FOUND", f"Product {sku} does not exist")
    return product


# ----- Suppliers -------------------------------------------------------------
@router.get("/suppliers", response_model=list[SupplierOut])
def list_suppliers(category: str | None = None, db: Session = Depends(get_db)):
    q = select(Supplier).order_by(Supplier.supplier_id)
    if category:
        q = q.where(Supplier.category == category)
    return db.scalars(q).all()


@router.get("/suppliers/{supplier_id}", response_model=SupplierOut)
def get_supplier(supplier_id: str, db: Session = Depends(get_db)):
    supplier = db.get(Supplier, supplier_id)
    if not supplier:
        raise api_error(404, "NOT_FOUND", f"Supplier {supplier_id} does not exist")
    return supplier


# ----- Sales orders (history) -------------------------------------------------
@router.get("/orders", response_model=list[SalesRecord])
def sales_history(
    sku: str | None = None,
    warehouse_id: str | None = None,
    date_from: date | None = Query(None, alias="from"),
    date_to: date | None = Query(None, alias="to"),
    limit: int = Query(5000, ge=1, le=50000),
    db: Session = Depends(get_db),
):
    """Daily sales. The Forecast agent calls this with one sku + warehouse."""
    if sku and not db.get(Product, sku):
        raise api_error(404, "NOT_FOUND", f"Product {sku} does not exist")
    if warehouse_id and not db.get(Warehouse, warehouse_id):
        raise api_error(404, "NOT_FOUND", f"Warehouse {warehouse_id} does not exist")
    if date_from and date_to and date_from > date_to:
        raise api_error(400, "INVALID", "'from' must be before 'to'")

    q = select(SalesHistory)
    if sku:
        q = q.where(SalesHistory.sku == sku)
    if warehouse_id:
        q = q.where(SalesHistory.warehouse_id == warehouse_id)
    if date_from:
        q = q.where(SalesHistory.date >= date_from)
    if date_to:
        q = q.where(SalesHistory.date <= date_to)
    q = q.order_by(SalesHistory.date).limit(limit)
    return db.scalars(q).all()


# ----- Purchase orders -------------------------------------------------------
@router.get("/purchase-orders", response_model=list[PurchaseOrderOut])
def list_purchase_orders(db: Session = Depends(get_db)):
    return db.scalars(select(PurchaseOrder).order_by(PurchaseOrder.po_id.desc())).all()


@router.post("/purchase-orders", response_model=PurchaseOrderOut, status_code=201)
def create_purchase_order(body: PurchaseOrderIn, db: Session = Depends(get_db)):
    rec = db.get(Recommendation, body.recommendation_id)
    if not rec:
        raise api_error(404, "NOT_FOUND", f"Recommendation {body.recommendation_id} does not exist")
    if rec.status != "approved":
        raise api_error(403, "NOT_APPROVED",
                        f"Recommendation {rec.rec_id} is '{rec.status}'. Only approved recommendations can become POs.")
    if not db.get(Supplier, body.supplier_id):
        raise api_error(404, "NOT_FOUND", f"Supplier {body.supplier_id} does not exist")
    stock = db.get(Inventory, (body.warehouse_id, body.sku))
    if not stock:
        raise api_error(404, "NOT_FOUND", f"{body.sku} is not stocked at {body.warehouse_id}")

    po = PurchaseOrder(**body.model_dump(), status="created")
    db.add(po)
    stock.in_transit += body.qty          # incoming stock is now on its way
    db.commit()
    db.refresh(po)
    return po