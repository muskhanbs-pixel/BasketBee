"""
BasketBee - API contracts (Pydantic schemas) for the mock ERP / WMS / TMS.

Every response the agents receive has a fixed, typed shape defined here.
If a field changes, it changes in ONE place - that's the "contracts
everywhere" principle from the architecture doc.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    """Lets Pydantic read straight from SQLAlchemy objects."""
    model_config = ConfigDict(from_attributes=True)


# ----- Errors ---------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str          # NOT_FOUND | NOT_APPROVED | INVALID | UPSTREAM_DOWN
    message: str


# ----- ERP ------------------------------------------------------------------
class ProductOut(ORMModel):
    sku: str
    name: str
    category: str
    pack_size: str
    unit_cost_inr: float
    selling_price_inr: float
    shelf_life_days: int
    supplier_id: str
    case_pack: int
    order_cost_inr: float
    holding_cost_pct_per_year: float


class SupplierOut(ORMModel):
    supplier_id: str
    name: str
    category: str
    lead_time_days: int
    on_time_rate: float
    defect_rate: float


class SalesRecord(ORMModel):
    date: date
    warehouse_id: str
    sku: str
    units_sold: int
    revenue_inr: float


class PurchaseOrderIn(BaseModel):
    recommendation_id: int = Field(..., description="Must point to an APPROVED recommendation")
    supplier_id: str
    warehouse_id: str
    sku: str
    qty: int = Field(..., gt=0)
    approved_by: str


class PurchaseOrderOut(ORMModel):
    po_id: int
    recommendation_id: int | None
    supplier_id: str
    warehouse_id: str
    sku: str
    qty: int
    status: str
    approved_by: str | None
    created_at: datetime


# ----- WMS ------------------------------------------------------------------
class WarehouseOut(ORMModel):
    warehouse_id: str
    name: str
    area: str
    lat: float
    lon: float
    capacity_units: int


class WarehouseUtilization(WarehouseOut):
    units_stored: int
    utilization_pct: float


class DarkStoreOut(ORMModel):
    store_id: str
    area: str
    lat: float
    lon: float
    served_by_warehouse: str


class StockOut(ORMModel):
    warehouse_id: str
    sku: str
    on_hand: int
    reserved: int
    in_transit: int
    available: int
    updated_at: datetime


# ----- TMS ------------------------------------------------------------------
class VehicleOut(ORMModel):
    vehicle_id: str
    warehouse_id: str
    type: str
    capacity_kg: int
    status: str


class ShipmentIn(BaseModel):
    origin_warehouse: str
    destination_store: str
    vehicle_id: str | None = None
    eta: datetime | None = None


class ShipmentOut(ORMModel):
    shipment_id: int
    origin_warehouse: str
    destination_store: str
    vehicle_id: str | None
    status: str
    eta: datetime | None
    created_at: datetime