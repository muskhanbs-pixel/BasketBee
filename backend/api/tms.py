"""
BasketBee - Mock TMS (M2).

Imitates a transportation management system:
    delivery vehicles and shipments from warehouses to dark stores.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.common import api_error, check_available
from backend.api.schemas import ShipmentIn, ShipmentOut, VehicleOut
from backend.db.database import get_db
from backend.db.models import DarkStore, Shipment, Vehicle, Warehouse

router = APIRouter(prefix="/tms", tags=["TMS (mock)"], dependencies=[Depends(check_available("tms"))])


@router.get("/vehicles", response_model=list[VehicleOut])
def list_vehicles(warehouse_id: str | None = None, status: str | None = None,
                  db: Session = Depends(get_db)):
    q = select(Vehicle).order_by(Vehicle.vehicle_id)
    if warehouse_id:
        q = q.where(Vehicle.warehouse_id == warehouse_id)
    if status:
        q = q.where(Vehicle.status == status)
    return db.scalars(q).all()


@router.get("/shipments", response_model=list[ShipmentOut])
def list_shipments(status: str | None = None, db: Session = Depends(get_db)):
    q = select(Shipment).order_by(Shipment.shipment_id.desc())
    if status:
        q = q.where(Shipment.status == status)
    return db.scalars(q).all()


@router.post("/shipments", response_model=ShipmentOut, status_code=201)
def create_shipment(body: ShipmentIn, db: Session = Depends(get_db)):
    if not db.get(Warehouse, body.origin_warehouse):
        raise api_error(404, "NOT_FOUND", f"Warehouse {body.origin_warehouse} does not exist")
    store = db.get(DarkStore, body.destination_store)
    if not store:
        raise api_error(404, "NOT_FOUND", f"Dark store {body.destination_store} does not exist")

    if body.vehicle_id:
        vehicle = db.get(Vehicle, body.vehicle_id)
        if not vehicle:
            raise api_error(404, "NOT_FOUND", f"Vehicle {body.vehicle_id} does not exist")
        if vehicle.warehouse_id != body.origin_warehouse:
            raise api_error(400, "INVALID", f"{vehicle.vehicle_id} belongs to {vehicle.warehouse_id}")
        if vehicle.status != "available":
            raise api_error(409, "INVALID", f"{vehicle.vehicle_id} is '{vehicle.status}', not available")
        vehicle.status = "on_route"

    shipment = Shipment(**body.model_dump(), status="planned")
    db.add(shipment)
    db.commit()
    db.refresh(shipment)
    return shipment