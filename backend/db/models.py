"""
BasketBee - database tables (M1 Data layer).

Master data      : Warehouse, DarkStore, Supplier, Product, Festival, Vehicle
Operational data : SalesHistory, Inventory
Agent workflow   : PurchaseOrder, Shipment, Recommendation, AuditLog
                   (empty at first - filled by the agents as they work)
"""

from datetime import date as  DateType, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


# ---------------------------------------------------------------------------
# Master data
# ---------------------------------------------------------------------------
class Warehouse(Base):
    __tablename__ = "warehouses"

    warehouse_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    area: Mapped[str] = mapped_column(String(100))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    capacity_units: Mapped[int] = mapped_column(Integer)
    demand_share: Mapped[float] = mapped_column(Float)

    dark_stores: Mapped[list["DarkStore"]] = relationship(back_populates="warehouse")


class DarkStore(Base):
    __tablename__ = "dark_stores"

    store_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    area: Mapped[str] = mapped_column(String(100))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    served_by_warehouse: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"))

    warehouse: Mapped["Warehouse"] = relationship(back_populates="dark_stores")


class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50))
    lead_time_days: Mapped[int] = mapped_column(Integer)
    on_time_rate: Mapped[float] = mapped_column(Float)
    defect_rate: Mapped[float] = mapped_column(Float)

    products: Mapped[list["Product"]] = relationship(back_populates="supplier")


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(50), index=True)
    pack_size: Mapped[str] = mapped_column(String(30))
    unit_cost_inr: Mapped[float] = mapped_column(Float)
    selling_price_inr: Mapped[float] = mapped_column(Float)
    shelf_life_days: Mapped[int] = mapped_column(Integer)
    base_daily_demand: Mapped[float] = mapped_column(Float)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"))
    holding_cost_pct_per_year: Mapped[float] = mapped_column(Float)
    order_cost_inr: Mapped[float] = mapped_column(Float)
    case_pack: Mapped[int] = mapped_column(Integer)

    supplier: Mapped["Supplier"] = relationship(back_populates="products")


class Festival(Base):
    __tablename__ = "festivals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    festival: Mapped[str] = mapped_column(String(50))
    date: Mapped[DateType] = mapped_column(Date)
    window_days: Mapped[int] = mapped_column(Integer)


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"))
    type: Mapped[str] = mapped_column(String(50))
    capacity_kg: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))      # available | maintenance | on_route


# ---------------------------------------------------------------------------
# Operational data
# ---------------------------------------------------------------------------
class SalesHistory(Base):
    __tablename__ = "sales_history"

    date: Mapped[DateType] = mapped_column(Date, primary_key=True)
    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"), primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), primary_key=True, index=True)
    units_sold: Mapped[int] = mapped_column(Integer)
    revenue_inr: Mapped[float] = mapped_column(Float)


class Inventory(Base):
    __tablename__ = "inventory"

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"), primary_key=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"), primary_key=True)
    on_hand: Mapped[int] = mapped_column(Integer)
    reserved: Mapped[int] = mapped_column(Integer)
    in_transit: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    @property
    def available(self) -> int:
        """Stock that can actually be sold right now."""
        return self.on_hand - self.reserved


# ---------------------------------------------------------------------------
# Agent workflow tables (start empty)
# ---------------------------------------------------------------------------
class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    po_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    supplier_id: Mapped[str] = mapped_column(ForeignKey("suppliers.supplier_id"))
    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"))
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"))
    qty: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="created")   # created | sent | received
    approved_by: Mapped[str | None] = mapped_column(String(100))
    recommendation_id: Mapped[int | None] = mapped_column(ForeignKey("recommendations.rec_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    origin_warehouse: Mapped[str] = mapped_column(ForeignKey("warehouses.warehouse_id"))
    destination_store: Mapped[str] = mapped_column(ForeignKey("dark_stores.store_id"))
    vehicle_id: Mapped[str | None] = mapped_column(ForeignKey("vehicles.vehicle_id"))
    status: Mapped[str] = mapped_column(String(20), default="planned")   # planned | in_transit | delivered | delayed
    eta: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Recommendation(Base):
    """Anything an agent proposes that a human must approve (e.g. a reorder)."""
    __tablename__ = "recommendations"

    rec_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50))                  # reorder | transfer | reroute
    payload_json: Mapped[str] = mapped_column(Text)                # details as JSON
    status: Mapped[str] = mapped_column(String(20), default="pending")   # pending | approved | rejected
    created_by_agent: Mapped[str] = mapped_column(String(50))
    decided_by: Mapped[str | None] = mapped_column(String(100))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AuditLog(Base):
    """Every agent step and tool call is recorded here."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    agent: Mapped[str] = mapped_column(String(50))
    tool: Mapped[str | None] = mapped_column(String(50))
    input_json: Mapped[str | None] = mapped_column(Text)
    output_json: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)