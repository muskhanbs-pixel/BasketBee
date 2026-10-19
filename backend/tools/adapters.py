"""
BasketBee - adapters between the tools and the enterprise systems.

Tools NEVER call URLs directly. They go through these adapter classes, so:
  * swapping the mock ERP for a real one (SAP, Odoo...) only changes this file
  * every HTTP problem becomes one clean AdapterError with a known code

Two ways to create an adapter:
    ERPAdapter()               -> talks to the running API server over HTTP
    ERPAdapter.in_process()    -> calls the API inside this Python process
                                  (no server needed - used by tests & scripts)
"""

import httpx
import pandas as pd

from backend.core.config import settings


class AdapterError(Exception):
    """Raised when an enterprise system can't give us what we asked for."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code          # NOT_FOUND | UPSTREAM_DOWN | INVALID
        self.message = message


class BaseAdapter:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=settings.api_base_url, timeout=15)

    @classmethod
    def in_process(cls):
        """Use the FastAPI app directly, without starting uvicorn."""
        from fastapi.testclient import TestClient   # imported lazily on purpose
        from backend.main import app
        return cls(TestClient(app))

    def _get(self, path: str, **params) -> list | dict:
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = self.client.get(path, params=params)
        except httpx.HTTPError:
            raise AdapterError("UPSTREAM_DOWN",
                               f"Could not reach BasketBee API at {settings.api_base_url}. Is uvicorn running?")
        if r.status_code >= 400:
            detail = r.json().get("detail", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
            code = detail.get("code", "UPSTREAM_DOWN" if r.status_code >= 500 else "INVALID") if isinstance(detail, dict) else "INVALID"
            message = detail.get("message", r.text) if isinstance(detail, dict) else str(detail)
            raise AdapterError(code, message)
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self.client.post(path, json=body)
        except httpx.HTTPError:
            raise AdapterError("UPSTREAM_DOWN",
                               f"Could not reach BasketBee API at {settings.api_base_url}. Is uvicorn running?")
        if r.status_code >= 400:
            detail = r.json().get("detail", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
            if isinstance(detail, dict):
                raise AdapterError(detail.get("code", "INVALID"), detail.get("message", r.text))
            raise AdapterError("INVALID", str(detail))
        return r.json()


class ERPAdapter(BaseAdapter):
    def create_purchase_order(self, recommendation_id: int, supplier_id: str, warehouse_id: str,
                              sku: str, qty: int, approved_by: str) -> dict:
        return self._post("/erp/purchase-orders", {
            "recommendation_id": recommendation_id, "supplier_id": supplier_id,
            "warehouse_id": warehouse_id, "sku": sku, "qty": qty, "approved_by": approved_by,
        })

    def product(self, sku: str) -> dict:
        return self._get(f"/erp/products/{sku}")

    def products(self, category: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(self._get("/erp/products", category=category))

    def suppliers(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("/erp/suppliers"))

    def warehouse_sales(self, warehouse_id: str) -> pd.DataFrame:
        """All products' daily sales for one warehouse in a single call (fast bulk load)."""
        df = pd.DataFrame(self._get("/erp/orders", warehouse_id=warehouse_id, limit=50000))
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "sku", "units_sold"]]

    def sales_history(self, sku: str, warehouse_id: str) -> pd.DataFrame:
        rows = self._get("/erp/orders", sku=sku, warehouse_id=warehouse_id, limit=50000)
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "units_sold"])
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "units_sold"]].sort_values("date").reset_index(drop=True)

    def festivals(self) -> pd.DataFrame:
        df = pd.DataFrame(self._get("/erp/festivals"))
        df["date"] = pd.to_datetime(df["date"])
        return df


class WMSAdapter(BaseAdapter):
    def stock(self, sku: str | None = None, warehouse_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(self._get("/wms/stock", sku=sku, warehouse_id=warehouse_id))

    def warehouses(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("/wms/warehouses"))