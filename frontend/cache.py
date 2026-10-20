"""Cached API calls, so moving between pages doesn't re-fetch everything."""

import streamlit as st

from frontend import api


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict | None:
    try:
        return api.health()
    except api.ApiError:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def products() -> list[dict]:
    return api.products()


@st.cache_data(ttl=600, show_spinner=False)
def suppliers() -> list[dict]:
    return api.suppliers()


@st.cache_data(ttl=10, show_spinner=False)
def recommendations(status: str = "all") -> list[dict]:
    return api.approvals(status)


@st.cache_data(ttl=180, show_spinner=False)
def dashboard(warehouse: str | None, days: int) -> dict:
    return api.dashboard(warehouse, days)


@st.cache_data(ttl=600, show_spinner=False)
def forecast(product: str, warehouse: str, days: int) -> dict:
    return api.forecast(product, warehouse, days)


@st.cache_data(ttl=10, show_spinner=False)
def purchase_orders() -> list[dict]:
    return api.purchase_orders()


def after_decision() -> None:
    """An approval changes stock in transit, pending counts and purchase orders."""
    recommendations.clear()
    dashboard.clear()
    purchase_orders.clear()


def pending_count() -> int:
    try:
        return sum(r["status"] == "pending" for r in recommendations("all"))
    except api.ApiError:
        return 0