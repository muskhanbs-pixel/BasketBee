"""Approvals - nothing is ordered until a person approves it."""

import pandas as pd
import streamlit as st

from frontend import api, cache, theme

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "OK": 3}

theme.header("Approvals", "Nothing is ordered until you approve it. Approving sends a purchase order "
                          "to the supplier; rejecting keeps your stock plan unchanged.")

try:
    recs = cache.recommendations("all")
    pos = cache.purchase_orders()
except api.ApiError as e:
    st.error(e.message, icon=":material/cloud_off:")
    st.stop()

pending = sorted([r for r in recs if r["status"] == "pending"], key=lambda r: (RISK_ORDER[r["risk"]], r["id"]))
theme.kpi_band([
    (str(len(pending)), "Reorders waiting", theme.GREEN),
    (theme.inr(sum(r["value_inr"] for r in pending)), "Value waiting", theme.GREEN),
    (str(sum(r["risk"] == "CRITICAL" for r in pending)), "Run out before restock", theme.RISK["CRITICAL"][0]),
    (str(len(pos)), "Purchase orders sent", theme.AMBER),
])


def approve(rec_id: int) -> None:
    try:
        res = api.approve(rec_id, st.session_state.get("user_name") or "Logistics manager")
        st.toast(f"Approved. Purchase order #{res['purchase_order']['po_id']} created.", icon=":material/check_circle:")
    except api.ApiError as e:
        st.toast(e.message, icon=":material/error:")
    cache.after_decision()


def reject(rec_id: int) -> None:
    reason = st.session_state.get(f"why-{rec_id}") or None
    try:
        api.reject(rec_id, st.session_state.get("user_name") or "Logistics manager", reason)
        st.toast("Rejected. Nothing was ordered.", icon=":material/block:")
    except api.ApiError as e:
        st.toast(e.message, icon=":material/error:")
    cache.after_decision()


theme.section("Waiting for you")
if not pending:
    st.html('<p class="bb-note">No reorders are waiting. Ask BasketBee what will run out, and its '
            'suggestions will appear here for you to approve.</p>')
    st.page_link("views/ask.py", label="Ask BasketBee", icon=":material/forum:")
for r in pending:
    with st.container(border=True, key=f"pend-{r['id']}"):
        info, ok, no = st.columns([5, 1.25, 0.95], vertical_alignment="center")
        info.html(theme.rec_html(r))
        ok.button("Approve", key=f"a-{r['id']}", type="primary", width="stretch",
                  on_click=approve, args=(r["id"],))
        with no.popover("Reject", width="stretch"):
            st.text_input("Reason (optional)", key=f"why-{r['id']}", placeholder="e.g. price too high")
            st.button("Reject reorder", key=f"r-{r['id']}", on_click=reject, args=(r["id"],))

theme.section("History")
t_approved, t_rejected, t_pos = st.tabs(["Approved", "Rejected", "Purchase orders"])


def decided_table(status: str):
    rows = [r for r in recs if r["status"] == status]
    if not rows:
        st.html(f'<p class="bb-note">No {status} reorders yet.</p>')
        return
    st.dataframe(pd.DataFrame([{
        "Product": r["product"], "Hub": theme.wh_name(r["warehouse_id"]), "Quantity": r["qty"],
        "Supplier": r["supplier"], "Value": theme.inr(r["value_inr"]), "Decided by": r["decided_by"],
        "Reason": r.get("comment") or "",
    } for r in rows]), hide_index=True, width="stretch")


with t_approved:
    decided_table("approved")
with t_rejected:
    decided_table("rejected")
with t_pos:
    if not pos:
        st.html('<p class="bb-note">No purchase orders yet. They appear here when you approve a reorder.</p>')
    else:
        names = {p["sku"]: p["name"] for p in cache.products()}
        suppliers = {s["supplier_id"]: s["name"] for s in cache.suppliers()}
        st.dataframe(pd.DataFrame([{
            "PO": f"#{p['po_id']}", "Product": names.get(p["sku"], p["sku"]), "Hub": theme.wh_name(p["warehouse_id"]),
            "Quantity": p["qty"], "Supplier": suppliers.get(p["supplier_id"], p["supplier_id"]),
            "Status": p["status"].title(),
            "Approved by": p["approved_by"], "Created": p["created_at"][:16].replace("T", " "),
        } for p in pos]), hide_index=True, width="stretch")