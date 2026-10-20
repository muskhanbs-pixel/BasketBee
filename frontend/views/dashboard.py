"""Stock health - run-out risks, warehouse space and the demand explorer."""

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from frontend import api, cache, theme

theme.header("Stock health", "What will run out, where, and what to reorder. Based on demand forecasts "
                             "for every product at every hub.")

c1, c2 = st.columns([2, 1.3], vertical_alignment="bottom")
hub = c1.segmented_control("Hub", ["All hubs", "South hub", "North hub", "East hub"], default="All hubs",
                           key="dash_hub")
days = c2.select_slider("Look ahead", options=[7, 14, 21, 30], value=14, format_func=lambda d: f"{d} days",
                        key="dash_days")
wh_param = None if hub in (None, "All hubs") else hub.split()[0].lower()

try:
    with st.spinner("Checking stock against the forecast for every product..."):
        d = cache.dashboard(wh_param, days)
except api.ApiError as e:
    st.error(e.message, icon=":material/cloud_off:")
    st.stop()

rc = d["risk_counts"]
theme.kpi_band([
    (str(rc["CRITICAL"]), "Run out before a new order can arrive", theme.RISK["CRITICAL"][0]),
    (str(rc["HIGH"]), "Need an order today", theme.RISK["HIGH"][0]),
    (str(rc["WATCH"]), "Need an order this week", theme.RISK["WATCH"][0]),
    (str(d["pending_approvals"]), f"Reorders waiting for approval ({theme.inr(d['pending_value_inr'])})",
     theme.GREEN),
])

# ----- warehouses ------------------------------------------------------------------
theme.section("Warehouse space")
cols = st.columns(len(d["warehouses"]))
for col, w in zip(cols, d["warehouses"]):
    col.html(theme.warehouse_bar(w))

# ----- run-out list --------------------------------------------------------------
theme.section(f"Products to act on in the next {days} days")
items = d["items"]
if not items:
    st.success("Nothing at risk in this window. Every product has enough stock.", icon=":material/check_circle:")
else:
    df = pd.DataFrame([{
        "Risk": theme.RISK[i["risk"]][2],
        "Product": i["name"],
        "Hub": theme.wh_name(i["warehouse_id"]),
        "Available": i["available"],
        "Selling per day": round(i["avg_daily_forecast"]),
        "Days of cover": i["days_of_cover"],
        "Runs out": theme.nice_date(i["stockout_date"]) if i["stockout_date"] else "-",
        "Suggested order": i["recommended_order_qty"],
        "Supplier": i["supplier_name"],
        "Order value": theme.inr(i["order_value_inr"]),
    } for i in items])
    risk_style = {label: f"color:{fg};background-color:{bg};font-weight:600"
                  for fg, bg, label in theme.RISK.values()}
    styled = df.style.map(lambda v: risk_style.get(v, ""), subset=["Risk"])
    st.dataframe(styled, hide_index=True, width="stretch", height=min(38 * len(df) + 40, 460),
                 column_config={
                     "Days of cover": st.column_config.ProgressColumn(
                         "Days of cover", format="%.1f", min_value=0, max_value=float(days)),
                     "Available": st.column_config.NumberColumn(format="%d"),
                     "Suggested order": st.column_config.NumberColumn(format="%d"),
                 })
    st.html(f'<p class="bb-note">Checked {d["items_checked"]} product and hub combinations in '
            f'{d["elapsed_seconds"]}s. Ask BasketBee to turn these into reorders you can approve.</p>')

# ----- demand explorer -------------------------------------------------------------
theme.section("Demand explorer")
try:
    product_names = sorted(p["name"] for p in cache.products())
except api.ApiError as e:
    st.error(e.message)
    st.stop()

e1, e2, e3 = st.columns([2, 1, 1])
product = e1.selectbox("Product", product_names, index=product_names.index("Diya Oil Lamps"), key="ex_product")
hub_ex = e2.selectbox("Hub", ["South hub", "North hub", "East hub"], index=1, key="ex_hub")
horizon = e3.selectbox("Forecast", [14, 28, 42], index=1, format_func=lambda x: f"{x} days", key="ex_days")

try:
    with st.spinner("Forecasting..."):
        f = cache.forecast(product, hub_ex.split()[0].lower(), horizon)
except api.ApiError as e:
    st.error(e.message)
    st.stop()

hist = pd.DataFrame(f["history"]).assign(series="Sold")
fc = pd.DataFrame(f["forecast"]).assign(series="Forecast")
data = pd.concat([hist, fc])
data["date"] = pd.to_datetime(data["date"])

colour = alt.Scale(domain=["Sold", "Forecast"], range=[theme.GREEN, "#E3A400"])
lines = alt.Chart(data).mark_line(strokeWidth=2.4, interpolate="monotone").encode(
    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelColor="#5E6B61", grid=False)),
    y=alt.Y("units:Q", title="Units per day", axis=alt.Axis(labelColor="#5E6B61", gridColor="#ECEEE5")),
    color=alt.Color("series:N", scale=colour, legend=alt.Legend(title=None, orient="top-left")),
    strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Sold", "Forecast"], range=[[1, 0], [6, 4]]),
                              legend=None),
    tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "series:N", alt.Tooltip("units:Q", format=",.0f")],
)
marks = [{"date": pd.Timestamp(d["business_date"]), "label": "Today"}]
if pd.Timestamp(d["festival_date"]) <= data["date"].max():
    marks.append({"date": pd.Timestamp(d["festival_date"]), "label": "Diwali"})
m = pd.DataFrame(marks)
rules = alt.Chart(m).mark_rule(color="#9A6412", strokeDash=[2, 3]).encode(x="date:T")
labels = alt.Chart(m).mark_text(align="left", dx=5, dy=-6, color="#9A6412", fontWeight="bold").encode(
    x="date:T", y=alt.value(12), text="label:N")
st.altair_chart((lines + rules + labels).properties(height=320), width="stretch")

recent = f["recent_avg_daily_units"]
lift = f["peak_units"] / recent if recent else 0
facts = st.columns(4)
facts[0].html(f'<div class="bb-fact"><div class="n">{f["total_units"]:,.0f}</div>'
              f'<div class="l">units forecast over {horizon} days</div></div>')
facts[1].html(f'<div class="bb-fact"><div class="n">{f["peak_units"]:,.0f}</div>'
              f'<div class="l">at the peak on {theme.nice_date(f["peak_date"])}'
              + (f', {lift:.0f}x a normal day' if lift >= 2 else '') + '</div></div>')
facts[2].html(f'<div class="bb-fact"><div class="n">{recent:,.0f}</div>'
              f'<div class="l">sold per day over the last 4 weeks</div></div>')
if f["wape_pct"] is not None:
    facts[3].html(f'<div class="bb-fact"><div class="n">{f["wape_pct"]:.0f}%</div>'
                  f'<div class="l">daily error over the last {max(horizon, 14)} days '
                  f'(a simple average scored {f["baseline_wape_pct"]:.0f}%)</div></div>')