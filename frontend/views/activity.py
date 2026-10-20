"""Activity log - every routing decision, tool call and approval."""

import json

import pandas as pd
import streamlit as st

from frontend import api, cache, theme

theme.header("Activity log", "Everything BasketBee did, newest first: how each question was routed, "
                             "which tools the agents used, and who approved what.")

WHO = {"Everything": None, "Supervisor": "supervisor", "Inventory agent": "inventory_agent",
       "Forecast agent": "forecast_agent", "People": "human"}
choice = st.segmented_control("Show", list(WHO), default="Everything", key="act_who")

try:
    rows = api.audit(limit=400, agent=WHO.get(choice or "Everything"))
    every = api.audit(limit=2000) if choice not in (None, "Everything") else rows
except api.ApiError as e:
    st.error(e.message, icon=":material/cloud_off:")
    st.stop()

questions = sum(r["agent"] == "supervisor" and r["tool"] == "route" for r in every)
tool_calls = sum(r["agent"].endswith("_agent") and r["tool"] not in (None, "") for r in every)
agent_runs = [r["latency_ms"] for r in every if r["agent"].endswith("_agent") and not r["tool"] and r["latency_ms"]]
decisions = sum(r["agent"] == "human" and r["tool"] in ("approve", "reject") for r in every)
theme.kpi_band([
    (str(questions), "Questions answered", theme.GREEN),
    (str(tool_calls), "Tool calls by agents", theme.GREEN),
    (f"{(sum(agent_runs) / len(agent_runs) / 1000):.0f}s" if agent_runs else "-", "Average agent time", theme.AMBER),
    (str(decisions), "Approvals and rejections", theme.GREEN),
])

ACTION = {"route": "Routed question", "compose": "Wrote answer", "approve": "Approved reorder",
          "reject": "Rejected reorder", "approve_failed": "Approval failed (ERP down)"}
WHO_LABEL = {"supervisor": "Supervisor", "inventory_agent": "Inventory agent", "forecast_agent": "Forecast agent",
             "human": "Person"}


try:
    REC_NAMES = {r["id"]: f'{r["product"]} at {theme.wh_name(r["warehouse_id"])}'
                 for r in cache.recommendations("all")}
except api.ApiError:
    REC_NAMES = {}


def _load(text: str | None):
    try:
        return json.loads(text) if text else None
    except ValueError:
        return text


def summary(text: str | None) -> str:
    data = _load(text)
    if data is None:
        return ""
    if isinstance(data, dict):
        for key in ("question", "answer", "intent", "message"):
            if key in data and isinstance(data[key], str):
                return data[key].replace("**", "").replace("\n", " ")[:160]
        return ", ".join(f"{k}: {v}" for k, v in list(data.items())[:4])[:160]
    return str(data)[:160]


def details(r: dict) -> str:
    if r["agent"] == "human":
        rec_id = (_load(r["input"]) or {}).get("recommendation_id")
        what = REC_NAMES.get(rec_id, f"Reorder #{rec_id}")
        out = _load(r["output"]) or {}
        if r["tool"] == "approve" and isinstance(out, dict) and out.get("po_id"):
            return f"{what}. Purchase order #{out['po_id']} for {out.get('qty', '?'):,} units."
        if r["tool"] == "reject" and isinstance(out, dict) and out.get("comment"):
            return f"{what}. Reason: {out['comment']}"
        return what
    return summary(r["output"]) if r["tool"] == "compose" else summary(r["input"])


if not rows:
    st.html('<p class="bb-note">Nothing here yet. Ask BasketBee a question and every step will be logged.</p>')
else:
    df = pd.DataFrame([{
        "Time": (r["ts"] or "")[:19].replace("T", " "),
        "Who": WHO_LABEL.get(r["agent"], r["agent"]),
        "Action": ACTION.get(r["tool"], theme.TOOL_NAMES.get(r["tool"], "Finished task") if r["tool"] else "Finished task"),
        "Details": details(r),
        "Time taken": f'{r["latency_ms"] / 1000:.1f}s' if r["latency_ms"] else "",
        "Session": r["session_id"],
    } for r in rows])
    st.dataframe(df, hide_index=True, width="stretch", height=min(36 * len(df) + 40, 560),
                 column_config={"Details": st.column_config.TextColumn(width="large")})