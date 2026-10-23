
"""Ask BasketBee - chat with the supervisor and its agents."""

import base64
from datetime import date
from pathlib import Path

import streamlit as st

from frontend import api, cache, theme

LOGO = str(Path(__file__).resolve().parents[1] / "assets" / "logo.png")
LOGO_B64 = base64.b64encode(Path(LOGO).read_bytes()).decode()
ICONS = [":material/inventory_2:", ":material/trending_up:", ":material/celebration:", ":material/balance:"]
SUGGESTIONS = [
    "What will run out in the south warehouse?",
    "How many diya lamps will North Hub sell before Diwali?",
    "Are we ready for Diwali?",
    "How much ghee will we sell next week and do we have enough stock?",
]


# ---------------------------------------------------------------------------
# Live progress lines
# ---------------------------------------------------------------------------
def describe(ev: dict) -> str | None:
    kind = ev["type"]
    if kind == "route":
        agents = [theme.AGENT_NAMES[p].lower() for p in ev["plan"] if p in theme.AGENT_NAMES]
        how = "keyword rules" if ev["method"] == "rules" else "the AI classifier"
        if agents:
            return f"Supervisor sent this to the {' and then the '.join(agents)} (chosen by {how})."
        return "Supervisor answered this one directly."
    if kind == "agent_start":
        return f"**{theme.AGENT_NAMES[ev['agent']]}** is working. The model is reading the question."
    if kind == "tool_call":
        detail = theme._args_summary(ev["tool"], ev["args"])
        return f"{theme.TOOL_NAMES.get(ev['tool'], ev['tool'])}" + (f" for {detail}" if detail else "") + "."
    if kind == "repair":
        return "Recovered a tool call the model wrote as text, and ran it."
    if kind == "tool_result":
        return None if ev["ok"] else f"{theme.TOOL_NAMES.get(ev['tool'], ev['tool'])} returned a problem."
    if kind == "agent_done":
        return f"{theme.AGENT_NAMES[ev['agent']]} finished in {ev['seconds']}s. Writing the answer."
    if kind == "recommendations":
        return f"Drafted {ev['count']} reorder{'s' if ev['count'] != 1 else ''} for your approval."
    return None


# ---------------------------------------------------------------------------
# One answer
# ---------------------------------------------------------------------------
def decide(rec_id: int, action: str) -> None:
    who = st.session_state.get("user_name") or "Logistics manager"
    try:
        if action == "approve":
            res = api.approve(rec_id, who)
            st.toast(f"Approved. Purchase order #{res['purchase_order']['po_id']} created.", icon=":material/check_circle:")
        else:
            api.reject(rec_id, who)
            st.toast("Rejected. Nothing was ordered.", icon=":material/block:")
    except api.ApiError as e:
        st.toast(e.message, icon=":material/error:")
    cache.after_decision()


def render_recs(recs: list[dict], msg_idx: int) -> None:
    if not recs:
        return
    try:
        status = {r["id"]: r for r in cache.recommendations("all")}
    except api.ApiError:
        status = {}
    theme.section("Reorders waiting for you")
    for r in recs:
        current = status.get(r["id"], {"status": "pending"})
        with st.container(border=True, key=f"rec-{msg_idx}-{r['id']}"):
            if current["status"] == "pending":
                info, ok, no = st.columns([5, 1.25, 0.85], vertical_alignment="center")
                info.html(theme.rec_html(r))
                ok.button("Approve", key=f"ap-{msg_idx}-{r['id']}", type="primary", width="stretch",
                          on_click=decide, args=(r["id"], "approve"))
                no.button("Reject", key=f"rj-{msg_idx}-{r['id']}", width="stretch",
                          on_click=decide, args=(r["id"], "reject"))
            else:
                info, done = st.columns([5, 2.1], vertical_alignment="center")
                info.html(theme.rec_html(r))
                colour, word = ("#24663F", "Approved") if current["status"] == "approved" else ("#A8261F", "Rejected")
                done.html(f'<div class="bb-status" style="color:{colour}">{word} by '
                          f'{theme.esc(current.get("decided_by") or "manager")}</div>')


def render_answer(msg: dict, idx: int) -> None:
    if msg.get("error"):
        st.error(msg["error"], icon=":material/error:")
        return
    result = msg["result"]
    st.markdown(result["answer"])
    st.html(theme.flight_path(result))
    v = result["verification"]
    if result["agents"]:
        if v["verified"]:
            st.html(f'<div class="bb-verified">Every figure above ({v["numbers_checked"]}) matches the system data. '
                    f'Answered in {result["seconds"]}s.</div>')
        else:
            st.html(f'<div class="bb-unverified">Check before acting: {", ".join(v["unverified"])} '
                    f'did not match the system data.</div>')
    render_recs(result["recommendations"], idx)
    if result["agents"]:
        with st.expander("See the data behind this answer"):
            for run in result["agents"]:
                for t in run["tools"]:
                    st.markdown(f"**{theme.TOOL_NAMES.get(t['tool'], t['tool'])}** "
                                f"(`{t['tool']}`, {theme.AGENT_NAMES[run['agent']].lower()})")
                    st.json(t["output"], expanded=False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
health = cache.health()
today = date.fromisoformat(health["business_date"]) if health else date(2026, 10, 18)
theme.header("Ask BasketBee",
             f"Stock, demand and reorders for the South, North and East hubs. "
             f"Today is {today.strftime('%A')}, {today.day} {today.strftime('%B')}.")

def pick(question: str) -> None:
    st.session_state.pending_question = question


messages = st.session_state.messages
if not messages and "pending_question" not in st.session_state:
    theme.welcome(LOGO_B64)
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, (s, icon) in enumerate(zip(SUGGESTIONS, ICONS)):
            cols[i % 2].button(s, key=f"sugg-{i}", icon=icon, width="stretch", on_click=pick, args=(s,))

for i, m in enumerate(messages):
    with st.chat_message(m["role"], avatar=LOGO if m["role"] == "assistant" else ":material/person:"):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render_answer(m, i)

prompt = st.chat_input("Ask about stock, demand or reorders")
prompt = prompt or st.session_state.pop("pending_question", None)

if prompt:
    # The question is saved together with its answer (below), so an interrupted answer
    # never leaves a lonely question in the history.
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=LOGO):
        status = st.status("BasketBee is on it", expanded=True)
        reply = None
        for ev in api.chat_stream(prompt, st.session_state.session_id):
            if ev["type"] == "final":
                reply = {"role": "assistant", "result": ev["result"]}
            elif ev["type"] == "error":
                reply = {"role": "assistant", "error": ev["message"]}
            elif (line := describe(ev)) is not None:
                status.write(line)
        if reply and "result" in reply:
            status.update(label=f"Answered in {reply['result']['seconds']}s", state="complete", expanded=False)
        else:
            reply = reply or {"role": "assistant", "error": "The answer stopped unexpectedly. Try asking again."}
            status.update(label="Couldn't answer", state="error", expanded=False)
        messages.extend([{"role": "user", "text": prompt}, reply])
        render_answer(reply, len(messages) - 1)
    cache.recommendations.clear()