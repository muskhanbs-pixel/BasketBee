
"""System status - check everything before a demo, and simulate failures on purpose."""

import streamlit as st

from frontend import api, cache, theme

theme.header("System status", "Check every part of BasketBee before a demo. You can also switch an "
                              "enterprise system off on purpose to show how the agents cope.")

try:
    health = api.health()
    outages = api.outages()
except api.ApiError as e:
    st.error(e.message, icon=":material/cloud_off:")
    st.stop()

SYSTEMS = {
    "erp": ("ERP", "Products, suppliers, sales history and purchase orders. The forecast agent and "
                   "approvals depend on it."),
    "wms": ("WMS", "Warehouses and stock levels. The inventory agent depends on it."),
    "tms": ("TMS", "Vehicles and shipments. Used by the transport agents in the next phase."),
}


def status_line(ok: bool, text_ok: str, text_bad: str) -> str:
    colour, word = ("#24663F", text_ok) if ok else ("#A8261F", text_bad)
    return (f'<span class="bb-dot" style="background:{colour}"></span>'
            f'<b style="color:{colour}">{theme.esc(word)}</b>')


def toggle(system: str) -> None:
    down = st.session_state[f"down-{system}"]
    try:
        api.set_outage(system, down)
        name = SYSTEMS[system][0]
        st.toast(f"{name} is now down. Ask BasketBee a question to see how it copes." if down
                 else f"{name} is back online.", icon=":material/power_settings_new:")
    except api.ApiError as e:
        st.toast(e.message, icon=":material/error:")
    cache.after_decision()
    cache.health.clear()      # the sidebar shows which systems are down


# ----- core services ----------------------------------------------------------------
theme.section("Core services")
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.html(f'<div class="bb-rec"><div class="prod">API</div>'
            f'<div class="meta">{status_line(True, "Running", "")} on 127.0.0.1:8000</div></div>')
with c2.container(border=True):
    db_ok = health["db"] == "ok"
    st.html(f'<div class="bb-rec"><div class="prod">Database</div>'
            f'<div class="meta">{status_line(db_ok, "Connected", "Error")}</div></div>')

# ----- language model ---------------------------------------------------------------
theme.section("Language model")
with st.container(border=True):
    left, right = st.columns([3, 1.2], vertical_alignment="center")
    left.html(f'<div class="bb-rec"><div class="prod">{theme.esc(health["llm"])}</div>'
              f'<div class="meta">Change it with LLM_PROVIDER and LLM_MODEL in the .env file, then restart.</div>'
              f'</div>')
    if right.button("Test the model", key="llm-test", type="primary", width="stretch"):
        with st.spinner("Asking the model a one-word question..."):
            st.session_state.llm_result = api.llm_check()
    res = st.session_state.get("llm_result")
    if res:
        if res["ok"]:
            st.success(f"The model answered in {res['seconds']}s. Ready for questions.",
                       icon=":material/check_circle:")
        else:
            st.error(res["message"], icon=":material/error:")

# ----- enterprise systems --------------------------------------------------------------
theme.section("Enterprise systems")
st.html('<p class="bb-note">Switch a system off to show failure handling live. The agents report what '
        'they couldn\'t reach instead of guessing, and approvals stay pending until the ERP is back.</p>')
for key, (name, desc) in SYSTEMS.items():
    down = outages.get(key, False)
    with st.container(border=True, key=f"sys-{key}"):
        info, sw = st.columns([4, 1.3], vertical_alignment="center")
        info.html(f'<div class="bb-rec"><div class="row1"><span class="prod">{name}</span>'
                  f'<span>{status_line(not down, "Online", "Down (simulated)")}</span></div>'
                  f'<div class="meta">{theme.esc(desc)}</div></div>')
        sw.toggle("Simulate outage", value=down, key=f"down-{key}", on_change=toggle, args=(key,))

# ----- demo data --------------------------------------------------------------------
theme.section("Demo data")
with st.container(border=True):
    info, btn = st.columns([4, 1.3], vertical_alignment="center")
    info.html('<div class="bb-rec"><div class="prod">Start fresh</div>'
              '<div class="meta">Deletes every reorder, approval, purchase order and log entry, restores the '
              'original stock levels, and switches all systems back on. Do this before each rehearsal.</div></div>')
    with btn.popover("Reset demo data", width="stretch"):
        st.write("This can't be undone. Continue?")
        if st.button("Reset everything", key="reset-yes", type="primary"):
            try:
                with st.spinner("Restoring the original data..."):
                    msg = api.reset_demo()["message"]
                st.session_state.messages = []
                st.session_state.pop("llm_result", None)
                st.cache_data.clear()
                st.toast(msg, icon=":material/restart_alt:")
                st.rerun()
            except api.ApiError as e:
                st.error(e.message)