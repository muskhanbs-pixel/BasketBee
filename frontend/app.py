"""
BasketBee - the app (M6).

Start everything from the project root:
    python run.py
or just the frontend (with the API already running):
    streamlit run frontend/app.py
"""

import base64
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend import cache, theme  # noqa: E402

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="BasketBee", page_icon=str(ASSETS / "favicon.png"), layout="wide",
                   initial_sidebar_state="auto")
theme.apply()

st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
st.session_state.setdefault("messages", [])

pages = {
    "ask": st.Page("views/ask.py", title="Ask BasketBee", icon=":material/forum:", default=True),
    "stock": st.Page("views/dashboard.py", title="Stock health", icon=":material/monitoring:", url_path="stock"),
    "approvals": st.Page("views/approvals.py", title="Approvals", icon=":material/task_alt:",
                         url_path="approvals"),
    "activity": st.Page("views/activity.py", title="Activity log", icon=":material/history:",
                        url_path="activity"),
}
nav = st.navigation(list(pages.values()), position="hidden")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
logo_b64 = base64.b64encode((ASSETS / "logo.png").read_bytes()).decode()
health = cache.health()

with st.sidebar:
    st.html(f'<div class="bb-brand"><img src="data:image/png;base64,{logo_b64}" alt="BasketBee logo">'
            f'<div><div class="name">BasketBee</div><div class="tag">Everyday essentials. Smarter supply.</div>'
            f'</div></div>')

    n_pending = cache.pending_count() if health else 0
    st.page_link(pages["ask"])
    st.page_link(pages["stock"])
    st.page_link(pages["approvals"], label=f"Approvals ({n_pending})" if n_pending else "Approvals")
    st.page_link(pages["activity"])

    business_day = date.fromisoformat(health["business_date"]) if health else date(2026, 10, 18)
    days_left = (date(2026, 11, 8) - business_day).days
    if days_left >= 0:
        st.html(f'<div class="bb-countdown"><div class="n">{days_left}</div>'
                f'<div class="t">days until Diwali on 8 November. Sweets, ghee and diyas sell fastest '
                f'in the last week.</div></div>')

    st.text_input("Approving as", value="Logistics manager", key="user_name", max_chars=60)

    if health:
        st.html(f'<div class="bb-side-meta"><span class="bb-dot" style="background:#5BD38A"></span>'
                f'<b>Connected</b><br>Model: {theme.esc(health["llm"])}<br>'
                f'Business day: {business_day.strftime("%a %d %b %Y")}</div>')
    else:
        st.html('<div class="bb-side-meta"><span class="bb-dot" style="background:#FF7A6E"></span>'
                '<b>API offline</b><br>Start everything with <b>python run.py</b></div>')

nav.run()