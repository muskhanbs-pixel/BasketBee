"""
BasketBee frontend - design system.

Palette (sampled from the logo):
    hive green  #1F4A33   headings, sidebar, primary buttons
    honey       #FFBB02   the one bold accent: the bee's flight path, highlights
    amber       #9A6412   secondary text accents (the tagline colour)
    leaf paper  #F5F6F0   page background
Type: Bricolage Grotesque (headings, numbers) + Figtree (body).

Signature element: the "flight path" - a dashed honey trail (like the one behind
the bee in the logo) that shows every step the agents took to answer.
"""

import html
from datetime import date

import streamlit as st

GREEN, HONEY, AMBER = "#1F4A33", "#FFBB02", "#9A6412"
RISK = {  # text colour, background tint, plain-language label
    "CRITICAL": ("#A8261F", "#FBE3E1", "Runs out before restock"),
    "HIGH": ("#A85400", "#FDEBD8", "Order today"),
    "WATCH": ("#7A5B00", "#FFF3C4", "Order this week"),
    "OK": ("#24663F", "#E1F1E6", "Healthy"),
}
WAREHOUSE_NAMES = {"WH-SOUTH": "South hub", "WH-NORTH": "North hub", "WH-EAST": "East hub"}
TOOL_NAMES = {
    "check_stock_risks": "Checked stock risks",
    "forecast_product_demand": "Forecast demand",
    "get_stock_level": "Read stock levels",
    "find_products": "Searched products",
}
AGENT_NAMES = {"forecast": "Forecast agent", "inventory": "Inventory agent"}

CSS = f"""
<style>
:root {{
  --green: {GREEN}; --honey: {HONEY}; --amber: {AMBER};
  --ink: #18261E; --muted: #5E6B61; --line: #DCE0D3; --paper: #F5F6F0; --card: #FFFFFF;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
@media (max-width: 760px) {{ .block-container {{ padding-top: 3.6rem; }} }}
h1, h2, h3 {{ letter-spacing: -0.015em; color: var(--green); }}

/* ---------- sidebar brand ---------- */
.bb-brand {{ margin: 0 0 20px; }}
.bb-brand img {{ width: 84px; height: 84px; background: #FBF8EF; border-radius: 20px; padding: 8px;
                 display: block; margin-bottom: 12px; }}
.bb-brand .name {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: 1.6rem;
                   color: #FFFFFF; line-height: 1.05; }}
.bb-brand .tag {{ color: #FFD666; font-size: .86rem; margin-top: 4px; }}
.bb-side-meta {{ font-size: .84rem; line-height: 1.55; color: #D7E2D6; }}
.bb-side-meta b {{ color: #FFFFFF; font-weight: 600; }}
.bb-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
.bb-countdown {{ background: #28583E; border-radius: 12px; padding: 12px 14px; margin: 8px 0 14px; }}
.bb-countdown .n {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 2rem; font-weight: 750;
                    color: var(--honey); line-height: 1; }}
.bb-countdown .t {{ font-size: .85rem; color: #EEF3EA; }}

/* ---------- page header ---------- */
.bb-head h1 {{ font-size: 2.35rem; margin: 0 0 .2rem; padding: 0; }}
.bb-head p {{ color: var(--muted); font-size: 1.02rem; margin: 0 0 1.2rem; max-width: 70ch; }}

/* ---------- risk chips ---------- */
.bb-chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
            font-weight: 600; white-space: nowrap; }}

/* ---------- KPI band: one band, cells split by rules (not a grid of cards) ---------- */
.bb-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); background: var(--card);
            border: 1px solid var(--line); border-radius: 16px; margin: .4rem 0 1.4rem; }}
.bb-kpi {{ padding: 16px 20px; border-left: 1px solid var(--line); }}
.bb-kpi:first-child {{ border-left: none; }}
.bb-kpi .n {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.3rem; font-weight: 750;
              line-height: 1.05; font-variant-numeric: tabular-nums; }}
.bb-kpi .l {{ color: var(--muted); font-size: .9rem; margin-top: 4px; }}
@media (max-width: 760px) {{
  .bb-kpis {{ grid-template-columns: repeat(2, 1fr); }}
  .bb-kpi:nth-child(3) {{ border-left: none; }}
  .bb-kpi:nth-child(n+3) {{ border-top: 1px solid var(--line); }}
}}

/* ---------- warehouse utilisation ---------- */
.bb-wh {{ padding: 4px 2px; }}
.bb-wh .top {{ display: flex; justify-content: space-between; align-items: baseline; }}
.bb-wh .name {{ font-weight: 650; color: var(--green); }}
.bb-wh .pct {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.25rem; }}
.bb-wh .bar {{ height: 10px; background: #ECEEE5; border-radius: 999px; overflow: hidden; margin: 6px 0; }}
.bb-wh .bar span {{ display: block; height: 100%; background: var(--honey); border-radius: 999px; }}
.bb-wh .sub {{ color: var(--muted); font-size: .84rem; }}

/* ---------- the flight path (signature element) ---------- */
.bb-flight {{ position: relative; display: flex; flex-wrap: wrap; gap: 0; margin: 14px 0 6px;
              padding: 4px 0 2px; }}
.bb-stop {{ position: relative; display: flex; flex-direction: column; min-width: 120px; flex: 1 1 0;
            padding-right: 14px; }}
.bb-stop::before {{ content: ""; position: absolute; top: 7px; left: 16px; right: 0;
                    border-top: 2px dashed #E7C25A; }}
.bb-stop:last-child::before {{ display: none; }}
.bb-stop .pin {{ position: relative; z-index: 1; width: 16px; height: 16px; border-radius: 50%;
                 background: var(--honey); border: 3px solid #FFF6D6; box-shadow: 0 0 0 1px #E3B10A; }}
.bb-stop.ok .pin {{ background: #2E7D4F; box-shadow: 0 0 0 1px #2E7D4F; border-color: #E1F1E6; }}
.bb-stop.warn .pin {{ background: #D9730D; box-shadow: 0 0 0 1px #D9730D; border-color: #FDEBD8; }}
.bb-stop.err .pin {{ background: #C2362F; box-shadow: 0 0 0 1px #C2362F; border-color: #FBE3E1; }}
.bb-stop .t {{ font-weight: 650; font-size: .86rem; color: var(--ink); margin-top: 7px; line-height: 1.2; }}
.bb-stop .s {{ font-size: .78rem; color: var(--muted); line-height: 1.3; margin-top: 2px; }}

/* ---------- reorder rows (compact: one decision per row) ---------- */
.bb-rec .row1 {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }}
.bb-rec .prod {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.08rem;
                 color: var(--green); }}
.bb-rec .line {{ color: var(--ink); }}
.bb-rec .val {{ color: var(--amber); font-weight: 650; font-variant-numeric: tabular-nums; }}
.bb-rec .meta {{ color: var(--muted); font-size: .85rem; margin-top: 3px; }}
.bb-rec .qty {{ font-variant-numeric: tabular-nums; font-weight: 650; }}
.bb-status {{ font-size: .9rem; font-weight: 600; text-align: right; }}
div[class*="st-key-rec-"], div[class*="st-key-pend-"] {{ background: var(--card); }}

/* ---------- welcome ---------- */
.bb-welcome {{ display: flex; gap: 18px; align-items: center; margin: .2rem 0 1.4rem; }}
.bb-welcome img {{ width: 92px; height: 92px; }}
.bb-welcome .hi {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: 1.6rem;
                   color: var(--green); }}
.bb-welcome p {{ color: var(--muted); margin: 4px 0 0; max-width: 56ch; }}
.st-key-suggestions button {{ justify-content: flex-start !important; text-align: left; padding: 14px 16px;
                              background: var(--card); min-height: 58px; height: auto; }}
.st-key-suggestions button > div, .st-key-suggestions button span {{ justify-content: flex-start !important; }}
.st-key-suggestions button * {{ white-space: normal !important; overflow: visible !important;
                                text-overflow: clip !important; text-align: left; }}
.st-key-suggestions button:hover {{ border-color: #E3B10A; background: #FFFBEA; }}
.st-key-suggestions button p {{ font-size: .95rem; }}

/* ---------- misc ---------- */
.bb-note {{ color: var(--muted); font-size: .88rem; }}
.bb-verified {{ font-size: .86rem; color: #24663F; }}
.bb-unverified {{ font-size: .86rem; color: #A85400; }}
.bb-section {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: var(--green);
               font-size: 1.25rem; margin: 1.4rem 0 .5rem; }}
.bb-fact .n {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.5rem;
               color: var(--green); font-variant-numeric: tabular-nums; }}
.bb-fact .l {{ color: var(--muted); font-size: .86rem; }}
[data-testid="stChatMessage"] {{ background: transparent; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def apply() -> None:
    st.html(CSS)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def inr(value: float) -> str:
    """Indian digit grouping: 917412 -> ₹9,17,412"""
    n = int(round(value))
    s, sign = str(abs(n)), "-" if n < 0 else ""
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}₹{s}"


def nice_date(iso: str | None) -> str:
    if not iso:
        return "-"
    d = date.fromisoformat(str(iso)[:10])
    return f"{d.day} {d.strftime('%b')}"


def wh_name(wh_id: str | None) -> str:
    return WAREHOUSE_NAMES.get(wh_id or "", wh_id or "All hubs")


def esc(text) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------
def chip(risk: str) -> str:
    fg, bg, label = RISK.get(risk, ("#5E6B61", "#ECEEE5", risk.title()))
    return f'<span class="bb-chip" style="color:{fg};background:{bg}">{label}</span>'


def header(title: str, subtitle: str) -> None:
    st.html(f'<div class="bb-head"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def section(title: str) -> None:
    st.html(f'<div class="bb-section">{esc(title)}</div>')


def kpi_band(cells: list[tuple[str, str, str]]) -> None:
    """cells: (number, label, colour)"""
    inner = "".join(f'<div class="bb-kpi"><div class="n" style="color:{c}">{esc(n)}</div>'
                    f'<div class="l">{esc(l)}</div></div>' for n, l, c in cells)
    st.html(f'<div class="bb-kpis">{inner}</div>')


def warehouse_bar(w: dict) -> str:
    pct = max(0.0, min(float(w["utilization_pct"]), 100.0))
    fill = "#D9730D" if pct >= 90 else HONEY          # nearly full hubs turn orange
    return (f'<div class="bb-wh"><div class="top"><span class="name">{esc(wh_name(w["warehouse_id"]))}</span>'
            f'<span class="pct">{w["utilization_pct"]:.0f}%</span></div>'
            f'<div class="bar"><span style="width:{pct}%;background:{fill}"></span></div>'
            f'<div class="sub">{esc(w["area"])}: {w["units_stored"]:,} of {w["capacity_units"]:,} units</div></div>')


def _args_summary(tool: str, args: dict) -> str:
    parts = []
    if args.get("product"):
        parts.append(str(args["product"]).title())
    if args.get("products"):
        parts.append(", ".join(str(p).title() for p in args["products"])[:40])
    if args.get("query"):
        parts.append(f'"{args["query"]}"')
    if "warehouse" in args:
        w = str(args.get("warehouse") or "all hubs").lower()
        parts.append(next((v for k, v in WAREHOUSE_NAMES.items() if k.split("-")[1].lower() in w), w.title()))
    if args.get("category"):
        parts.append(str(args["category"]).title())
    if args.get("days"):
        parts.append(f'{args["days"]} days')
    return ", ".join(parts)


def flight_path(result: dict) -> str:
    stops = []
    method = {"rules": "keyword rules", "llm": "AI classifier", "default": "default"}.get(result["route_method"], "")
    plan = [AGENT_NAMES.get(p, "") for p in result["plan"] if p in AGENT_NAMES]
    stops.append(("", "Supervisor", f"Routed by {method}" + (f" to {len(plan)} agent{'s' if len(plan) > 1 else ''}"
                                                              if plan else "")))
    for run in result["agents"]:
        stops.append(("", AGENT_NAMES.get(run["agent"], run["agent"]), f'{run["seconds"]}s'))
        for t in run["tools"]:
            stops.append(("" if t["ok"] else "err", TOOL_NAMES.get(t["tool"], t["tool"]),
                          _args_summary(t["tool"], t["args"]) or ("ok" if t["ok"] else "failed")))
    if result["recommendations"]:
        n = len(result["recommendations"])
        stops.append(("", f"Drafted {n} reorder{'s' if n > 1 else ''}", "Waiting for you"))
    v = result.get("verification") or {}
    if result["agents"]:
        if v.get("verified", True):
            stops.append(("ok", "Numbers checked", f'{v.get("numbers_checked", 0)} match system data'))
        else:
            stops.append(("warn", "Numbers checked", f'{len(v["unverified"])} could not be verified'))
    else:
        stops.append(("ok", "Answered directly", "No agents needed"))

    inner = "".join(f'<div class="bb-stop {cls}"><div class="pin"></div><div class="t">{esc(t)}</div>'
                    f'<div class="s">{esc(s)}</div></div>' for cls, t, s in stops)
    return f'<div class="bb-flight" role="list" aria-label="Steps BasketBee took">{inner}</div>'


def rec_html(r: dict) -> str:
    runs = f'runs out {nice_date(r.get("runs_out"))}' if r.get("runs_out") else "below reorder point"
    return (f'<div class="bb-rec"><div class="row1">{chip(r["risk"])}'
            f'<span class="prod">{esc(r["product"])}</span>'
            f'<span class="line">Order <span class="qty">{r["qty"]:,}</span> from {esc(r["supplier"])}</span>'
            f'<span class="val">{inr(r["value_inr"])}</span></div>'
            f'<div class="meta">{esc(wh_name(r["warehouse_id"]))}, {runs}. '
            f'{r.get("available", 0):,} left, selling about {r.get("per_day", 0):,} a day.</div></div>')


def welcome(logo_b64: str) -> None:
    st.html(f'<div class="bb-welcome"><img src="data:image/png;base64,{logo_b64}" alt="">'
            f'<div><div class="hi">Hi, I\'m BasketBee.</div>'
            f'<p>I keep an eye on stock at all three hubs, forecast demand and draft reorders. '
            f'You decide what actually gets ordered.</p></div></div>')