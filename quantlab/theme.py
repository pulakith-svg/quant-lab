"""Nifty-Desk style terminal theme: black ground, amber accent, monospace type."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

C = dict(bg="#000000", surface="#0a0a0a", surface2="#111111", line="#2a2a2a", line2="#444444",
         text="#e8e2d4", bright="#ffffff", muted="#9a8f7a", accent="#ff9900", accent_dim="#8a5200",
         accent_soft="#261700", bull="#00e676", bear="#ff3b3b", warn="#ffe14d", info="#4db8ff")

# ------------------------------------------------------------------ plotly
pio.templates["desk"] = go.layout.Template(layout=dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(family="JetBrains Mono, monospace", color=C["text"], size=11),
    title=dict(font=dict(color=C["accent"], size=12)),
    xaxis=dict(gridcolor="#1a1a1a", linecolor=C["line"], zerolinecolor=C["line"]),
    yaxis=dict(gridcolor="#1a1a1a", linecolor=C["line"], zerolinecolor=C["line"]),
    colorway=[C["accent"], C["info"], C["bull"], C["bear"], C["warn"], "#b388ff", "#80cbc4", "#f48fb1"],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["accent"], font=dict(family="JetBrains Mono")),
))
TEMPLATE = "desk"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
  font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace !important; }}
.stApp {{ background:{C['bg']}; color:{C['text']}; }}
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
.block-container {{ padding-top:3.2rem !important; max-width:1400px; }}

/* command bar */
.cmdbar {{ position:fixed; top:0; left:0; right:0; z-index:999; display:flex; height:38px; align-items:stretch;
  background:{C['bg']}; border-bottom:1px solid {C['line2']}; font-size:11px; overflow-x:auto; }}
.cmdbar .seg {{ display:flex; align-items:center; gap:7px; padding:0 15px; white-space:nowrap;
  border-right:1px solid {C['line']}; color:{C['muted']}; text-transform:uppercase; letter-spacing:.08em; }}
.cmdbar .brand {{ background:{C['accent']}; color:#000; font-weight:700; font-size:13px; letter-spacing:.12em; }}
.cmdbar .val {{ color:{C['bright']}; font-size:12px; letter-spacing:0; }}
.cmdbar .amb {{ color:{C['accent']}; }} .cmdbar .spacer {{ flex:1; border-right:none; }}
.dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}

/* masthead */
.mast {{ display:flex; align-items:flex-end; justify-content:space-between; border-bottom:2px solid {C['accent']};
  padding:18px 0 12px; margin-bottom:14px; }}
.mast h1 {{ color:{C['bright']} !important; margin:0; font-size:30px; font-weight:700; letter-spacing:.06em; color:{C['bright']}; line-height:1; }}
.mast h1 b {{ color:{C['accent']} !important; }}
.mast .sub {{ color:{C['muted']}; font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; margin-left:12px; }}
.mast .clock {{ color:{C['muted']}; font-size:12px; letter-spacing:.08em; }}
.disc {{ border:1px solid {C['line2']}; border-left:3px solid {C['warn']}; background:{C['surface']};
  padding:10px 16px; font-size:11.5px; color:{C['muted']}; margin-bottom:16px; line-height:1.6; }}
.disc b {{ color:{C['warn']}; }}

/* tabs like the desk tab strip */
[role="tablist"] {{ gap:0; border:1px solid {C['line2']}; background:{C['surface']}; flex-wrap:wrap; }}
[role="tab"] {{ height:36px; padding:0 15px; border-right:1px solid {C['line']}; border-radius:0;
  color:{C['accent']}; text-transform:uppercase; letter-spacing:.08em; font-size:11.5px; font-weight:600; background:transparent; }}
[role="tab"] p {{ font-size:11.5px; font-weight:600; text-transform:uppercase; color:inherit; }}
[role="tab"][aria-selected="true"] {{ background:{C['accent']} !important; color:#000 !important; }}
[role="tab"] {{ display:flex; align-items:center; cursor:pointer; }}
[role="tab"][aria-selected="true"] p {{ color:#000 !important; }}
[role="tablist"] + div[class*="highlight"], [role="tablist"] ~ div[aria-hidden="true"] {{ display:none; }}

/* panels */
.panel {{ border:1px solid {C['line2']}; background:{C['surface']}; margin:14px 0 6px; }}
.panel .ph {{ padding:14px 20px 12px; border-bottom:1px solid {C['line']}; }}
.panel .ph .t {{ color:{C['accent']}; font-size:13px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; }}
.panel .ph .s {{ color:{C['muted']}; font-size:11px; margin-top:4px; }}
.panel .pb {{ padding:12px 20px; font-size:12px; color:{C['muted']}; line-height:1.7; }}
.panel .pb b {{ color:{C['bright']}; }}
h1,h2,h3,h4 {{ color:{C['accent']} !important; text-transform:uppercase; letter-spacing:.08em; font-weight:600 !important; }}
h3 {{ font-size:15px !important; }} h4 {{ font-size:13px !important; }}

/* controls */
[data-testid="stWidgetLabel"] p {{ color:{C['muted']}; text-transform:uppercase; font-size:10px !important; letter-spacing:.1em; }}
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{ background:{C['surface']} !important;
  border-radius:0 !important; border-color:{C['line2']} !important; }}
.stButton button {{ border-radius:0; border:1px solid {C['accent']}; color:{C['accent']}; background:transparent;
  text-transform:uppercase; letter-spacing:.08em; font-size:11.5px; }}
.stButton button:hover {{ background:{C['accent']}; color:#000; }}

/* metrics as tiles */
[data-testid="stMetric"] {{ border:1px solid {C['line2']}; background:{C['surface']}; padding:10px 14px; }}
[data-testid="stMetricLabel"] p {{ color:{C['muted']} !important; text-transform:uppercase; font-size:10px !important; letter-spacing:.1em; }}
[data-testid="stMetricValue"] {{ color:{C['bright']}; font-size:20px !important; }}

/* expanders = checklist rows */
[data-testid="stExpander"] details {{ border:1px solid {C['line']} !important; border-radius:0 !important; background:{C['surface']}; }}
[data-testid="stExpander"] summary p {{ font-size:12px; }}
[data-testid="stExpander"] summary:hover p {{ color:{C['accent']}; }}
[data-testid="stDataFrame"], .stPlotlyChart {{ border:1px solid {C['line2']}; }}
[data-testid="stAlert"] {{ border-radius:0; }}
.pill {{ display:inline-block; border:1px solid; padding:0 8px; font-size:11px; font-weight:600; line-height:20px; letter-spacing:.06em; }}
.bull {{ color:{C['bull']}; }} .bear {{ color:{C['bear']}; }} .neu {{ color:{C['muted']}; }} .amb {{ color:{C['accent']}; }}
footer {{ visibility:hidden; }}
</style>
"""


def _ist_now():
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def market_status(now_ist: datetime) -> tuple[str, str]:
    wd, hm = now_ist.weekday(), now_ist.hour * 60 + now_ist.minute
    if wd >= 5:
        return "MARKET CLOSED · WEEKEND", C["muted"]
    if 9 * 60 <= hm < 9 * 60 + 15:
        return "PRE-OPEN", C["warn"]
    if 9 * 60 + 15 <= hm < 15 * 60 + 30:
        return "MARKET OPEN", C["bull"]
    return "MARKET CLOSED", C["muted"]


def apply(feed_live: bool, ticker: str, price: str, asof: str):
    st.markdown(CSS, unsafe_allow_html=True)
    now = _ist_now()
    status, col = market_status(now)
    feed = ("LIVE · DELAYED", C["warn"]) if feed_live else ("SYNTHETIC", C["bear"])
    st.markdown(f"""
<div class="cmdbar">
  <div class="seg brand">QUANT LAB</div>
  <div class="seg"><span class="dot" style="background:{col};box-shadow:0 0 6px {col}"></span>
     <span style="color:{col}">{status}</span></div>
  <div class="seg">IST <span class="val">{now:%H:%M:%S}</span></div>
  <div class="seg">FEED <span class="val" style="color:{feed[1]}">{feed[0]}</span></div>
  <div class="seg">SYMBOL <span class="val amb">{ticker}</span></div>
  <div class="seg">LAST <span class="val">{price}</span></div>
  <div class="seg">AS OF <span class="val">{asof}</span></div>
  <div class="seg spacer"></div>
  <div class="seg">FX · METALS · CRYPTO · EQUITIES</div>
</div>
<div class="mast"><div style="display:flex;align-items:flex-end">
  <h1><b>QUANT</b>LAB</h1>
  <span class="sub">Multi-asset technical &amp; quant terminal · stocks · indices · forex · gold · crypto</span></div>
  <div class="clock">{now:%d %b %Y · %H:%M} IST</div></div>
<div class="disc">⚠ <b>Not investment advice.</b> This terminal runs mechanical rules over free, delayed,
unofficial Yahoo Finance data. Patterns, Elliott counts and cycles are heuristics. Everything below describes
what the data shows — never a recommendation to buy or sell. Verify against your broker's terminal before acting.</div>
""", unsafe_allow_html=True)


def panel(title: str, sub: str = "", body: str = ""):
    st.markdown(f'<div class="panel"><div class="ph"><div class="t">{title}</div>'
                + (f'<div class="s">{sub}</div>' if sub else "") + "</div>"
                + (f'<div class="pb">{body}</div>' if body else "") + "</div>", unsafe_allow_html=True)


def pill(text: str, verdict: str) -> str:
    cls = {"Bullish": "bull", "Bearish": "bear"}.get(verdict, "neu")
    return f'<span class="pill {cls}">{text}</span>'
