"""Streamlit rendering for the Technical Checklist tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import checklist as ck

from . import theme
T = theme.TEMPLATE
COL = {"Bullish": "#00e676", "Bearish": "#ff3b3b", "Neutral": "#9e9e9e"}
ICON = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}


@st.cache_data(ttl=900, show_spinner=False)
def _run(ticker, _loader, with_context):
    return ck.analyze(ticker, loader=_loader, with_context=with_context)


def _gauge(value, title, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                        "bgcolor": theme.C["surface2"], "bordercolor": theme.C["line2"]}))
    fig.update_layout(template=T, height=210, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def _chart(r):
    f = r["df"].tail(300)
    fig = go.Figure(go.Candlestick(x=f.index, open=f.Open, high=f.High, low=f.Low, close=f.Close,
                                   name="Price", increasing_line_color=theme.C["bull"],
                                   decreasing_line_color=theme.C["bear"]))
    for n, cl in ((50, theme.C["info"]), (200, "#b388ff")):
        fig.add_trace(go.Scatter(x=f.index, y=r["df"].Close.rolling(n).mean().tail(300),
                                 name=f"MA{n}", line=dict(width=1, color=cl)))
    for l in r["sr"]["support"]:
        fig.add_hline(l["price"], line=dict(color="#00e676", dash="dot", width=1),
                      annotation_text=f"S {l['price']:.5g}", annotation_position="right")
    for l in r["sr"]["resistance"]:
        fig.add_hline(l["price"], line=dict(color="#ff3b3b", dash="dot", width=1),
                      annotation_text=f"R {l['price']:.5g}", annotation_position="right")
    for k, v in r["fib"].items():
        fig.add_hline(v, line=dict(color=theme.C["accent"], width=0.6, dash="dash"),
                      annotation_text=f"Fib {k}", annotation_position="left")
    ch = r["channel"]
    idx = ch["index"]
    for off, nm in ((0, "Channel mid"), (ch["k"] * ch["sd"], "Channel up"), (-ch["k"] * ch["sd"], "Channel low")):
        fig.add_trace(go.Scatter(x=idx, y=ch["fit"] + off, name=nm,
                                 line=dict(color="#90a4ae", width=1, dash="dot" if off else "solid")))
    for key, colr in (("up", "#00e676"), ("down", "#ff3b3b")):
        t = r["trendlines"].get(key)
        if t:
            fig.add_trace(go.Scatter(x=[t["from"].date, r["df"].index[-1]],
                                     y=[t["from"].price, t["value_now"]], name=f"{key}-trendline",
                                     line=dict(color=colr, width=2)))
    pv = [p for p in r["pivots"] if p.date >= f.index[0]]
    fig.add_trace(go.Scatter(x=[p.date for p in pv], y=[p.price for p in pv], mode="lines+markers",
                             name="Swings", line=dict(color="white", width=0.8)))
    for d, lab in r["elliott"].get("wave_labels", []):
        p = next((q for q in r["pivots"] if q.date == d), None)
        if p and d >= f.index[0]:
            fig.add_annotation(x=d, y=p.price, text=f"<b>{lab}</b>", showarrow=False,
                               yshift=14 if p.kind == "H" else -14, font=dict(color=theme.C["accent"]))
    plan = r["plans"]["Swing (days–weeks)"]
    if plan["action"] in ("BUY", "SELL"):
        for k_, colr in (("entry", "#ffffff"), ("stop", "#ff1744"), ("t1", "#00e676"), ("t2", "#00e676")):
            fig.add_hline(plan[k_], line=dict(color=colr, width=1.5),
                          annotation_text=f"{k_.upper()} {plan[k_]:.5g}", annotation_position="top left")
    fig.update_layout(template=T, height=700, xaxis_rangeslider_visible=False, margin=dict(t=20),
                      legend=dict(orientation="h"))
    return fig


def _pf_chart(pf):
    cols = pf["columns"][-40:]
    xs, ys, txt = [], [], []
    for i, c in enumerate(cols):
        n = int(round((c["hi"] - c["lo"]) / pf["box"])) + 1
        for j in range(n):
            xs.append(i); ys.append(c["lo"] + j * pf["box"]); txt.append(c["type"])
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="text", text=txt,
                               textfont=dict(color=["#00e676" if t == "X" else "#ff3b3b" for t in txt])))
    fig.update_layout(template=T, height=380, title="Point & Figure (3-box reversal)",
                      xaxis_title="column", margin=dict(t=40))
    return fig


def render(ticker: str, loader):
    theme.panel(f"Technical checklist — {ticker}",
                "Market · trend · levels · volume · patterns · oscillators · Elliott · cycles · P&F → scores and trade plan")
    ctx = st.checkbox("Include market & sector context (slower: downloads benchmark + sectors)", True)
    with st.spinner("Running the full technical checklist…"):
        r = _run(ticker, loader, ctx)
    if not r["live"]:
        st.warning("Live data unavailable — results below use SYNTHETIC data.")

    c = st.columns([1, 1, 1.4])
    c[0].plotly_chart(_gauge(r["buy_score"], "BUY SCORE", theme.C["bull"]), width="stretch")
    c[1].plotly_chart(_gauge(r["sell_score"], "SELL SCORE", theme.C["bear"]), width="stretch")
    with c[2]:
        vcol = theme.C["bull"] if r["bias"] == "BUY" else theme.C["bear"] if r["bias"] == "SELL" else theme.C["warn"]
        st.markdown(f"""<div class="panel" style="margin-top:0"><div class="ph"><div class="t">Desk verdict</div></div>
<div class="pb"><div style="font-size:26px;font-weight:700;color:{vcol};letter-spacing:.06em">{r['bias']}</div>
MARKET <b>{r['market']}</b> · LAST <b>{ck.fmt(r['price'])}</b><br>
OUTLOOK (NEXT MONTHS) <b>{r['outlook']}</b><br>
NET SCORE <b>{r['net']:+.2f}</b> <span style="font-size:10.5px">(−1 strong sell … +1 strong buy)</span></div></div>""",
                    unsafe_allow_html=True)

    theme.panel("Trade plans", "Intraday (hourly) · swing (daily) · position (weekly) — entry / stop / targets")
    rows = []
    for k, p in r["plans"].items():
        rows.append({"Horizon": k, "Action": p["action"],
                     "Score": p["score"], "Entry": p["entry"], "Stop": p["stop"],
                     "Target 1": p["t1"], "Target 2": p["t2"], "R:R": p["rr"]})
    st.dataframe(pd.DataFrame(rows).style.format({c_: "{:,.5g}" for c_ in
                                                  ("Score", "Entry", "Stop", "Target 1", "Target 2", "R:R")},
                                                 na_rep="—"), width="stretch", hide_index=True)
    best = max(r["plans"].values(), key=lambda p: abs(p["score"]) if pd.notna(p["score"]) and p["action"] in ("BUY", "SELL") else -1)
    if best["action"] in ("BUY", "SELL"):
        st.success(f"Best-aligned horizon: **{best['horizon']}** — {best['entry_note']} {best['stop_note']} {best['target_note']}")
    else:
        st.info("No horizon has a clear edge right now — standing aside is a position too.")

    st.plotly_chart(_chart(r), width="stretch")

    theme.panel("The checklist", "Every question answered from the data — click a row to expand")
    cur = None
    for it in r["items"]:
        if it.section != cur:
            cur = it.section
            st.markdown(f'<div style="color:{theme.C["accent"]};font-size:11px;letter-spacing:.14em;'
                        f'text-transform:uppercase;margin:14px 0 4px;border-bottom:1px solid {theme.C["line"]};'
                        f'padding-bottom:4px">{cur}</div>', unsafe_allow_html=True)
        with st.expander(f"{ICON[it.verdict]} {it.question}", expanded=it.section == "Decision"):
            st.markdown(theme.pill(it.verdict.upper(), it.verdict), unsafe_allow_html=True)
            st.markdown(it.answer)
            if it.weight:
                st.caption(f"Verdict: {it.verdict} · score {it.score:+.2f} · weight {it.weight:g}")
            if "table" in it.details:
                st.dataframe(it.details["table"].style.format({"1M": "{:.1%}", "3M": "{:.1%}", "Score": "{:+.2f}"}),
                             width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_pf_chart(r["pf"]), width="stretch")
    with c2:
        sc = pd.DataFrame([{"Factor": i.question[:55], "Score": i.score * i.weight}
                           for i in r["items"] if i.weight > 0])
        fig = go.Figure(go.Bar(x=sc.Score, y=sc.Factor, orientation="h",
                               marker_color=np.where(sc.Score > 0, "#00e676", "#ff3b3b")))
        fig.update_layout(template=T, height=380, title="Weighted evidence", margin=dict(t=40, l=10),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=9)))
        st.plotly_chart(fig, width="stretch")
    st.caption("Automated, rule-based analysis. Patterns, Elliott counts and cycles are heuristics — "
               "always confirm on the chart. Not investment advice.")
