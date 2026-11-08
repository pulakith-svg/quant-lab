"""Streamlit view: LTP, prints, delta/CVD, footprint, VWAP, volume & TPO profile, AMT read."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import orderflow as of
from . import theme

T, C = theme.TEMPLATE, theme.C


@st.cache_data(ttl=60, show_spinner=False)
def _load(ticker):
    return of.load_intraday(ticker)


def _f(x):
    return "—" if x is None or not np.isfinite(x) else (f"{x:,.5f}" if abs(x) < 10 else f"{x:,.2f}")


def render(ticker: str):
    theme.panel(f"Order flow & auction — {ticker}",
                "LTP · prints · delta / CVD · footprint · VWAP · volume profile · TPO market profile · auction market theory")
    c0 = st.columns([1, 1, 1, 2])
    bar = c0[0].selectbox("Footprint / delta bar", ["5min", "15min", "30min", "60min"], index=1)
    if c0[1].button("↻ Refresh feed"):
        _load.clear()
    with st.spinner("Loading 1-minute data…"):
        df, meta = _load(ticker)
    c0[3].markdown(f'<div style="padding-top:28px;font-size:11px;color:{C["muted"]}">SOURCE · '
                   f'<span style="color:{C["bull"] if meta["real_delta"] else C["warn"]}">{meta["source"]}</span></div>',
                   unsafe_allow_html=True)
    if not meta["has_volume"]:
        st.warning("This instrument publishes no volume (cash index / spot FX). Delta, footprint and volume "
                   "profile are not meaningful — the profile below is TIME-based (TPO). For volume, use the "
                   "futures or an ETF (e.g. NIFTYBEES.NS, GC=F, 6E=F).")
    if not meta["real_delta"] and meta["has_volume"]:
        st.info("Delta is ESTIMATED from 1-minute bar structure — true bid/ask aggression is not in free equity / "
                "FX / futures feeds. Crypto pairs use real Binance taker-buy volume when reachable.")

    a = of.auction_read(df)
    if "error" in a:
        st.error(a["error"]); return
    today = of.delta_stats(a["today_df"])
    bars = of.bar_summary(today, bar)
    vw = of.vwap(today)
    ltp = today.Close.iloc[-1]
    prev_close = a["prev_df"].Close.iloc[-1]
    tpo, pv, tv = a["tpo"], a["prev"], a["today"]

    # ---------------------------------------------------------------- tiles
    k = st.columns(6)
    k[0].metric("LTP", _f(ltp), f"{ltp / prev_close - 1:+.2%}")
    k[1].metric("Session VWAP", _f(vw.vwap.iloc[-1]), f"{(ltp / vw.vwap.iloc[-1] - 1):+.2%} vs VWAP")
    k[2].metric("Session delta", f"{today.delta.sum():+,.0f}" if meta["has_volume"] else "n/a",
                f"{today.delta.sum() / max(today.Volume.sum(), 1e-9):+.1%} of vol" if meta["has_volume"] else None)
    k[3].metric("Today POC", _f(tv["poc"]), f"VA {_f(tv['val'])} – {_f(tv['vah'])}", delta_color="off")
    k[4].metric("Prior POC", _f(pv["poc"]), f"VA {_f(pv['val'])} – {_f(pv['vah'])}", delta_color="off")
    k[5].metric("Initial balance", f"{_f(tpo['ib_low'])} – {_f(tpo['ib_high'])}",
                ("ext ↑ " if tpo["ext_up"] else "") + ("ext ↓" if tpo["ext_down"] else "") or "inside IB", delta_color="off")

    # ----------------------------------------------------------- AMT verdict
    sc = a["score"]
    verdict = "BUY-SIDE CONTROL" if sc > 0.3 else "SELL-SIDE CONTROL" if sc < -0.3 else "TWO-SIDED / BALANCE"
    vc = C["bull"] if sc > 0.3 else C["bear"] if sc < -0.3 else C["warn"]
    plan = _plan(a, ltp, vw.vwap.iloc[-1], sc)
    st.markdown(f"""<div class="panel"><div class="ph"><div class="t">Auction market read</div>
<div class="s">Where is price vs value · who is in control · what the auction is likely to do next</div></div>
<div class="pb"><div style="font-size:22px;font-weight:700;color:{vc};letter-spacing:.06em">{verdict}
<span style="font-size:12px;color:{C['muted']}">score {sc:+.2f}</span></div>
OPEN TYPE <b>{a['open_type']}</b><br>
OPEN LOCATION <b>{a['open_location']}</b> · PRICE NOW <b>{a['price_location']}</b><br>
DAY TYPE <b>{tpo['day_type']}</b><br>
PROFILE SHAPE <b>{tv['shape']}</b><br>
VALUE MIGRATION <b>{a['value_migration']}</b><br>
MARKET STATE <b>{a['market_state']}</b><br>
{'<br>'.join(re.sub(r'[*][*](.+?)[*][*]', r'<b>\1</b>', n) for n in a['notes'])}<br>
SINGLE PRINTS <b>{', '.join(_f(x) for x in tpo['single_prints'][:6]) or 'none'}</b> ·
POOR HIGH <b>{'YES (unfinished auction — likely revisited)' if tpo['poor_high'] else 'no (excess)'}</b> ·
POOR LOW <b>{'YES (unfinished auction — likely revisited)' if tpo['poor_low'] else 'no (excess)'}</b><br>
NAKED POCs (untested magnets) <b>{', '.join(_f(x) for x in a['naked_pocs']) or 'none'}</b><br>
HVN <b>{', '.join(_f(x) for x in tv['hvn'][:5]) or '—'}</b> · LVN <b>{', '.join(_f(x) for x in tv['lvn'][:5]) or '—'}</b>
</div></div>""", unsafe_allow_html=True)
    pc = C["bull"] if plan["side"] == "LONG" else C["bear"] if plan["side"] == "SHORT" else C["warn"]
    st.markdown(f"""<div class="panel"><div class="ph"><div class="t">Intraday auction plan</div></div>
<div class="pb"><b style="color:{pc};font-size:16px">{plan['side']}</b> — {plan['why']}<br>
ENTRY <b>{_f(plan['entry'])}</b> · STOP <b style="color:{C['bear']}">{_f(plan['stop'])}</b> ·
TARGET 1 <b style="color:{C['bull']}">{_f(plan['t1'])}</b> · TARGET 2 <b style="color:{C['bull']}">{_f(plan['t2'])}</b></div></div>""",
                unsafe_allow_html=True)

    # ---------------------------------------- price + VWAP + volume profile
    b5 = today.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, column_widths=[0.8, 0.2],
                        row_heights=[0.72, 0.28], horizontal_spacing=0.005, vertical_spacing=0.03,
                        specs=[[{}, {}], [{}, None]])
    fig.add_trace(go.Candlestick(x=b5.index, open=b5.Open, high=b5.High, low=b5.Low, close=b5.Close, name="5m",
                                 increasing_line_color=C["bull"], decreasing_line_color=C["bear"]), 1, 1)
    for col, dash, nm in (("vwap", "solid", "VWAP"), ("u1", "dot", "+1σ"), ("l1", "dot", "−1σ"),
                          ("u2", "dash", "+2σ"), ("l2", "dash", "−2σ")):
        fig.add_trace(go.Scatter(x=vw.index, y=vw[col], name=nm, line=dict(color=C["info"], dash=dash,
                                 width=1.6 if col == "vwap" else 0.8)), 1, 1)
    for y, nm, colr in ((pv["poc"], "pPOC", C["accent"]), (pv["vah"], "pVAH", C["muted"]), (pv["val"], "pVAL", C["muted"]),
                        (tpo["ib_high"], "IBH", C["warn"]), (tpo["ib_low"], "IBL", C["warn"])):
        fig.add_hline(y, line=dict(color=colr, width=1, dash="dash"), annotation_text=nm,
                      annotation_position="left", row=1, col=1)
    for p in a["naked_pocs"]:
        fig.add_hline(p, line=dict(color="#b388ff", width=1, dash="dot"), annotation_text="nPOC", row=1, col=1)
    lv = tv["levels"]
    in_va = (lv >= tv["val"]) & (lv <= tv["vah"])
    if tv["by"] == "volume":
        fig.add_trace(go.Bar(y=lv, x=tv["buy"], orientation="h", name="Ask vol", marker_color=C["bull"],
                             marker_opacity=np.where(in_va, .9, .35)), 1, 2)
        fig.add_trace(go.Bar(y=lv, x=tv["sell"], orientation="h", name="Bid vol", marker_color=C["bear"],
                             marker_opacity=np.where(in_va, .9, .35)), 1, 2)
    else:
        fig.add_trace(go.Bar(y=lv, x=tv["volume"], orientation="h", name="TPO", marker_color=C["accent"],
                             marker_opacity=np.where(in_va, .9, .35)), 1, 2)
    fig.add_hline(tv["poc"], line=dict(color=C["accent"], width=2), annotation_text="POC", row=1, col=2)
    fig.add_trace(go.Bar(x=bars.index, y=bars.delta, name="Delta",
                         marker_color=np.where(bars.delta > 0, C["bull"], C["bear"])), 2, 1)
    fig.update_layout(template=T, height=760, barmode="stack", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h"), margin=dict(t=20, l=10, r=10))
    st.plotly_chart(fig, width="stretch")
    if meta["has_volume"]:
        cv = go.Figure(go.Scatter(x=today.index, y=today.cvd, fill="tozeroy", name="CVD",
                                  line=dict(color=C["accent"], width=1.5)))
        cv.update_layout(template=T, height=220, title="Cumulative volume delta (CVD) — session",
                         margin=dict(t=36, l=10, r=10))
        st.plotly_chart(cv, width="stretch")

    # -------------------------------------------------------------- footprint
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fp = of.footprint(today.tail(int(pd.Timedelta(bar) / pd.Timedelta("1min")) * 12), bar)
        if len(fp) and meta["has_volume"]:
            piv = fp.pivot(index="price", columns="bar", values="delta").sort_index(ascending=False)
            kf = lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"
            txt = fp.assign(t=[f"{kf(b)}×{kf(a_)}" for b, a_ in zip(fp.bid, fp.ask)]) \
                .pivot(index="price", columns="bar", values="t").reindex_like(piv).fillna("")
            m = np.nanmax(np.abs(piv.values)) or 1
            fig = go.Figure(go.Heatmap(z=piv.values, x=[t.strftime("%H:%M") for t in piv.columns], y=piv.index,
                                       text=txt.values, texttemplate="%{text}", textfont=dict(size=8),
                                       colorscale=[[0, C["bear"]], [0.5, "#111111"], [1, C["bull"]]], zmid=0,
                                       zmin=-m, zmax=m, colorbar=dict(title="Δ")))
            fig.update_layout(template=T, height=560, title=f"Footprint (bid × ask, colour = delta) · {bar}",
                              yaxis=dict(dtick=fp.attrs["box"] * 2), margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
            imb = fp[fp.buy_imb | fp.sell_imb]
            st.caption(f"Stacked imbalances (≥3:1 diagonal): {int(fp.buy_imb.sum())} buy-side, {int(fp.sell_imb.sum())} sell-side "
                       f"in the window shown." + (" Latest at " + _f(imb.price.iloc[-1]) if len(imb) else ""))
        else:
            st.info("Footprint needs volume.")
    with c2:
        # TPO letters
        lv_ = tpo["levels"][::-1]
        lines = []
        for p in lv_:
            tag = " ◀POC" if abs(p - tpo["tpo_poc"]) < 1e-9 else (" ◀VAH" if abs(p - tpo["tpo_vah"]) < 1e-9 else
                                                                   " ◀VAL" if abs(p - tpo["tpo_val"]) < 1e-9 else "")
            col = C["accent"] if tag else (C["text"] if tpo["tpo_val"] <= p <= tpo["tpo_vah"] else C["muted"])
            lines.append(f'<span style="color:{C["muted"]}">{_f(p):>11}</span> '
                         f'<span style="color:{col}">{tpo["letters"][p]}{tag}</span>')
        st.markdown(f'<div class="panel" style="margin-top:0"><div class="ph"><div class="t">TPO market profile</div>'
                    f'<div class="s">30-min periods A, B, C… · IB = A+B</div></div>'
                    f'<div class="pb" style="font-size:11px;line-height:1.25;white-space:pre;overflow:auto;max-height:520px">'
                    + "\n".join(lines) + "</div></div>", unsafe_allow_html=True)

    # ------------------------------------------------ signals & prints tables
    c3, c4 = st.columns(2)
    with c3:
        theme.panel("Order-flow signals", "Absorption · initiative activity · CVD divergence")
        sig = of.orderflow_signals(bars) if meta["has_volume"] else []
        if sig:
            st.dataframe(pd.DataFrame(sig).sort_values("time", ascending=False), width="stretch", hide_index=True)
        else:
            st.caption("No absorption, initiative or divergence signal in recent bars.")
    with c4:
        theme.panel("Large prints · time & sales",
                    "Real trades (crypto)" if meta["prints"] is not None else "Volume-spike 1-min bars (no tick feed)")
        lp = of.large_prints(meta["prints"], today) if meta["has_volume"] else pd.DataFrame()
        if len(lp):
            st.dataframe(lp.style.map(lambda v: f"color:{C['bull']}" if v == "BUY" else f"color:{C['bear']}" if v == "SELL" else "",
                                      subset=["side"]).format({"price": "{:,.5g}", "size": "{:,.2f}"}),
                         width="stretch", hide_index=True, height=360)
        else:
            st.caption("No volume data for prints.")
    st.caption("AMT rules of thumb: acceptance = time + volume at a price; rejection = fast move away (excess, single prints). "
               "Open inside value → rotation; open outside value → look for acceptance or a return (80% rule). "
               "Poor highs/lows and naked POCs act as magnets. Heuristic read — not investment advice.")


def _plan(a, ltp, vwap_now, sc):
    pv, tv = a["prev"], a["today"]
    box = pv["box"]
    if sc > 0.3:
        entry = max(vwap_now, tv["poc"]) if ltp > max(vwap_now, tv["poc"]) else ltp
        stop = min(tv["val"], vwap_now) - 2 * box
        t1 = max(ltp, tv["vah"], pv["vah"]) + (pv["vah"] - pv["val"]) * 0.5
        t2 = t1 + (pv["vah"] - pv["val"])
        return {"side": "LONG", "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                "why": "price and value are above prior value / VWAP — buy pullbacks to VWAP or today's POC."}
    if sc < -0.3:
        entry = min(vwap_now, tv["poc"]) if ltp < min(vwap_now, tv["poc"]) else ltp
        stop = max(tv["vah"], vwap_now) + 2 * box
        t1 = min(ltp, tv["val"], pv["val"]) - (pv["vah"] - pv["val"]) * 0.5
        t2 = t1 - (pv["vah"] - pv["val"])
        return {"side": "SHORT", "entry": entry, "stop": stop, "t1": t1, "t2": t2,
                "why": "price and value are below prior value / VWAP — sell rallies into VWAP or today's POC."}
    near_top = ltp > (pv["poc"] + pv["vah"]) / 2
    if near_top:
        return {"side": "FADE SHORT AT VAH", "entry": pv["vah"], "stop": pv["vah"] + 3 * box, "t1": pv["poc"],
                "t2": pv["val"], "why": "balanced auction — fade the value-area high back to POC / VAL."}
    return {"side": "FADE LONG AT VAL", "entry": pv["val"], "stop": pv["val"] - 3 * box, "t1": pv["poc"],
            "t2": pv["vah"], "why": "balanced auction — fade the value-area low back to POC / VAH."}
