"""QuantLab - multi-asset quant research terminal for FX, Gold and Crypto.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quantlab import backtest as bt
from quantlab import checklist as ck
from quantlab import data, portfolio as pf, risk
from quantlab import indicators as ind
from quantlab import signals as sg

st.set_page_config(page_title="Quant Lab", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(ttl=900, show_spinner=False)
def get_ohlcv(ticker, period, interval):
    return data.load_ohlcv(ticker, period, interval)


@st.cache_data(ttl=900, show_spinner=False)
def get_closes(tickers, period):
    return data.load_closes(list(tickers), period)


def pct(x): return f"{x:.2%}" if pd.notna(x) else "—"
def num(x, d=2): return f"{x:,.{d}f}" if pd.notna(x) else "—"


# ---------------- Controls (top strip, Nifty-Desk style) ----------------
from quantlab import theme
TEMPLATE = theme.TEMPLATE
_hdr = st.container()
_c = st.columns([1.1, 1.4, 1.2, 0.7, 0.7])
cls = _c[0].selectbox("Asset class", list(data.UNIVERSE), index=list(data.UNIVERSE).index("Indices"))
name = _c[1].selectbox("Instrument", list(data.UNIVERSE[cls]))
custom = _c[2].text_input("Any Yahoo ticker", "", placeholder="e.g. TCS.NS, NVDA, GC=F")
interval = _c[3].selectbox("Interval", ["1d", "1h", "1wk"])
period = _c[4].selectbox("History", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)
ticker = custom.strip().upper() or data.UNIVERSE[cls][name]
if interval == "1h" and period not in ("1y", "2y"):
    period = "2y"
ppy = data.periods_per_year(ticker, interval)

raw, live = get_ohlcv(ticker, period, interval)
f = ind.add_all(raw)
sig = sg.composite(f)
reg = sg.regime(f)
_px = f["Close"].iloc[-1]
with _hdr:
    theme.apply(live, ticker, f"{_px:,.5f}" if _px < 10 else f"{_px:,.2f}", str(f.index[-1].date()))
if not live:
    st.warning("Live data unavailable — showing SYNTHETIC data.")

tabs = st.tabs(["✓ Checklist", "Order Flow & Auction", "Dashboard", "Chart & Signals", "Volatility & Regime",
                "Risk", "Backtest", "Cross-Asset", "Portfolio", "Position Sizer"])

with tabs[0]:
    from quantlab.ui_checklist import render
    render(ticker, get_ohlcv)

with tabs[1]:
    from quantlab import ui_orderflow
    ui_orderflow.render(ticker)

# ---------------- Dashboard ----------------
with tabs[2]:
    last = f.iloc[-1]
    score = sig["composite"].iloc[-1]
    c = st.columns(6)
    c[0].metric(ticker, num(last.Close, 5 if last.Close < 10 else 2),
                pct(np.exp(last.ret) - 1))
    c[1].metric("Signal", sg.label(score), num(score))
    c[2].metric("Regime", reg.iloc[-1])
    c[3].metric("EWMA vol (ann.)", pct(ind.ewma_vol(f.ret, ppy=ppy).iloc[-1]))
    c[4].metric("RSI(14)", num(last.rsi, 1))
    h = ind.hurst(f.Close)
    c[5].metric("Hurst", num(h), "trending" if h > 0.55 else "mean-rev" if h < 0.45 else "random walk")

    st.subheader("Signal components (latest)")
    comp = sig.iloc[-1].drop("composite")
    fig = go.Figure(go.Bar(x=comp.values, y=comp.index, orientation="h",
                           marker_color=["#00e676" if v > 0 else "#ff3b3b" for v in comp]))
    fig.update_layout(template=TEMPLATE, height=260, xaxis_range=[-1, 1], margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Market scanner")
    if st.button("Scan full universe"):
        rows = []
        prog = st.progress(0.0)
        all_t = [(c_, n_, t_) for c_, d_ in data.UNIVERSE.items() for n_, t_ in d_.items()]
        for i, (c_, n_, t_) in enumerate(all_t):
            d, _ = get_ohlcv(t_, "2y", "1d")
            ff = ind.add_all(d)
            s_ = sg.composite(ff)["composite"].iloc[-1]
            r_ = ff.ret
            rows.append({"Class": c_, "Instrument": n_, "Last": ff.Close.iloc[-1],
                         "1D": np.exp(r_.iloc[-1]) - 1, "1M": np.exp(r_.tail(21).sum()) - 1,
                         "3M": np.exp(r_.tail(63).sum()) - 1,
                         "Vol": ind.ewma_vol(r_, ppy=data.periods_per_year(t_)).iloc[-1],
                         "RSI": ff.rsi.iloc[-1], "Regime": sg.regime(ff).iloc[-1],
                         "Score": s_, "Signal": sg.label(s_)})
            prog.progress((i + 1) / len(all_t))
        tbl = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(tbl.style.format({"Last": "{:,.4f}", "1D": "{:.2%}", "1M": "{:.2%}",
                                       "3M": "{:.2%}", "Vol": "{:.1%}", "RSI": "{:.1f}",
                                       "Score": "{:+.2f}"})
                     .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-1, vmax=1),
                     width="stretch", hide_index=True)

# ---------------- Chart ----------------
with tabs[3]:
    n = st.slider("Bars shown", 100, len(f), min(400, len(f)))
    v = f.tail(n)
    s = sig.tail(n)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2],
                        vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low,
                                 close=v.Close, name="Price"), 1, 1)
    for col, colr in [("ema20", "#ffb74d"), ("ema50", "#4fc3f7"), ("ema200", "#ba68c8")]:
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name=col.upper(),
                                 line=dict(width=1, color=colr)), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.upper, line=dict(width=0.5, color="gray"),
                             name="BB up", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.lower, line=dict(width=0.5, color="gray"),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
                             name="Bollinger"), 1, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.rsi, name="RSI", line=dict(color="#90caf9")), 2, 1)
    fig.add_hline(70, line_dash="dot", row=2, col=1); fig.add_hline(30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=v.index, y=v["hist"], name="MACD hist",
                         marker_color=np.where(v["hist"] > 0, "#00e676", "#ff3b3b")), 3, 1)
    fig.add_trace(go.Scatter(x=v.index, y=v.macd, name="MACD", line=dict(width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=s.index, y=s.composite, name="Composite", fill="tozeroy",
                             line=dict(color="#ffd54f")), 4, 1)
    fig.update_layout(template=TEMPLATE, height=850, xaxis_rangeslider_visible=False,
                      margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sig.tail(10).style.format("{:+.2f}"), width="stretch")

# ---------------- Volatility & Regime ----------------
with tabs[4]:
    r = f.ret.dropna()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Realised vs EWMA volatility")
        rv = pd.DataFrame({"RV 20": r.rolling(20).std() * np.sqrt(ppy),
                           "RV 60": r.rolling(60).std() * np.sqrt(ppy),
                           "EWMA λ=.94": ind.ewma_vol(r, ppy=ppy)})
        fig = go.Figure([go.Scatter(x=rv.index, y=rv[k], name=k) for k in rv])
        fig.update_layout(template=TEMPLATE, height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("GARCH(1,1)")
        with st.spinner("Fitting GARCH…"):
            g = ind.garch11(r.tail(1500))
        st.table(pd.DataFrame({
            "Value": [num(g["omega"], 8), num(g["alpha"], 4), num(g["beta"], 4),
                      num(g["persistence"], 4),
                      pct(np.sqrt(g["next_var"] * ppy)), pct(np.sqrt(g["long_run_var"] * ppy))]},
            index=["ω", "α", "β", "α+β (persistence)", "Next-bar vol (ann.)", "Long-run vol (ann.)"]))
        hl = ind.half_life(f.Close.tail(500))
        st.metric("OU half-life (last 500 bars)", "∞ (no reversion)" if np.isinf(hl) else f"{hl:.1f} bars")
    st.subheader("Regime map")
    rr = reg.tail(750)
    fig = go.Figure()
    for lbl in sorted(rr.unique()):
        m = rr == lbl
        fig.add_trace(go.Scatter(x=rr.index[m], y=f.Close.loc[rr.index][m], mode="markers",
                                 marker=dict(size=4), name=lbl))
    fig.update_layout(template=TEMPLATE, height=420)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(reg.value_counts(normalize=True).rename("share").to_frame()
                 .style.format("{:.1%}"))

# ---------------- Risk ----------------
with tabs[5]:
    r = f.ret.dropna()
    lvl = st.select_slider("Confidence", [0.9, 0.95, 0.975, 0.99], value=0.95)
    vc = risk.var_cvar(r, lvl)
    m = risk.metrics(r, ppy)
    c = st.columns(4)
    c[0].metric(f"Historical VaR {lvl:.1%}", pct(vc["hist_var"]))
    c[1].metric("Expected Shortfall", pct(vc["hist_cvar"]))
    c[2].metric("Gaussian VaR", pct(vc["param_var"]))
    c[3].metric("Cornish-Fisher VaR", pct(vc["cf_var"]))
    c = st.columns(5)
    for i, k in enumerate(["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD"]):
        c[i].metric(k + " (buy&hold)", pct(m[k]) if k in ("CAGR", "Vol", "MaxDD") else num(m[k]))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=r, nbinsx=120, histnorm="probability density", name="returns"))
        x = np.linspace(r.min(), r.max(), 300)
        from scipy.stats import norm
        fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, r.mean(), r.std()), name="Normal"))
        fig.add_vline(-vc["hist_var"], line_color="red", annotation_text="VaR")
        fig.update_layout(template=TEMPLATE, height=380, title=f"Return distribution · skew {m['Skew']:.2f} · excess kurt {m['Kurtosis']:.2f}")
        st.plotly_chart(fig, width="stretch")
    with c2:
        dd = risk.drawdown(np.exp(r.cumsum()))
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line_color="#ff3b3b"))
        fig.update_layout(template=TEMPLATE, height=380, title="Underwater curve", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Monte Carlo (block bootstrap)")
    hz = st.slider("Horizon (bars)", 20, 756, 252)
    paths = risk.monte_carlo(r, hz, 2000)
    qs = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    for q, nm in zip(qs, ["P5", "P25", "Median", "P75", "P95"]):
        fig.add_trace(go.Scatter(y=q, name=nm))
    fig.update_layout(template=TEMPLATE, height=380, yaxis_title="Growth of 1")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"P(loss at horizon) = {(paths[:, -1] < 1).mean():.1%} · "
               f"P(drawdown > 20%) = {((paths / np.maximum.accumulate(paths, axis=1)).min(axis=1) < 0.8).mean():.1%}")

# ---------------- Backtest ----------------
with tabs[6]:
    c = st.columns(5)
    tv = c[0].slider("Target vol", 0.05, 0.5, 0.15, 0.01)
    ml = c[1].slider("Max leverage", 0.5, 10.0, 3.0, 0.5)
    cost = c[2].number_input("Cost (bps/turnover)", 0.0, 50.0, 2.0 if cls == "Forex" else 5.0)
    th = c[3].slider("Signal threshold", 0.0, 0.6, 0.1, 0.05)
    lo = c[4].checkbox("Long only", cls == "Crypto")
    strat = st.selectbox("Strategy", ["composite", "trend", "momentum", "mean_rev", "breakout"])
    cfg = bt.BTConfig(tv, ml, cost, th, ppy, lo)
    res = bt.run(f, sig[strat], cfg)
    mm, bm = res["metrics"], res["bench_metrics"]
    st.dataframe(pd.DataFrame({"Strategy": mm, "Buy & Hold": bm}).T
                 .style.format("{:.3f}"), width="stretch")
    psr = risk.probabilistic_sharpe(res["returns"])
    st.caption(f"Probabilistic Sharpe Ratio (SR>0): **{psr:.1%}** · Kelly fraction: {risk.kelly_fraction(res['returns']):.2f}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"], name="Strategy"), 1, 1)
    fig.add_trace(go.Scatter(x=res["bench"].index, y=res["bench"], name="Buy & Hold"), 1, 1)
    fig.add_trace(go.Scatter(x=res["position"].index, y=res["position"], name="Position",
                             fill="tozeroy"), 2, 1)
    fig.update_layout(template=TEMPLATE, height=600, yaxis_type="log")
    st.plotly_chart(fig, width="stretch")
    mret = res["returns"].resample("ME").sum()
    heat = pd.DataFrame({"y": mret.index.year, "m": mret.index.month, "r": np.exp(mret.values) - 1})
    heat = heat.pivot(index="y", columns="m", values="r")
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="RdYlGn",
                               zmid=0, text=np.vectorize(lambda x: f"{x:.1%}" if pd.notna(x) else "")(heat.values),
                               texttemplate="%{text}"))
    fig.update_layout(template=TEMPLATE, height=300, title="Monthly returns")
    st.plotly_chart(fig, width="stretch")
    if st.checkbox("Run walk-forward optimisation (composite)"):
        wf = bt.walk_forward(raw, cfg, train=min(500, len(f) // 2), test=125)
        if wf.empty:
            st.info("Not enough history for walk-forward.")
        else:
            st.dataframe(wf, width="stretch")
            oos = wf.attrs["oos_returns"]
            st.metric("Stitched OOS Sharpe", num(risk.metrics(oos, ppy)["Sharpe"]))

# ---------------- Cross-asset ----------------
default = ["EURUSD=X", "GBPUSD=X", "JPY=X", "DX-Y.NYB", "GC=F", "SI=F", "BTC-USD", "ETH-USD"]
all_tickers = {t: n for d in data.UNIVERSE.values() for n, t in d.items()}
with tabs[7]:
    sel = st.multiselect("Instruments", list(all_tickers), default,
                         format_func=lambda t: all_tickers[t])
    if len(sel) >= 2:
        closes, live_x = get_closes(tuple(sel), period)
        rets = np.log(closes).diff().dropna()
        rets.columns = [all_tickers[t] for t in rets.columns]
        win = st.slider("Rolling window", 20, 250, 60)
        corr = rets.tail(win).corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale="RdBu", text=corr.round(2).values, texttemplate="%{text}"))
        fig.update_layout(template=TEMPLATE, height=520, title=f"Correlation (last {win} bars)")
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        x1 = a.selectbox("Asset", rets.columns, index=min(4, len(rets.columns) - 1))
        x2 = b.selectbox("vs", rets.columns, index=0)
        rc = rets[x1].rolling(win).corr(rets[x2])
        beta = pf.rolling_beta(rets[x1], rets[x2], win)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=rc.index, y=rc, name="Rolling corr"), 1, 1)
        fig.add_trace(go.Scatter(x=beta.index, y=beta, name="Rolling beta"), 2, 1)
        fig.update_layout(template=TEMPLATE, height=450)
        st.plotly_chart(fig, width="stretch")
        perf = np.exp(rets.cumsum())
        fig = go.Figure([go.Scatter(x=perf.index, y=perf[k], name=k) for k in perf])
        fig.update_layout(template=TEMPLATE, height=420, yaxis_type="log", title="Relative performance")
        st.plotly_chart(fig, width="stretch")

# ---------------- Portfolio ----------------
with tabs[8]:
    sel = st.multiselect("Universe", list(all_tickers), ["EURUSD=X", "GC=F", "BTC-USD", "JPY=X", "ETH-USD"],
                         format_func=lambda t: all_tickers[t], key="pf")
    if len(sel) >= 2:
        closes, _ = get_closes(tuple(sel), period)
        rets = np.log(closes).diff().dropna()
        rets.columns = [all_tickers[t] for t in rets.columns]
        cov = pf.shrunk_cov(rets) * 252
        mu = rets.mean() * 252
        W = pd.DataFrame({"Equal": pd.Series(1 / len(mu), mu.index),
                          "Risk Parity": pf.risk_parity(cov),
                          "Min Variance": pf.min_variance(cov),
                          "Max Sharpe": pf.max_sharpe(mu, cov)})
        st.dataframe(W.style.format("{:.1%}"), width="stretch")
        stats_ = {}
        for k in W:
            pr = rets @ W[k]
            stats_[k] = risk.metrics(pr, 252)
        st.dataframe(pd.DataFrame(stats_).T[["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD"]]
                     .style.format("{:.3f}"), width="stretch")
        c1, c2 = st.columns(2)
        ef = pf.efficient_frontier(mu, cov)
        fig = go.Figure(go.Scatter(x=ef.vol, y=ef.ret, mode="markers",
                                   marker=dict(color=ef.sharpe, colorscale="Viridis", size=3,
                                               showscale=True), name="random"))
        for k in W:
            w = W[k].values
            fig.add_trace(go.Scatter(x=[np.sqrt(w @ cov.values @ w)], y=[w @ mu.values],
                                     mode="markers+text", text=[k], marker=dict(size=12), name=k))
        fig.update_layout(template=TEMPLATE, height=450, xaxis_title="Vol", yaxis_title="Return",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        c1.plotly_chart(fig, width="stretch")
        rc = pd.DataFrame({k: pf.risk_contributions(W[k], cov) for k in W})
        fig = go.Figure([go.Bar(x=rc.columns, y=rc.loc[a], name=a) for a in rc.index])
        fig.update_layout(template=TEMPLATE, barmode="stack", height=450, title="Risk contribution",
                          yaxis_tickformat=".0%")
        c2.plotly_chart(fig, width="stretch")
        st.caption("Note: Max-Sharpe uses historical means, which are noisy — prefer risk-based weights.")

# ---------------- Position sizer ----------------
with tabs[9]:
    a_ = float(f.atr.iloc[-1]); px = float(f.Close.iloc[-1])
    c = st.columns(4)
    eq = c[0].number_input("Account equity ($)", 100.0, 1e9, 10000.0)
    rp = c[1].number_input("Risk per trade (%)", 0.1, 10.0, 1.0) / 100
    entry = c[2].number_input("Entry", value=px, format="%.5f")
    mult = c[3].number_input("Stop = ATR ×", 0.5, 10.0, 2.0)
    direction = "Long" if sig["composite"].iloc[-1] >= 0 else "Short"
    stop = entry - mult * a_ if direction == "Long" else entry + mult * a_
    tp = entry + 2 * (entry - stop)
    contract = {"Forex": 100000.0, "Metals": 100.0}.get(cls, 1.0) if not custom else 1.0
    ps = risk.position_size(eq, rp, entry, stop, contract)
    c = st.columns(5)
    c[0].metric("Bias", direction)
    c[1].metric("Stop-loss", num(stop, 5))
    c[2].metric("Take-profit (2R)", num(tp, 5))
    c[3].metric("Size", f"{ps['units']:.3f} " + ("lots" if contract > 1 else "units"))
    c[4].metric("Notional / Leverage", f"${ps['notional']:,.0f}", f"{ps['leverage']:.1f}x")
    st.caption(f"ATR(14) = {a_:.5f}. Contract size = {contract:g}. Risk amount = ${ps['risk_amount']:,.2f}.")

st.markdown('<div style="border-top:1px solid #2a2a2a;margin-top:30px;padding-top:12px;color:#9a8f7a;font-size:11px">'
            "SRC: Yahoo Finance (delayed EOD/intraday) · indicators computed locally · unofficial feed — verify against your broker terminal</div>", unsafe_allow_html=True)
