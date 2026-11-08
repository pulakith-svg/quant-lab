"""The classic technical-analysis checklist, answered automatically.

`analyze(ticker)` runs every question (market & sector direction, multi-
timeframe trends, S/R, trendlines, volume, retracements, patterns & targets,
moving averages, oscillators, divergences, sentiment, Elliott, Fibonacci,
cycles, computer trend, P&F, candlesticks) and turns the evidence into
Buy / Sell scores and a trade plan for intraday, swing and position horizons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from . import indicators as ind
from . import patterns as pt
from . import signals as sg

# ---------------------------------------------------------- market context
CONTEXT = {
    "US Stocks": {"benchmark": ("S&P 500", "^GSPC"), "sectors": {
        "Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Health Care": "XLV",
        "Cons. Discretionary": "XLY", "Cons. Staples": "XLP", "Industrials": "XLI",
        "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication": "XLC"}},
    "India Stocks": {"benchmark": ("NIFTY 50", "^NSEI"), "sectors": {
        "Bank Nifty": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "Pharma": "^CNXPHARMA",
        "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL", "Energy": "^CNXENERGY",
        "Realty": "^CNXREALTY", "Media": "^CNXMEDIA"}},
    "Forex": {"benchmark": ("US Dollar Index", "DX-Y.NYB"), "sectors": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X", "USD/CHF": "CHF=X", "NZD/USD": "NZDUSD=X"}},
    "Metals": {"benchmark": ("Gold", "GC=F"), "sectors": {
        "Silver": "SI=F", "Platinum": "PL=F", "Copper": "HG=F", "US Dollar Index": "DX-Y.NYB",
        "Gold Miners (GDX)": "GDX", "US 10Y Yield": "^TNX"}},
    "Crypto": {"benchmark": ("Bitcoin", "BTC-USD"), "sectors": {
        "Ethereum": "ETH-USD", "Solana": "SOL-USD", "BNB": "BNB-USD", "XRP": "XRP-USD",
        "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD", "Avalanche": "AVAX-USD"}},
}


def market_of(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith((".NS", ".BO")) or t.startswith("^NSE") or t.startswith("^CNX") or t == "^BSESN":
        return "India Stocks"
    cls = data.asset_class(ticker)
    if cls in ("Forex", "Metals", "Crypto"):
        return cls
    return "US Stocks"


# ---------------------------------------------------------------- helpers
@dataclass
class Item:
    section: str
    question: str
    answer: str
    verdict: str = "Neutral"          # Bullish / Bearish / Neutral
    weight: float = 1.0
    score: float = 0.0                # -1 .. +1
    details: dict = field(default_factory=dict)


def _v(score: float, th: float = 0.15) -> str:
    return "Bullish" if score > th else "Bearish" if score < -th else "Neutral"


def trend_state(close: pd.Series, n: int) -> tuple[str, float]:
    """Direction of a trend over n bars using MA slope + price vs MA + regression r2."""
    if len(close) < n + 5:
        n = max(5, len(close) // 2)
    ma = close.rolling(n).mean()
    slope = (ma.iloc[-1] / ma.iloc[-max(2, n // 4)] - 1) if pd.notna(ma.iloc[-1]) else 0
    above = 1 if close.iloc[-1] > ma.iloc[-1] else -1
    y = np.log(close.tail(n).values); x = np.arange(len(y))
    b = np.polyfit(x, y, 1)[0]
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(y) > 2 else 0
    vol = np.std(np.diff(y)) or 1e-9
    t = np.tanh(b / vol * np.sqrt(n) * 0.5) * (0.5 + r2)
    s = float(np.clip(0.5 * t + 0.3 * above + 0.2 * np.sign(slope), -1, 1))
    label = "UP" if s > 0.25 else "DOWN" if s < -0.25 else "SIDEWAYS"
    return label, s


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    lo, hi = df["Low"].rolling(n).min(), df["High"].rolling(n).max()
    k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.5f}" if abs(x) < 10 else f"{x:,.2f}"


# ----------------------------------------------------------------- analysis
def analyze(ticker: str, df: pd.DataFrame | None = None, loader=None,
            with_context: bool = True) -> dict:
    load = loader or (lambda t, p="5y", i="1d": data.load_ohlcv(t, p, i))
    live = True
    if df is None:
        df, live = load(ticker, "5y", "1d")
    df = df.dropna(subset=["Close"]).copy()
    has_vol = df["Volume"].fillna(0).sum() > 0
    f = ind.add_all(df)
    c = f["Close"]; px = float(c.iloc[-1])
    atr = float(f["atr"].iloc[-1])
    pivots = pt.zigzag(df)
    minor_piv = pt.zigzag(df, atr_mult=1.5)
    items: list[Item] = []
    mkt = market_of(ticker)

    # 1-2. market & sectors --------------------------------------------------
    sector_rows = []
    if with_context:
        name, bt = CONTEXT[mkt]["benchmark"]
        bdf, _ = load(bt, "2y", "1d")
        lbl, s = trend_state(bdf["Close"], 100)
        lbl_s, s_s = trend_state(bdf["Close"], 20)
        items.append(Item("Market", "What is the direction of the overall market?",
                          f"{name} ({bt}) primary trend is **{lbl}**, short-term {lbl_s}.",
                          _v(s), 2.0 if mkt != "Forex" else 1.0, s if mkt != "Forex" else 0,
                          {"benchmark": bt}))
        for sec, st in CONTEXT[mkt]["sectors"].items():
            try:
                sdf, _ = load(st, "1y", "1d")
                l1, s1 = trend_state(sdf["Close"], 50)
                r1 = sdf["Close"].iloc[-1] / sdf["Close"].iloc[-21] - 1
                r3 = sdf["Close"].iloc[-1] / sdf["Close"].iloc[-63] - 1
                sector_rows.append({"Sector": sec, "Ticker": st, "Trend": l1,
                                    "Score": round(s1, 2), "1M": r1, "3M": r3})
            except Exception:
                continue
        if sector_rows:
            sr = pd.DataFrame(sector_rows).sort_values("Score", ascending=False)
            up = (sr.Trend == "UP").sum(); dn = (sr.Trend == "DOWN").sum()
            breadth = (up - dn) / len(sr)
            items.append(Item("Market", "What is the direction of the various market sectors?",
                              f"{up} up, {dn} down, {len(sr) - up - dn} sideways. "
                              f"Leaders: {', '.join(sr.Sector.head(3))}. Laggards: {', '.join(sr.Sector.tail(3))}.",
                              _v(breadth), 1.0 if mkt in ("US Stocks", "India Stocks", "Crypto") else 0.5,
                              breadth if mkt != "Forex" else 0, {"table": sr}))

    # 3-4. weekly / monthly & major/intermediate/minor ---------------------------
    wk, mo = resample(df, "W-FRI"), resample(df, "ME")
    lw, sw = trend_state(wk["Close"], 26)
    lm, sm = trend_state(mo["Close"], 12)
    items.append(Item("Trend", "What are the weekly and monthly charts showing?",
                      f"Weekly trend **{lw}** (26-week), monthly trend **{lm}** (12-month). "
                      f"Weekly RSI {ind.rsi(wk['Close']).iloc[-1]:.0f}, monthly RSI {ind.rsi(mo['Close']).iloc[-1]:.0f}.",
                      _v((sw + sm) / 2), 2.0, (sw + sm) / 2))
    major = trend_state(c, 200); inter = trend_state(c, 50); minor = trend_state(c, 15)
    items.append(Item("Trend", "Are the major, intermediate and minor trends up, down or sideways?",
                      f"Major (≈200d): **{major[0]}** · Intermediate (≈50d): **{inter[0]}** · Minor (≈15d): **{minor[0]}**",
                      _v(0.5 * major[1] + 0.3 * inter[1] + 0.2 * minor[1]), 3.0,
                      0.5 * major[1] + 0.3 * inter[1] + 0.2 * minor[1],
                      {"major": major[0], "intermediate": inter[0], "minor": minor[0]}))

    # 5. support / resistance --------------------------------------------------
    sr_lv = pt.support_resistance(df, pivots)
    s1 = sr_lv["support"][0]["price"] if sr_lv["support"] else px - 3 * atr
    r1 = sr_lv["resistance"][0]["price"] if sr_lv["resistance"] else px + 3 * atr
    room = ((r1 - px) - (px - s1)) / max(r1 - s1, 1e-12)
    items.append(Item("Levels", "Where are important support and resistance levels?",
                      "Support: " + (", ".join(f"{fmt(l['price'])} ({l['touches']}×)" for l in sr_lv["support"]) or "none nearby")
                      + " · Resistance: " + (", ".join(f"{fmt(l['price'])} ({l['touches']}×)" for l in sr_lv["resistance"]) or "none — at highs"),
                      _v(room * 0.5), 1.0, room * 0.5, sr_lv))

    # 6. trendlines & channels -------------------------------------------------
    ch = pt.regression_channel(c, 100)
    tl = pt.trendlines(df, pivots)
    parts = [f"100-bar regression channel slope {ch['slope_pct_per_bar']:+.3f}%/bar (R² {ch['r2']:.2f}); "
             f"price at {ch['position']:+.2f} of the band ({fmt(ch['lower'])} – {fmt(ch['upper'])})."]
    tsc = np.sign(ch["slope_pct_per_bar"]) * min(1, ch["r2"] * 1.5)
    if "up" in tl:
        u = tl["up"]
        parts.append(f"Up-trendline from {u['from'].date.date()} now at {fmt(u['value_now'])} — "
                     + ("**BROKEN** (bearish)." if u["broken"] else "intact."))
        tsc += -0.7 if u["broken"] else 0.3
    if "down" in tl:
        d = tl["down"]
        parts.append(f"Down-trendline from {d['from'].date.date()} now at {fmt(d['value_now'])} — "
                     + ("**BROKEN** (bullish)." if d["broken"] else "intact."))
        tsc += 0.7 if d["broken"] else -0.3
    tsc = float(np.clip(tsc, -1, 1))
    items.append(Item("Levels", "Where are important trendlines or channels?", " ".join(parts),
                      _v(tsc), 1.5, tsc, {"channel": ch, "trendlines": tl}))

    # 7. volume ------------------------------------------------------------------
    if has_vol:
        o = obv(df)
        n = 50
        p_ch = c.iloc[-1] / c.iloc[-n] - 1
        o_sl = np.polyfit(np.arange(n), o.tail(n).values, 1)[0]
        upv = df["Volume"].where(c.diff() > 0, 0).tail(n).sum()
        dnv = df["Volume"].where(c.diff() < 0, 0).tail(n).sum()
        ud = upv / dnv if dnv else np.inf
        vol_ratio = df["Volume"].tail(5).mean() / df["Volume"].tail(50).mean()
        confirm = np.sign(p_ch) == np.sign(o_sl)
        vs = float(np.clip(np.sign(o_sl) * 0.5 + np.tanh(np.log(ud + 1e-9)) * 0.5, -1, 1))
        msg = (f"{'YES' if confirm else 'NO'} — price {p_ch:+.1%} over {n} bars, OBV trend "
               f"{'rising' if o_sl > 0 else 'falling'}; up/down volume ratio {ud:.2f}; "
               f"recent volume {vol_ratio:.1f}× average. "
               + ("Volume is confirming the move." if confirm else "**Volume is NOT confirming — warning of weakness.**"))
        items.append(Item("Volume", "Are volume and price action confirming each other?", msg,
                          _v(vs), 1.5, vs, {"obv": o, "confirm": bool(confirm)}))
    else:
        items.append(Item("Volume", "Are volume and price action confirming each other?",
                          "No reliable volume for this instrument (spot FX has no central volume). "
                          "Use futures volume or tick volume from your broker.", "Neutral", 0, 0))

    # 8. retracements (33/50/66) & Fibonacci ---------------------------------
    big = [p for p in pivots[-6:]]
    hi_p = max(big, key=lambda p: p.price); lo_p = min(big, key=lambda p: p.price)
    up_leg = lo_p.idx < hi_p.idx
    H, L = hi_p.price, lo_p.price
    rng = H - L
    lv = lambda f_: H - f_ * rng if up_leg else L + f_ * rng
    rets = {f"{int(k*100) if k not in (0.382, 0.618, 0.786) else k*100:g}%": lv(k)
            for k in (0.333, 0.382, 0.5, 0.618, 0.666, 0.786)}
    retr_now = (H - px) / rng if up_leg else (px - L) / rng
    ext = {f"{k:g}x": (L + k * rng if up_leg else H - k * rng) for k in (1.272, 1.618, 2.0, 2.618)}
    rsc = (0.4 if retr_now < 0.5 else -0.4 if retr_now > 0.66 else 0) * (1 if up_leg else -1)
    items.append(Item("Fibonacci", "Where are the 33%, 50% and 66% retracements?",
                      f"Last major swing {'UP' if up_leg else 'DOWN'} {fmt(L if up_leg else H)} → {fmt(H if up_leg else L)}. "
                      f"33%: {fmt(lv(0.333))} · 50%: {fmt(lv(0.5))} · 66%: {fmt(lv(0.666))}. "
                      f"Price has retraced {retr_now:.0%} of the swing"
                      + (" (shallow — trend healthy)." if retr_now < 0.4 else " (deep — trend at risk)." if retr_now > 0.66 else "."),
                      _v(rsc), 1.0, rsc, {"levels": rets, "extensions": ext, "up_leg": up_leg,
                                          "H": H, "L": L, "hi": hi_p, "lo": lo_p}))
    items.append(Item("Fibonacci", "What about Fibonacci retracements or projections?",
                      "Retracements: " + ", ".join(f"{k} {fmt(v)}" for k, v in rets.items() if k in ("38.2%", "50%", "61.8%"))
                      + " · Projections: " + ", ".join(f"{k} {fmt(v)}" for k, v in ext.items()),
                      "Neutral", 0, 0))

    # 9-11. patterns & price objectives -----------------------------------------
    pats = pt.chart_patterns(df, pivots) + [dict(p, name=p["name"] + " (minor)")
                                            for p in pt.chart_patterns(df, minor_piv)]
    rev = [p for p in pats if p["type"] == "Reversal"]
    cont = [p for p in pats if p["type"] == "Continuation"]
    def psc(arr):
        s = sum((1 if p["bias"] == "Bullish" else -1) * (1.0 if p["confirmed"] else 0.5) for p in arr)
        return float(np.clip(s, -1, 1))
    def pdesc(arr):
        return "; ".join(f"**{p['name']}** ({p['bias']}, {'CONFIRMED' if p['confirmed'] else 'forming'} at {fmt(p['level'])})"
                         for p in arr)
    rev_txt = pdesc(rev) or "No major reversal pattern detected."
    fb = [p for p in rev if not p["confirmed"] and p["bias"] == "Bullish"]
    fs = [p for p in rev if not p["confirmed"] and p["bias"] == "Bearish"]
    if fb and fs:
        lo_ = min(p["level"] for p in fs); hi_ = max(p["level"] for p in fb)
        rev_txt += (f". **Bullish and bearish setups are both forming → price is boxed in a range "
                    f"{fmt(min(lo_, hi_))} – {fmt(max(lo_, hi_))}. Trade the breakout: above → bullish, below → bearish.**")
    items.append(Item("Patterns", "Are there any major reversal patterns visible?",
                      rev_txt, _v(psc(rev)), 2.0, psc(rev), {"patterns": rev}))
    items.append(Item("Patterns", "Are there any continuation patterns visible?",
                      pdesc(cont) or "No continuation pattern detected.", _v(psc(cont)), 1.5, psc(cont), {"patterns": cont}))
    items.append(Item("Patterns", "What are the price objectives from those patterns?",
                      "; ".join(f"{p['name']} → **{fmt(p['target'])}** ({(p['target']/px-1):+.1%})" for p in pats)
                      or "No pattern objectives (no patterns).", "Neutral", 0, 0, {"patterns": pats}))

    # 12. moving averages --------------------------------------------------------
    ma_rows, mscore = [], 0.0
    for n in (10, 20, 50, 100, 200):
        m = c.rolling(n).mean()
        if m.notna().sum() < 10:
            continue
        slope = m.iloc[-1] / m.iloc[-6] - 1
        ma_rows.append(f"{n}: {'↑' if slope > 0 else '↓'} {'above' if px > m.iloc[-1] else 'below'}")
        mscore += (np.sign(slope) + (1 if px > m.iloc[-1] else -1)) / 10
    cross = ""
    m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
    if m200.notna().sum() > 5:
        g = (m50 > m200).astype(int).diff().tail(20)
        if (g == 1).any(): cross = " **Golden cross** in last 20 bars."
        if (g == -1).any(): cross = " **Death cross** in last 20 bars."
    items.append(Item("Moving Averages", "Which way are the moving averages pointed?",
                      "Price vs MA (slope): " + " · ".join(ma_rows) + "." + cross,
                      _v(mscore), 2.0, float(np.clip(mscore, -1, 1))))

    # 13. oscillators --------------------------------------------------------------
    k, dline = stochastic(df)
    rsi_v, k_v = float(f["rsi"].iloc[-1]), float(k.iloc[-1])
    macd_h = float(f["hist"].iloc[-1])
    ob = []
    if rsi_v > 70: ob.append(f"RSI {rsi_v:.0f} OVERBOUGHT")
    elif rsi_v < 30: ob.append(f"RSI {rsi_v:.0f} OVERSOLD")
    else: ob.append(f"RSI {rsi_v:.0f} neutral")
    ob.append(f"Stochastic %K {k_v:.0f} " + ("OVERBOUGHT" if k_v > 80 else "OVERSOLD" if k_v < 20 else "neutral"))
    ob.append(f"MACD histogram {'positive' if macd_h > 0 else 'negative'} and "
              f"{'rising' if f['hist'].iloc[-1] > f['hist'].iloc[-2] else 'falling'}")
    osc_s = float(np.clip((50 - rsi_v) / 40 * 0.5 + (50 - k_v) / 60 * 0.3 + np.sign(macd_h) * 0.3, -1, 1)) \
        if (rsi_v > 70 or rsi_v < 30) else float(np.clip((rsi_v - 50) / 40 + np.sign(macd_h) * 0.3, -1, 1))
    items.append(Item("Oscillators", "Are the oscillators overbought or oversold?", " · ".join(ob) + ".",
                      _v(osc_s), 1.5, osc_s))

    # 14. divergences ----------------------------------------------------------------
    divs = pt.divergences(c, f["rsi"], minor_piv) + \
        [d_.replace("oscillator", "MACD") for d_ in pt.divergences(c, f["macd"], minor_piv)]
    dsc = sum(0.6 if "Bullish" in d_ or "bullish" in d_ else -0.6 for d_ in divs if "Hidden" not in d_) + \
        sum(0.3 if "bullish" in d_ else -0.3 for d_ in divs if "Hidden" in d_)
    items.append(Item("Oscillators", "Are any divergences apparent on the oscillators?",
                      "; ".join(divs) if divs else "No divergence on RSI or MACD at recent swings.",
                      _v(dsc), 1.5, float(np.clip(dsc, -1, 1))))

    # 15. contrary opinion -------------------------------------------------------------
    z200 = float(((c - c.rolling(200).mean()) / c.rolling(200).std()).iloc[-1]) if len(c) > 220 else float(f["z50"].iloc[-1])
    streak = int((np.sign(c.diff()).tail(15)[::-1].cumprod() == np.sign(c.diff().iloc[-1])).sum())
    ext_msg = ("EXTREME OPTIMISM (contrarian sell warning)" if z200 > 2.2 else
               "EXTREME PESSIMISM (contrarian buy warning)" if z200 < -2.2 else "no extreme")
    csc = -0.8 if z200 > 2.2 else 0.8 if z200 < -2.2 else 0
    items.append(Item("Sentiment", "Are contrary-opinion numbers showing any extremes?",
                      f"Sentiment proxy: price is {z200:+.1f}σ from its 200-bar mean, {streak}-bar streak → **{ext_msg}**. "
                      "(True sentiment data — COT, put/call, Fear & Greed — is not available from free feeds; "
                      "check them manually.)", _v(csc), 1.0, csc))

    # 16-17. Elliott wave -------------------------------------------------------------
    ew = pt.elliott(pivots)
    esc = 0.5 * ew.get("direction", 0)
    items.append(Item("Elliott Wave", "What is the Elliott Wave pattern showing?",
                      f"{ew['pattern']}. {ew['detail']} (Automated count — treat as a hint, not a fact.)",
                      _v(esc), 1.0, esc, ew))
    labels = ew.get("wave_labels", [])
    items.append(Item("Elliott Wave", "Are there any obvious 3- or 5-wave patterns?",
                      ("Yes — " + ("5-wave" if len(labels) >= 5 else "3-wave") + " structure: "
                       + ", ".join(f"{lab}@{d_.date()}" for d_, lab in labels)) if labels else "No obvious 3 or 5 wave pattern.",
                      "Neutral", 0, 0))

    # 18. cycles -----------------------------------------------------------------------
    cy = pt.dominant_cycle(c)
    if cy.get("period"):
        nb = cy["bars_to_bottom"]; nt = cy["bars_to_top"]
        nxt = "BOTTOM" if nb < nt else "TOP"
        bars = min(nb, nt)
        when = (df.index[-1] + pd.tseries.offsets.BDay(int(round(bars)))).date()
        cs = (0.5 if nxt == "TOP" else -0.5) * cy["strength"] * 2
        items.append(Item("Cycles", "Are there any cycle tops or bottoms due?",
                          f"Dominant cycle ≈ {cy['period']:.0f} bars (strength {cy['strength']:.0%}). "
                          f"Next cycle **{nxt}** due in ~{bars:.0f} bars (≈ {when}).",
                          _v(cs), 0.75, float(np.clip(cs, -1, 1)), cy))

    # 19. computer trend ------------------------------------------------------------------
    comp = sg.composite(f)
    cs_ = float(comp["composite"].iloc[-1])
    items.append(Item("Systems", "Which way is the computer trend moving?",
                      f"Systematic trend model: **{sg.label(cs_)}** (score {cs_:+.2f}); "
                      f"trend {comp['trend'].iloc[-1]:+.2f}, momentum {comp['momentum'].iloc[-1]:+.2f}, "
                      f"breakout {comp['breakout'].iloc[-1]:+.2f}, mean-reversion {comp['mean_rev'].iloc[-1]:+.2f}.",
                      _v(cs_), 2.0, cs_))

    # 20. P&F + candlesticks ------------------------------------------------------------------
    pf = pt.point_figure(c, box=atr)
    cnd = pt.candlesticks(df, 5)
    pfs = 0.6 if "BUY" in pf["signal"] else -0.6 if "SELL" in pf["signal"] else (0.2 if pf["current"] == "X" else -0.2)
    csn = sum(1 if x["bias"] == "Bullish" else -1 if x["bias"] == "Bearish" else 0 for x in cnd)
    sc_ = float(np.clip(pfs + 0.25 * csn, -1, 1))
    items.append(Item("Charts", "What are the point & figure charts and candlesticks showing?",
                      f"P&F (box {fmt(pf['box'])}, 3-box reversal): current column **{pf['current']}s**, signal **{pf['signal']}**, "
                      f"vertical-count objective {fmt(pf['objective'])}. Candles (last 5 bars): "
                      + (", ".join(f"{x['pattern']} ({x['bias']}, {x['date'].date()})" for x in cnd) or "no notable patterns") + ".",
                      _v(sc_), 1.0, sc_, {"pf": pf, "candles": cnd}))

    # ---------------------------------------------------------------- scores
    W = sum(i.weight for i in items if i.weight > 0)
    net = sum(i.weight * i.score for i in items) / W
    buy = sum(i.weight * max(i.score, 0) for i in items) / W * 100
    sell = sum(i.weight * max(-i.score, 0) for i in items) / W * 100
    bias = "BUY" if net > 0.12 else "SELL" if net < -0.12 else "STAND ASIDE / NO TRADE"

    # outlook over next several months ---------------------------------------------------
    long_s = 0.4 * major[1] + 0.3 * (sw + sm) / 2 + 0.2 * mscore + 0.1 * esc
    outlook = ("UP" if long_s > 0.2 else "DOWN" if long_s < -0.2 else "SIDEWAYS / RANGE")
    items.append(Item("Decision", "Which way will this market trend over the next several months?",
                      f"Likely **{outlook}** (long-horizon score {long_s:+.2f}, from major trend, weekly/monthly charts, MAs, Elliott).",
                      _v(long_s), 0, 0))

    plans = trade_plans(ticker, df, net, sr_lv, rets, ext, pats, atr, load)
    side = "Long" if net >= 0 else "Short"
    sw_plan = plans["Swing (days–weeks)"]
    items.append(Item("Decision", "Am I going to buy or sell this market?",
                      f"**{bias}** — net score {net:+.2f}. Buy score {buy:.0f}/100 · Sell score {sell:.0f}/100.",
                      _v(net), 0, 0))
    items.append(Item("Decision", "Where will I enter the market?", sw_plan["entry_note"], "Neutral", 0, 0))
    items.append(Item("Decision", "Where will I place my protective stops?", sw_plan["stop_note"], "Neutral", 0, 0))
    items.append(Item("Decision", "What is my profit objective?", sw_plan["target_note"], "Neutral", 0, 0))

    return {"ticker": ticker, "market": mkt, "price": px, "live": live, "df": f,
            "items": items, "net": net, "buy_score": buy, "sell_score": sell, "bias": bias,
            "outlook": outlook, "plans": plans, "pivots": pivots, "sr": sr_lv, "fib": rets,
            "fib_ext": ext, "patterns": pats, "channel": ch, "trendlines": tl,
            "sectors": pd.DataFrame(sector_rows), "weekly": wk, "monthly": mo,
            "elliott": ew, "pf": pf, "side": side}


# ------------------------------------------------------------- trade plans
def _plan(df: pd.DataFrame, side: int, score: float, label: str, atr_mult_stop: float,
          rr: float, supports: list[float], resistances: list[float], targets: list[float]) -> dict:
    f = ind.add_all(df)
    px = float(f["Close"].iloc[-1]); a = float(f["atr"].iloc[-1])
    if side == 0:
        return {"horizon": label, "action": "NO TRADE", "score": score,
                "entry_note": "No edge — wait for a clearer setup.", "stop_note": "—", "target_note": "—",
                "entry": np.nan, "stop": np.nan, "t1": np.nan, "t2": np.nan, "rr": np.nan}
    side = int(side)
    if side > 0:
        pull = max([s for s in supports if px - 2 * a < s < px] or [px - 0.5 * a])
        entry = min(px, pull + 0.1 * a) if score < 0.4 else px
        stop_struct = max([s for s in supports if s < entry - 0.3 * a] or [entry - atr_mult_stop * a])
        stop = min(stop_struct - 0.25 * a, entry - atr_mult_stop * a * 0.6)
        risk = entry - stop
        cands = sorted([t for t in resistances + targets if t > entry + risk])
        t1 = cands[0] if cands else entry + rr * risk
        t2 = cands[1] if len(cands) > 1 else entry + 2 * rr * risk
    else:
        pull = min([r for r in resistances if px < r < px + 2 * a] or [px + 0.5 * a])
        entry = max(px, pull - 0.1 * a) if score > -0.4 else px
        stop_struct = min([r for r in resistances if r > entry + 0.3 * a] or [entry + atr_mult_stop * a])
        stop = max(stop_struct + 0.25 * a, entry + atr_mult_stop * a * 0.6)
        risk = stop - entry
        cands = sorted([t for t in supports + targets if t < entry - risk], reverse=True)
        t1 = cands[0] if cands else entry - rr * risk
        t2 = cands[1] if len(cands) > 1 else entry - 2 * rr * risk
    rr_ = abs(t1 - entry) / risk if risk else np.nan
    act = "BUY" if side > 0 else "SELL"
    how = "at market" if entry == px else f"on a pullback to {fmt(entry)} (limit order)"
    return {"horizon": label, "action": act, "score": score, "entry": entry, "stop": stop,
            "t1": t1, "t2": t2, "rr": rr_,
            "entry_note": f"{act} {how}; current price {fmt(px)}.",
            "stop_note": f"Stop at **{fmt(stop)}** ({abs(stop / entry - 1):.2%} away) — beyond structure and ≥{atr_mult_stop * 0.6:.1f}×ATR.",
            "target_note": f"Target 1 **{fmt(t1)}** ({abs(t1 / entry - 1):.2%}, R:R {rr_:.1f}), Target 2 **{fmt(t2)}**. "
                           "Take partial profit at T1 and trail the stop."}


def trade_plans(ticker, df, net, sr_lv, rets, ext, pats, atr, load) -> dict:
    sup = [l["price"] for l in sr_lv["support"]] + list(rets.values())
    res = [l["price"] for l in sr_lv["resistance"]] + list(rets.values())
    tgts = [p["target"] for p in pats] + list(ext.values())
    plans = {}
    # intraday: hourly bars over the last 60 days
    try:
        idf, _ = load(ticker, "60d", "1h")
        if len(idf) < 60:
            raise ValueError
        fi = ind.add_all(idf)
        s_i = float(sg.composite(fi)["composite"].iloc[-1])
        lvl = pt.support_resistance(idf, pt.zigzag(idf, atr_mult=2))
        side = 1 if s_i > 0.15 and net > -0.3 else -1 if s_i < -0.15 and net < 0.3 else 0
        plans["Intraday (hours)"] = _plan(idf, side, s_i, "Intraday", 1.5, 1.5,
                                          [l["price"] for l in lvl["support"]],
                                          [l["price"] for l in lvl["resistance"]], [])
    except Exception:
        plans["Intraday (hours)"] = {"horizon": "Intraday", "action": "N/A", "score": np.nan,
                                     "entry_note": "Intraday data unavailable.", "stop_note": "—", "target_note": "—",
                                     "entry": np.nan, "stop": np.nan, "t1": np.nan, "t2": np.nan, "rr": np.nan}
    side = 1 if net > 0.12 else -1 if net < -0.12 else 0
    plans["Swing (days–weeks)"] = _plan(df, side, net, "Swing", 2.0, 2.0, sup, res, tgts)
    wk = resample(df, "W-FRI")
    fw = ind.add_all(wk)
    s_w = float(sg.composite(fw)["composite"].iloc[-1]) if len(fw) > 60 else net
    pos_s = 0.5 * s_w + 0.5 * net
    side = 1 if pos_s > 0.15 else -1 if pos_s < -0.15 else 0
    plans["Position (few months)"] = _plan(wk, side, pos_s, "Position", 2.0, 3.0, sup, res, tgts)
    return plans
