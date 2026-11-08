"""Order flow, VWAP, volume profile, TPO market profile and auction-market theory.

Data sources
------------
* Crypto (BTC-USD, ETH-USD, ...): Binance public REST — 1-minute klines carry the
  real *taker-buy volume*, so delta is exact at the 1-minute level, and aggTrades
  give real prints (time & sales) with aggressor side.
* Everything else: Yahoo 1-minute bars. True bid/ask aggression is not public,
  so delta is ESTIMATED with the close-location rule
  (buy volume = volume x (close-low)/(high-low)). The UI labels this clearly.
* Instruments with no volume (cash indices, spot FX) fall back to time-based TPO
  profiles; volume-based studies are marked unavailable.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import data

BINANCE = ("https://data-api.binance.vision", "https://api.binance.com")


# ======================================================================= data
def binance_symbol(ticker: str) -> str | None:
    t = ticker.upper()
    if t.endswith("-USD") and data.asset_class(t) == "Crypto" or t.endswith("-USD"):
        return t.replace("-USD", "USDT")
    if t.endswith("USDT"):
        return t
    return None


def _get(path: str, params: dict) -> list:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for host in BINANCE:
        try:
            req = urllib.request.Request(f"{host}{path}?{q}", headers={"User-Agent": "quantlab"})
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read())
        except Exception as e:  # try next host
            last = e
    raise ConnectionError(last)


def binance_bars(symbol: str, minutes: int = 2880) -> pd.DataFrame:
    rows, end = [], None
    while len(rows) < minutes:
        p = {"symbol": symbol, "interval": "1m", "limit": 1000}
        if end:
            p["endTime"] = end
        k = _get("/api/v3/klines", p)
        if not k:
            break
        rows = k + rows
        end = k[0][0] - 1
        if len(k) < 1000:
            break
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close", "Volume", "ct", "qv", "n",
                                     "tbv", "tbq", "x"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.t, unit="ms")
    df = df[["Open", "High", "Low", "Close", "Volume", "tbv"]].astype(float)
    df["buy"] = df.tbv
    df["sell"] = df.Volume - df.tbv
    return df.drop(columns="tbv").tail(minutes)


def binance_prints(symbol: str, n: int = 1000) -> pd.DataFrame:
    t = _get("/api/v3/aggTrades", {"symbol": symbol, "limit": min(n, 1000)})
    df = pd.DataFrame(t)
    out = pd.DataFrame({"time": pd.to_datetime(df["T"], unit="ms"), "price": df["p"].astype(float),
                        "size": df["q"].astype(float)})
    out["side"] = np.where(df["m"], "SELL", "BUY")   # m = buyer is maker -> seller aggressed
    return out


def estimate_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Close-location value split of bar volume into buy/sell (estimate)."""
    rng = (df.High - df.Low).replace(0, np.nan)
    clv = ((df.Close - df.Low) / rng).fillna(0.5)
    # blend with tick rule so flat bars still get direction
    tick = np.sign(df.Close.diff()).fillna(0)
    w = (0.75 * clv + 0.25 * (tick + 1) / 2).clip(0, 1)
    out = df.copy()
    out["buy"] = df.Volume * w
    out["sell"] = df.Volume - out["buy"]
    return out


def load_intraday(ticker: str) -> tuple[pd.DataFrame, dict]:
    """1-minute bars with buy/sell columns + metadata on data quality."""
    sym = binance_symbol(ticker)
    if sym:
        try:
            df = binance_bars(sym)
            if len(df) > 100:
                try:
                    prints = binance_prints(sym)
                except Exception:
                    prints = None
                return df, {"source": f"Binance {sym} (real taker-buy delta)", "real_delta": True,
                            "has_volume": True, "prints": prints, "live": True}
        except Exception:
            pass
    df, live = data.load_ohlcv(ticker, "5d", "1m")
    if not live or len(df) < 100:
        df = _synthetic_intraday(ticker)
        live = False
    has_vol = df.Volume.fillna(0).sum() > 0
    df = estimate_delta(df.fillna({"Volume": 0}))
    return df, {"source": "Yahoo 1-minute bars (delta ESTIMATED from bar structure)" if live
                else "SYNTHETIC intraday data", "real_delta": False, "has_volume": bool(has_vol),
                "prints": None, "live": live}


def _synthetic_intraday(ticker: str, days: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    idx = []
    for d in pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=375, freq="1min"))
    idx = pd.DatetimeIndex(idx)
    r = rng.standard_t(4, len(idx)) * 0.0006
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, .0004, len(c))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, .0004, len(c))))
    v = rng.lognormal(8, .6, len(c)) * (1 + 2 * np.exp(-((np.arange(len(c)) % 375) / 40)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def sessions(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(df.index.date) if len(g) > 10]


# ================================================================== order flow
def tick_size(df: pd.DataFrame, bins: int = 60) -> float:
    rng = df.High.max() - df.Low.min()
    raw = rng / bins
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def delta_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta"] = out.buy - out.sell
    out["cvd"] = out["delta"].cumsum()
    return out


def footprint(df: pd.DataFrame, bar: str = "15min", box: float | None = None) -> pd.DataFrame:
    """Bid x Ask volume per (time bar, price level). Volume of each 1-min bar is
    spread evenly over the price levels it traded."""
    box = box or tick_size(df, 40)
    rows = []
    for t, r in df.iterrows():
        lo, hi = np.floor(r.Low / box) * box, np.floor(r.High / box) * box
        lv = np.arange(lo, hi + box / 2, box)
        k = len(lv)
        for p in lv:
            rows.append((t.floor(bar), round(p, 10), r.buy / k, r.sell / k))
    fp = pd.DataFrame(rows, columns=["bar", "price", "ask", "bid"])
    fp = fp.groupby(["bar", "price"]).sum().reset_index()
    fp["delta"] = fp.ask - fp.bid
    fp["total"] = fp.ask + fp.bid
    # diagonal imbalance: ask at price vs bid one level below (>=3:1)
    fp = fp.sort_values(["bar", "price"])
    fp["bid_below"] = fp.groupby("bar")["bid"].shift(1)
    fp["ask_above"] = fp.groupby("bar")["ask"].shift(-1)
    fp["buy_imb"] = fp.ask >= 3 * fp.bid_below.fillna(np.inf).clip(lower=1e-12)
    fp["sell_imb"] = fp.bid >= 3 * fp.ask_above.fillna(np.inf).clip(lower=1e-12)
    fp.attrs["box"] = box
    return fp


def bar_summary(df: pd.DataFrame, bar: str = "15min") -> pd.DataFrame:
    g = df.resample(bar).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                              "Volume": "sum", "buy": "sum", "sell": "sum"}).dropna(subset=["Close"])
    g["delta"] = g.buy - g.sell
    g["delta_pct"] = g.delta / g.Volume.replace(0, np.nan)
    g["cvd"] = g.delta.cumsum()
    return g


def orderflow_signals(bars: pd.DataFrame) -> list[dict]:
    """Absorption, exhaustion, delta divergence on the bar summary."""
    out = []
    if len(bars) < 6:
        return out
    b = bars.tail(40)
    vmed = b.Volume.median()
    body = (b.Close - b.Open)
    for t, r in b.tail(12).iterrows():
        big = r.Volume > 1.8 * vmed
        if big and abs(r.delta_pct) > 0.25 and np.sign(r.Close - r.Open) != np.sign(r.delta) and r.Close != r.Open:
            side = "buyers" if r.delta > 0 else "sellers"
            out.append({"time": t, "signal": "Absorption",
                        "detail": f"Aggressive {side} (delta {r.delta:+,.0f}) absorbed — price closed the other way",
                        "bias": "Bearish" if r.delta > 0 else "Bullish"})
        if big and abs(r.delta_pct) > 0.45:
            out.append({"time": t, "signal": "Initiative " + ("buying" if r.delta > 0 else "selling"),
                        "detail": f"Delta {r.delta_pct:+.0%} of volume on {r.Volume / vmed:.1f}x volume",
                        "bias": "Bullish" if r.delta > 0 else "Bearish"})
    # CVD divergence over last 12 bars
    w = b.tail(12)
    if w.High.iloc[-1] >= w.High.max() and w.cvd.iloc[-1] < w.cvd.max() - 0.25 * abs(w.cvd).max():
        out.append({"time": w.index[-1], "signal": "Bearish CVD divergence",
                    "detail": "New price high without a new cumulative-delta high — buyers fading", "bias": "Bearish"})
    if w.Low.iloc[-1] <= w.Low.min() and w.cvd.iloc[-1] > w.cvd.min() + 0.25 * abs(w.cvd).max():
        out.append({"time": w.index[-1], "signal": "Bullish CVD divergence",
                    "detail": "New price low without a new cumulative-delta low — sellers fading", "bias": "Bullish"})
    return out


def large_prints(prints: pd.DataFrame | None, bars: pd.DataFrame, q: float = 0.98) -> pd.DataFrame:
    """Large trades (real prints) or, without tick data, volume-spike bars."""
    if prints is not None and len(prints):
        th = prints["size"].quantile(q)
        return prints[prints["size"] >= th].sort_values("time", ascending=False).head(25)
    th = bars.Volume.quantile(q)
    s = bars[bars.Volume >= th].copy()
    return pd.DataFrame({"time": s.index, "price": s.Close, "size": s.Volume,
                         "side": np.where(s.delta > 0, "BUY", "SELL")}).sort_values("time", ascending=False).head(25)


# ======================================================================== VWAP
def vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df.High + df.Low + df.Close) / 3
    v = df.Volume.replace(0, np.nan).fillna(1e-9) if df.Volume.sum() > 0 else pd.Series(1.0, df.index)
    cv = (tp * v).cumsum() / v.cumsum()
    var = ((tp - cv) ** 2 * v).cumsum() / v.cumsum()
    sd = np.sqrt(var)
    return pd.DataFrame({"vwap": cv, "u1": cv + sd, "l1": cv - sd, "u2": cv + 2 * sd, "l2": cv - 2 * sd})


def anchored_vwap(df: pd.DataFrame, anchor) -> pd.Series:
    return vwap(df.loc[anchor:])["vwap"]


# ============================================================== volume profile
def volume_profile(df: pd.DataFrame, box: float | None = None, va: float = 0.70,
                   by: str = "volume") -> dict:
    box = box or tick_size(df, 60)
    lo, hi = np.floor(df.Low.min() / box) * box, np.floor(df.High.max() / box) * box
    levels = np.round(np.arange(lo, hi + box / 2, box), 10)
    vol = np.zeros(len(levels)); buy = np.zeros(len(levels))
    for r in df.itertuples():
        a = int(round((np.floor(r.Low / box) * box - lo) / box))
        b = int(round((np.floor(r.High / box) * box - lo) / box))
        k = b - a + 1
        w = r.Volume if by == "volume" else 1.0
        vol[a:b + 1] += w / k
        if by == "volume":
            buy[a:b + 1] += getattr(r, "buy", w / 2) / k
    if vol.sum() == 0:
        return volume_profile(df, box, va, by="time")
    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i; acc = vol[poc_i]; tot = vol.sum()
    while acc < va * tot and (lo_i > 0 or hi_i < len(vol) - 1):
        up = vol[hi_i + 1:hi_i + 3].sum() if hi_i < len(vol) - 1 else -1
        dn = vol[max(lo_i - 2, 0):lo_i].sum() if lo_i > 0 else -1
        if up >= dn:
            step = min(2, len(vol) - 1 - hi_i); acc += vol[hi_i + 1:hi_i + 1 + step].sum(); hi_i += step
        else:
            step = min(2, lo_i); acc += vol[lo_i - step:lo_i].sum(); lo_i -= step
    sm = pd.Series(vol).rolling(5, center=True, min_periods=1).mean().values
    hvn = [levels[i] for i in range(2, len(sm) - 2) if sm[i] == sm[i - 2:i + 3].max() and sm[i] > np.median(sm) * 1.3]
    lvn = [levels[i] for i in range(2, len(sm) - 2) if sm[i] == sm[i - 2:i + 3].min() and sm[i] < np.median(sm) * 0.6
           and lo_i < i < hi_i + 5]
    pos = (poc_i) / max(len(vol) - 1, 1)
    shape = ("P-shape (buying / short-covering, value at top)" if pos > 0.65 else
             "b-shape (selling / long liquidation, value at bottom)" if pos < 0.35 else
             "D-shape (balanced, two-sided trade)")
    return {"levels": levels, "volume": vol, "buy": buy, "sell": vol - buy, "box": box, "by": by,
            "poc": levels[poc_i], "vah": levels[hi_i], "val": levels[lo_i],
            "hvn": hvn, "lvn": lvn, "shape": shape}


# ================================================================= TPO profile
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def tpo_profile(sess: pd.DataFrame, period: str = "30min", box: float | None = None) -> dict:
    box = box or tick_size(sess, 50)
    per = sess.resample(period).agg({"High": "max", "Low": "min", "Open": "first", "Close": "last"}).dropna()
    lo = np.floor(sess.Low.min() / box) * box
    hi = np.floor(sess.High.max() / box) * box
    levels = np.round(np.arange(lo, hi + box / 2, box), 10)
    letters = {p: "" for p in levels}
    for i, (_, r) in enumerate(per.iterrows()):
        ch = LETTERS[i % len(LETTERS)]
        for p in levels:
            if np.floor(r.Low / box) * box - 1e-12 <= p <= np.floor(r.High / box) * box + 1e-12:
                letters[p] += ch
    counts = np.array([len(letters[p]) for p in levels])
    vp = volume_profile(sess, box, by="time")
    ib = per.iloc[:2]
    ib_hi, ib_lo = ib.High.max(), ib.Low.min()
    s_hi, s_lo = sess.High.max(), sess.Low.min()
    ext_up, ext_dn = s_hi > ib_hi + 1e-12, s_lo < ib_lo - 1e-12
    single = [levels[i] for i in range(1, len(levels) - 1) if counts[i] == 1
              and counts[:i].max() > 1 and counts[i + 1:].max() > 1]
    poor_high = counts[-1] >= 2
    poor_low = counts[0] >= 2
    rng, ibr = s_hi - s_lo, ib_hi - ib_lo
    close = sess.Close.iloc[-1]
    cpos = (close - s_lo) / rng if rng else 0.5
    if ext_up and ext_dn:
        day = "Neutral day (range extension both sides)" + (" — Neutral Extreme" if cpos > .85 or cpos < .15 else "")
    elif rng >= 2 * ibr and (cpos > 0.85 or cpos < 0.15):
        day = "Trend day (one-sided, closed at the extreme)"
    elif len(single) >= 3 and rng >= 1.5 * ibr:
        day = "Double-distribution day (single prints separate two value areas)"
    elif rng <= 1.15 * ibr:
        day = "Normal day (initial balance held the range)"
    else:
        day = "Normal-variation day (moderate one-sided range extension)"
    return {"levels": levels, "letters": letters, "counts": counts, "box": box,
            "tpo_poc": vp["poc"], "tpo_vah": vp["vah"], "tpo_val": vp["val"],
            "ib_high": ib_hi, "ib_low": ib_lo, "ext_up": ext_up, "ext_down": ext_dn,
            "single_prints": single, "poor_high": poor_high, "poor_low": poor_low, "day_type": day,
            "periods": per}


# ======================================================= auction market theory
def open_type(sess: pd.DataFrame, prev_vp: dict | None) -> str:
    first = sess.iloc[:30]
    o = sess.Open.iloc[0]
    if len(first) < 10:
        return "n/a"
    hi, lo, c = first.High.max(), first.Low.min(), first.Close.iloc[-1]
    rng = hi - lo or 1e-12
    if (c - lo) / rng > 0.8 and (o - lo) / rng < 0.15:
        return "Open-Drive UP (strong initiative buying from the open)"
    if (hi - c) / rng > 0.8 and (hi - o) / rng < 0.15:
        return "Open-Drive DOWN (strong initiative selling from the open)"
    if (o - lo) / rng < 0.35 and c > o and lo < o:
        return "Open-Test-Drive UP (tested lower, then drove up)"
    if (hi - o) / rng < 0.35 and c < o and hi > o:
        return "Open-Test-Drive DOWN (tested higher, then drove down)"
    if abs(c - o) / rng < 0.25 and lo < o < hi:
        return "Open-Auction (rotational — no conviction yet)"
    return "Open-Rejection-Reverse" if np.sign(c - o) != np.sign(first.Close.iloc[10] - o) else "Open-Auction"


def auction_read(df: pd.DataFrame) -> dict:
    ss = sessions(df)
    if len(ss) < 2:
        return {"error": "Need at least two sessions of intraday data."}
    today, prev = ss[-1], ss[-2]
    use_vol = df.Volume.sum() > 0
    by = "volume" if use_vol else "time"
    vp_prev = volume_profile(prev, by=by)
    vp_today = volume_profile(today, box=vp_prev["box"], by=by)
    tpo = tpo_profile(today, box=vp_prev["box"])
    px, o = today.Close.iloc[-1], today.Open.iloc[0]
    vah, val, poc = vp_prev["vah"], vp_prev["val"], vp_prev["poc"]

    loc = lambda p: "ABOVE prior value" if p > vah else "BELOW prior value" if p < val else "INSIDE prior value"
    open_loc = loc(o)
    notes = []
    if open_loc.startswith("INSIDE"):
        notes.append("Opened inside yesterday's value → expect balance / rotation between VAH and VAL unless value is rejected.")
    else:
        notes.append(f"Opened {open_loc.split()[0].lower()} yesterday's value → potential imbalance; "
                     "acceptance outside value (time + volume) means the move continues, rejection means a return to value.")
        reentered = (today.Close.iloc[60:] < vah).any() if o > vah else (today.Close.iloc[60:] > val).any()
        held = (today.Close.iloc[-60:].between(val, vah)).mean() > 0.7 if len(today) > 120 else False
        if reentered and held:
            notes.append(f"**80% rule active:** price re-entered and held inside prior value → high odds of traversing to "
                         f"the other side ({'VAL ' + f'{val:.5g}' if o > vah else 'VAH ' + f'{vah:.5g}'}).")
    # value migration across sessions
    pocs = [volume_profile(s, box=vp_prev["box"], by=by)["poc"] for s in ss[-5:]]
    vas = [(volume_profile(s, box=vp_prev["box"], by=by)) for s in ss[-3:]]
    overlap = all(max(a["val"], b["val"]) <= min(a["vah"], b["vah"]) for a, b in zip(vas, vas[1:]))
    mig = ("HIGHER (buyers in control)" if vp_today["poc"] > vah else
           "LOWER (sellers in control)" if vp_today["poc"] < val else "OVERLAPPING (balance)")
    overlap = overlap and mig.startswith("OVERLAPPING")
    state = ("BALANCE — overlapping value areas; fade the edges (sell VAH, buy VAL) until a breakout is accepted"
             if overlap else "IMBALANCE / TREND — value is migrating; trade with the direction of value, buy/sell pullbacks to VWAP or POC")
    # bias
    score = 0.0
    score += 0.4 if px > vah else -0.4 if px < val else 0
    score += 0.3 if vp_today["poc"] > poc else -0.3 if vp_today["poc"] < poc else 0
    v = vwap(today)["vwap"].iloc[-1]
    score += 0.3 if px > v else -0.3
    return {"prev": vp_prev, "today": vp_today, "tpo": tpo, "open_type": open_type(today, vp_prev),
            "open_location": open_loc, "price_location": loc(px), "notes": notes,
            "value_migration": mig, "market_state": state, "poc_history": pocs,
            "naked_pocs": [p for p, s in zip(pocs[:-1], ss[-5:-1])
                           if not ((today.Low <= p) & (today.High >= p)).any()],
            "score": float(np.clip(score, -1, 1)), "today_df": today, "prev_df": prev, "vwap": v}
