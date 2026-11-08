"""Chart-pattern recognition: swings, S/R, trendlines, classic patterns,
candlesticks, point & figure, divergences, Elliott-wave heuristics, cycles.

Everything here is rule-based and heuristic. Each detector returns plain
Python structures so the checklist can explain *why* it reached a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- swings
@dataclass
class Pivot:
    idx: int
    date: pd.Timestamp
    price: float
    kind: str  # "H" or "L"


def zigzag(df: pd.DataFrame, pct: float | None = None, atr_mult: float = 3.0) -> list[Pivot]:
    """Alternating swing highs/lows; a reversal must exceed `pct` (or ATR multiple)."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    if pct is None:
        tr = (df["High"] - df["Low"]).rolling(14).mean().bfill().values
        thr = atr_mult * tr / c
    else:
        thr = np.full(len(c), pct)
    piv: list[Pivot] = []
    P = lambda i, k: Pivot(i, df.index[i], float(h[i] if k == "H" else l[i]), k)
    trend, hi_i, lo_i = 0, 0, 0
    for i in range(1, len(c)):
        if trend == 0:
            if h[i] > h[hi_i]: hi_i = i
            if l[i] < l[lo_i]: lo_i = i
            if hi_i > lo_i and h[hi_i] > l[lo_i] * (1 + thr[i]):
                piv.append(P(lo_i, "L")); trend = 1
            elif lo_i > hi_i and l[lo_i] < h[hi_i] * (1 - thr[i]):
                piv.append(P(hi_i, "H")); trend = -1
        elif trend == 1:
            if h[i] >= h[hi_i]: hi_i = i
            elif l[i] < h[hi_i] * (1 - thr[i]):
                piv.append(P(hi_i, "H")); trend, lo_i = -1, i
        else:
            if l[i] <= l[lo_i]: lo_i = i
            elif h[i] > l[lo_i] * (1 + thr[i]):
                piv.append(P(lo_i, "L")); trend, hi_i = 1, i
    if trend == 1: piv.append(P(hi_i, "H"))
    elif trend == -1: piv.append(P(lo_i, "L"))
    return piv


# ------------------------------------------------------ support/resistance
def support_resistance(df: pd.DataFrame, pivots: list[Pivot], n: int = 4) -> dict:
    """Cluster pivot prices; score clusters by touches and recency."""
    px = df["Close"].iloc[-1]
    tol = (df["High"] - df["Low"]).tail(100).mean() * 1.0
    levels: list[dict] = []
    for p in pivots:
        for lv in levels:
            if abs(lv["price"] - p.price) <= tol:
                lv["prices"].append(p.price); lv["touches"] += 1
                lv["last"] = max(lv["last"], p.idx)
                lv["price"] = float(np.mean(lv["prices"]))
                break
        else:
            levels.append({"price": p.price, "prices": [p.price], "touches": 1, "last": p.idx})
    N = len(df)
    for lv in levels:
        lv["strength"] = lv["touches"] + 2 * lv["last"] / N
    sup = sorted([l for l in levels if l["price"] < px], key=lambda x: -x["price"])
    res = sorted([l for l in levels if l["price"] > px], key=lambda x: x["price"])
    pick = lambda arr: [{"price": round(l["price"], 6), "touches": l["touches"]}
                        for l in sorted(arr[:n * 2], key=lambda x: -x["strength"])[:n]]
    s, r = pick(sup), pick(res)
    return {"support": sorted(s, key=lambda x: -x["price"]),
            "resistance": sorted(r, key=lambda x: x["price"])}


# ------------------------------------------------------------- trendlines
def regression_channel(close: pd.Series, n: int = 100, k: float = 2.0) -> dict:
    y = close.tail(n).values
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    fit = a + b * x
    sd = np.std(y - fit)
    pos = (y[-1] - fit[-1]) / (k * sd) if sd else 0.0
    r2 = 1 - np.var(y - fit) / np.var(y) if np.var(y) else 0.0
    return {"slope_pct_per_bar": b / y.mean() * 100, "mid": fit[-1],
            "upper": fit[-1] + k * sd, "lower": fit[-1] - k * sd,
            "position": float(np.clip(pos, -1.5, 1.5)), "r2": r2,
            "index": close.tail(n).index, "fit": fit, "sd": sd, "k": k}


def trendlines(df: pd.DataFrame, pivots: list[Pivot]) -> dict:
    """Line through the last two higher lows (up-trendline) / lower highs."""
    N = len(df) - 1
    out = {}
    lows = [p for p in pivots if p.kind == "L"]
    highs = [p for p in pivots if p.kind == "H"]
    if len(lows) >= 2 and lows[-1].price > lows[-2].price:
        a, b = lows[-2], lows[-1]
        slope = (b.price - a.price) / (b.idx - a.idx)
        val = b.price + slope * (N - b.idx)
        out["up"] = {"from": a, "to": b, "value_now": val, "slope": slope,
                     "broken": df["Close"].iloc[-1] < val}
    if len(highs) >= 2 and highs[-1].price < highs[-2].price:
        a, b = highs[-2], highs[-1]
        slope = (b.price - a.price) / (b.idx - a.idx)
        val = b.price + slope * (N - b.idx)
        out["down"] = {"from": a, "to": b, "value_now": val, "slope": slope,
                       "broken": df["Close"].iloc[-1] > val}
    return out


# ------------------------------------------------------- classic patterns
def _near(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b))


def chart_patterns(df: pd.DataFrame, pivots: list[Pivot], tol: float = 0.03) -> list[dict]:
    """Reversal & continuation patterns on the most recent pivots, with targets."""
    px = df["Close"].iloc[-1]
    P = pivots
    found = []
    seen = set()
    for w in (P[-3:], P[-4:-1]):
        if len(w) < 3:
            continue
        t1, mid, t2 = w
        if t1.kind == "H" and t2.kind == "H" and _near(t1.price, t2.price, tol) and "DT" not in seen:
            seen.add("DT")
            found.append(dict(name="Double Top", type="Reversal", bias="Bearish", level=mid.price,
                              target=mid.price - (max(t1.price, t2.price) - mid.price),
                              confirmed=px < mid.price))
        if t1.kind == "L" and t2.kind == "L" and _near(t1.price, t2.price, tol) and "DB" not in seen:
            seen.add("DB")
            found.append(dict(name="Double Bottom", type="Reversal", bias="Bullish", level=mid.price,
                              target=mid.price + (mid.price - min(t1.price, t2.price)),
                              confirmed=px > mid.price))
    for w in (P[-5:], P[-6:-1]):
        if len(w) < 5:
            continue
        s1, n1, hd, n2, s2 = w
        if hd.kind == "H" and s1.kind == "H" and hd.price > max(s1.price, s2.price) \
                and _near(s1.price, s2.price, tol * 2) and "HS" not in seen:
            seen.add("HS")
            neck = (n1.price + n2.price) / 2
            found.append(dict(name="Head & Shoulders Top", type="Reversal", bias="Bearish",
                              level=neck, target=neck - (hd.price - neck), confirmed=px < neck))
        if hd.kind == "L" and s1.kind == "L" and hd.price < min(s1.price, s2.price) \
                and _near(s1.price, s2.price, tol * 2) and "IHS" not in seen:
            seen.add("IHS")
            neck = (n1.price + n2.price) / 2
            found.append(dict(name="Inverse Head & Shoulders", type="Reversal", bias="Bullish",
                              level=neck, target=neck + (neck - hd.price), confirmed=px > neck))
    # Triangles / rectangles from last 4 pivots' highs & lows
    if len(P) >= 5:
        hs = [p for p in P[-5:] if p.kind == "H"][-2:]
        ls = [p for p in P[-5:] if p.kind == "L"][-2:]
        if len(hs) == 2 and len(ls) == 2:
            hi_flat = _near(hs[0].price, hs[1].price, tol / 2)
            lo_flat = _near(ls[0].price, ls[1].price, tol / 2)
            lower_hi = hs[1].price < hs[0].price * (1 - tol / 2)
            higher_lo = ls[1].price > ls[0].price * (1 + tol / 2)
            height = max(h.price for h in hs) - min(l.price for l in ls)
            top, bot = hs[1].price, ls[1].price
            prior_up = P[-5].kind == "L"
            name = None
            if hi_flat and higher_lo: name, bias = "Ascending Triangle", "Bullish"
            elif lo_flat and lower_hi: name, bias = "Descending Triangle", "Bearish"
            elif lower_hi and higher_lo: name, bias = "Symmetrical Triangle", "Bullish" if prior_up else "Bearish"
            elif hi_flat and lo_flat: name, bias = "Rectangle", "Bullish" if prior_up else "Bearish"
            if name:
                tgt = top + height if bias == "Bullish" else bot - height
                found.append(dict(name=name, type="Continuation", bias=bias,
                                  level=top if bias == "Bullish" else bot, target=tgt,
                                  confirmed=px > top if bias == "Bullish" else px < bot))
    # Flag / pennant: sharp pole then tight drift over last ~15 bars
    c = df["Close"]
    if len(c) > 40:
        pole = c.iloc[-15] / c.iloc[-35] - 1
        drift = c.iloc[-1] / c.iloc[-15] - 1
        vol = c.pct_change().tail(60).std() * np.sqrt(20)
        rng = (df["High"].tail(15).max() - df["Low"].tail(15).min()) / c.iloc[-1]
        if abs(pole) > 2.5 * vol and rng < abs(pole) * 0.6 and np.sign(drift) != np.sign(pole):
            bull = pole > 0
            hgt = abs(c.iloc[-15] - c.iloc[-35])
            lvl = df["High"].tail(15).max() if bull else df["Low"].tail(15).min()
            found.append(dict(name="Bull Flag" if bull else "Bear Flag", type="Continuation",
                              bias="Bullish" if bull else "Bearish", level=lvl,
                              target=lvl + hgt if bull else lvl - hgt,
                              confirmed=c.iloc[-1] > lvl if bull else c.iloc[-1] < lvl))
    return [p for p in found if _still_valid(p, px)]


def _still_valid(p: dict, px: float) -> bool:
    """Drop patterns whose objective is already hit or whose price has left the pattern."""
    lvl, tgt = p["level"], p["target"]
    h = abs(tgt - lvl)
    if p["bias"] == "Bullish":
        if px >= tgt:
            return False
        return p["confirmed"] or (lvl - h * 1.05 <= px <= lvl)
    if px <= tgt:
        return False
    return p["confirmed"] or (lvl <= px <= lvl + h * 1.05)


# ------------------------------------------------------------- candlesticks
def candlesticks(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    out = []
    n = len(c)
    avg_body = np.mean(np.abs(c[-30:] - o[-30:])) or 1e-12
    for i in range(max(2, n - lookback), n):
        body, rng = abs(c[i] - o[i]), h[i] - l[i] or 1e-12
        up_sh, lo_sh = h[i] - max(c[i], o[i]), min(c[i], o[i]) - l[i]
        prev_trend = c[i - 1] - c[max(0, i - 6)]
        d = df.index[i]
        if body < 0.1 * rng:
            out.append((d, "Doji", "Indecision"))
        if lo_sh > 2 * body and up_sh < body and prev_trend < 0:
            out.append((d, "Hammer", "Bullish"))
        if up_sh > 2 * body and lo_sh < body and prev_trend > 0:
            out.append((d, "Shooting Star", "Bearish"))
        if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]:
            out.append((d, "Bullish Engulfing", "Bullish"))
        if c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] <= o[i - 1] and o[i] >= c[i - 1]:
            out.append((d, "Bearish Engulfing", "Bearish"))
        if i >= 2:
            b2 = abs(c[i - 2] - o[i - 2]); b1 = abs(c[i - 1] - o[i - 1])
            if c[i - 2] < o[i - 2] and b2 > avg_body and b1 < 0.5 * avg_body \
                    and c[i] > o[i] and c[i] > (o[i - 2] + c[i - 2]) / 2:
                out.append((d, "Morning Star", "Bullish"))
            if c[i - 2] > o[i - 2] and b2 > avg_body and b1 < 0.5 * avg_body \
                    and c[i] < o[i] and c[i] < (o[i - 2] + c[i - 2]) / 2:
                out.append((d, "Evening Star", "Bearish"))
        if body > 2 * avg_body and up_sh < 0.1 * rng and lo_sh < 0.1 * rng:
            out.append((d, "Marubozu", "Bullish" if c[i] > o[i] else "Bearish"))
    return [{"date": d, "pattern": p, "bias": b} for d, p, b in out]


# --------------------------------------------------------- point & figure
def point_figure(close: pd.Series, box: float | None = None, reversal: int = 3) -> dict:
    c = close.dropna().values
    if box is None:
        box = float(np.median(np.abs(np.diff(c[-200:])))) * 2 or c[-1] * 0.01
    cols: list[dict] = []
    cur = {"type": "X", "lo": np.floor(c[0] / box) * box, "hi": np.floor(c[0] / box) * box}
    for p in c[1:]:
        if cur["type"] == "X":
            if p >= cur["hi"] + box:
                cur["hi"] = np.floor(p / box) * box
            elif p <= cur["hi"] - reversal * box:
                cols.append(cur)
                cur = {"type": "O", "hi": cur["hi"] - box, "lo": np.ceil(p / box) * box}
        else:
            if p <= cur["lo"] - box:
                cur["lo"] = np.ceil(p / box) * box
            elif p >= cur["lo"] + reversal * box:
                cols.append(cur)
                cur = {"type": "X", "lo": cur["lo"] + box, "hi": np.floor(p / box) * box}
    cols.append(cur)
    signal = "None"
    xs = [k for k in cols if k["type"] == "X"]
    os_ = [k for k in cols if k["type"] == "O"]
    if cur["type"] == "X" and len(xs) >= 2 and cur["hi"] > xs[-2]["hi"]:
        signal = "Double-Top BUY"
    elif cur["type"] == "O" and len(os_) >= 2 and cur["lo"] < os_[-2]["lo"]:
        signal = "Double-Bottom SELL"
    # vertical count objective from the current column's prior column length
    prev = cols[-2] if len(cols) >= 2 else cur
    boxes = round((prev["hi"] - prev["lo"]) / box) + 1
    if cur["type"] == "X":
        target = prev["lo"] + boxes * box * reversal
    else:
        target = prev["hi"] - boxes * box * reversal
    return {"box": box, "columns": cols, "current": cur["type"], "signal": signal,
            "objective": target}


# ------------------------------------------------------------- divergence
def divergences(close: pd.Series, osc: pd.Series, pivots: list[Pivot], lookback: int = 6) -> list[str]:
    out = []
    hs = [p for p in pivots[-lookback:] if p.kind == "H"]
    ls = [p for p in pivots[-lookback:] if p.kind == "L"]
    if len(hs) >= 2:
        a, b = hs[-2], hs[-1]
        if b.price > a.price and osc.iloc[b.idx] < osc.iloc[a.idx]:
            out.append(f"Bearish divergence: price higher high ({b.date.date()}) but oscillator lower high")
        if b.price < a.price and osc.iloc[b.idx] > osc.iloc[a.idx]:
            out.append(f"Hidden bearish divergence at {b.date.date()}")
    if len(ls) >= 2:
        a, b = ls[-2], ls[-1]
        if b.price < a.price and osc.iloc[b.idx] > osc.iloc[a.idx]:
            out.append(f"Bullish divergence: price lower low ({b.date.date()}) but oscillator higher low")
        if b.price > a.price and osc.iloc[b.idx] < osc.iloc[a.idx]:
            out.append(f"Hidden bullish divergence at {b.date.date()}")
    return out


# ------------------------------------------------------------ Elliott wave
def elliott(pivots: list[Pivot]) -> dict:
    """Test the latest swings against the core Elliott impulse rules.
    Heuristic: wave counts are subjective; this reports the best rule-consistent fit."""
    P = pivots
    res = {"pattern": "No clear count", "detail": "", "wave_labels": []}
    for k in (6, 5):  # 6 points = 5 completed waves; 5 points = wave 5 in progress
        if len(P) < k:
            continue
        pts = P[-k:]
        up = pts[0].kind == "L"
        pr = [p.price if up else -p.price for p in pts]
        w = np.diff(pr)
        if k == 6:
            w1, w2, w3, w4, w5 = w
            ok = (w1 > 0 and pr[2] > pr[0] and w3 > 0 and pr[4] > pr[1]
                  and abs(w3) >= min(abs(w1), abs(w5)))
            if ok:
                res = {"pattern": f"5-wave impulse {'UP' if up else 'DOWN'} likely COMPLETE",
                       "detail": "Expect an A-B-C correction against the trend.",
                       "wave_labels": list(zip([p.date for p in pts], ["0", "1", "2", "3", "4", "5"])),
                       "direction": -1 if up else 1}
                return res
        else:
            w1, w2, w3, w4 = w
            ok = w1 > 0 and pr[2] > pr[0] and w3 > 0 and pr[4] > pr[1] and w3 > w1 * 0.8
            if ok:
                tgt = pts[4].price + (pts[1].price - pts[0].price)
                res = {"pattern": f"Wave 5 {'UP' if up else 'DOWN'} in progress",
                       "detail": f"Typical wave-5 target ≈ {tgt:.5g} (wave 5 = wave 1).",
                       "wave_labels": list(zip([p.date for p in pts], ["0", "1", "2", "3", "4"])),
                       "direction": 1 if up else -1, "target": tgt}
                return res
    if len(P) >= 4:
        pts = P[-4:]
        x0, A, B, C = (p.price for p in pts)
        if (pts[0].kind == "H" and A < x0 and B < x0 and C < A) or \
           (pts[0].kind == "L" and A > x0 and B > x0 and C > A):
            res = {"pattern": "3-wave A-B-C correction (zigzag) visible",
                   "detail": "Corrections often end near the 50-61.8% retracement.",
                   "wave_labels": list(zip([p.date for p in pts], ["X", "A", "B", "C"])),
                   "direction": 1 if pts[0].kind == "H" else -1}
    return res


# ------------------------------------------------------------------ cycles
def dominant_cycle(close: pd.Series, min_p: int = 10, max_p: int = 250) -> dict:
    y = np.log(close.dropna().values[-1024:])
    x = np.arange(len(y))
    y = y - np.polyval(np.polyfit(x, y, 2), x)        # detrend
    y = y * np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y))
    mask = (freqs > 0) & (1 / np.maximum(freqs, 1e-12) >= min_p) & (1 / np.maximum(freqs, 1e-12) <= max_p)
    if not mask.any():
        return {"period": None}
    i = np.argmax(np.where(mask, spec, 0))
    period = 1 / freqs[i]
    phase = np.angle(np.fft.rfft(y)[i])
    # phase of the sine at the last bar -> bars until next trough / peak
    t = len(y) - 1
    theta = (2 * np.pi * freqs[i] * t + phase) % (2 * np.pi)
    bars_to_peak = ((0 - theta) % (2 * np.pi)) / (2 * np.pi) * period
    bars_to_trough = ((np.pi - theta) % (2 * np.pi)) / (2 * np.pi) * period
    strength = spec[i] / spec[mask].sum()
    return {"period": float(period), "bars_to_top": float(bars_to_peak),
            "bars_to_bottom": float(bars_to_trough), "strength": float(strength)}
