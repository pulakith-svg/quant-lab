"""Regime detection and a multi-factor signal ensemble.

Each component signal lies in [-1, 1]; the composite is a weighted average.
All signals use information available at bar close t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def regime(df: pd.DataFrame, vol_window: int = 20) -> pd.Series:
    """Classify each bar as Trend-Up / Trend-Down / Range, tagged High/Low vol."""
    f = df if "adx" in df else ind.add_all(df)
    trending = f["adx"] > 25
    direction = np.where(f["ema20"] > f["ema50"], "Up", "Down")
    rv = f["ret"].rolling(vol_window).std()
    high_vol = rv > rv.rolling(250, min_periods=60).median()
    base = np.where(trending, "Trend-" + pd.Series(direction, index=f.index), "Range")
    tag = np.where(high_vol, " / High-Vol", " / Low-Vol")
    return pd.Series(base, index=f.index) + tag


def trend_signal(f: pd.DataFrame) -> pd.Series:
    """Time-series momentum: vol-normalised blend of EMA crossovers."""
    c, a = f["Close"], f["atr"]
    s = ((f["ema20"] - f["ema50"]) / a).clip(-3, 3) / 3
    l = ((f["ema50"] - f["ema200"]) / a).clip(-3, 3) / 3
    return (0.6 * s + 0.4 * l).fillna(0)


def momentum_signal(f: pd.DataFrame, lookbacks=(20, 60, 120)) -> pd.Series:
    r = f["ret"]
    vol = r.rolling(60).std()
    parts = [np.tanh(r.rolling(n).sum() / (vol * np.sqrt(n))) for n in lookbacks]
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0)


def mean_reversion_signal(f: pd.DataFrame) -> pd.Series:
    z = -(f["z50"].clip(-3, 3) / 3)
    r = -((f["rsi"] - 50) / 50)
    return (0.5 * z + 0.5 * r).fillna(0)


def breakout_signal(f: pd.DataFrame, n: int = 55) -> pd.Series:
    hi, lo = f["High"].rolling(n).max().shift(), f["Low"].rolling(n).min().shift()
    pos = (f["Close"] - lo) / (hi - lo)
    return (2 * pos.clip(0, 1) - 1).fillna(0)


def composite(df: pd.DataFrame, weights: dict | None = None,
              regime_aware: bool = True) -> pd.DataFrame:
    f = df if "adx" in df else ind.add_all(df)
    sig = pd.DataFrame({
        "trend": trend_signal(f),
        "momentum": momentum_signal(f),
        "mean_rev": mean_reversion_signal(f),
        "breakout": breakout_signal(f),
    }, index=f.index)
    w = weights or {"trend": 0.35, "momentum": 0.3, "mean_rev": 0.15, "breakout": 0.2}
    if regime_aware:
        # Trend-following in trending markets, mean-reversion in ranges.
        trending = (f["adx"] > 25).astype(float)
        wt = pd.DataFrame({k: v for k, v in w.items()}, index=f.index)
        wt["mean_rev"] = wt["mean_rev"] * (1 - trending) * 2.5
        for k in ("trend", "momentum", "breakout"):
            wt[k] = wt[k] * (0.5 + trending)
        wt = wt.div(wt.sum(axis=1), axis=0)
        sig["composite"] = (sig[list(w)] * wt).sum(axis=1)
    else:
        tot = sum(w.values())
        sig["composite"] = sum(sig[k] * v for k, v in w.items()) / tot
    return sig


def label(score: float) -> str:
    if score > 0.5: return "STRONG BUY"
    if score > 0.15: return "BUY"
    if score < -0.5: return "STRONG SELL"
    if score < -0.15: return "SELL"
    return "NEUTRAL"
