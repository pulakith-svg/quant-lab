"""Technical and statistical features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    m, sd = close.rolling(n).mean(), close.rolling(n).std()
    return pd.DataFrame({"mid": m, "upper": m + k * sd, "lower": m - k * sd})


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def zscore(s: pd.Series, n: int = 50) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def ewma_vol(returns: pd.Series, lam: float = 0.94, ppy: int = 252) -> pd.Series:
    """RiskMetrics EWMA volatility, annualised."""
    var = (returns**2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * ppy)


def garch11(returns: pd.Series) -> dict:
    """Fit GARCH(1,1) by MLE; return params and next-step conditional vol."""
    from scipy.optimize import minimize

    r = returns.dropna().values
    r = r - r.mean()
    v0 = r.var()

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = np.empty_like(r)
        h[0] = v0
        for t in range(1, len(r)):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + r**2 / h)

    res = minimize(nll, x0=[v0 * 0.05, 0.08, 0.9], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-8})
    w, a, b = res.x
    h = v0
    for x in r:
        h = w + a * x**2 + b * h
    return {"omega": w, "alpha": a, "beta": b, "persistence": a + b,
            "next_var": h, "long_run_var": w / max(1 - a - b, 1e-9)}


def hurst(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent via variance of lagged differences (log prices).
    <0.5 mean-reverting, ~0.5 random walk, >0.5 trending."""
    x = np.log(series.dropna().values)
    lags = np.arange(2, min(max_lag, len(x) // 2))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def half_life(series: pd.Series) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life (in bars)."""
    y = np.log(series.dropna())
    dy, lag = y.diff().dropna(), y.shift().dropna()
    lag = lag.loc[dy.index]
    beta = np.polyfit(lag.values, dy.values, 1)[0]
    return float(-np.log(2) / beta) if beta < 0 else float("inf")


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    for n in (20, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["rsi"] = rsi(c)
    out = out.join(macd(c)).join(bollinger(c))
    out["atr"] = atr(out)
    out["adx"] = adx(out)
    out["z50"] = zscore(c, 50)
    out["ret"] = np.log(c).diff()
    return out
