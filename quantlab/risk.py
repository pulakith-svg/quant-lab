"""Performance and risk analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def var_cvar(returns: pd.Series, level: float = 0.95) -> dict:
    r = returns.dropna()
    q = 1 - level
    hist_var = -np.quantile(r, q)
    hist_cvar = -r[r <= -hist_var].mean()
    mu, sd = r.mean(), r.std()
    param_var = -(mu + sd * stats.norm.ppf(q))
    # Cornish-Fisher adjusted for skew / kurtosis
    s, k = stats.skew(r), stats.kurtosis(r)
    z = stats.norm.ppf(q)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return {"hist_var": hist_var, "hist_cvar": hist_cvar,
            "param_var": param_var, "cf_var": -(mu + sd * zcf)}


def metrics(returns: pd.Series, ppy: int = 252) -> dict:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return {k: 0.0 for k in ("CAGR", "Vol", "Sharpe", "Sortino", "Calmar",
                                 "MaxDD", "HitRate", "Skew", "Kurtosis", "ProfitFactor")}
    eq = np.exp(r.cumsum())
    years = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(ppy)
    downside = r[r < 0].std() * np.sqrt(ppy)
    mdd = drawdown(eq).min()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    active = r[r != 0]
    return {
        "CAGR": cagr, "Vol": vol,
        "Sharpe": r.mean() / r.std() * np.sqrt(ppy),
        "Sortino": r.mean() * ppy / downside if downside > 0 else np.nan,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "MaxDD": mdd,
        "HitRate": (active > 0).mean() if len(active) else 0.0,
        "Skew": stats.skew(r), "Kurtosis": stats.kurtosis(r),
        "ProfitFactor": gains / losses if losses > 0 else np.nan,
    }


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR: P(true per-period SR > benchmark)."""
    r = returns.dropna()
    n = len(r)
    sr = r.mean() / r.std()
    s, k = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - s * sr + (k - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / denom))


def kelly_fraction(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.mean() / r.var()) if r.var() > 0 else 0.0


def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  contract_size: float = 1.0) -> dict:
    risk_amt = equity * risk_pct
    per_unit = abs(entry - stop) * contract_size
    units = risk_amt / per_unit if per_unit > 0 else 0.0
    return {"risk_amount": risk_amt, "units": units,
            "notional": units * entry * contract_size,
            "leverage": units * entry * contract_size / equity if equity else 0.0}


def monte_carlo(returns: pd.Series, horizon: int = 252, sims: int = 2000,
                seed: int = 7) -> np.ndarray:
    """Stationary block bootstrap of returns -> simulated equity paths."""
    r = returns.dropna().values
    rng = np.random.default_rng(seed)
    block = 10
    paths = np.empty((sims, horizon))
    for i in range(sims):
        out = []
        while len(out) < horizon:
            s = rng.integers(0, len(r) - block)
            out.extend(r[s:s + block])
        paths[i] = np.exp(np.cumsum(out[:horizon]))
    return paths
