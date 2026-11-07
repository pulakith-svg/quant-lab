"""Cross-asset analytics and portfolio construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def shrunk_cov(returns: pd.DataFrame, delta: float = 0.2) -> pd.DataFrame:
    """Simple constant-correlation shrinkage of the sample covariance."""
    s = returns.cov()
    sd = np.sqrt(np.diag(s))
    corr = returns.corr().values
    avg = (corr.sum() - len(sd)) / (len(sd) ** 2 - len(sd)) if len(sd) > 1 else 0
    target = avg * np.outer(sd, sd)
    np.fill_diagonal(target, sd**2)
    return pd.DataFrame(delta * target + (1 - delta) * s.values, s.index, s.columns)


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    n = len(cov)
    c = cov.values

    def obj(w):
        rc = w * (c @ w)
        return ((rc - rc.mean()) ** 2).sum() * 1e6

    res = minimize(obj, np.ones(n) / n, bounds=[(1e-6, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, cov.index)


def min_variance(cov: pd.DataFrame) -> pd.Series:
    n = len(cov)
    res = minimize(lambda w: w @ cov.values @ w, np.ones(n) / n, bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, cov.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    n = len(cov)
    res = minimize(lambda w: -(w @ mu.values) / np.sqrt(w @ cov.values @ w),
                   np.ones(n) / n, bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, cov.index)


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    mrc = cov.values @ w.values
    rc = w.values * mrc
    return pd.Series(rc / rc.sum(), w.index)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n: int = 3000,
                       seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(mu)), n)
    ret = w @ mu.values
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.values, w))
    return pd.DataFrame({"ret": ret, "vol": vol, "sharpe": ret / vol})


def rolling_beta(asset: pd.Series, bench: pd.Series, window: int = 60) -> pd.Series:
    return asset.rolling(window).cov(bench) / bench.rolling(window).var()
