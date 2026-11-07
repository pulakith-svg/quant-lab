"""Vectorised, look-ahead-free backtester with vol targeting and costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from . import signals as sg
from .risk import metrics


@dataclass
class BTConfig:
    target_vol: float = 0.15       # annualised
    max_leverage: float = 3.0
    cost_bps: float = 2.0          # per unit of turnover
    threshold: float = 0.1         # |signal| below this -> flat
    ppy: int = 252
    long_only: bool = False


def run(df: pd.DataFrame, signal: pd.Series, cfg: BTConfig = BTConfig()) -> dict:
    f = df if "ret" in df else ind.add_all(df)
    ret = f["ret"].fillna(0)
    s = signal.where(signal.abs() >= cfg.threshold, 0.0)
    if cfg.long_only:
        s = s.clip(lower=0)
    vol = ind.ewma_vol(ret, ppy=cfg.ppy).replace(0, np.nan)
    lev = (cfg.target_vol / vol).clip(upper=cfg.max_leverage).fillna(0)
    pos = (s * lev).shift(1).fillna(0)          # trade on next bar: no look-ahead
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turnover * cfg.cost_bps / 1e4
    trades = int((np.sign(pos).diff().fillna(0) != 0).sum())
    return {"returns": strat, "position": pos,
            "equity": np.exp(strat.cumsum()), "bench": np.exp(ret.cumsum()),
            "metrics": metrics(strat, cfg.ppy) | {"Trades": trades,
                                                  "AvgExposure": pos.abs().mean()},
            "bench_metrics": metrics(ret, cfg.ppy)}


def walk_forward(df: pd.DataFrame, cfg: BTConfig, train: int = 500, test: int = 125,
                 thresholds=(0.05, 0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Pick the best threshold in-sample, apply it out-of-sample, roll forward."""
    f = ind.add_all(df)
    comp = sg.composite(f)["composite"]
    rows, oos = [], []
    for start in range(0, len(f) - train - test + 1, test):
        tr = slice(start, start + train)
        te = slice(start + train, start + train + test)
        best, best_sr = thresholds[0], -np.inf
        for th in thresholds:
            c = BTConfig(**{**cfg.__dict__, "threshold": th})
            sr = run(f.iloc[tr], comp.iloc[tr], c)["metrics"]["Sharpe"]
            if sr > best_sr:
                best, best_sr = th, sr
        c = BTConfig(**{**cfg.__dict__, "threshold": best})
        res = run(f.iloc[start:te.stop], comp.iloc[start:te.stop], c)
        r_oos = res["returns"].iloc[train:]
        oos.append(r_oos)
        rows.append({"oos_start": f.index[te.start], "threshold": best,
                     "is_sharpe": best_sr,
                     "oos_sharpe": metrics(r_oos, cfg.ppy)["Sharpe"]})
    table = pd.DataFrame(rows)
    table.attrs["oos_returns"] = pd.concat(oos) if oos else pd.Series(dtype=float)
    return table
