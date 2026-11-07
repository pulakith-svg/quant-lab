import numpy as np
import pandas as pd

from quantlab import backtest as bt, data, indicators as ind, portfolio as pf, risk, signals as sg


def frame():
    return data.synthetic_ohlcv("BTC-USD", n=900, seed=1)


def test_indicators_and_signals_bounded():
    f = ind.add_all(frame())
    assert f["rsi"].dropna().between(0, 100).all()
    s = sg.composite(f)
    assert s.abs().max().max() <= 1 + 1e-9
    assert sg.regime(f).notna().all()


def test_no_lookahead():
    f = ind.add_all(frame())
    sig = sg.composite(f)["composite"]
    a = bt.run(f, sig)["returns"]
    # perturbing future prices must not change past strategy returns
    g = frame(); g.iloc[-50:, :4] *= 1.5
    g = ind.add_all(g)
    b = bt.run(g, sg.composite(g)["composite"])["returns"]
    assert np.allclose(a.iloc[:-60], b.iloc[:-60])


def test_risk_and_portfolio():
    r = ind.add_all(frame())["ret"].dropna()
    v = risk.var_cvar(r)
    assert v["hist_cvar"] >= v["hist_var"] > 0
    assert 0 <= risk.probabilistic_sharpe(r) <= 1
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, [0.01, 0.02, 0.04], (500, 3)),
                        columns=list("abc"))
    cov = pf.shrunk_cov(rets)
    w = pf.risk_parity(cov)
    assert abs(w.sum() - 1) < 1e-6 and w["a"] > w["c"]
    assert np.allclose(pf.risk_contributions(w, cov), 1 / 3, atol=0.02)


def test_garch_and_hurst():
    r = ind.add_all(frame())["ret"].dropna()
    g = ind.garch11(r)
    assert 0 < g["persistence"] < 1
    rw = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0, .01, 3000))))
    assert 0.4 < ind.hurst(rw) < 0.6


def test_walk_forward():
    wf = bt.walk_forward(frame(), bt.BTConfig(), train=300, test=100)
    assert len(wf) >= 4 and len(wf.attrs["oos_returns"]) == 100 * len(wf)
