# QuantLab — Multi-Asset Quant Research Terminal

A quant-desk-style analysis app for **Forex, Gold/Metals and Crypto**, built with Streamlit + Plotly.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Data comes from Yahoo Finance (any Yahoo ticker works). If the network is unavailable, the app
falls back to clearly-flagged synthetic data so every feature still runs.

## Features
| Tab | What it does |
|---|---|
| Dashboard | Live signal, regime, EWMA vol, RSI, Hurst exponent; one-click scanner ranking the full universe |
| Chart & Signals | Candles, EMA 20/50/200, Bollinger, RSI, MACD, composite signal |
| Volatility & Regime | Realised vs EWMA (RiskMetrics) vol, GARCH(1,1) MLE fit & forecast, OU half-life, regime map |
| Risk | Historical / Gaussian / Cornish-Fisher VaR, Expected Shortfall, fat-tail stats, drawdowns, block-bootstrap Monte Carlo |
| Backtest | Vol-targeted, cost-aware, look-ahead-free backtester; Sharpe/Sortino/Calmar, Probabilistic Sharpe, Kelly, monthly heatmap, walk-forward optimisation |
| Cross-Asset | Rolling correlation matrix, rolling corr/beta between any pair, relative performance |
| Portfolio | Shrinkage covariance; Equal / Risk-Parity / Min-Variance / Max-Sharpe weights, efficient frontier, risk contributions |
| Position Sizer | ATR-based stop, 2R target, lot sizing from account risk % |

### Signal model
Four normalised factors in [-1, 1] — **trend** (vol-scaled EMA spreads), **time-series momentum**
(multi-horizon, vol-normalised), **mean reversion** (z-score + RSI), **breakout** (Donchian position) —
blended with **regime-aware weights** (ADX > 25 ⇒ favour trend; otherwise favour mean reversion).

## Layout
```
app.py                  Streamlit UI
quantlab/data.py        universe, Yahoo loader, synthetic fallback
quantlab/indicators.py  TA, EWMA vol, GARCH, Hurst, half-life
quantlab/signals.py     factors, composite, regime classifier
quantlab/risk.py        metrics, VaR/ES, PSR, Kelly, sizing, Monte Carlo
quantlab/backtest.py    vectorised backtester + walk-forward
quantlab/portfolio.py   covariance shrinkage, optimisers, frontier
tests/                  pytest suite (incl. a no-look-ahead test)
```

> Research/education tool only — not investment advice.
