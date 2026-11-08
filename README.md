# QuantLab — Multi-Asset Quant Research Terminal

A quant-desk-style analysis app for **Stocks (US & India), Indices, Forex, Gold/Metals and Crypto**, built with Streamlit + Plotly.

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
| **Technical Checklist** | Answers the full technical-analysis checklist automatically: market & sector direction, weekly/monthly charts, major/intermediate/minor trends, support/resistance, trendlines & channels, volume confirmation (OBV, up/down volume), 33/50/66% and Fibonacci retracements & projections, reversal & continuation patterns (double tops/bottoms, head & shoulders, triangles, rectangles, flags) with price objectives, moving-average direction, overbought/oversold oscillators, RSI/MACD divergences, contrary-opinion extremes, Elliott 3/5-wave counts, dominant cycle top/bottom timing, computer trend, point & figure and candlestick patterns. Outputs BUY/SELL scores (0-100), outlook for the next months, and entry / stop / targets for **intraday, swing and position** trades |
| **Order Flow & Auction** | LTP, session VWAP with ±1σ/±2σ bands, delta per bar and cumulative delta (CVD), footprint (bid × ask per price, stacked 3:1 imbalances), large prints / time & sales, absorption, initiative buying/selling and CVD divergence; volume profile (POC, VAH/VAL, HVN/LVN, P/b/D shape) and TPO market profile (letters, initial balance, range extension, single prints, poor highs/lows); auction-market read: open type, open vs prior value, day type, value migration, balance vs imbalance, 80% rule, naked POCs, and an intraday auction plan. Crypto uses real Binance taker-buy delta and trades; other markets estimate delta from 1-minute bars |
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
quantlab/patterns.py    swings, S/R, trendlines, chart & candle patterns, P&F, Elliott, cycles
quantlab/orderflow.py   order flow, VWAP, volume/TPO profile, auction market theory
quantlab/checklist.py   the technical checklist engine, scores and trade plans
quantlab/ui_checklist.py  Streamlit view of the checklist
tests/                  pytest suite (incl. a no-look-ahead test)
```

> Research/education tool only — not investment advice.
