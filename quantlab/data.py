"""Market data loading (Yahoo Finance) with a synthetic offline fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE: dict[str, dict[str, str]] = {
    "Indices": {
        "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI",
        "NIFTY 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN",
    },
    "US Stocks": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA", "JPMorgan": "JPM",
    },
    "India Stocks": {
        "Reliance": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS", "TCS": "TCS.NS", "SBI": "SBIN.NS", "Tata Motors": "TATAMOTORS.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X", "USD/CHF": "CHF=X", "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
        "DXY (Dollar Index)": "DX-Y.NYB",
    },
    "Metals": {
        "Gold (XAU/USD)": "GC=F", "Silver (XAG/USD)": "SI=F",
        "Platinum": "PL=F", "Copper": "HG=F",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
        "BNB": "BNB-USD", "XRP": "XRP-USD", "Cardano": "ADA-USD",
        "Dogecoin": "DOGE-USD", "Avalanche": "AVAX-USD",
    },
}

# Trading periods per year, used to annualise statistics.
PERIODS_PER_YEAR = {"1d": 252, "1h": 252 * 24, "1wk": 52}
CRYPTO_PERIODS_PER_YEAR = {"1d": 365, "1h": 365 * 24, "1wk": 52}


def asset_class(ticker: str) -> str:
    for cls, items in UNIVERSE.items():
        if ticker in items.values():
            return cls
    return "Crypto" if ticker.endswith("-USD") else "Other"


def periods_per_year(ticker: str, interval: str = "1d") -> int:
    table = CRYPTO_PERIODS_PER_YEAR if asset_class(ticker) == "Crypto" else PERIODS_PER_YEAR
    return table.get(interval, 252)


def synthetic_ohlcv(ticker: str, n: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Regime-switching GBM with fat tails, so the app works offline."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % 2**32)
    base = {"Forex": (1.1, 0.07), "Metals": (2000.0, 0.15), "Crypto": (30000.0, 0.65)}
    p0, vol = base.get(asset_class(ticker), (100.0, 0.2))
    dt = 1 / 252
    regimes = np.cumsum(rng.random(n) < 0.01) % 3
    drift = np.array([0.10, -0.10, 0.0])[regimes]
    vols = vol * np.array([0.8, 1.4, 1.0])[regimes]
    shocks = rng.standard_t(df=4, size=n) / np.sqrt(2)
    rets = (drift - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * shocks
    close = p0 * np.exp(np.cumsum(rets))
    open_ = np.r_[p0, close[:-1]]
    rng_hl = np.abs(rng.normal(0, vols * np.sqrt(dt), n)) * close
    high = np.maximum(open_, close) + rng_hl
    low = np.minimum(open_, close) - rng_hl
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    volume = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": volume}, index=idx)


def load_ohlcv(ticker: str, period: str = "3y", interval: str = "1d") -> tuple[pd.DataFrame, bool]:
    """Return (frame, is_live). Falls back to synthetic data if download fails."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        if len(df) > 50:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df, True
    except Exception:
        pass
    return synthetic_ohlcv(ticker), False


def load_closes(tickers: list[str], period: str = "3y") -> tuple[pd.DataFrame, bool]:
    frames, live = {}, True
    for t in tickers:
        df, ok = load_ohlcv(t, period)
        live &= ok
        frames[t] = df["Close"]
    return pd.DataFrame(frames).ffill().dropna(how="all"), live
