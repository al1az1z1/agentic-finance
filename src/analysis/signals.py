from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from typing import Optional
from ..config.settings import SETTINGS

BASE = "https://www.alphavantage.co/query"
KEYS = {"SMA": "Technical Analysis: SMA", "RSI": "Technical Analysis: RSI"}

# Local computations
def compute_sma(prices: pd.DataFrame, window: int) -> pd.Series:
    """Compute simple moving average on close prices."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    return prices["close"].rolling(window=window).mean()

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) from close prices."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    delta = prices["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Align with price index so downstream merges don't misalign
    gain_s = pd.Series(gain, index=prices.index)
    loss_s = pd.Series(loss, index=prices.index)
    avg_gain = gain_s.rolling(window=window).mean()
    avg_loss = loss_s.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _fallback_from_prices(symbol: str, indicator: str, time_period: int) -> pd.DataFrame:
    """Compute indicator locally using yfinance prices."""
    from ..data_io.prices import fetch_prices
    prices = fetch_prices(symbol, None, None)
    if prices is None or prices.empty:
        return pd.DataFrame()

    if indicator == "SMA":
        df = pd.DataFrame({"date": prices["date"], "SMA": compute_sma(prices, window=time_period)})
    elif indicator == "RSI":
        df = pd.DataFrame({"date": prices["date"], "RSI": compute_rsi(prices, window=time_period)})
    else:
        return pd.DataFrame()

    # Normalize shape like AV path
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df

# Primary fetch with fallback
def fetch_indicator(symbol: str, indicator: str, time_period: int = 14) -> pd.DataFrame:
    key = KEYS.get(indicator)

    # If no API key, go straight to fallback
    if not SETTINGS.alpha_api_key or key is None:
        return _fallback_from_prices(symbol, indicator, time_period)

    params = {
        "function": indicator,
        "symbol": symbol,
        "interval": "daily",
        "time_period": time_period,
        "series_type": "close",
        "apikey": SETTINGS.alpha_api_key,
    }

    data: Optional[dict] = None
    try:
        resp = requests.get(BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # Network/API error -> fallback
        return _fallback_from_prices(symbol, indicator, time_period)

    # Missing/empty payload -> fallback
    if not data or key not in data or not data[key]:
        return _fallback_from_prices(symbol, indicator, time_period)

    # Alpha Vantage happy path
    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "date"})
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure newest is last so iloc[-1] is the latest value
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df
