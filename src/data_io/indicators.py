from __future__ import annotations
import requests
import pandas as pd
from typing import Optional
from ..config.settings import SETTINGS
from .prices import fetch_prices
from ..analysis.features import compute_sma, compute_rsi
from ..data_io.cache import load_cache, save_cache

BASE = "https://www.alphavantage.co/query"
KEYS = {"SMA": "Technical Analysis: SMA", "RSI": "Technical Analysis: RSI"}


# If AV isn’t available (no key/limit), our code falls back to computing indicators locally from prices using our compute_sma / compute_rsi.
def _fallback_from_prices(symbol: str, indicator: str, time_period: int) -> pd.DataFrame:
    prices = fetch_prices(symbol, None, None)
    if prices is None or prices.empty:
        return pd.DataFrame()

    if indicator == "SMA":
        df = pd.DataFrame({"date": prices["date"], "SMA": compute_sma(prices, window=time_period)})
    elif indicator == "RSI":
        df = pd.DataFrame({"date": prices["date"], "RSI": compute_rsi(prices, window=time_period)})
    else:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("date", ascending=True).reset_index(drop=True)
    return df

def fetch_indicator(symbol: str, indicator: str, time_period: int = 14) -> pd.DataFrame:
    key = KEYS.get(indicator)

    # Try cache first
    cache_key = f"indicator_{symbol}_{indicator}_{time_period}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return pd.DataFrame(cached)

    if not SETTINGS.alpha_api_key or key is None:
        df = _fallback_from_prices(symbol, indicator, time_period)
        save_cache(cache_key, df.to_dict(orient="records"))
        return df

    params = {
        "function": indicator,
        "symbol": symbol,
        "interval": "daily",
        "time_period": time_period,
        "series_type": "close",
        "apikey": SETTINGS.alpha_api_key,
    }
    try:
        resp = requests.get(BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage quota message handling:
        if (not data or key not in data or not data[key] or "Note" in data or "Information" in data or "Error Message" in data):
            df = _fallback_from_prices(symbol, indicator, time_period)
            save_cache(cache_key, df.to_dict(orient="records"))
            return df
    except Exception:
        df = _fallback_from_prices(symbol, indicator, time_period)
        save_cache(cache_key, df.to_dict(orient="records"))
        return df

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "date"})
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=True).reset_index(drop=True)
    save_cache(cache_key, df.to_dict(orient="records"))
    return df
