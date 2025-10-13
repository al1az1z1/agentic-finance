from __future__ import annotations
import requests, pandas as pd
from ..config.settings import SETTINGS

BASE = "https://www.alphavantage.co/query"
KEYS = {"SMA": "Technical Analysis: SMA", "RSI": "Technical Analysis: RSI"}

def fetch_indicator(symbol: str, indicator: str, time_period: int = 14) -> pd.DataFrame:
    if not SETTINGS.alpha_api_key:
        return pd.DataFrame()
    params = {
        "function": indicator, "symbol": symbol, "interval": "daily",
        "time_period": time_period, "series_type": "close",
        "apikey": SETTINGS.alpha_api_key
    }
    data = requests.get(BASE, params=params, timeout=30).json()
    key = KEYS.get(indicator)
    if key not in data:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "date"})
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
