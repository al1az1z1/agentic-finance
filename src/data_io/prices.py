from __future__ import annotations
import pandas as pd
import yfinance as yf
from ..data_io.cache import load_cache, save_cache
from ..config.settings import SETTINGS

def fetch_prices(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    cache_key = f"prices_{symbol}_{start}_{end}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return pd.DataFrame(cached)
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    df = df.reset_index().rename(columns={
        "Date": "date", "open":"open","high":"high","low":"low","close":"close","adj close":"adj_close","volume":"volume"
    })
    df["date"] = df["date"].astype(str)
    save_cache(cache_key, df.to_dict(orient="records"))
    return df
