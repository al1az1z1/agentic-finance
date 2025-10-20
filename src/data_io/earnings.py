# src/data_io/earnings.py
from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import Dict, Any
from ..data_io.cache import load_cache, save_cache
from ..config.settings import SETTINGS

def fetch_earnings(symbol: str) -> pd.DataFrame:
    """
    Quarterly earnings with EPS actual/estimate/surprise.
    Columns: ['date','EPS Estimate','Reported EPS','Surprise(%)']
    """
    cache_key = f"earnings_{symbol}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        tk = yf.Ticker(symbol)
        df = tk.earnings_dates
        if df is None or getattr(df, "empty", True):
            df = pd.DataFrame(columns=["Earnings Date","EPS Estimate","Reported EPS","Surprise(%)"])

        # Normalize column named 'Earnings Date' -> 'date'
        if "Earnings Date" in df.columns:
            df = df.reset_index(drop=True).rename(columns={"Earnings Date": "date"})
        elif df.index.name == "Earnings Date":
            df = df.reset_index().rename(columns={"Earnings Date": "date"})
        else:
            if "date" not in df.columns:
                df = df.reset_index().rename(columns={"index": "date"})

        keep = ["date","EPS Estimate","Reported EPS","Surprise(%)"]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        df = df[keep].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        df = pd.DataFrame(columns=["date","EPS Estimate","Reported EPS","Surprise(%)"])

    save_cache(cache_key, df.to_dict(orient="records"))
    return df
