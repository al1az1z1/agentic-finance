from __future__ import annotations
import os, requests, pandas as pd
from ..data_io.cache import load_cache, save_cache
from ..config.settings import SETTINGS

BASE = "https://www.alphavantage.co/query"


def fetch_news(symbol: str) -> pd.DataFrame:
    if not SETTINGS.alpha_api_key:
        return pd.DataFrame()  # safe fail
    cache_key = f"news_{symbol}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return pd.DataFrame(cached)

    params = {"function":"NEWS_SENTIMENT","tickers":symbol,"apikey":SETTINGS.alpha_api_key}
    r = requests.get(BASE, params=params, timeout=30)
    data = r.json()
    if "feed" not in data:
        return pd.DataFrame()
    
    rows = []
    for item in data.get("feed", []):
        tickers = item.get("ticker_sentiment", []) or []
        # keep only if our symbol is explicitly mentioned
        keep = any(t.get("ticker", "").upper() == symbol.upper() and float(t.get("relevance_score", 0) or 0) >= 0.30
                   for t in tickers)
        if not keep:
            continue

        rows.append({
            "published_at": item.get("time_published"),
            "source": item.get("source"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "url": item.get("url"),
            "overall_sentiment": item.get("overall_sentiment_label")
        })

    # ====== Forth APPROACH =====
    df = pd.DataFrame(rows)
    save_cache(cache_key, df.to_dict(orient="records"))
    return df




