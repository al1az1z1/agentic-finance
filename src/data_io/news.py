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
    # ====== FORTH APPROACH =====
    rows = []
    for item in data["feed"]:
        rows.append({
            "published_at": item.get("time_published"),
            "source": item.get("source"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "url": item.get("url"),
            "overall_sentiment": item.get("overall_sentiment_label")
        })
    # rows = []
    # for item in data.get("feed", []):
    #     tickers = item.get("ticker_sentiment", []) or []
    #     # keep only if our symbol is explicitly mentioned
    #     keep = any(t.get("ticker", "").upper() == symbol.upper() and float(t.get("relevance_score", 0) or 0) >= 0.30
    #                for t in tickers)
    #     if not keep:
    #         continue

    #     rows.append({
    #         "published_at": item.get("time_published"),
    #         "source": item.get("source"),
    #         "title": item.get("title"),
    #         "summary": item.get("summary"),
    #         "url": item.get("url"),
    #         "overall_sentiment": item.get("overall_sentiment_label")
    #     })

    # ====== Forth APPROACH =====
    df = pd.DataFrame(rows)
    save_cache(cache_key, df.to_dict(orient="records"))
    return df

# fifth approach



# def fetch_news(symbol: str) -> pd.DataFrame:
#     if not SETTINGS.alpha_api_key:
#         return pd.DataFrame()  # safe fail

#     cache_key = f"news_{symbol}"
#     cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
#     if cached is not None:
#         return pd.DataFrame(cached)

#     params = {"function":"NEWS_SENTIMENT","tickers":symbol,"apikey":SETTINGS.alpha_api_key}
#     r = requests.get(BASE, params=params, timeout=30)
#     data = r.json()
#     if "feed" not in data:
#         return pd.DataFrame()

#     rows = []
#     sym_up = symbol.upper()
#     for item in data["feed"]:
#         # keep only if our ticker is present with decent relevance
#         ts = item.get("ticker_sentiment") or []
#         keep = any(
#             (t.get("ticker","").upper() == sym_up) and \
#             (float(t.get("relevance_score", 0) or 0) >= 0.30)
#             for t in ts
#         )
#         if not keep:
#             continue

#         rows.append({
#             "published_at": item.get("time_published"),
#             "source": item.get("source"),
#             "title": item.get("title"),
#             "summary": item.get("summary"),
#             "url": item.get("url"),
#             "overall_sentiment": item.get("overall_sentiment_label"),
#         })

#     df = pd.DataFrame(rows)
#     save_cache(cache_key, df.to_dict(orient="records"))
#     return df



# # Sixth approach

# def fetch_news(symbol: str) -> pd.DataFrame:
#     if not SETTINGS.alpha_api_key:
#         return pd.DataFrame()  # safe fail

#     cache_key = f"news_{symbol}"
#     cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
#     if cached is not None:
#         return pd.DataFrame(cached)

#     params = {
#         "function": "NEWS_SENTIMENT",
#         "tickers": symbol,
#         "sort": "LATEST",
#         "limit": 50,  # get enough to filter
#         "apikey": SETTINGS.alpha_api_key,
#     }
#     r = requests.get(BASE, params=params, timeout=30)
#     data = r.json()

#     # Handle rate limit / info responses
#     if isinstance(data, dict) and any(k in data for k in ("Note", "Information", "Error Message")):
#         # Don't cache this; just return empty to allow a retry next call
#         return pd.DataFrame()

#     if "feed" not in data or not data["feed"]:
#         return pd.DataFrame()

#     rows = []
#     sym_up = symbol.upper()
#     name_hints = {
#         "AMZN": ["amazon"],
#         "MSFT": ["microsoft"],
#         "AAPL": ["apple"],
#         "GOOGL": ["alphabet", "google"],
#         "META": ["meta", "facebook"],
#         # add more if you need
#     }
#     aliases = name_hints.get(sym_up, [])

#     # relaxed threshold; AV relevance can be low even for real mentions
#     MIN_REL = 0.20

#     for item in data["feed"]:
#         title = (item.get("title") or "").strip()
#         summary = (item.get("summary") or "").strip()
#         text_l = f"{title} {summary}".lower()

#         ts = item.get("ticker_sentiment") or []
#         has_ticker = any(
#             (t.get("ticker", "").upper() == sym_up) and
#             (float(t.get("relevance_score", 0) or 0) >= MIN_REL)
#             for t in ts
#         )

#         # Fallback: headline/summary keyword match (“Amazon”, “AMZN”, etc.)
#         has_keyword = (sym_up in title.upper()) or any(alias in text_l for alias in aliases)

#         if not (has_ticker or has_keyword):
#             continue

#         rows.append({
#             "published_at": item.get("time_published"),
#             "source": item.get("source"),
#             "title": title,
#             "summary": summary,
#             "url": item.get("url"),
#             "overall_sentiment": item.get("overall_sentiment_label"),
#         })

#     df = pd.DataFrame(rows)

#     # Only cache if we actually have rows (avoid caching empty due to rate limits / strict filters)
#     if not df.empty:
#         save_cache(cache_key, df.to_dict(orient="records"))
#     return df


# Seventh approach


# def fetch_news(symbol: str) -> pd.DataFrame:
#     if not SETTINGS.alpha_api_key:
#         return pd.DataFrame()  # requires ALPHAVANTAGE_API_KEY

#     cache_key = f"news_{symbol.upper()}"
#     cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
#     if cached is not None:
#         return pd.DataFrame(cached)

#     params = {
#         "function": "NEWS_SENTIMENT",
#         "tickers": symbol,
#         "sort": "LATEST",
#         "limit": 50,
#         "apikey": SETTINGS.alpha_api_key
#     }
#     r = requests.get(BASE, params=params, timeout=30)
#     data = r.json()

#     # Rate limit / error responses return "Note"/"Information"/"Error Message"
#     if isinstance(data, dict) and any(k in data for k in ("Note","Information","Error Message")):
#         return pd.DataFrame()

#     feed = data.get("feed") or []
#     if not feed:
#         return pd.DataFrame()

#     rows = []
#     for item in feed:
#         rows.append({
#             "published_at": item.get("time_published"),  # parsed later
#             "source": item.get("source"),
#             "title": (item.get("title") or "").strip(),
#             "summary": (item.get("summary") or "").strip(),
#             "url": item.get("url"),
#             "overall_sentiment": item.get("overall_sentiment_label"),
#         })

#     df = pd.DataFrame(rows)

 
#     if not df.empty:
#         save_cache(cache_key, df.to_dict(orient="records"))

#     return df
