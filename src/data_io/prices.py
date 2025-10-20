# from __future__ import annotations
# import pandas as pd
# import yfinance as yf
# from ..data_io.cache import load_cache, save_cache
# from ..config.settings import SETTINGS

# def fetch_prices(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
#     cache_key = f"prices_{symbol}_{start}_{end}"
#     cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
#     if cached is not None:
#         return pd.DataFrame(cached)
#     df = yf.download(
#         symbol,
#         start=start,
#         end=end, 
#         progress=False,
#         auto_adjust=False,
#         threads=False,
#     )
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = [c[0].lower() for c in df.columns]
#     df = df.reset_index().rename(columns={
#         "Date": "date", "open":"open","high":"high","low":"low","close":"close","adj close":"adj_close","volume":"volume"
#     })
#     df["date"] = df["date"].astype(str)
#     save_cache(cache_key, df.to_dict(orient="records"))
#     return df


# src/data_io/prices.py
from __future__ import annotations

import os
import time
import logging
import random
from typing import Optional
import pandas as pd
import yfinance as yf
import requests
from datetime import date
import io

from ..data_io.cache import load_cache, save_cache
from ..config.settings import SETTINGS

_LOG = logging.getLogger(__name__)

YF_MAX_RETRIES = 4
YF_BASE_SLEEP = 1.2  # seconds
HTTP_TIMEOUT = 30


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance-like result columns and structure."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    out = pd.DataFrame({
        "date": df["date"].astype(str),
        "open": df.get("open"),
        "high": df.get("high"),
        "low": df.get("low"),
        "close": df.get("close"),
        "adj_close": df.get("adj close") if "adj close" in df.columns else df.get("close"),
        "volume": df.get("volume"),
    })
    out = out.dropna(subset=["date", "close"]).reset_index(drop=True)
    return out


def _fetch_prices_yfinance(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Try yfinance; custom session (User-Agent), shorter history, backoff + jitter."""
    last_exc = None

    # Create a session with a browsery User-Agent (some networks/CDNs block default)
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })

    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                repair=True,
                group_by="column",
                session=session,  # <— important
            )
            if df is not None and not df.empty:
                return _normalize_ohlcv(df)
            else:
                _LOG.warning("yfinance returned empty for %s (attempt %d)", symbol, attempt)
        except Exception as e:
            last_exc = e
            _LOG.warning("yfinance error for %s (attempt %d/%d): %s", symbol, attempt, YF_MAX_RETRIES, e)
        sleep = YF_BASE_SLEEP * (2 ** (attempt - 1)) + random.uniform(0, 0.4)
        time.sleep(sleep)

    # Alternate code path via Ticker().history; use a lighter window than 'max'
    try:
        tk = yf.Ticker(symbol, session=session)
        if start or end:
            h = tk.history(start=start, end=end, auto_adjust=False, interval="1d", prepost=False)
        else:
            # last ~2y is usually enough for risk/indicators
            h = tk.history(period="2y", auto_adjust=False, interval="1d", prepost=False)
        if h is not None and not h.empty:
            return _normalize_ohlcv(h)
    except Exception as e:
        last_exc = e
        _LOG.warning("Ticker().history failed for %s: %s", symbol, e)

    if last_exc:
        raise last_exc
    raise RuntimeError("yfinance returned empty data repeatedly")


def _fetch_prices_alphavantage(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Fallback using Alpha Vantage TIME_SERIES_DAILY_ADJUSTED.
    Requires SETTINGS.alpha_api_key.
    """
    api_key = getattr(SETTINGS, "alpha_api_key", None) or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        _LOG.warning("Alpha Vantage API key not set; cannot fallback for %s", symbol)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    # Choose compact for short windows to reduce payload / rate-limit risk
    outputsize = "full"
    try:
        if start and end:
            sd = date.fromisoformat(start)
            ed = date.fromisoformat(end)
            if (ed - sd).days <= 120:
                outputsize = "compact"
    except Exception:
        pass

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    # small pacing knob (helps with free-tier)
    if getattr(SETTINGS, "av_pacing_seconds", 0) > 0:
        time.sleep(SETTINGS.av_pacing_seconds)

    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        _LOG.error("Alpha Vantage HTTP error for %s: %s", symbol, e)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    key = "Time Series (Daily)"
    # rate-limit payloads often contain 'Note' or 'Information'
    if key not in data:
        _LOG.warning("Alpha Vantage fallback missing time series for %s. Keys: %s", symbol, list(data.keys()))
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    rows = []
    ts = data[key]
    for d, vals in ts.items():
        try:
            rows.append({
                "date": d,
                "open": float(vals.get("1. open")),
                "high": float(vals.get("2. high")),
                "low": float(vals.get("3. low")),
                "close": float(vals.get("4. close")),
                "adj_close": float(vals.get("5. adjusted close")),
                "volume": int(vals.get("6. volume")),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    return df.sort_values("date").reset_index(drop=True)


def _fetch_prices_stooq(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Free CSV daily prices from Stooq.
    Symbols are typically like 'aapl.us'. We'll try both 'symbol' and 'symbol.us'.
    """
    def _try(sym: str) -> pd.DataFrame:
        url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 or not r.text or r.text.strip().startswith("<!"):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["adj_close"] = df["close"]
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        return df.sort_values("date").reset_index(drop=True)

    df = _try(symbol)
    if df.empty and not symbol.lower().endswith(".us"):
        df = _try(f"{symbol}.us")
    return df


def fetch_prices(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    """
    Fetch OHLCV with yfinance; fallback to Alpha Vantage; finally Stooq CSV.
    Cached for performance.
    """
    cache_key = f"prices_{symbol}_{start}_{end}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return pd.DataFrame(cached)

    df = pd.DataFrame()

    # 1) Try Yahoo (yfinance)
    try:
        df = _fetch_prices_yfinance(symbol, start, end)
    except Exception as yf_err:
        _LOG.error("yfinance failed for %s: %s; attempting Alpha Vantage fallback.", symbol, yf_err)

    # 2) Alpha Vantage fallback
    if df is None or df.empty:
        df = _fetch_prices_alphavantage(symbol, start, end)

    # 3) Stooq final fallback
    if df is None or df.empty:
        df = _fetch_prices_stooq(symbol, start, end)

    # 4) Final safety: ensure correct columns even if empty
    if df is None or df.empty:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    save_cache(cache_key, df.to_dict(orient="records"))
    return df
