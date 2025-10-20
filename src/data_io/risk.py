from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from ..data_io.cache import load_cache, save_cache
from ..config.settings import SETTINGS
from ..data_io.prices import fetch_prices

def _daily_returns(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "close" not in df:
        return pd.Series(dtype=float)
    return df["close"].astype(float).pct_change()

def _max_drawdown_pct(prices: pd.DataFrame) -> float:
    if prices is None or prices.empty or "close" not in prices:
        return float("nan")
    series = prices["close"].astype(float)
    roll_max = series.cummax()
    drawdown = (series / roll_max) - 1.0
    mdd = drawdown.min()
    return float(round(mdd * 100.0, 3))

def _beta_vs_bench(asset_rets: pd.Series, bench_rets: pd.Series) -> float:
    m = pd.concat([asset_rets, bench_rets], axis=1).dropna()
    if m.empty:
        return float("nan")
    cov = np.cov(m.iloc[:, 0], m.iloc[:, 1])[0, 1]
    var = np.var(m.iloc[:, 1])
    if var == 0:
        return float("nan")
    return float(cov / var)

def fetch_risk_metrics(symbol: str, start: Optional[str], end: Optional[str], benchmark: str = "^GSPC") -> Dict[str, Any]:
    cache_key = f"risk_{symbol}_{start}_{end}"
    cached = load_cache(cache_key, ttl_minutes=SETTINGS.cache_ttl_minutes)
    if cached is not None:
        return cached

    prices = fetch_prices(symbol, start, end)
    if prices is None or prices.empty:
        save_cache(cache_key, {})
        return {}

    rets = _daily_returns(prices).dropna()
    if rets.empty:
        save_cache(cache_key, {})
        return {}

    mean_ret = float(rets.mean())
    vol = float(rets.std())
    sharpe = float(mean_ret / vol) if vol > 0 else float("nan")
    mdd_pct = _max_drawdown_pct(prices)
    var_5 = float(np.nanpercentile(rets.values, 5))

    # Download benchmark with explicit auto_adjust to avoid FutureWarning
    beta = float("nan")
    try:
        bench = yf.download(
            benchmark,
            start=prices["date"].min(),
            end=prices["date"].max(),
            progress=False,
            auto_adjust=False,   # <— key change
            threads=False,
        )
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = [c[0].lower() for c in bench.columns]
        else:
            bench.columns = [c.lower() for c in bench.columns]
        bench = bench.reset_index().rename(columns={"Date": "date"})
        bench["date"] = bench["date"].astype(str)
        bench_rets = bench["close"].astype(float).pct_change().dropna()
        n = min(len(rets), len(bench_rets))
        beta = _beta_vs_bench(rets.tail(n).reset_index(drop=True), bench_rets.tail(n).reset_index(drop=True))
    except Exception:
        beta = float("nan")

    metrics = {
        "avg_daily_return": round(mean_ret, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else float("nan"),
        "max_drawdown": mdd_pct,       # percent (negative)
        "var_5": round(var_5, 6),
        "beta": round(beta, 3) if not np.isnan(beta) else float("nan"),
    }
    save_cache(cache_key, metrics)
    return metrics
