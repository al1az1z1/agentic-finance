# src/analysis/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

def compute_sma(prices: pd.DataFrame, window: int) -> pd.Series:
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    return prices["close"].rolling(window=window).mean()

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    delta = prices["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_s = pd.Series(gain, index=prices.index)
    loss_s = pd.Series(loss, index=prices.index)
    avg_gain = gain_s.rolling(window=window).mean()
    avg_loss = loss_s.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_max_drawdown(prices: pd.DataFrame) -> float:
    """
    Max drawdown in percent (negative number, e.g., -22.5 for -22.5%).
    Assumes 'close' column exists.
    """
    if prices is None or prices.empty or "close" not in prices:
        return float("nan")
    series = prices["close"].astype(float)
    roll_max = series.cummax()
    drawdown = (series / roll_max) - 1.0
    mdd = drawdown.min()
    return float(round(mdd * 100.0, 3))  # percent