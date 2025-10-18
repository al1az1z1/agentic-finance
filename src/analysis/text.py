# Second approach

from __future__ import annotations
import re
import pandas as pd
from datetime import datetime, timedelta
from ..config.settings import SETTINGS

# -----------------------------
# Existing tagging / preprocessing
# -----------------------------

TAG_RULES = {
    "earnings": ["earnings", "eps", "guidance", "outlook", "quarter", "revenue"],
    "product":  ["launch", "iphone", "chip", "feature", "service"],
    "legal":    ["lawsuit", "regulator", "antitrust", "fine", "settlement"],
    "macro":    ["inflation", "rates", "fed", "recession", "gdp"]
}

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "published_at","source","title","summary","url",
            "overall_sentiment","tags","numbers"
        ])

    df = df.copy()

    # Alpha Vantage format is like "20251017T200143"
    # Parse with explicit format; keep timezone-aware for safety
    df["published_at"] = pd.to_datetime(
        df["published_at"], format="%Y%m%dT%H%M%S", errors="coerce", utc=True
    )

    # Drop rows with no title/url; keep others (don’t drop NaT here — the date filter happens later)
    df = df.dropna(subset=["title","url"]).drop_duplicates(subset=["url"])
    df["summary"] = df["summary"].fillna("")
    return df

def classify_tags(text: str) -> list[str]:
    text_l = text.lower()
    tags = [k for k, kws in TAG_RULES.items() if any(kw in text_l for kw in kws)]
    return tags or ["general"]

NUM_RE = re.compile(r'(\$?\b\d+(?:\.\d+)?%?)')

def extract_numbers(text: str) -> list[str]:
    return NUM_RE.findall(text or "")[:6]

def add_tags_and_numbers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["tags"] = (df["title"] + " " + df["summary"]).apply(classify_tags)
    df["numbers"] = (df["title"] + " " + df["summary"]).apply(extract_numbers)
    return df

def recent_topk(df: pd.DataFrame, topk: int, days: int, required_tags: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    # Make an aware UTC cutoff; df['published_at'] is already UTC-aware
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    f = df[df["published_at"] >= cutoff]

    if required_tags:
        want = [t.strip().lower() for t in required_tags]
        f_tags = f[f["tags"].apply(lambda ts: any(t in [x.lower() for x in ts] for t in want))]
        f = f_tags if not f_tags.empty else f

    return f.sort_values("published_at", ascending=False).head(topk)

# -----------------------------
# NEW: shared agent utilities
# -----------------------------

import json

def strip_code_fences(s: str) -> str:
    """Remove leading/trailing ``` blocks (optionally ```json)."""
    if not isinstance(s, str):
        return s
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)

def to_float(x, default: float = 0.0) -> float:
    """Best-effort conversion of model outputs or strings to float."""
    try:
        if isinstance(x, str):
            xs = x.strip().lower()
            # map common words to numeric anchors
            if xs in ("high", "strong", "bullish", "overbought"): 
                return 0.8
            if xs in ("medium", "moderate", "neutral"):
                return 0.5
            if xs in ("low", "weak", "bearish", "oversold"):
                return 0.2
        return float(x)
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_score(v: float) -> float:
    """
    Normalize arbitrary score ranges to [-1, 1].
    Heuristics:
      - If already in [-1,1], keep.
      - If in [0,1], map to [-1,1] via (v-0.5)*2.
      - If in (1,100], treat as percent.
      - If in (1,10], treat as 0-10 and map.
      - Else, clamp.
    """
    try:
        v = float(v)
    except Exception:
        return 0.0
    if -1.0 <= v <= 1.0:
        return v
    if 0.0 <= v <= 1.0:
        return (v - 0.5) * 2.0
    if 1.0 < v <= 100.0:
        v01 = v / 100.0
        return (v01 - 0.5) * 2.0
    if 1.0 < v <= 10.0:
        v01 = v / 10.0
        return (v01 - 0.5) * 2.0
    return clamp(v, -1.0, 1.0)

def normalize_conf(v) -> float:
    """Normalize any confidence-like value to [0,1]."""
    f = to_float(v, 0.7)
    if 1.0 < f <= 100.0:
        f = f / 100.0
    return clamp(f, 0.0, 1.0)

# Optional: helpers to render structured dicts into strings (for external tools)
def pretty_json_block(obj: dict, max_chars: int = 4000) -> str:
    """Return a fenced JSON markdown block, truncated for UI safety."""
    try:
        js = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        js = str(obj)
    if len(js) > max_chars:
        js = js[: max_chars - 20] + "\n... (truncated)"
    return f"```json\n{js}\n```"
