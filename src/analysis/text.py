from __future__ import annotations
import re, pandas as pd
from datetime import datetime, timedelta
from ..config.settings import SETTINGS

TAG_RULES = {
    "earnings": ["earnings", "eps", "guidance", "outlook", "quarter", "revenue"],
    "product":  ["launch", "iphone", "chip", "feature", "service"],
    "legal":    ["lawsuit", "regulator", "antitrust", "fine", "settlement"],
    "macro":    ["inflation", "rates", "fed", "recession", "gdp"]
}

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"])
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
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
    if df.empty: return df
    df = df.copy()
    df["tags"] = (df["title"] + " " + df["summary"]).apply(classify_tags)
    df["numbers"] = (df["title"] + " " + df["summary"]).apply(extract_numbers)
    return df

def recent_topk(df: pd.DataFrame, topk: int, days: int, required_tags: list[str] | None = None) -> pd.DataFrame:
    if df.empty: return df
    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=days))
    f = df[df["published_at"] >= cutoff]
    if required_tags:
        f = f[f["tags"].apply(lambda ts: any(t in ts for t in required_tags))]
    f = f.sort_values("published_at", ascending=False).head(topk)
    return f

def summarize_rows(df: pd.DataFrame) -> list[str]:
    bullets = []
    for _, r in df.iterrows():
        nums = ", ".join(r.get("numbers") or [])
        bullets.append(f"- {r['title']} ({r['source']}) [{r['url']}] {(' | nums: ' + nums) if nums else ''}")
    return bullets[:5]
