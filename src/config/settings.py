from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# project root = repo root
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "data" / "cache"
    runs_dir: Path = ROOT / "data" / "runs"
    alpha_api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    news_window_days: int = 14
    topk_news: int = 5
    cache_ttl_minutes: int = 60

SETTINGS = Settings()
SETTINGS.cache_dir.mkdir(parents=True, exist_ok=True)
SETTINGS.runs_dir.mkdir(parents=True, exist_ok=True)
