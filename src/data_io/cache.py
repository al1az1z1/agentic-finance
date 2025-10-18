from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
from ..config.settings import SETTINGS

def _cache_path(key: str) -> Path:
    return SETTINGS.cache_dir / f"{key}.json"

def load_cache(key: str, ttl_minutes: int | None = None) -> Any | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if ttl_minutes is None:
            return obj
        if (time.time() - obj.get("_ts", 0)) <= ttl_minutes * 60:
            return obj.get("data")
    except Exception:
        return None
    return None

def save_cache(key: str, data: Any) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps({"_ts": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
