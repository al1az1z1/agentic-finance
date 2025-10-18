# cache.py
from __future__ import annotations
import json, time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from ..config.settings import SETTINGS

def _cache_path(key: str) -> Path:
    return SETTINGS.cache_dir / f"{key}.json"

def _json_default(o: Any):
    # datetime & pandas.Timestamp (subclass of datetime) → ISO 8601
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # Fallback: make a best-effort string (covers Decimal, Path, Enum, etc.)
    try:
        return str(o)
    except Exception:
        return repr(o)

def load_cache(key: str, ttl_minutes: int | None = None) -> Any | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if ttl_minutes is None:
            return obj.get("data")  # consistent: always return payload
        if (time.time() - obj.get("_ts", 0)) <= ttl_minutes * 60:
            return obj.get("data")
    except Exception:
        return None
    return None

def save_cache(key: str, data: Any) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {"_ts": time.time(), "data": data}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(p)  # atomic on most OS/filesystems
