from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..config.settings import SETTINGS

MEM_PATH = SETTINGS.runs_dir / "run_notes.jsonl"

# keep a history of runs, critiques, scores, notes
def append_memory(record: dict[str, Any]) -> None:
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
