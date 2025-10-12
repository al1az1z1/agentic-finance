from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Project root 
try:
    # .../src/config/settings.py -> repo root
    ROOT_PATH = Path(__file__).resolve().parents[2]
except NameError:
    
    ROOT_PATH = Path.cwd().resolve()

# --- Load only the .env at the repo root --------------------------------------
load_dotenv(ROOT_PATH / ".env", override=False)

# --- Typed config --------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    ROOT: str = str(ROOT_PATH)
    DATA: str = str(ROOT_PATH / "data")
    CACHE: str = str(ROOT_PATH / "data" / "cache")
    RUNS: str = str(ROOT_PATH / "data" / "runs")

@dataclass(frozen=True)
class Models:
    CHAT_MODEL: str = os.getenv("AGENTIC_MODEL", "gpt-4o-mini")

@dataclass(frozen=True)
class FeatureFlags:
    ENABLE_CRITIC: bool = True
    ENABLE_MEMORY: bool = True

@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "AAPL")
    PATHS: Paths = Paths()
    MODELS: Models = Models()
    FLAGS: FeatureFlags = FeatureFlags()

SETTINGS = Settings()

# Create folders on first run
os.makedirs(SETTINGS.PATHS.CACHE, exist_ok=True)
os.makedirs(SETTINGS.PATHS.RUNS, exist_ok=True)

# Fail fast if API key is missing
if not SETTINGS.OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the project's .env at the repo root.")
