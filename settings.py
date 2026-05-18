from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CHUNKY_DIR = Path(os.environ.get("BESTCHOICE_CHUNKY_DIR", PROJECT_DIR.parent / "chunkymonkey"))

MARKET_DB = Path(os.environ.get("BESTCHOICE_MARKET_DB", CHUNKY_DIR / "data/market.duckdb"))
SMART_DB = Path(os.environ.get("BESTCHOICE_SMART_DB", CHUNKY_DIR / "data/smartmoney.duckdb"))
SCRIPTS_DIR = Path(os.environ.get("BESTCHOICE_SCRIPTS_DIR", PROJECT_DIR / "scripts"))
CACHE_DIR = Path(os.environ.get("BESTCHOICE_CACHE_DIR", PROJECT_DIR))

CACHE_MAX_AGE = int(os.environ.get("BESTCHOICE_CACHE_MAX_AGE", "86400"))
MAX_WARMUP_WORKERS = max(1, int(os.environ.get("BESTCHOICE_MAX_WARMUP_WORKERS", "1")))

