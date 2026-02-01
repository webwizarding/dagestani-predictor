"""Project configuration and paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
PARSED_DIR = DATA_DIR / "parsed"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

UFCSTATS_BASE = "http://ufcstats.com"
USER_AGENT = "dagestani-predictor/0.1 (+https://example.com)"


@dataclass(frozen=True)
class ScrapeConfig:
    min_delay_s: float = 0.75
    max_delay_s: float = 1.5
    max_retries: int = 4
    backoff_factor: float = 1.7
    timeout_s: int = 20
    respect_robots: bool = True
    cache_enabled: bool = True
    max_events: int | None = None
    max_fights_per_event: int | None = None
    dev_mode: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    min_prior_fights: int = 0
    ewma_alpha: float = 0.35


@dataclass(frozen=True)
class ModelConfig:
    test_size_recent: int = 200
    random_state: int = 42
    logistic_c: float = 1.0

