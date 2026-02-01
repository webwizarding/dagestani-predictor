"""HTTP fetcher with caching, throttling, and robots.txt support."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from src.config import RAW_HTML_DIR, ScrapeConfig, USER_AGENT

logger = logging.getLogger(__name__)


class CacheStore:
    """Simple file cache for HTML responses keyed by URL hash."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.jsonl"

    def _key_to_path(self, url: str) -> Path:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.base_dir / f"{key}.html"

    def has(self, url: str) -> bool:
        return self._key_to_path(url).exists()

    def read(self, url: str) -> str:
        return self._key_to_path(url).read_text(encoding="utf-8")

    def write(self, url: str, content: str, status: int) -> Path:
        path = self._key_to_path(url)
        path.write_text(content, encoding="utf-8")
        record = {
            "url": url,
            "path": str(path),
            "status": status,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return path


class RobotsCache:
    """Fetches and caches robots.txt per domain."""

    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser] = {}

    def allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._parsers:
            robots_url = f"{base}/robots.txt"
            parser = RobotFileParser()
            try:
                parser.set_url(robots_url)
                parser.read()
            except Exception as exc:  # pragma: no cover - network edge
                logger.warning("robots.txt fetch failed for %s: %s", base, exc)
            self._parsers[base] = parser
        return self._parsers[base].can_fetch(user_agent, url)


class CachedSession:
    """Session wrapper with caching, throttling, and retry/backoff."""

    def __init__(self, config: ScrapeConfig) -> None:
        self.config = config
        self.cache = CacheStore(RAW_HTML_DIR)
        self.robots = RobotsCache()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _throttle(self) -> None:
        delay = random.uniform(self.config.min_delay_s, self.config.max_delay_s)
        time.sleep(delay)

    def get(self, url: str) -> str:
        if self.config.cache_enabled and self.cache.has(url):
            logger.debug("cache hit: %s", url)
            return self.cache.read(url)

        if self.config.respect_robots and not self.robots.allowed(url, USER_AGENT):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.config.timeout_s)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"status {resp.status_code}")
                text = resp.text
                if self.config.cache_enabled:
                    self.cache.write(url, text, resp.status_code)
                return text
            except Exception as exc:  # pragma: no cover - retries
                last_exc = exc
                backoff = (self.config.backoff_factor ** attempt) + random.random()
                logger.warning("fetch failed (%s), backing off %.2fs", exc, backoff)
                time.sleep(backoff)

        raise RuntimeError(f"failed to fetch {url}: {last_exc}")


def save_raw_html(url: str, html: str, target_dir: Path = RAW_HTML_DIR) -> Path:
    """Save a raw HTML blob to a deterministic path for traceability."""
    store = CacheStore(target_dir)
    return store.write(url, html, status=200)


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


