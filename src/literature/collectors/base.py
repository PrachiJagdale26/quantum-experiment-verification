"""
Shared machinery for scholarly API collectors: a normalised record type, polite rate limiting,
retries on rate limits / server errors, and an on-disk response cache so searches are reproducible.

Rules (from the project brief): never fabricate metadata; keep source URLs and retrieval
timestamps; record missing abstracts and unavailable full text explicitly; respect rate limits.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR = os.getenv("LITERATURE_CACHE_DIR") or os.path.join(ROOT, "data", "cache", "http")
USER_AGENT = "research-gap-framework/0.2 (academic research prototype)"

logger = logging.getLogger(__name__)

try:
    import diskcache
    _CACHE = diskcache.Cache(CACHE_DIR, size_limit=512 * 1024 * 1024)
except Exception:  # pragma: no cover - cache is an optimisation, never a requirement
    _CACHE = None


@dataclass
class PublicationRecord:
    """One record as returned by one source, normalised but not merged."""
    source: str
    title: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    arxiv_version: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    year: Optional[int] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    source_url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    referenced_works: List[str] = field(default_factory=list)   # OpenAlex ids, when the source provides them
    cited_by_count: Optional[int] = None
    versions: List[Dict[str, Any]] = field(default_factory=list)  # [{"label": "v2", "date": "..."}]
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    source: str
    query: str
    status: str                      # ok | cached | rate_limited | failed | skipped
    records: List[PublicationRecord] = field(default_factory=list)
    total_available: Optional[int] = None
    error: Optional[str] = None
    url: Optional[str] = None


class RateLimitedError(RuntimeError):
    pass


class BaseCollector:
    """HTTP GET with a per-source minimum interval, retries, and a response cache."""
    name = "base"
    min_interval = 1.0           # seconds between requests to this source
    cache_ttl = 7 * 24 * 3600    # responses reused for a week -> reproducible re-runs

    _locks: Dict[str, threading.Lock] = {}
    _last_call: Dict[str, float] = {}

    def __init__(self, client: Optional[httpx.Client] = None, use_cache: bool = True):
        self.client = client or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        self.use_cache = use_cache and _CACHE is not None

    def _wait_turn(self) -> None:
        lock = self._locks.setdefault(self.name, threading.Lock())
        with lock:
            wait = self.min_interval - (time.monotonic() - self._last_call.get(self.name, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_call[self.name] = time.monotonic()

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, as_json: bool = True,
            cache_params: Optional[Dict[str, Any]] = None) -> tuple:
        """Returns (payload, from_cache). `cache_params` excludes secrets (API keys) from the cache key."""
        key_params = cache_params if cache_params is not None else params
        key = f"{self.name}|{url}|{json.dumps(key_params or {}, sort_keys=True)}"
        if self.use_cache and key in _CACHE:
            return _CACHE[key], True

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            self._wait_turn()
            try:
                resp = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                if attempt == 2 or retry_after > 30:
                    raise RateLimitedError(f"{self.name} rate limit (HTTP 429)")
                time.sleep(max(retry_after, 2 ** (attempt + 1)))
                continue
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"{self.name} HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 404:
                return None, False
            resp.raise_for_status()
            payload = resp.json() if as_json else resp.text
            if self.use_cache:
                _CACHE.set(key, payload, expire=self.cache_ttl)
            return payload, False
        raise last_exc or RuntimeError(f"{self.name} request failed")

    def run_search(self, query: str, fn, **kwargs) -> SearchResult:
        """Wrap a source-specific search so every outcome (including failure) is a SearchResult."""
        try:
            records, total, cached, url = fn(query, **kwargs)
            return SearchResult(self.name, query, "cached" if cached else "ok", records, total, url=url)
        except RateLimitedError as exc:
            logger.warning(str(exc))
            return SearchResult(self.name, query, "rate_limited", error=str(exc))
        except Exception as exc:
            logger.warning(f"{self.name} search failed: {exc}")
            return SearchResult(self.name, query, "failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
