"""
Production-oriented HTTP access to Singapore open-data hosts: rate limiting, optional cache,
host allowlisting, and metrics. Use for all data.gov.sg / geo.data.gov.sg calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from singapore_eda.gov_limits import (
    min_interval_datastore_sec,
    min_interval_file_sec,
    pace_config_public,
)

logger = logging.getLogger("singapore_eda.gov_http")

# Only these hosts may be called through GovClient (defence in depth for SSRF).
_ALLOWED_SUFFIXES = (".data.gov.sg", "data.gov.sg")
_DEFAULT_CACHE_TTL = 3600


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean from the environment (1/true/yes/on)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def is_allowed_gov_url(url: str) -> bool:
    """True for https to data.gov.sg or *.data.gov.sg (e.g. geo)."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("https",):
        return False
    host = (p.netloc or "").lower().split("@")[-1]
    if not host:
        return False
    if ":" in host:
        host = host.split(":", 1)[0]
    return bool(host) and (host == "data.gov.sg" or host.endswith(_ALLOWED_SUFFIXES))


def redact_secrets(s: str) -> str:
    """Mask values that look like API keys in log strings (best-effort)."""
    if not s:
        return s
    # Common env-style keys in URLs/headers
    for needle in (
        "api_key=",
        "X-API-Key",
        "api-key",
        "token=",
    ):
        if needle.lower() in s.lower():
            return "[REDACTED: possible secret]"
    return s


@dataclass
class HttpMetrics:
    """In-process counters (OK for a single app / worker; use an exporter for k8s)."""

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    requests_total: int = 0
    response_bytes_in: int = 0
    status_2xx: int = 0
    status_429: int = 0
    status_5xx: int = 0
    other_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_request_monotonic: float = 0.0
    last_error: str = ""

    def record_request_start(self) -> None:
        with self._lock:
            self.requests_total += 1

    def record_response(self, status: int, body_len: int) -> None:
        with self._lock:
            self.response_bytes_in += body_len
            if 200 <= status < 300:
                self.status_2xx += 1
            elif status == 429:
                self.status_429 += 1
            elif 500 <= status < 600:
                self.status_5xx += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def record_other_error(self, msg: str) -> None:
        with self._lock:
            self.other_errors += 1
            self.last_error = msg[:500]

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "response_bytes_in": self.response_bytes_in,
                "status_2xx": self.status_2xx,
                "status_429": self.status_429,
                "status_5xx": self.status_5xx,
                "other_errors": self.other_errors,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "last_error": self.last_error,
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP singapore_eda_http_requests_total Outbound requests via GovClient",
                "# TYPE singapore_eda_http_requests_total counter",
                f"singapore_eda_http_requests_total {self.requests_total}",
                "# HELP singapore_eda_http_cache_hits_total Disk cache hits",
                "# TYPE singapore_eda_http_cache_hits_total counter",
                f"singapore_eda_http_cache_hits_total {self.cache_hits}",
                "# HELP singapore_eda_http_cache_misses_total Disk cache misses",
                "# TYPE singapore_eda_http_cache_misses_total counter",
                f"singapore_eda_http_cache_misses_total {self.cache_misses}",
                "# HELP singapore_eda_http_429_total Too Many Requests responses",
                "# TYPE singapore_eda_http_429_total counter",
                f"singapore_eda_http_429_total {self.status_429}",
            ]
        return "\n".join(lines) + "\n"


_GLOBAL_METRICS = HttpMetrics()


def get_metrics() -> HttpMetrics:
    return _GLOBAL_METRICS


class _TokenBucketPace:
    """Ensure a minimum wall-clock interval between *successful* call sites (client-side)."""

    def __init__(self, min_interval_sec: float) -> None:
        self._min = max(0.0, float(min_interval_sec))
        self._lock = threading.RLock()
        self._next_ok = 0.0

    def wait_turn(self) -> None:
        if self._min <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
                now = time.monotonic()
            self._next_ok = now + self._min


class _DiskCache:
    def __init__(self, root: Path, ttl_sec: int) -> None:
        self._root = root
        self._ttl = max(0, int(ttl_sec))
        self._lock = threading.RLock()

    @staticmethod
    def _key(method: str, url: str, params: dict[str, Any] | None) -> str:
        return _make_request_cache_key(method, url, params)

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self._ttl <= 0:
            return None
        path = self._root / f"{key}.json"
        with self._lock:
            if not path.is_file():
                return None
            try:
                raw = path.read_text(encoding="utf-8")
                obj = json.loads(raw)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return None
        exp = float(obj.get("exp", 0))
        if time.time() > exp:
            return None
        return obj.get("data")

    def set_json(self, key: str, data: dict[str, Any]) -> None:
        if self._ttl <= 0:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{key}.json"
        payload = {
            "exp": time.time() + self._ttl,
            "data": data,
        }
        with self._lock:
            path.write_text(json.dumps(payload), encoding="utf-8")

    def clear(self) -> int:
        """Remove cached entries; return count removed."""
        n = 0
        with self._lock:
            if not self._root.is_dir():
                return 0
            for p in self._root.glob("*.json"):
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n

    def approx_size_bytes(self) -> int:
        with self._lock:
            if not self._root.is_dir():
                return 0
            return sum(f.stat().st_size for f in self._root.glob("*.json") if f.is_file())


def _normalize_url_params(url: str, params: dict[str, Any]) -> str:
    p = urlparse(url)
    q = parse_qsl(p.query, keep_blank_values=True)
    extra = list(params.items()) if params else []
    merged = sorted(list(q) + [(str(a), str(b)) for a, b in extra])
    newq = urlencode(merged, doseq=True)
    p2 = p._replace(query=newq)
    return urlunparse(p2)


def _make_request_cache_key(method: str, url: str, params: dict[str, Any] | None) -> str:
    norm = _normalize_url_params(url, params or {})
    return hashlib.sha256(f"{method}:{norm}".encode()).hexdigest()


class GovClient:
    """
    Shared client for data.gov.sg API calls with:
    - host allowlist (SSRF hardening);
    - tier-based minimum intervals: Datastore search vs file download (see ``gov_limits``);
    - optional on-disk JSON response cache;
    - 429/503 retry with bounded backoff;
    - metrics (in-process) suitable for a sidecar /scrape in simple deployments.
    """

    def __init__(
        self,
        *,
        min_interval_sec: float | None = None,
        min_interval_datastore: float | None = None,
        min_interval_file: float | None = None,
        cache_dir: Path | None = None,
        cache_enabled: bool | None = None,
        cache_ttl_sec: int | None = None,
        session: requests.Session | None = None,
        metrics: HttpMetrics | None = None,
    ) -> None:
        if min_interval_sec is not None:
            ds_ival = fi_ival = max(0.0, float(min_interval_sec))
        else:
            ds_ival = (
                min_interval_datastore
                if min_interval_datastore is not None
                else min_interval_datastore_sec()
            )
            fi_ival = (
                min_interval_file if min_interval_file is not None else min_interval_file_sec()
            )
        self._pace_ds = _TokenBucketPace(ds_ival)
        self._pace_file = _TokenBucketPace(fi_ival)
        self._session = session or requests.Session()
        self._metrics = metrics or get_metrics()
        if cache_enabled is None:
            en = env_flag("SINGAPORE_EDA_HTTP_CACHE_ENABLED", True)
        else:
            en = cache_enabled
        if cache_ttl_sec is not None:
            ttl = cache_ttl_sec
        else:
            ttl = _env_int("SINGAPORE_EDA_HTTP_CACHE_TTL_SEC", _DEFAULT_CACHE_TTL)
        cdir = cache_dir
        if cdir is None:
            cdir = Path(
                os.environ.get(
                    "SINGAPORE_EDA_HTTP_CACHE_DIR",
                    str(Path("data/processed") / ".http_cache"),
                )
            )
        self._cache: _DiskCache | None
        if en:
            self._cache = _DiskCache(cdir, ttl)
        else:
            self._cache = None

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 120,
        use_cache: bool = True,
        use_file_pace: bool = False,
    ) -> dict[str, Any]:
        if not is_allowed_gov_url(url):
            raise ValueError("URL host is not in the allowed government open-data list")
        cache_key = _make_request_cache_key("GET", url, params) if self._cache else ""
        if self._cache and use_cache and cache_key:
            hit = self._cache.get_json(cache_key)
            if hit is not None:
                self._metrics.record_cache_hit()
                return hit
            self._metrics.record_cache_miss()

        (self._pace_file if use_file_pace else self._pace_ds).wait_turn()
        hdrs = dict(headers) if headers else {}
        r = _http_get_with_retry(
            self._session, url, params=params, headers=hdrs, timeout=timeout, metrics=self._metrics
        )
        with self._metrics._lock:  # noqa: SLF001
            self._metrics.last_request_monotonic = time.monotonic()
        try:
            data = r.json()
        except (json.JSONDecodeError, TypeError) as e:
            self._metrics.record_other_error(f"json decode: {e!s}")
            raise
        if self._cache and use_cache and r.status_code == 200 and cache_key:
            self._cache.set_json(cache_key, data)
        return data

    def cache_info(self) -> dict[str, Any]:
        base: dict[str, Any] = {"pace": pace_config_public()}
        if self._cache is None:
            return {**base, "enabled": False, "path": None, "approx_bytes": 0}
        return {
            **base,
            "enabled": True,
            "path": str(self._cache._root),
            "approx_bytes": self._cache.approx_size_bytes(),
        }

    def get_bytes(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> bytes:
        if not is_allowed_gov_url(url):
            raise ValueError("URL host is not in the allowed government open-data list")
        self._pace_file.wait_turn()
        r = _http_get_with_retry(
            self._session, url, params=None, headers=headers, timeout=timeout, metrics=self._metrics
        )
        with self._metrics._lock:  # noqa: SLF001
            self._metrics.last_request_monotonic = time.monotonic()
        return bytes(r.content)


def _retry_after_wait_sec(last: requests.Response, *, back_max: float, attempt: int) -> float:
    ra = (last.headers.get("Retry-After") or "").strip()
    if ra:
        if ra.isdigit():
            return min(back_max, float(ra))
        try:
            dt = parsedate_to_datetime(ra)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                w = (dt - datetime.now(UTC)).total_seconds()
                return min(back_max, max(0.0, w))
        except (TypeError, ValueError, OSError):
            pass
    return min(back_max, 2.0 ** (attempt + 1))


def _http_get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: int,
    metrics: HttpMetrics,
    max_attempts: int = 10,
) -> requests.Response:
    back_max = 90.0
    last: requests.Response | None = None
    for attempt in range(max_attempts):
        metrics.record_request_start()
        try:
            last = session.get(
                url, params=params, headers=headers, timeout=timeout, allow_redirects=True
            )
        except requests.RequestException as e:
            metrics.record_other_error(redact_secrets(repr(e)))
            raise
        n = int(last.headers.get("content-length", 0) or 0) or len(last.content)
        metrics.record_response(last.status_code, n)
        final = last.url
        if not is_allowed_gov_url(final):
            err = f"final URL not allowed: {redact_secrets(str(final)[:200])}"
            metrics.record_other_error(err)
            raise ValueError("Redirect produced a disallowed host (possible misconfiguration)")

        if last.status_code in (429, 503) and attempt < max_attempts - 1:
            wait = _retry_after_wait_sec(last, back_max=back_max, attempt=attempt)
            logger.warning(
                "http_retry status=%s attempt=%s wait_s=%.1f url=%s",
                last.status_code,
                attempt,
                wait,
                redact_secrets(url),
            )
            time.sleep(wait)
            continue
        last.raise_for_status()
        return last
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("request failed with no response")


_gov: GovClient | None = None
_gov_lock = threading.RLock()


def get_gov_client() -> GovClient:
    global _gov
    with _gov_lock:
        if _gov is None:
            _gov = GovClient()
        return _gov


def clear_http_cache() -> int:
    c = get_gov_client()
    if c._cache is None:  # noqa: SLF001
        return 0
    return c._cache.clear()  # noqa: SLF001


def verify_admin_token(provided: str, expected: str) -> bool:
    """Constant-time compare for admin UI."""
    if not expected or not provided:
        return False
    try:
        a, b = provided.encode("utf-8"), expected.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(a) > 10_000 or len(a) != len(b):
        return False
    return secrets.compare_digest(a, b)
