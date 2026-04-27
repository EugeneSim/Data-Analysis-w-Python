"""Liveness / readiness style checks for batch jobs, Streamlit, or a small health HTTP server."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from singapore_eda.constants import HDB_DATASTORE_SEARCH
from singapore_eda.gov_http import get_gov_client, is_allowed_gov_url

_DEFAULT_WRITE = Path("data/processed") / ".healthcheck_write"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class HealthReport:
    status: str  # "ok" | "degraded" | "fail"
    python: str
    platform: str
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "python": self.python,
            "platform": self.platform,
            "checks": [c.__dict__ for c in self.checks],
        }


def _check_imports() -> CheckResult:
    need = (
        "pandas",
        "numpy",
        "requests",
    )
    missing: list[str] = []
    for m in need:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    if missing:
        return CheckResult("imports", False, f"missing: {', '.join(missing)}")
    return CheckResult("imports", True, "core deps importable")


def _check_paths_writable() -> CheckResult:
    try:
        p = Path(os.environ.get("SINGAPORE_EDA_HEALTH_WRITE_PATH", str(_DEFAULT_WRITE)))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok", encoding="utf-8")
        p.unlink(missing_ok=True)
        return CheckResult("disk_write", True, str(p.parent))
    except OSError as e:
        return CheckResult("disk_write", False, str(e)[:200])


def _check_gov_reachable() -> CheckResult:
    if os.environ.get("SINGAPORE_EDA_HEALTH_SKIP_HTTP", "").strip() == "1":
        return CheckResult("data_gov_reachable", True, "skipped (SINGAPORE_EDA_HEALTH_SKIP_HTTP=1)")
    if not is_allowed_gov_url(HDB_DATASTORE_SEARCH):
        return CheckResult("data_gov_reachable", False, "HDB URL allowlist check failed (bug?)")
    try:
        c = get_gov_client()
        c.get_json(
            HDB_DATASTORE_SEARCH,
            params={"resource_id": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc", "limit": 1, "offset": 0},
            timeout=30,
            use_cache=True,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult("data_gov_reachable", False, str(e)[:200])
    return CheckResult("data_gov_reachable", True, "ckan small probe ok")


def run_health() -> HealthReport:
    rep = HealthReport(
        status="ok",
        python=sys.version.split()[0],
        platform=sys.platform,
    )
    rep.checks.append(_check_imports())
    rep.checks.append(_check_paths_writable())
    rep.checks.append(_check_gov_reachable())
    for c in rep.checks:
        if c.name in ("imports", "disk_write") and not c.ok:
            rep.status = "fail"
            return rep
    for c in rep.checks:
        if c.name == "data_gov_reachable" and not c.ok and "skipped" not in c.detail:
            rep.status = "degraded"
            return rep
    rep.status = "ok"
    return rep
