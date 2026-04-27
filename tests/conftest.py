"""Test defaults: avoid flaky outbound health checks unless a test clears the flag."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _default_skip_outbound_health_probe() -> None:
    os.environ.setdefault("SINGAPORE_EDA_HEALTH_SKIP_HTTP", "1")
