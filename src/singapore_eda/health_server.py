"""Optional ASGI app for liveness, readiness, and Prometheus text metrics (pip install [api]).

Run: uvicorn singapore_eda.health_server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from typing import Any

from singapore_eda.gov_http import get_metrics
from singapore_eda.health import run_health

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError:  # pragma: no cover - optional extra
    FastAPI = None  # type: ignore[assignment, misc]
    app = None  # type: ignore[assignment, misc]
else:
    app = FastAPI(
        title="singapore-eda-ops",
        description="Liveness, readiness, and in-process HTTP metrics for open-data fetches",
        version="0.1.0",
    )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        """Liveness: process up."""
        return {"status": "live"}

    @app.get("/ready", tags=["ops"])
    def ready() -> JSONResponse:
        """Readiness: deps + optional CKAN smoke check."""
        h = run_health()
        code = 200 if h.status in ("ok", "degraded") else 503
        return JSONResponse(h.to_dict(), status_code=code)

    @app.get("/ops", tags=["ops"])
    def ops() -> JSONResponse:
        h = run_health()
        m = get_metrics().as_dict()
        return JSONResponse({"health": h.to_dict(), "http_client": m})

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        body = get_metrics().prometheus_text()
        return PlainTextResponse(
            content=body, media_type="text/plain; version=0.0.4; charset=utf-8"
        )


def main() -> None:
    import sys

    if FastAPI is None:
        print(
            "Install the api extra: pip install 'singapore-eda[api]' (or fastapi+uvicorn).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    import uvicorn

    uvicorn.run("singapore_eda.health_server:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
