"""Feedback logging for forecaster iteration loop."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FEEDBACK_COLUMNS: tuple[str, ...] = (
    "feedback_id",
    "feedback_at_utc",
    "model_version",
    "model_family",
    "predicted_price",
    "actual_price",
    "user_rating",
    "user_comment",
    "input_payload_json",
)
MAX_COMMENT_LENGTH = 500


def _parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _redact_comment(comment: str, *, min_digits: int = 6) -> str:
    txt = str(comment).strip()
    txt = txt[:MAX_COMMENT_LENGTH]
    pattern = rf"\b\d{{{max(4, int(min_digits))},}}\b"
    return re.sub(pattern, "[REDACTED_NUMERIC]", txt)


def _feedback_id(row: dict[str, Any]) -> str:
    key = "|".join(
        str(row.get(k, ""))
        for k in (
            "feedback_at_utc",
            "model_version",
            "model_family",
            "predicted_price",
            "input_payload_json",
        )
    )
    return sha256(key.encode("utf-8")).hexdigest()[:16]


def validate_feedback_row(row: dict[str, Any]) -> dict[str, Any]:
    if not (1 <= int(row["user_rating"]) <= 5):
        raise ValueError("user_rating must be between 1 and 5.")
    if float(row["predicted_price"]) <= 0:
        raise ValueError("predicted_price must be positive.")
    if row.get("actual_price") is not None and float(row["actual_price"]) <= 0:
        raise ValueError("actual_price must be positive when provided.")
    _parse_utc(str(row["feedback_at_utc"]))
    # Ensure payload is valid JSON object.
    payload = json.loads(str(row["input_payload_json"]))
    if not isinstance(payload, dict):
        raise ValueError("input_payload_json must encode an object.")
    out = dict(row)
    out["user_comment"] = _redact_comment(str(row.get("user_comment", "")))
    out["predicted_price"] = float(row["predicted_price"])
    out["actual_price"] = (
        float(row["actual_price"]) if row.get("actual_price") is not None else None
    )
    out["user_rating"] = int(row["user_rating"])
    return out


def append_feedback(
    *,
    store_path: Path | str,
    model_version: str,
    model_family: str,
    predicted_price: float,
    user_rating: int,
    user_comment: str,
    input_payload: dict[str, Any],
    actual_price: float | None = None,
) -> Path:
    p = Path(store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "feedback_at_utc": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "model_family": model_family,
        "predicted_price": float(predicted_price),
        "actual_price": float(actual_price) if actual_price is not None else None,
        "user_rating": int(user_rating),
        "user_comment": str(user_comment).strip(),
        "input_payload_json": json.dumps(input_payload, sort_keys=True),
    }
    row["feedback_id"] = _feedback_id(row)
    row = validate_feedback_row(row)
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out = out.drop_duplicates(subset=["feedback_id"], keep="last")
    out = out[list(REQUIRED_FEEDBACK_COLUMNS)]
    out.to_csv(p, index=False)
    return p


def materialize_feedback_views(
    *,
    raw_path: Path | str,
    validated_path: Path | str,
    retraining_path: Path | str,
    retention_days: int = 365,
    min_comment_redact_digits: int = 6,
) -> dict[str, int]:
    raw = Path(raw_path)
    validated = Path(validated_path)
    retraining = Path(retraining_path)
    validated.parent.mkdir(parents=True, exist_ok=True)
    retraining.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        empty = pd.DataFrame(columns=REQUIRED_FEEDBACK_COLUMNS + ("quality_status",))
        empty.to_csv(validated, index=False)
        empty.to_csv(retraining, index=False)
        return {"raw_rows": 0, "validated_rows": 0, "retraining_rows": 0}

    src = pd.read_csv(raw)
    rows: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC).timestamp() - max(1, int(retention_days)) * 86400
    for _, rec in src.iterrows():
        row = {k: rec.get(k) for k in src.columns}
        try:
            row.setdefault("feedback_id", _feedback_id(row))
            row["feedback_at_utc"] = str(row.get("feedback_at_utc", ""))
            row["user_comment"] = _redact_comment(
                str(row.get("user_comment", "")),
                min_digits=min_comment_redact_digits,
            )
            row["input_payload_json"] = str(row.get("input_payload_json", "{}"))
            row["model_version"] = str(row.get("model_version", ""))
            row["model_family"] = str(row.get("model_family", ""))
            row["predicted_price"] = float(row.get("predicted_price"))
            row["actual_price"] = (
                float(row["actual_price"]) if pd.notna(row.get("actual_price")) else None
            )
            row["user_rating"] = int(row.get("user_rating"))
            parsed = validate_feedback_row(row)
            if _parse_utc(parsed["feedback_at_utc"]).timestamp() < cutoff:
                continue
            parsed["quality_status"] = "validated"
            rows.append(parsed)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    validated_df = pd.DataFrame(rows)
    if validated_df.empty:
        validated_df = pd.DataFrame(columns=list(REQUIRED_FEEDBACK_COLUMNS) + ["quality_status"])
    else:
        validated_df = validated_df.drop_duplicates(subset=["feedback_id"], keep="last")
        validated_df = validated_df.sort_values("feedback_at_utc").reset_index(drop=True)
        validated_df = validated_df[list(REQUIRED_FEEDBACK_COLUMNS) + ["quality_status"]]
    validated_df.to_csv(validated, index=False)

    retraining_df = validated_df.loc[
        validated_df["actual_price"].notna() & (validated_df["actual_price"].astype(float) > 0)
    ].copy()
    if not retraining_df.empty:
        retraining_df["absolute_error"] = (
            retraining_df["actual_price"].astype(float)
            - retraining_df["predicted_price"].astype(float)
        ).abs()
    retraining_df.to_csv(retraining, index=False)
    return {
        "raw_rows": int(len(src)),
        "validated_rows": int(len(validated_df)),
        "retraining_rows": int(len(retraining_df)),
    }
