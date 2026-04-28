#!/usr/bin/env python3
"""Quick reproducibility check for forecaster v1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from singapore_eda.forecaster_v1 import predict_with_explain


def main() -> None:
    model_path = Path("models/forecaster_v1/model.joblib")
    metadata_path = Path("models/forecaster_v1/metadata.json")
    payload = {
        "month": "2026-04-01",
        "town": "ANG MO KIO",
        "flat_type": "4 ROOM",
        "flat_model": "Model A",
        "storey_range": "04 TO 06",
        "floor_area_sqm": 93.0,
        "lease_commence_date": 1998,
        "remaining_lease_years": 68.0,
    }
    p1 = predict_with_explain(payload, model_path=model_path, metadata_path=metadata_path)
    p2 = predict_with_explain(payload, model_path=model_path, metadata_path=metadata_path)
    out = {
        "prediction_run_1": p1["prediction"],
        "prediction_run_2": p2["prediction"],
        "equal_prediction": float(p1["prediction"]) == float(p2["prediction"]),
        "model_version": p1.get("model_version"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
