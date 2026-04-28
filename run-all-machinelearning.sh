#!/usr/bin/env bash
# Mac / Linux: one-shot ML setup, train forecaster, verify, and optionally launch Streamlit.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

if [[ "${RUN_ALL_SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -U pip
  pip install -e ".[dev]"
else
  echo "=== RUN_ALL_SKIP_INSTALL=1, skipping package install ==="
fi

mkdir -p data/raw models/forecaster_v1
if [[ ! -f data/raw/median_rent_hdb.csv ]]; then
  echo ""
  echo "=== Downloading official median rent (needs network) ==="
  hdb-rent-download -o data/raw/median_rent_hdb.csv || true
fi
if [[ ! -f data/raw/hdb_resale_2017_onwards.csv ]]; then
  echo ""
  if [[ -n "${HDB_MAX_ROWS:-}" ]]; then
    echo "=== Downloading resale CSV (HDB_MAX_ROWS=$HDB_MAX_ROWS; needs network) ==="
    hdb-download -o data/raw/hdb_resale_2017_onwards.csv --max-rows "$HDB_MAX_ROWS" --latest-first || true
  else
    echo "=== Downloading full Jan-2017+ resale tranche (set HDB_MAX_ROWS to cap) ==="
    hdb-download -o data/raw/hdb_resale_2017_onwards.csv || true
  fi
fi

echo ""
echo "=== Downloading BTO historical + future supply datasets (needs network) ==="
hdb-bto-download -o data/reference || true

if [[ ! -f data/raw/hdb_resale_2017_onwards.csv ]]; then
  echo ""
  echo "=== No download; using fixture data for model build ==="
  INPUT_CSV="tests/fixtures/hdb_sample.csv"
else
  INPUT_CSV="data/raw/hdb_resale_2017_onwards.csv"
fi

echo ""
echo "=== Lint ==="
python -m ruff check src tests streamlit_app.py scripts

echo ""
echo "=== Tests ==="
python -m pytest tests -q

echo ""
echo "=== Build Forecaster V1 Artifacts ==="
python scripts/build_forecaster_v1.py --input "$INPUT_CSV" --config configs/forecaster_v1.yaml

if [[ "${ML_RUN_DEEPCHECKS:-0}" == "1" ]]; then
  echo ""
  echo "=== Deepchecks Validation ==="
  python scripts/run_deepchecks_forecaster.py
fi

echo ""
echo "=== Inference Smoke Check ==="
python - <<'PY'
from singapore_eda.forecaster_v1 import predict_with_explain

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
out = predict_with_explain(payload)
print(f"prediction=${out['prediction']:,.0f}")
print(
    "interval="
    f"${out['prediction_interval']['p10']:,.0f}.."
    f"${out['prediction_interval']['p90']:,.0f}"
)
print(f"contributors={len(out['top_contributors'])}")
print(f"warnings={len(out['warnings'])}")
PY

echo ""
echo "=== Near-term Evaluation Gate ==="
if [[ "${ML_ENFORCE_NEAR_TERM_GATE:-0}" == "1" ]]; then
  python scripts/run_forecaster_near_term_eval.py --input "$INPUT_CSV" --enforce-gate
else
  python scripts/run_forecaster_near_term_eval.py --input "$INPUT_CSV"
fi

echo ""
echo "=== Feedback Governance Materialization ==="
python scripts/prepare_feedback_dataset.py

echo ""
echo "=== Vulnerability Audit (requirements scope) ==="
python -m pip_audit -r requirements.txt || true

if [[ "${ML_OPEN_STREAMLIT:-1}" == "1" ]]; then
  echo ""
  echo "=== Streamlit (Ctrl+C to stop) ==="
  streamlit run streamlit_app.py
else
  echo ""
  echo "=== Done (ML_OPEN_STREAMLIT=0, skipped launching Streamlit) ==="
fi
