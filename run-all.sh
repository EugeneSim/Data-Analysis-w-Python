#!/usr/bin/env bash
# Mac / Linux: one-shot setup, tests, analysis, and Streamlit.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# Speed up repeat runs: skip dependency install unless missing or forced.
if [[ "${RUN_ALL_SKIP_INSTALL:-0}" != "1" ]]; then
  if ! python -c "import singapore_eda" >/dev/null 2>&1; then
    python -m pip install -U pip
    pip install -e ".[dev]"
  fi
else
  echo "=== RUN_ALL_SKIP_INSTALL=1, skipping package install ==="
fi

mkdir -p data/raw
if [[ ! -f data/raw/median_rent_hdb.csv ]]; then
  echo ""
  echo "=== Downloading official median rent (quarterly, full history; needs network) ==="
  hdb-rent-download -o data/raw/median_rent_hdb.csv || true
fi
if [[ ! -f data/raw/hdb_resale_2017_onwards.csv ]]; then
  echo ""
  if [[ -n "${HDB_MAX_ROWS:-}" ]]; then
    echo "=== Downloading resale CSV (HDB_MAX_ROWS=$HDB_MAX_ROWS; needs network) ==="
    hdb-download -o data/raw/hdb_resale_2017_onwards.csv --max-rows "$HDB_MAX_ROWS" --latest-first || true
  else
    echo "=== Downloading full Jan-2017+ resale tranche (comprehensive; can take a while; set HDB_MAX_ROWS to cap) ==="
    hdb-download -o data/raw/hdb_resale_2017_onwards.csv || true
  fi
fi
if [[ ! -f data/raw/hdb_resale_2017_onwards.csv ]]; then
  echo ""
  echo "=== No download; using tests/fixtures/hdb_sample.csv for analysis ==="
  ANL=(python scripts/run_analysis.py --input tests/fixtures/hdb_sample.csv)
else
  ANL=(python scripts/run_analysis.py)
fi

echo ""
echo "=== Tests ==="
python -m pytest tests -q

echo ""
echo "=== Analysis (console summary) ==="
"${ANL[@]}"

echo ""
echo "=== Streamlit (Ctrl+C to stop; opens in your browser) ==="
streamlit run streamlit_app.py
