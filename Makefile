PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test ruff app report web clean-data rent-data bto-data health-serve forecaster-v1 forecaster-v1-verify forecaster-v1-deepchecks forecaster-v1-near-term-eval forecaster-feedback-prepare fetch-mrt-reference fetch-mrt-travel-times

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests -q

ruff:
	$(PYTHON) -m ruff check src tests streamlit_app.py scripts

# Download full HDB dataset (slow: paginated API with throttling; optional max rows for dev)
clean-data:
	$(PYTHON) -m singapore_eda.download_data --output data/raw/hdb_resale_2017_onwards.csv

# HDB median rent by town/flat (data.gov.sg); needed for yield tab
rent-data:
	$(PYTHON) -m singapore_eda.rent_ingest -o data/raw/median_rent_hdb.csv

# HDB BTO historical + future-facing datasets (launch pricing, completion status, property base)
bto-data:
	$(PYTHON) -m singapore_eda.bto_ingest -o data/reference

app:
	$(PYTHON) -m streamlit run streamlit_app.py

# Requires Quarto CLI: https://quarto.org/docs/get-started/
report:
	cd quarto && quarto render

analysis:
	$(PYTHON) scripts/run_analysis.py

fetch-geo:
	$(PYTHON) scripts/fetch_reference_geo.py

fetch-mrt-reference:
	$(PYTHON) scripts/fetch_mrt_reference.py

fetch-mrt-travel-times:
	$(PYTHON) scripts/fetch_mrt_travel_times.py

# Optional: pip install -e ".[api]" — liveness/readiness + /metrics for operations
health-serve:
	$(PYTHON) -m singapore_eda.health_server

forecaster-v1:
	$(PYTHON) scripts/build_forecaster_v1.py --input data/raw/hdb_resale_2017_onwards.csv

forecaster-v1-verify:
	$(PYTHON) scripts/verify_forecaster_repro.py

forecaster-v1-deepchecks:
	$(PYTHON) scripts/run_deepchecks_forecaster.py

forecaster-v1-near-term-eval:
	$(PYTHON) scripts/run_forecaster_near_term_eval.py --input data/raw/hdb_resale_2017_onwards.csv

forecaster-feedback-prepare:
	$(PYTHON) scripts/prepare_feedback_dataset.py
