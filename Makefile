PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install test ruff app report web clean-data rent-data health-serve

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

app:
	$(PYTHON) -m streamlit run streamlit_app.py

# Requires Quarto CLI: https://quarto.org/docs/get-started/
report:
	cd quarto && quarto render

analysis:
	$(PYTHON) scripts/run_analysis.py

fetch-geo:
	$(PYTHON) scripts/fetch_reference_geo.py

# Optional: pip install -e ".[api]" — liveness/readiness + /metrics for operations
health-serve:
	$(PYTHON) -m singapore_eda.health_server
