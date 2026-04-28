# Data-Analysis-w-Python

## Problem and what this project solves

**Singapore’s HDB public resale market** is easy to *describe* in headlines but hard to *explore* rigorously. Official data arrives as **large, evolving CSVs** and **CKAN API pages**, with **rate limits** and multiple **reference dimensions** (town, planning area, maturity, MRT access, optional policy overlays). Ad-hoc notebooks and one-off scripts tend to **duplicate** download logic, break when tranches change, and make it **unclear** how numbers were produced or whether API use was **safe and reproducible**.

This project’s goal is to **close that gap** for **exploratory analysis and teaching**:

- **Reproducibility:** A small library (`singapore_eda`) turns raw resale (and optional rent) data into a **documented path**: ingest → clean → feature engineering → joins to **in-repo and optional geo reference** data → tables and models used consistently by the batch script, tests, and app.
- **Operability:** HTTP to **data.gov.sg** is centralized with **tier-aware pacing**, retries, and optional caching—so you can refresh data without hand-tuning sleep timers or tripping limits.
- **Usability for insight:** A **Streamlit** app bundles **maps, clustering, rental gross yield, simple forecasts, and a town–correlation graph** so you can *interact* with the same pipeline rather than re-plumbing plots each time. A **Quarto** site supports a **static, shareable** story (e.g. on GitHub Pages). **Jupyter** materials under `course_materials/` support **foundational** Python data skills in parallel with the “real” pipeline in `src/`.

**In short:** the problem is *fragmented, fragile analysis of a rich public dataset*; the solution is *one coherent stack*—governed ingestion, a maintainable EDA/insights library, and multiple surfaces (CLI, app, report, course notebooks) on top of it.

---

End-to-end exploratory analysis of **HDB public resale transaction data** (Singapore): modular pipeline in `singapore_eda`, **Streamlit** (maps, clusters, rental gross yield, forecasts, town correlation graph), and a **Quarto** static report for **GitHub Pages**.

Course-style Jupyter notebooks (NumPy, Pandas, cleaning) live under [`course_materials/`](course_materials/). New work is code-first in [`src/singapore_eda/`](src/singapore_eda/).

> **Heads-up:** This repository is **experimental** and **educational**. Outputs are **not** validated for real-world property, financial, or policy decisions. Read [Scope, limitations, data accuracy, and disclaimers](#scope-limitations-data-accuracy-and-disclaimers) before relying on any number, map, or forecast.

## Quick start (one command)

From the repository root, the scripts create `.venv`, install the package, optionally pull a small resale sample from data.gov.sg, run **tests** and the **batch analysis** script, then start **Streamlit** (stays running until you press Ctrl+C).

| OS | Command |
|----|--------|
| **Windows** | Double-click or run: `run-all-machinelearning.cmd` |
| **macOS / Linux** | `chmod +x run-all-machinelearning.sh` once, then `./run-all-machinelearning.sh` |

- For faster repeat runs, set `RUN_ALL_SKIP_INSTALL=1` to skip dependency bootstrap when your `.venv` is already ready.
- In the Streamlit UI, the **Project & method** tab states the **problem, data, storage, models, and how users interact** (dashboard / deployment). For a full **planning-area map** (not the two-polygon test fixture), run `python scripts/fetch_reference_geo.py` and set the map path to `data/reference/planning_areas.geojson`.
- First run with network: downloads up to **20,000** rows into `data/raw/hdb_resale_2017_onwards.csv` if that file is missing. Offline or blocked download: analysis uses `tests/fixtures/hdb_sample.csv`; Streamlit already falls back to the same fixture when the default raw path is missing.
- **Quarto** static site is not in these scripts. Install [Quarto](https://quarto.org/docs/get-started/) and run `make report` (output under `quarto/_site`).

## Manual install (if you prefer not to use the scripts)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Then: `make test` · `hdb-download -o data/raw/hdb_resale_2017_onwards.csv` (or `--max-rows 20000` for a faster sample) · `hdb-bto-download -o data/reference` · `make analysis` or `python scripts/run_analysis.py --refresh-bto` · `streamlit run streamlit_app.py`.

## HDB Forecaster V1 (Postgres + Streamlit)

This repository now includes a dedicated V1 regression forecaster path with:
- **Leakage-safe temporal split** (train/validation/test ordered by transaction month),
- **LightGBM + XGBoost candidate training** with automatic winner selection by validation holdout performance,
- **SHAP local explainability** for top prediction drivers,
- **Prediction interval** from split-conformal absolute-residual calibration (default nominal coverage 0.80),
- **Optional Postgres curated table write** for a production-style feature store.
- **Empirical lease time-decay feature** fitted from historical data (captures
  age/lease effects without forcing literal zero-value end state).
- **MLflow experiment tracking** (params, metrics, artifacts) via config.
- **Deepchecks data integrity validation** hook via config.
- **Feedback loop capture** from Streamlit predictions to local feedback store.
- **Optional location/accessibility features** when available in enriched data
  (`planning_area`, `region_ocr`, `mrt_station_count`, `nearest_mrt_km_proxy`).
- **Leakage-safe lag market features** (`lag_town_median_price`, segment lag medians,
  and prior transaction-count liquidity proxies).

### Build artifacts

Run either:

```bash
python scripts/build_forecaster_v1.py --input data/raw/hdb_resale_2017_onwards.csv
```

or:

```bash
make forecaster-v1
```

Artifacts are written to `models/forecaster_v1/`:
- `model.joblib` (model + feature columns + guardrail bounds),
- `metadata.json` (metrics, split windows, provenance, feature contract).
- `train_config.json` (full reproducible training config snapshot).
- `model_card.md` (run-level model summary, benchmark, and intended-use notes).

### Configuration management

All forecaster hyperparameters and tracking toggles are version-controlled in:

- `configs/forecaster_v1.yaml`

This includes:
- train split and seed,
- candidate model list and model-specific hyperparameters,
- MLflow enablement and experiment name,
- Deepchecks output settings,
- feedback storage path and governed feedback materialization paths.
- MLflow tracking backend now defaults to local SQLite (`sqlite:///mlflow.db`) to avoid deprecated filesystem tracking store warnings.

### One-command ML end-to-end runners

- **macOS / Linux:** `./run-all-machinelearning.sh`
- **Windows:** `run-all-machinelearning.cmd`
- Refresh official MRT references: `make fetch-mrt-reference`
- Fetch official MRT-to-CBD travel times (OneMap auth required): `make fetch-mrt-travel-times`
  - Local dev: put `ONEMAP_API_EMAIL` and `ONEMAP_API_PASSWORD` in repo-root `.env` (gitignored),
    or export them in your shell.

What these scripts do:
- prepare `.venv` and install dependencies,
- fetch resale/rent data when missing (with fixture fallback),
- run lint + tests,
- build forecaster artifacts,
- run inference smoke-check,
- run vulnerability audit (`pip-audit` on `requirements.txt`),
- launch Streamlit unless `ML_OPEN_STREAMLIT=0`.
- Optional hard gate: set `ML_ENFORCE_NEAR_TERM_GATE=1` to fail the run when near-term promotion gate does not pass.

### Optional Postgres load

To write curated training rows to Postgres before model training:

```bash
python scripts/build_forecaster_v1.py \
  --input data/raw/hdb_resale_2017_onwards.csv \
  --postgres-dsn "postgresql://user:password@localhost:5432/hdb"
```

### Streamlit usage

After artifacts exist, start the app and use the **Forecaster V1** tab:

```bash
streamlit run streamlit_app.py
```

The tab returns:
- point estimate,
- p10-p90 uncertainty interval,
- top SHAP contributors,
- range warnings when inputs fall outside training bounds.
- constrained dropdowns for `storey_range` (no free-text entry),
- correlated input guardrails for `flat_type` + `flat_model` + floor-area range,
- floor-area input in either `sqm` or `sqft` (auto-converted to model input),
- lease guardrails (commence year cannot exceed transaction year; optional auto 99-year remaining lease),
- `Resale` / `BTO` profile mode with BTO naming + lease prefill when BTO reference rows are available,
- graph-analytics context (town trend + percentile position),
- optional market-anchor blend toward recent comparable medians for stability.
- lease time-decay coefficient display for transparency of leasehold adjustment.
- nearest comparable-transactions panel with similarity distance and median/IQR context.
- reliability score driven by comparables density, interval width, OOD warnings, and segment RMSE.
- segment error slices from `metadata.json` (town and flat-type) for transparency.
- human-readable segment benchmark tables with plain-language guidance (`lower error is better`, `higher sample size is better`).
- location/connectivity premium context (planning area, region tier, MRT proxy, and CBD access heuristic).
- optional scenario overlay controls for school-zone, shopping-hub, and fast-CBD-access premium assumptions.
- new **Housing economics details** tab for end-to-end ownership estimation:
  - MOP timeline and estimated sale timing checks
  - grants checklist, estimated return-to-government, and levy
  - HDB/bank loan cashflow schedule with CPF vs cash split
  - COV/premium, renovation, taxes/fees, and itemized cost ledger
  - profit waterfall and breakdown tables
  - bank package modelling for common `2-year fixed -> SORA + spread` path
  - repricing/refinancing simulation with admin/legal/valuation/clawback and lock-in penalty impacts

### Housing economics details (estimator)

The app now includes an estimator-focused page for BTO/resale/EC/private planning using policy defaults in:

- `configs/housing_finance_v1.yaml`

Core implementation lives in:

- `src/singapore_eda/housing_finance/models.py`
- `src/singapore_eda/housing_finance/policy_defaults.py`
- `src/singapore_eda/housing_finance/calculators.py`
- `src/singapore_eda/housing_finance/formatters.py`

Run and verify:

```bash
streamlit run streamlit_app.py
pytest tests/test_housing_finance.py
```

For detailed formulas, assumptions, and interpretation guidance, see:

- `docs/housing_details_v1_runbook.md`

### Verification checklist

```bash
make test
make ruff
python -m pip_audit -r requirements.txt
python scripts/run_forecaster_near_term_eval.py --input data/raw/hdb_resale_2017_onwards.csv
python scripts/prepare_feedback_dataset.py
```

`pip-audit` is required before release handoff; record critical/high findings and remediation.

### Verified end-to-end run (latest)

Validated with:

```bash
ML_OPEN_STREAMLIT=0 ML_RUN_DEEPCHECKS=1 RUN_ALL_SKIP_INSTALL=1 ./run-all-machinelearning.sh
```

Observed status:
- lint + tests passed (`48 passed`)
- forecaster artifacts built
- inference smoke-check passed
- Deepchecks step executed (environment warning may be emitted by upstream deepchecks)
- vulnerability scan passes after upgrading to a secure `mlflow` line

Security note:
- `mlflow` baseline is pinned to `>=3.11.1,<4` to clear known CVEs reported against the previous `2.22.4` line.
- Keep `python -m pip_audit -r requirements.txt` in release checks to catch newly disclosed vulnerabilities.

### Reproducibility and recovery

- Repro check: `make forecaster-v1-verify`
- Deepchecks (on demand): `make forecaster-v1-deepchecks`
- Runbook: [`docs/forecaster_v1_runbook.md`](docs/forecaster_v1_runbook.md)
- Decision log (what we chose and why): [`docs/forecaster_v1_decisions.md`](docs/forecaster_v1_decisions.md)
- Model benchmark evidence (LightGBM vs XGBoost): [`docs/forecaster_v1_model_benchmark.md`](docs/forecaster_v1_model_benchmark.md)
- Near-term rolling backtest / calibration / ablation artifacts: `reports/forecaster_v1/near_term_eval.{json,md}`
- Near-term eval now enforces a minimum rolling-window count and emits a promotion gate (`coverage_pass`, `width_pass`, `pass`) in JSON.
- MRT references sourced from official data.gov.sg APIs via `scripts/fetch_mrt_reference.py`:
  - URA MP14 MRT Station Symbol (`d_649357d5cb04ddbef9166dfcf1fa8d21`)
  - URA Master Plan 2003 MRT Name (`d_dbc192abee39f51efecc0adbe9f1a75d`)
- Official travel-time file via `scripts/fetch_mrt_travel_times.py`:
  - output: `data/reference/mrt_travel_time_to_cbd.csv`
  - credentials: set `ONEMAP_API_EMAIL` and `ONEMAP_API_PASSWORD`
  - source API: OneMap routing (`routeType=pt`, `mode=TRANSIT`)

**Geo (Graph tab, planning-area edges):** `pip install -e ".[geo]"` and run `make fetch-geo` or `scripts/fetch_reference_geo.py` to pull the official [MP2025 planning-area boundary (No Sea)](https://data.gov.sg/datasets/d_2cc750190544007400b2cfd5d7f53209/view) GeoJSON (old static `geo.data.gov.sg` MP14 links are no longer served). **Rent / gross yield:** set `HDB_MEDIAN_RENT_RESOURCE_ID` and `hdb-rent-download -o data/raw/median_rent_hdb.csv`, or use [`tests/fixtures/median_rent_sample.csv`](tests/fixtures/median_rent_sample.csv). **BTO historical + future supply references:** run `hdb-bto-download -o data/reference` (writes launch price range, completion status, property information, plus `hdb_bto_reference.csv`). Override staleness in the app with `SINGAPORE_EDA_RENT_TTL_HOURS`. Tranche merge and other CLI details: `hdb-download --help`, `hdb-resale-merge --help`.

Forecaster training/inference now consumes BTO-derived town/year context features when these files are present:
- `bto_launch_count_town_3y`
- `bto_avg_price_range_mid_town_3y`
- `bto_under_construction_units_town`
- `bto_completed_units_town`

## Data

**Primary resale table:** [Resale flat prices (registration) from Jan-2017 onwards](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) (HDB, [Open Data Licence](https://data.gov.sg/open-data-licence)).

**Reference tables (in-repo):** [`data/reference/`](data/reference/) — town → planning area / OCR, mature vs young labels, MRT access proxy, optional EIP block stub, and official MRT reference exports fetched from government APIs.

## Live demo (two URLs)

- **GitHub Pages** (static report only): enable [`.github/workflows/publish-quarto.yml`](.github/workflows/publish-quarto.yml) and set **Settings → Pages** to **GitHub Actions**. URL shape: `https://<user>.github.io/<repo>/`.
- **Streamlit Community Cloud** (interactive app): main file `streamlit_app.py`; Python dependencies are in [`requirements.txt`](requirements.txt) (`-e .`) and project metadata in [`pyproject.toml`](pyproject.toml). Set secrets in the Cloud UI (e.g. `DATA_GOV_SG_API_KEY`) as in [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example). List the Cloud app URL in this README next to the Pages URL.

### Publish checklist (GitHub Pages + Streamlit Cloud)

1. Push your repository to GitHub and keep `main` as the default branch.
2. In GitHub: **Settings → Pages → Source = GitHub Actions**.
3. Push to `main` (or run the Pages workflow manually). The workflow [`.github/workflows/publish-quarto.yml`](.github/workflows/publish-quarto.yml) builds `quarto/_site` and deploys it.
4. Confirm your static URL opens:
   - `https://<your-github-username>.github.io/<your-repo-name>/`
5. In Streamlit Community Cloud:
   - Click **New app**
   - Repository: this repo
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - (Optional) Python version: 3.12
6. Add secrets in Streamlit Cloud (**App settings → Secrets**) using [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).
   - For OneMap travel-time fetches, include:
     - `ONEMAP_API_EMAIL`
     - `ONEMAP_API_PASSWORD`
7. Deploy and copy your Streamlit URL (typically `https://<app-name>.streamlit.app`).
8. Replace the placeholders below and commit:

```md
- GitHub Pages (static): https://<your-github-username>.github.io/<your-repo-name>/
- Streamlit app (interactive): https://<your-app-name>.streamlit.app
```

9. Add the same Streamlit URL into `quarto/index.qmd` under the "Static site vs interactive app" section so users on Pages can jump to the live app.

### Streamlit deployment notes (lessons learned)

These defaults now reflect issues encountered during deployment and the fixes applied in-repo:

- **Pin Python for Cloud builds:** keep both [`runtime.txt`](runtime.txt) and [`.python-version`](.python-version) set to `3.12`. This avoids unsupported runtime combinations where dependencies (for example `pyarrow`) may fall back to source builds.
- **Dependency files:** Streamlit installs Python deps from [`requirements.txt`](requirements.txt). Keep `packages.txt` for apt/system packages only; do **not** put pip specifiers there.
- **CSV schema guardrails:** the app expects HDB-like resale columns (at least `month`, `resale_price`). If you point the sidebar to a non-HDB CSV, the app now warns clearly and avoids silent bad fallbacks.
- **Fallback order:** when the default raw resale file is missing, the app tries auto-download (if enabled), then safe built-in alternatives, and uses tiny fixtures only as a last resort.
- **Map and rent auto-bootstrap:** optional Cloud toggles can auto-fetch full planning-area GeoJSON and official median-rent CSV so **Map & blocks** and **Rental yield** do not stay on minimal fixtures.

Recommended Streamlit Cloud secrets (copy from [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)):

```toml
SINGAPORE_EDA_AUTO_DOWNLOAD_ON_MISSING = "1"
SINGAPORE_EDA_BOOTSTRAP_MAX_ROWS = "20000"
SINGAPORE_EDA_AUTO_DOWNLOAD_RENT_ON_MISSING = "1"
SINGAPORE_EDA_RENT_BOOTSTRAP_MAX_ROWS = "200000"
SINGAPORE_EDA_AUTO_FETCH_GEO_ON_TINY = "1"
```

Optional OneMap secrets for official travel-time refresh jobs:

```toml
ONEMAP_API_EMAIL = "your_onemap_email"
ONEMAP_API_PASSWORD = "your_onemap_password"
```

Local-only development alternative:
- put the same values into repo-root `.env` (already gitignored),
- run `make fetch-mrt-travel-times`.

## Production: data.gov.sg limits, cache, health API, ops UI

Official **per-10-second** limits (see [`API rate limits`](https://guide.data.gov.sg/developer-guide/api-overview/api-rate-limits)) depend on whether you use an API key and its **tier**:

| API (per 10s) | No key | Dev key | Prod key |
|----------------|--------|---------|----------|
| **Datastore search** (CKAN `datastore_search`) | 4 | 8 | 20 |
| **Dataset / file downloads** (e.g. poll-download, large files) | 2 | 4 | 10 |

This project maps those rows to **minimum spacing** between calls in `singapore_eda.gov_limits` (with `SINGAPORE_EDA_RATE_HEADROOM`, default 12% extra). Set **`SINGAPORE_EDA_API_TIER=none|dev|prod`**; if unset, **no key → `none`**, **key set → `dev`** (use `prod` when your key is production). Override with **`SINGAPORE_EDA_MIN_INTERVAL_DATASTORE_SEC`** / **`SINGAPORE_EDA_MIN_INTERVAL_FILE_SEC`**, or **`SINGAPORE_EDA_MIN_INTERVAL_SEC`** for one value applied to **both** paces.

All **\*.data.gov.sg** HTTP calls go through **`singapore_eda.gov_http`**: **HTTPS allowlist**, **separate pacing** for CKAN JSON vs file-style GETs, **429/503 retries**, optional **on-disk JSON cache**, in-process **metrics**.

| Mechanism | Notes |
|-----------|--------|
| **Call only when needed** | **Skip fresh files:** `hdb-download --skip-if-fresh-hours H` or `SINGAPORE_EDA_SKIP_DOWNLOAD_IF_FRESH_HOURS` (no API calls if the CSV is newer than H hours). **`hdb-rent-download`** supports the same flag. Planning GeoJSON: `scripts/fetch_reference_geo.py --skip-if-fresh-days D` or `SINGAPORE_EDA_GEO_SKIP_IF_FRESH_DAYS`. |
| **Fewer paginated calls** | `PAGE_SIZE` is 10,000 so each **datastore_search** returns more rows per request (within CKAN limits). |
| **Per-page cache** | `SINGAPORE_EDA_DATASTORE_PAGE_CACHE=1` caches each CKAN page (dev/replay; can use large disk on full imports). |
| **Response cache** | `SINGAPORE_EDA_HTTP_CACHE_*` — TTL’d JSON cache for identical GETs. |
| **Health** | `SINGAPORE_EDA_HEALTH_SKIP_HTTP=1` in CI skips the online probe. |
| **Metrics HTTP (optional)** | `pip install -e ".[api]"` then `singapore-eda-health` or `make health-serve` — `/health`, `/ready`, `/ops`, `/metrics`. **Per process**; scrape all replicas in Kubernetes. |
| **Admin (Streamlit)** | **`pages/2_Ops_and_Admin.py`** — set `SINGAPORE_EDA_ADMIN_TOKEN` for cache clear. **Never** commit real tokens. |

## Package layout (high level)

| Module | Role |
|--------|------|
| `ingest`, `clean`, `features`, `stats`, `viz` | Core EDA and OLS |
| `geo_join` | Planning area, maturity, MRT proxy |
| `storey` | Storey bands / stratified medians |
| `numerology` | Descriptive digit flags (block / storey) |
| `eip` | Optional join to `eip_block_stub.csv` |
| `sun_exposure` | Placeholder only unless `ENABLE_SUN_PROXY=1` |
| `rental_yields` | Gross yield vs aligned medians |
| `rent_ingest` | Paginated download of rent resource |
| `clustering` | k-means segments |
| `forecasting` | Monthly median + ETS forecast / backtest RMSE |
| `graph_analytics` | Town correlation graph + modularity communities |
| `mapviz` | Folium choropleth by name |
| `datastore` | Shared paginated CKAN download (via `gov_http`) |
| `gov_http` / `gov_limits` | Tiered pacing vs data.gov.sg table; allowlist; cache; metrics |
| `health` / `health_server` | Readiness-style checks; optional ASGI app |
| `history` | Tranche standardization / concat; `merge_resale_tranches` |
| `rent_cache` | Rent CSV age / TTL for the app |
| `pipeline` | `load_enriched()` = clean → features → reference joins |

## Scope, limitations, data accuracy, and disclaimers

This section is intentionally direct: the project is useful for **learning and exploration**, not as a sole basis for high-stakes choices.

### What this project is and is not

| | |
|--|--|
| **It is** | A **reproducible** stack for EDA and teaching: scripted ingest, cleaning, feature joins, simple models, and multiple UIs (Streamlit, Quarto, notebooks) over **public** HDB and reference data. |
| **It is not** | A **guaranteed-accurate** market database, a **licensed** valuation or investment tool, a **real-time** trading system, or a substitute for **professional** advice. Code and labels like “mature town” or MRT access are **convenience features** for analysis, not authoritative classifications from HDB or UDA. |

### Strengths and pros (why it is worth using)

- **Reproducibility:** A single library path (ingest → clean → features → joins) is shared by tests, the batch script, and the app, which reduces ad-hoc copy-paste errors.
- **Operability on public APIs:** Centralized HTTP with tier-aware spacing, retries, and optional caching supports **safer** refresh workflows than one-off scripts.
- **Multiple surfaces:** CLI download/merge, Streamlit for interaction, Quarto for static narrative, and course notebooks for pedagogy.
- **Tests and fixtures:** Automated tests and small CSV fixtures make offline development and CI possible without a full data pull.
- **Modular design:** Features (e.g. clustering, simple forecasts, graph analytics) are separated so you can **inspect** and **criticize** them independently.

### Weaknesses and cons (what to be careful about)

- **Exploratory, not a full research pipeline:** Methods are **standard EDA and simple models** (e.g. OLS-style stats, k-means, ETS). They are **not** a complete econometric or ML validation programme; assumptions are not fully audited in this repo.
- **Reference data and proxies:** In-repo tables (town→planning area, MRT access, maturity labels) and optional files (e.g. EIP **stub**) are simplifications where noted. Even when sourced from official datasets/APIs, joins and transformations can still be wrong, incomplete, or stale relative to current ground truth.
- **Optional or stubbed features:** Some modules are explicitly limited (e.g. **sun exposure** is largely placeholder unless enabled and configured; see package layout). Treat their outputs as **illustrative**.
- **Sample and default limits:** Default downloads cap row counts; partial samples **do not** represent the full market. Stale or cached data may not reflect the latest government release.
- **Upstream and maintenance risk:** API formats, tranche boundaries, and CKAN behaviour **change**; the pipeline may need maintenance after upstream changes.
- **Not investment or legal advice:** Nothing here authorizes a property purchase, sale, lease, or any regulated activity.

### Data accuracy and how seriously to take the numbers

Be **neutral** about accuracy: the **primary** resale series comes from the official HDB / data.gov.sg distribution named in this README, but **any** number in this project after that point depends on **your** file version, **merge keys**, **joins to reference files**, **imputations**, **aggregations**, and **model choices**.

| Aspect | Realistic expectation |
|--------|------------------------|
| **Source data** | Official open datasets are the **starting point**; they can still have **reporting lags**, **revisions**, or **schema** changes. This project does not certify row-level truth against HDB’s internal systems. |
| **In-repo and derived reference** | Town/planning/MRT/maturity and similar fields are **heuristic** or **simplified**; boundary and naming mismatches can create **join errors** or **mis-labelling**. |
| **Aggregates and maps** | Choropleths and tables aggregate potentially noisy points; small-N areas can be **unstable** or **misleading** when compared visually. |
| **Rents, yields, forecasts** | Rent merges and **gross yield** depend on **alignment** and **median** logic; **forecasts** are **illustrative** and carry **uncertainty** (backtests in code do not guarantee future performance). |
| **Test fixtures** | `tests/fixtures/*.csv` are **tiny, synthetic, or subsampled** for CI and demos—not representative of the market. |

**Bottom line:** Treat all figures as **indicative** and **uncertainty-heavy**. For decisions with money or legal exposure, use **primary sources**, **current** official data, and **qualified** professionals; verify critical fields manually where it matters.

### General disclaimers (please read)

- **No warranty:** The software and documentation are provided **“as is”** under the project license, **without** warranty of any kind, express or implied, including **accuracy**, **fitness for a particular purpose**, or **non-infringement**. Authors and contributors are **not** liable for any damages or losses arising from use of this project.
- **Not professional advice:** Nothing here is **financial**, **property**, **legal**, **tax**, or **estate** advice. **Do not** treat dashboards, reports, or forecasts as recommendations.
- **Your responsibility:** You are responsible for how you use downloads, API keys, and outputs, including **compliance** with data.gov.sg terms, API limits, and applicable laws.
- **Experimental / teaching focus:** The project is intended for **experimentation, teaching, and transparency about methods**. It should **not** be taken as authoritative for serious economic, property, or policy conclusions without **independent** verification.
- **Third-party and government data** remain under their own licences and terms; cite original sources in any public work.

## License

New code: [MIT License](LICENSE). `course_materials/` may include separate licenses. Third-party and government data remain under their own terms; cite official sources in reports.

## References

- Freecodecamp / Data Analysis (legacy notebooks): [YouTube](https://www.youtube.com/watch?v=r-uOLxNrNk8)
