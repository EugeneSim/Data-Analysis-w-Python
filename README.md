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
| **Windows** | Double-click or run: `run-all.cmd` |
| **macOS / Linux** | `chmod +x run-all.sh` once, then `./run-all.sh` |

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

Then: `make test` · `hdb-download -o data/raw/hdb_resale_2017_onwards.csv` (or `--max-rows 20000` for a faster sample) · `make analysis` or `python scripts/run_analysis.py` · `streamlit run streamlit_app.py`.

**Geo (Graph tab, planning-area edges):** `pip install -e ".[geo]"` and run `make fetch-geo` or `scripts/fetch_reference_geo.py` to pull the official [MP2025 planning-area boundary (No Sea)](https://data.gov.sg/datasets/d_2cc750190544007400b2cfd5d7f53209/view) GeoJSON (old static `geo.data.gov.sg` MP14 links are no longer served). **Rent / gross yield:** set `HDB_MEDIAN_RENT_RESOURCE_ID` and `hdb-rent-download -o data/raw/median_rent_hdb.csv`, or use [`tests/fixtures/median_rent_sample.csv`](tests/fixtures/median_rent_sample.csv). Override staleness in the app with `SINGAPORE_EDA_RENT_TTL_HOURS`. Tranche merge and other CLI details: `hdb-download --help`, `hdb-resale-merge --help`.

## Data

**Primary resale table:** [Resale flat prices (registration) from Jan-2017 onwards](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) (HDB, [Open Data Licence](https://data.gov.sg/open-data-licence)).

**Reference tables (in-repo):** [`data/reference/`](data/reference/) — town → planning area / OCR, mature vs young labels, MRT access proxy, optional EIP block stub, future MRT example rows.

## Live demo (two URLs)

- **GitHub Pages** (static report only): enable [`.github/workflows/publish-quarto.yml`](.github/workflows/publish-quarto.yml) and set **Settings → Pages** to **GitHub Actions**. URL shape: `https://<user>.github.io/<repo>/`.
- **Streamlit Community Cloud** (interactive app): main file `streamlit_app.py`; use [`packages.txt`](packages.txt) or `pip install -e .`. Set secrets in the Cloud UI (e.g. `DATA_GOV_SG_API_KEY`) as in [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example). List the Cloud app URL in this README next to the Pages URL.

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
7. Deploy and copy your Streamlit URL (typically `https://<app-name>.streamlit.app`).
8. Replace the placeholders below and commit:

```md
- GitHub Pages (static): https://<your-github-username>.github.io/<your-repo-name>/
- Streamlit app (interactive): https://<your-app-name>.streamlit.app
```

9. Add the same Streamlit URL into `quarto/index.qmd` under the "Static site vs interactive app" section so users on Pages can jump to the live app.

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
- **Reference data and proxies:** In-repo tables (town→planning area, MRT access, maturity labels) and optional files (e.g. EIP **stub**, future MRT **examples**) are **simplifications** or **placeholders** where noted in code or docs. They can be **wrong**, **incomplete**, or **out of date** relative to ground truth.
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
