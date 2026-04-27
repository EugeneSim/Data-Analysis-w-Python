from pathlib import Path
from typing import Any

# data.gov.sg CKAN: Resale flat prices (registration) from Jan 2017
HDB_RESALE_RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
HDB_DATASTORE_SEARCH = "https://data.gov.sg/api/action/datastore_search"
HDB_CITATION_URL = "https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view"

# data.gov.sg CKAN: Median rent by town and flat type (quarterly; full history; updated by HDB)
# https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view
HDB_MEDIAN_RENT_RESOURCE_ID = "d_23000a00c52996c55106084ed0339566"
HDB_MEDIAN_RENT_CITATION_URL = (
    "https://data.gov.sg/datasets/d_23000a00c52996c55106084ed0339566/view"
)

# Other resale tranche resource IDs; validate on data.gov.sg (collection 189)
# and merge with `hdb-resale-merge` if schemas align.
HDB_RESALE_TRANCHES: list[dict[str, Any]] = [
    {
        "id": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
        "label": "resale_2017_onwards",
        "coverage": "2017+",
        "month_field": "month",
    },
    {
        "id": "d_ea9ed51da2787afaf8e51f827c304208",
        "label": "resale_2015_2016_reg",
        "coverage": "2015-01 to 2016-12 (registration; verify on portal)",
        "month_field": "month",
    },
]

# URA planning-area polygon (no sea) — data.gov.sg dataset + poll-download API.
# Legacy `https://geo.data.gov.sg/gis/MP14...` is no longer served.
PLANNING_AREA_GEOJSON_DATASET_ID = "d_2cc750190544007400b2cfd5d7f53209"
PLANNING_AREA_GEOJSON_POLL_URL = (
    f"https://api-open.data.gov.sg/v1/public/api/datasets/"
    f"{PLANNING_AREA_GEOJSON_DATASET_ID}/poll-download"
)

# Paths relative to repo root when running from project root
DEFAULT_RAW_CSV = Path("data/raw/hdb_resale_2017_onwards.csv")
DEFAULT_PROCESSED_PARQUET = Path("data/processed/hdb_clean.parquet")
DEFAULT_RENT_CSV = Path("data/raw/median_rent_hdb.csv")

# Legacy doc only — spacing is tier-based in ``singapore_eda.gov_limits`` (match data.gov.sg table).
API_THROTTLE_SEC = 10.5
# Largest page reduces **number of API calls** (subject to CKAN max; see portal if capped lower).
PAGE_SIZE = 10_000

# Singapore floor area: CSV uses m²; some UI copy uses square feet.
SQM_TO_SQFT = 10.763910416709722
