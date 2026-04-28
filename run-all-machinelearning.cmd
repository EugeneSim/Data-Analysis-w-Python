@echo off
setlocal
cd /d "%~dp0"

set "VENV=.venv\Scripts\python.exe"
if not exist "%VENV%" (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
)
call .venv\Scripts\activate.bat

if "%RUN_ALL_SKIP_INSTALL%"=="1" (
  echo === RUN_ALL_SKIP_INSTALL=1, skipping package install ===
) else (
  python -m pip install -U pip
  pip install -e ".[dev]"
)

if not exist "data\raw" mkdir "data\raw"
if not exist "models\forecaster_v1" mkdir "models\forecaster_v1"

if not exist "data\raw\median_rent_hdb.csv" (
  echo.
  echo === Downloading official median rent ^(needs network^) ===
  hdb-rent-download -o data\raw\median_rent_hdb.csv
)
if not exist "data\raw\hdb_resale_2017_onwards.csv" (
  echo.
  if defined HDB_MAX_ROWS (
    echo === Downloading resale CSV ^(HDB_MAX_ROWS=%HDB_MAX_ROWS%; needs network^) ===
    hdb-download -o data\raw\hdb_resale_2017_onwards.csv --max-rows %HDB_MAX_ROWS% --latest-first
  ) else (
    echo === Downloading full Jan-2017+ tranche ^(set HDB_MAX_ROWS to cap^) ===
    hdb-download -o data\raw\hdb_resale_2017_onwards.csv
  )
)

echo.
echo === Downloading BTO historical + future supply datasets ^(needs network^) ===
hdb-bto-download -o data\reference

if not exist "data\raw\hdb_resale_2017_onwards.csv" (
  echo.
  echo === No download; using fixture data for model build ===
  set "INPUT_CSV=tests\fixtures\hdb_sample.csv"
) else (
  set "INPUT_CSV=data\raw\hdb_resale_2017_onwards.csv"
)

echo.
echo === Lint ===
python -m ruff check src tests streamlit_app.py scripts
if errorlevel 1 exit /b 1

echo.
echo === Tests ===
python -m pytest tests -q
if errorlevel 1 exit /b 1

echo.
echo === Build Forecaster V1 Artifacts ===
python scripts\build_forecaster_v1.py --input "%INPUT_CSV%" --config configs\forecaster_v1.yaml
if errorlevel 1 exit /b 1

if "%ML_RUN_DEEPCHECKS%"=="1" (
  echo.
  echo === Deepchecks Validation ===
  python scripts\run_deepchecks_forecaster.py
  if errorlevel 1 exit /b 1
)

echo.
echo === Inference Smoke Check ===
python -c "import json; from singapore_eda.forecaster_v1 import predict_with_explain; p={'month':'2026-04-01','town':'ANG MO KIO','flat_type':'4 ROOM','flat_model':'Model A','storey_range':'04 TO 06','floor_area_sqm':93.0,'lease_commence_date':1998,'remaining_lease_years':68.0}; o=predict_with_explain(p); print(json.dumps({'prediction':o['prediction'],'prediction_interval':o['prediction_interval'],'contributors':len(o['top_contributors']),'warnings':len(o['warnings'])}, indent=2))"
if errorlevel 1 exit /b 1

echo.
echo === Near-term Evaluation Gate ===
if "%ML_ENFORCE_NEAR_TERM_GATE%"=="1" (
  python scripts\run_forecaster_near_term_eval.py --input "%INPUT_CSV%" --enforce-gate
) else (
  python scripts\run_forecaster_near_term_eval.py --input "%INPUT_CSV%"
)
if errorlevel 1 exit /b 1

echo.
echo === Feedback Governance Materialization ===
python scripts\prepare_feedback_dataset.py
if errorlevel 1 exit /b 1

echo.
echo === Vulnerability Audit ^(requirements scope^) ===
python -m pip_audit -r requirements.txt

if "%ML_OPEN_STREAMLIT%"=="0" (
  echo.
  echo === Done ^(ML_OPEN_STREAMLIT=0, skipped launching Streamlit^) ===
  goto :eof
)

echo.
echo === Streamlit ^(Ctrl+C to stop; opens in your browser^) ===
streamlit run streamlit_app.py

endlocal
