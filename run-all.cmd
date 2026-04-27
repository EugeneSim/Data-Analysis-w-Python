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
  python -c "import singapore_eda" >nul 2>&1
  if errorlevel 1 (
    python -m pip install -U pip
    pip install -e ".[dev]"
  )
)

if not exist "data\raw" mkdir "data\raw"
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
if not exist "data\raw\hdb_resale_2017_onwards.csv" (
  echo.
  echo === No download; using tests\fixtures\hdb_sample.csv for analysis ===
  set "ANL=python scripts\run_analysis.py --input tests\fixtures\hdb_sample.csv"
) else (
  set "ANL=python scripts\run_analysis.py"
)

echo.
echo === Tests ===
python -m pytest tests -q
if errorlevel 1 exit /b 1

echo.
echo === Analysis ^(console summary^) ===
%ANL%
if errorlevel 1 exit /b 1

echo.
echo === Streamlit ^(Ctrl+C to stop; opens in your browser^) ===
streamlit run streamlit_app.py

endlocal
