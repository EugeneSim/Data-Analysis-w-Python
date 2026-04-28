# Forecaster V1 Runbook

This runbook covers build, validation, and incident recovery for the V1 forecaster pipeline.

## Build and verify

1. Build artifacts:
   - `python scripts/build_forecaster_v1.py --input data/raw/hdb_resale_2017_onwards.csv`
2. Verify reproducibility smoke check:
   - `python scripts/verify_forecaster_repro.py`
3. Run quality gates:
   - `make ruff`
   - `make test`
   - `python -m pip_audit -r requirements.txt`
   - `python scripts/run_forecaster_near_term_eval.py --input data/raw/hdb_resale_2017_onwards.csv`
4. Materialize governed feedback datasets:
   - `python scripts/prepare_feedback_dataset.py`
4. (Optional) refresh official MRT references:
   - `make fetch-mrt-reference`
5. (Optional) refresh official MRT-to-CBD travel-time features:
   - `make fetch-mrt-travel-times`
   - Credentials required:
     - local: repo-root `.env` (`ONEMAP_API_EMAIL`, `ONEMAP_API_PASSWORD`)
     - Streamlit Cloud: App Settings -> Secrets with same keys

## Full end-to-end validation command

Use this command to run the full machine-learning path without opening Streamlit:

```bash
ML_OPEN_STREAMLIT=0 ML_RUN_DEEPCHECKS=1 RUN_ALL_SKIP_INSTALL=1 ./run-all-machinelearning.sh
```

Expected phases:
- environment/bootstrap,
- lint + tests,
- artifact build,
- Deepchecks validation stage,
- inference smoke-check,
- vulnerability audit.

Near-term eval release criteria:
- rolling windows must satisfy configured minimum (`validation.rolling_min_windows`),
- promotion gate in `near_term_eval.json` must report `pass=true`,
- if gate fails, do not promote artifacts without documented exception.
- enforce hard failure in CI/release jobs with `python scripts/run_forecaster_near_term_eval.py --input ... --enforce-gate`.

## Artifacts and contract

- Model artifact: `models/forecaster_v1/model.joblib`
- Metadata: `models/forecaster_v1/metadata.json`
- Config snapshot: `models/forecaster_v1/train_config.json`
- Model card: `models/forecaster_v1/model_card.md`
- Near-term evaluation: `reports/forecaster_v1/near_term_eval.{json,md}`
- Official MRT references: `data/reference/mrt_station_symbol_ura.geojson`
- Official MRT name annotations: `data/reference/mrt_masterplan2003_name_ura.geojson`
- Official MRT travel-time features: `data/reference/mrt_travel_time_to_cbd.csv`

Expected metadata fields include:
- split windows and counts,
- aggregate and segment metrics,
- source file provenance (`source_path`, `source_sha256`, `ingested_at_utc`),
- training configuration.

## Incident recovery

### Build fails with XGBoost OpenMP error on macOS

- Symptom: missing `libomp.dylib`.
- Fix: `brew install libomp`.

### Build succeeds but Streamlit forecaster tab says model missing

- Ensure both files exist:
  - `models/forecaster_v1/model.joblib`
  - `models/forecaster_v1/metadata.json`
- Re-run builder command if either file is missing.

### Prediction request fails in Streamlit

1. Check inference inputs are valid (positive area/lease and known categoricals).
2. Rebuild artifacts to refresh feature map:
   - `python scripts/build_forecaster_v1.py --input data/raw/hdb_resale_2017_onwards.csv`
3. Re-run smoke check:
   - `python scripts/verify_forecaster_repro.py`

### Segment metrics unexpectedly degrade

1. Confirm source dataset hash changed (`metadata.json` provenance block).
2. Compare current vs previous metrics JSON snapshots.
3. Validate ingestion schema did not change unexpectedly.
4. Roll back to prior known-good artifact set if needed.

### Near-term calibration gate fails

1. Inspect `reports/forecaster_v1/near_term_eval.json`:
   - `rolling_origin.overall_interval_coverage_p10_p90`
   - `rolling_origin.mean_interval_width`
   - `promotion_gate`
2. Compare against config thresholds in `configs/forecaster_v1.yaml`.
3. If width is high or coverage is low, retrain with updated features/hyperparameters and rerun eval.
4. Keep previous artifact set as active until the gate passes.

### Deepchecks stage does not generate HTML report

- Check for fallback file:
  - `reports/deepchecks/forecaster_v1/deepchecks_unavailable.txt`
- This indicates environment-level incompatibility in upstream Deepchecks dependencies.
- Keep training pipeline running, but treat Deepchecks as degraded until dependency compatibility is resolved.

### Vulnerability gate reports MLflow CVEs

- Current baseline is `mlflow>=3.11.1,<4` in `pyproject.toml`.
- Tracking backend default is `sqlite:///mlflow.db` (configured in `configs/forecaster_v1.yaml`) to avoid deprecated filesystem backend warnings.
- If `pip-audit` reports new MLflow vulnerabilities:
  1. upgrade to remediated line,
  2. rebuild environment,
  3. re-run `python -m pip_audit -r requirements.txt`.
  4. record findings/remediation in release notes.

### OneMap travel-time fetch fails

- `400 Bad Request` during auth:
  - verify `.env` or secrets keys are present and not placeholder values.
- `401/403` during routing:
  - refresh credentials/token and retry.
- `404` for route edge case:
  - script now guards self-anchor routing and should not fail on identical start/end.

### Feedback governance checks fail

1. Re-run materialization: `python scripts/prepare_feedback_dataset.py`.
2. Validate outputs:
   - `data/feedback/forecaster_feedback_validated.csv`
   - `data/feedback/forecaster_feedback_retraining.csv`
3. Ensure retention and comment redaction settings are correct in `configs/forecaster_v1.yaml`.
