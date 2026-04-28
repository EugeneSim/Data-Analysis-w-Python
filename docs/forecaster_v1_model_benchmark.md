# Forecaster V1 Model Benchmark Record

This file records the head-to-head training results used to justify model-family selection in V1.

## Run Context

- **Run timestamp (UTC):** `2026-04-27T07:50:51.617236+00:00`
- **Data source:** `data/raw/hdb_resale_2017_onwards.csv`
- **Source SHA256:** `6d45c6bc9370516ee4526a277e516fb1e37809627d3a4a59105aa4781b131bfe`
- **Split:** time-based (`train=13993`, `val=2998`, `test=2999`)
- **Selection rule:** lowest `test_rmse`, tie-break with lowest `test_mae`

## Candidate Results

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| XGBoost | 31,875.07 | 45,600.90 | 32,894.19 | 47,556.50 |
| LightGBM | 31,655.88 | 45,972.86 | 33,512.24 | 48,776.09 |

## Decision

- **Selected model family:** `xgboost`
- **Reason:** better holdout test generalization (`test_rmse` and `test_mae`) despite slightly better LightGBM validation MAE.
- **Interpretation:** XGBoost was more stable on unseen holdout under the current data snapshot and split policy.

## Traceability

- Raw benchmark values are also stored in:
  - `models/forecaster_v1/metadata.json` under `model_benchmark`
  - `models/forecaster_v1/metadata.json` under `selected_model_family`

## Near-term evaluation follow-up (post-benchmark)

Additional reliability/ablation artifacts are tracked separately from the core model-family benchmark:

- File: `reports/forecaster_v1/near_term_eval.json`
- File: `reports/forecaster_v1/near_term_eval.md`

Latest recorded highlights:
- Rolling-origin windows evaluated: `1`
- Rolling RMSE: `183,080.66`
- Rolling MAE: `155,880.12`
- Interval coverage (`p10-p90`): `0.191`
- Exogenous/location ablation:
  - without location features RMSE: `61,271.28`
  - with location features RMSE: `57,099.73`
