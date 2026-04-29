# Near-term Evaluation Report

- Generated: `2026-04-28T08:28:43.574563+00:00`
- Rolling windows evaluated: `1`
- Mean rolling RMSE: `206598.59`
- Mean rolling MAE: `165988.40`
- Interval coverage (p10-p90): `0.217`
- Mean interval width: `96066.74`

## Exogenous/Location Ablation (test split)
- Without location features RMSE: `94168.71`
- With location features RMSE: `66123.28`

Artifacts:
- JSON: `reports/forecaster_v1/near_term_eval.json`
- Promotion gate: `FAIL` (coverage_pass=False, width_pass=True)
