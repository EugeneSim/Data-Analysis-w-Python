# Near-term Evaluation Report

- Generated: `2026-04-28T03:01:41.502719+00:00`
- Rolling windows evaluated: `1`
- Mean rolling RMSE: `208574.04`
- Mean rolling MAE: `176651.03`
- Interval coverage (p10-p90): `0.138`
- Mean interval width: `97319.55`

## Exogenous/Location Ablation (test split)
- Without location features RMSE: `49404.03`
- With location features RMSE: `48376.61`

Artifacts:
- JSON: `reports/forecaster_v1/near_term_eval.json`
- Promotion gate: `FAIL` (coverage_pass=False, width_pass=True)
