# Forecaster V1 Decision Log

This file records major implementation decisions for V1, including alternatives considered and rationale.

## D-001: Use temporal split instead of random split

- **Decision:** Train/validation/test are split by chronological `month`.
- **Alternatives considered:** random split, stratified random split.
- **Why:** random splits leak future market conditions into training and overstate generalization for time-evolving prices. Temporal splits better approximate real deployment behavior.
- **Trade-off:** evaluation is usually harsher and can vary across regimes.

## D-002: Keep V1 feature scope focused on core transactional attributes

- **Decision:** Start with core columns (town, flat type/model, storey range, area, lease-related fields, month-derived year/month).
- **Alternatives considered:** include many external features immediately (school ranks, URA overlays, macro rates).
- **Why:** smaller feature surface reduces integration risk and debugging complexity; enables clear attribution of model lift in V1.
- **Trade-off:** potential ceiling on predictive performance until richer features are added.

## D-003: Use Postgres curated schema contract

- **Decision:** Add explicit curated schema (`feature_store.py`) and validate non-null constraints before writes.
- **Alternatives considered:** write ad-hoc dataframes directly without schema checks.
- **Why:** explicit contracts prevent silent schema drift and support reproducible downstream training.
- **Trade-off:** slightly more upfront boilerplate.

## D-004: Persist provenance metadata with each training run

- **Decision:** Store `source_path`, `source_sha256`, and ingestion timestamp in model metadata.
- **Alternatives considered:** keep only model metrics.
- **Why:** reproducibility and auditability require traceability to exact input data snapshot.
- **Trade-off:** metadata payload is larger.

## D-005: Benchmark both LightGBM and XGBoost, select winner automatically

- **Decision:** Train both model families and select by validation holdout performance (`val_rmse`, then `val_mae`); keep test metrics for one-time final reporting only.
- **Alternatives considered:** fix to one model family.
- **Why:** avoids framework bias and bases final choice on observed data performance.
- **Trade-off:** longer training runtime and extra dependency footprint.

### Current benchmark outcome

- **Selected model family:** `xgboost`
- **Observed test performance:** lower RMSE/MAE than LightGBM on current dataset snapshot.
- **Note:** this is not permanent; selection can change with data distribution shifts.
- **Recorded evidence:** [`docs/forecaster_v1_model_benchmark.md`](docs/forecaster_v1_model_benchmark.md)

## D-006: Model on log-price and invert at prediction time

- **Decision:** train with `log1p(resale_price)` target and convert back via `expm1`.
- **Alternatives considered:** train directly on price levels.
- **Why:** log target improves stability for right-skewed price distributions and often reduces sensitivity to high-price outliers.
- **Trade-off:** interpretation requires care when mapping back to absolute-dollar errors.

## D-007: Use residual-quantile interval for uncertainty

- **Decision:** use split-conformal absolute-residual calibration on validation holdout for prediction intervals.
- **Alternatives considered:** conformal prediction, quantile regression models, Bayesian approaches.
- **Why:** simple, fast, and sufficiently interpretable for V1 while keeping implementation maintainable.
- **Trade-off:** interval calibration can degrade under distribution shift.

## D-020: Use month-bucket temporal split boundaries

- **Decision:** split train/validation/test on unique month buckets, never row-count boundaries.
- **Alternatives considered:** row-count split on sorted rows.
- **Why:** month-level boundary purity prevents mixed-month leakage between adjacent splits and better reflects forecast deployment cadence.
- **Trade-off:** split row counts can be less balanced when month volumes vary.

## D-021: Disable lag-market features by default until online lag state is available

- **Decision:** keep lag-market features off by default (`train.use_lag_features: false`) and warn during inference if model expects lag fields but request omits them.
- **Alternatives considered:** always-on lag features with training-median defaults at inference.
- **Why:** avoids hidden train-serve skew from synthetic default lag values.
- **Trade-off:** may reduce raw accuracy ceiling until a deterministic online lag-state service is introduced.

## D-022: Govern feedback into validated and retraining-eligible datasets

- **Decision:** keep raw feedback log, plus deterministic materialization into validated and retraining-eligible datasets with retention and redaction.
- **Alternatives considered:** direct retraining from raw feedback CSV.
- **Why:** raw user feedback is noisy and may include sensitive free text; governance improves quality and compliance posture.
- **Trade-off:** added operational step (`scripts/prepare_feedback_dataset.py`) before feedback-assisted retraining.

## D-008: Use SHAP for local explainability

- **Decision:** produce top local contributors from SHAP values per inference request.
- **Alternatives considered:** permutation importance only, gain-only feature importance.
- **Why:** local explanations are more useful for end-user case-level reasoning than global-only importance.
- **Trade-off:** added dependency/runtime overhead.

## D-009: Add input guardrails using training-range checks

- **Decision:** warn when numeric inputs fall outside observed training bounds.
- **Alternatives considered:** no warnings, hard rejection.
- **Why:** soft warnings preserve usability while signaling extrapolation risk.
- **Trade-off:** warnings are heuristic and not full out-of-distribution detection.

## D-010: Surface model family/version and disclaimers in Streamlit

- **Decision:** show model metadata and decision-support disclaimer in UI.
- **Alternatives considered:** output prediction only.
- **Why:** transparency and responsible usage are essential for trust and interview-grade rigor.
- **Trade-off:** more UI text, but clearer expectations.

## D-011: Add reproducibility and incident runbook

- **Decision:** include `forecaster_v1_runbook.md` and reproducibility smoke check script.
- **Alternatives considered:** rely on informal chat/tribal workflow.
- **Why:** operational reliability requires explicit procedures for build, verification, and recovery.
- **Trade-off:** documentation maintenance overhead.

## D-012: Scope vulnerability checks to repository requirements for release gate

- **Decision:** use `pip-audit -r requirements.txt` as release gate.
- **Alternatives considered:** audit full global environment only.
- **Why:** repository-scoped dependencies are the deployable surface; global environment includes unrelated packages that can obscure actionable project risk.
- **Trade-off:** misses vulnerabilities in unrelated local global packages by design.

## D-013: Use MLflow (not ZenML) for V1 experiment tracking

- **Decision:** integrate MLflow logging directly in training path.
- **Alternatives considered:** ZenML pipeline orchestration, no experiment tracker.
- **Why:** MLflow provides lightweight and fast adoption for parameter/metric/artifact tracking with minimal architecture change.
- **Trade-off:** orchestration remains script-driven (not full DAG/pipeline management as ZenML would provide).
- **Security follow-up:** `pip-audit` currently reports known CVEs on the pinned MLflow line; upgrade is required before production posture.

## D-014: Make hyperparameters and toggles YAML-driven

- **Decision:** all train splits, candidate models, and model-specific params are maintained in `configs/forecaster_v1.yaml`.
- **Alternatives considered:** hardcoded Python constants only.
- **Why:** version-controlled config is easier to review, compare, and reproduce across runs.
- **Trade-off:** additional config parsing layer must be kept in sync with code.

## D-015: Use optional Deepchecks data-integrity gate

- **Decision:** deepchecks validation is integrated and configurable (`deepchecks_enabled`).
- **Alternatives considered:** no formal data validation, custom one-off checks only.
- **Why:** Deepchecks gives standardized, inspectable data quality diagnostics before training.
- **Trade-off:** extra dependency/runtime overhead; enabled by toggle to keep default runs fast.

## D-016: Capture user feedback in-product for retraining loop

- **Decision:** add Streamlit feedback submission (rating, optional actual price, comments, payload snapshot) to local CSV.
- **Alternatives considered:** external feedback tools only, no feedback capture in app.
- **Why:** closes the loop between predictions and observed user outcomes for future iteration prioritization.
- **Trade-off:** feedback quality depends on user participation and data hygiene.

## D-017: Use official government MRT references (replace sample placeholders)

- **Decision:** fetch MRT reference files from official data.gov.sg APIs (`scripts/fetch_mrt_reference.py`) and regenerate local reference files.
- **Alternatives considered:** keep static sample CSV rows.
- **Why:** sample placeholders are not acceptable for production-like analytics; official agency-published data improves traceability and trust.
- **Trade-off:** dependency on network/API availability and rate-limit behavior.

## D-018: Add official OneMap routing travel-time feature pipeline with secret-based auth

- **Decision:** implement `scripts/fetch_mrt_travel_times.py` using OneMap routing API and credentials from secrets/env/.env.
- **Alternatives considered:** only heuristic travel-time proxy from distance/region.
- **Why:** official routing durations provide higher-fidelity accessibility features than static distance-only proxies.
- **Trade-off:** authenticated API dependency, credential management overhead, and variable runtime due to route calls.

## D-019: Keep scenario overlays as user-assumption controls, not learned coefficients

- **Decision:** school-zone, shopping, and CBD-access premiums are exposed as optional UI overlays and clearly labeled as scenario assumptions.
- **Alternatives considered:** hard-code pseudo-premium coefficients directly into model predictions.
- **Why:** avoids presenting unvalidated policy/lifestyle premiums as model-ground truth; preserves user flexibility while maintaining methodological transparency.
- **Trade-off:** users must set assumptions explicitly; overlay outputs are not "fully model-inferred" values.
