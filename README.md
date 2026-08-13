# ClaimGuard — Auto Insurance Claims Risk, Severity & Tail-Loss Intelligence

Auto insurance risk platform benchmarking actuarial GLMs against machine learning for claim frequency, severity, pure premium, and extreme-loss risk. Built on real French motor TPL claims with calibration, EVT tail modeling, risk segmentation, and portfolio-level validation.

## Setup and data acquisition

ClaimGuard uses Python 3.11+ and the pinned freMTPL2 OpenML snapshots. Create an
isolated environment, install the dependencies, and download the raw data:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m preprocessing.acquire_data
```

The acquisition command validates each table's schema and writes Parquet files
plus a checksum manifest under `data/raw/`. It is idempotent: existing files are
validated and reused. Pass `--force` to download fresh copies.

Verify the setup with:

```bash
pytest -q
```

## Data audit and reconciliation

Profile data quality and reconcile policy claim counts against observed claim
records with:

```bash
python -m preprocessing.audit_data
```

This produces JSON and Markdown reports plus row-level flags under `data/audit/`.
Every rule has an explicit valid, correctable, ambiguous, or excluded handling
decision. Raw source files remain unchanged.

## Actuarial modeling tables

Construct policy-frequency, claim-severity, and policy-loss datasets with:

```bash
python -m preprocessing.build_tables
```

The generated tables retain audit lineage, join policy characteristics onto
individual claims, and reconcile individual claim amounts to policy total loss.

## Portfolio analysis

Generate portfolio summaries, exposure-adjusted frequency tables, severity
statistics, loss-concentration metrics, and diagnostic charts with:

```bash
python -m portfolio.exploratory_analysis
```

The reproducible report is written to `reports/portfolio_analysis/`.

## Actuarial baselines

Generate constant-rate frequency, severity, and pure-premium benchmarks with:

```bash
python -m integration.actuarial_baselines
```

The output includes policy- and claim-level predictions plus transparent
in-sample diagnostics under `reports/actuarial_baselines/`.

## Traditional claim-frequency models

Fit Poisson and Negative Binomial GLMs with log exposure offsets using:

```bash
python -m frequency.traditional_models
```

Held-out metrics, coefficient tables, risk-decile calibration, predictions, and
a calibration chart are written to `reports/traditional_frequency/`.

## Nonlinear claim-frequency models

Train exposure-aware HistGradientBoosting and XGBoost Poisson models with:

```bash
python -m frequency.ml_models
```

The resulting report compares nonlinear performance with the naive rate and
traditional GLMs on the same policy holdout.

## Traditional claim-severity models

Fit Gamma GLM and bias-corrected lognormal severity models with:

```bash
python -m severity.traditional_models
```

The policy-grouped holdout evaluation includes monetary errors, Gamma deviance,
aggregate bias, coefficient tables, predictions, and risk-decile calibration.

## Nonlinear claim-severity models

Train Random Forest, Gamma-loss HistGradientBoosting, and Gamma-objective
XGBoost models with:

```bash
python -m severity.ml_models
```

The output compares all severity models on the same policy-grouped holdout and
stores reusable model artifacts, predictions, and calibration tables.

## Component pure-premium models

Refit the evaluated frequency and severity components on the full analytical
portfolio and generate annual policy pure premiums with:

```bash
python -m pure_premium.component_models
```

The output contains Poisson–Gamma, Poisson–lognormal, HistGradientBoosting, and
XGBoost frequency×severity estimates, exposure-period losses, and reusable
nonlinear scoring artifacts. Actuarial models are stored as portable coefficient
tables with explicit formula and link metadata.

## Direct Tweedie pure premium

Compare component-based expected loss with a direct compound Poisson–Gamma
Tweedie GLM using:

```bash
python -m pure_premium.tweedie_model
```

All candidates use a common policy holdout. The command also refits and stores a
direct Tweedie scoring pipeline for the complete portfolio.

## Nonlinear pure premium

Compare traditional expected-loss models with component-based gradient boosting
and direct Tweedie boosting using:

```bash
python -m pure_premium.ml_model
```

The common-holdout report covers Tweedie deviance, MAE, aggregate calibration,
normalized Gini, top-decile loss capture, and calibration by predicted risk.

## Policy risk deciles

Create equal-sized held-out risk groups for every pure-premium model with:

```bash
python -m segmentation.risk_deciles
```

Each model report includes exposure, recorded and linked claims, predicted and
observed loss, severity, relative loss cost, calibration, and ranking diagnostics.

## Large-loss classification

Model the probability that claim severity exceeds development-sample 90th, 95th,
and 99th percentile thresholds with:

```bash
python -m tail_risk.large_loss_classification
```

Logistic regression and gradient boosting are evaluated on policy-grouped
holdout claims using PR-AUC, ROC-AUC, recall, precision, Brier score, and
probability calibration.

## Severity tail audit

Compare every severity model across ordinary claims and overlapping top-loss
segments with:

```bash
python -m tail_risk.tail_performance
```

The audit uses development-defined thresholds and reports held-out monetary
error, bias, aggregate underprediction, and predicted-to-observed loss.

## Extreme-value modeling

Fit a Peaks Over Threshold Generalized Pareto model and generate threshold
diagnostics, exceedance probabilities, high quantiles, and return levels with:

```bash
python -m tail_risk.extreme_value
```

EVT tail estimates are compared with empirical, Gamma, and lognormal severity
quantiles, with threshold sensitivity reported explicitly.

## Conditional severity quantiles

Estimate policy-level 50th, 75th, 90th, and 95th percentile claim severities
with:

```bash
python -m severity.quantile_models
```

The policy-grouped holdout report compares pinball loss with unconditional
benchmarks, evaluates coverage and crossing, and stores full-portfolio scoring
models and predictions.

## Interpretable risk segments

Create actuarial frequency/severity quadrants enriched with pure premium and
large-loss probability using:

```bash
python -m segmentation.risk_segments
```

The output assigns every policy to standard risk, frequent claimant,
catastrophic exposure, or critical risk without introducing arbitrary clustering.

## Bonus-Malus analysis

Evaluate observed and controlled frequency, severity, and pure premium across
French Bonus-Malus levels with:

```bash
python -m portfolio.bonus_malus_analysis
```

Nested held-out GLMs test whether Bonus-Malus adds predictive value after
controlling for driver, vehicle, and geographic rating characteristics.

## Geographic risk analysis

Decompose observed insurance risk across region, area, and population-density
bands with:

```bash
python -m portfolio.geographic_analysis
```

The report separates frequency from severity, assigns descriptive geographic
risk profiles, and avoids causal interpretation of observed associations.

## Model explainability

Compare GLM coefficients and relativities with held-out permutation importance
and SHAP contributions using:

```bash
python -m explainability.model_explainability
```

Frequency and severity reports align raw-feature ranks across actuarial and
nonlinear models while keeping predictive importance distinct from causality.
