# ClaimGuard — Auto Insurance Claims Risk, Severity & Tail-Loss Intelligence

ClaimGuard is an end-to-end actuarial risk platform for French motor third-party
liability insurance. It benchmarks transparent generalized linear models against
nonlinear machine learning for claim frequency, conditional severity, pure
premium, risk ranking, and extreme-loss behavior—then serves the fitted system
through an inference API and interactive dashboard.

> Can modern machine learning improve auto-insurance risk estimation over
> traditional actuarial frequency–severity models, and do those improvements
> survive calibration and extreme-loss testing?

The answer is nuanced: ML improves several held-out accuracy and ranking metrics,
but the Tweedie GLM is far better calibrated at portfolio level, and none of the
policy-level models predicts the most extreme claims reliably. The most
defensible design is therefore hybrid rather than winner-takes-all.

## Why ClaimGuard?

Average predictive error is not enough for insurance. A useful risk system must
distinguish how often claims occur, how costly they are when they occur, whether
predictions balance in aggregate, how well policies are ordered by risk, and how
much capital uncertainty comes from rare losses.

ClaimGuard treats these as related but distinct questions:

```text
Policy features
      │
      ├── Claim frequency ──┐
      │                     ├── Expected annual loss ── Calibration and ranking
      └── Claim severity ───┘                │
                                             ├── Large-loss probability
                                             ├── EVT tail scenarios
                                             └── Portfolio stress testing
```

For policy \(i\), annual pure premium is modeled as:

\[
\operatorname{E}[L_i]
= \operatorname{E}[N_i]\times\operatorname{E}[Y_i\mid N_i>0]
\]

Exposure-period expected loss is annual pure premium multiplied by earned
exposure. These are modeled insurance losses—not commercial prices.

## Empirical portfolio

The project uses the public
[freMTPL2 datasets](https://dutangc.github.io/CASdatasets/reference/freMTPL.html):

| Quantity | Observed value |
|---|---:|
| Policies | 678,013 |
| Earned exposure | 358,360 policy-years |
| Recorded claims | 36,102 |
| Linked positive severities | 26,444 |
| Annual claim frequency | 0.1007 |
| Linked claim loss | €59.91 million |
| Mean / median severity | €2,266 / €1,172 |
| Maximum severity | €4.08 million |
| Observed linked loss cost | €167.18 per exposure |

The loss distribution is exceptionally concentrated: the largest 10% of claims
produce 59.9% of loss, the largest 5% produce 52.1%, and the largest 1% produce
38.0%. This concentration is why the project evaluates the tail separately from
ordinary predictive performance.

### Modeling tables

The frequency source contains policy ID, exposure, claim count, driver age,
vehicle age and power, vehicle brand and fuel, Bonus-Malus, area, density, and
region. The severity source contains individual claim amounts linked by policy
ID. Reproducible preprocessing creates:

- one policy-frequency row per policy;
- one claim-severity row per matched individual claim;
- one policy-loss row with linked claims aggregated and zero loss assigned to
  policies without a linked claim.

## Data quality and reconciliation

Raw inputs are preserved. Audit rules classify records as valid, correctable,
ambiguous, or excluded, with row-level flags and explicit decisions.

| Finding | Count | Treatment |
|---|---:|---|
| Duplicate policy IDs | 0 | None found |
| Exposure above one year | 1,224 | Capped at one in analytical tables |
| Unmatched severity records | 195 | Excluded from policy-linked models |
| Recorded/linked claim-count mismatches | 9,117 | Retained and reported; neither source is overwritten |
| Possible duplicate claims | 496 | Retained because no claim identifier exists |
| Claims above the empirical 99.5th percentile | 134 | Retained and flagged for tail analysis |

Sensitivity analysis confirms that cleaning choices materially alter the target.
For example, strict filtering lowers observed loss cost from €167.18 to €111.96
and top-1% loss concentration from 38.0% to 12.1%. ClaimGuard therefore reports
results under the minimally altered portfolio and makes alternative scenarios
visible instead of treating cleaning as neutral.

## Research questions

The research notebook answers twelve questions and labels each conclusion as
robust, suggestive, exploratory, or data-limited:

1. Which policy characteristics are associated with claim frequency?
2. Which characteristics are associated with claim severity?
3. How different are the drivers of frequency and severity?
4. How well does Bonus-Malus separate actual risk?
5. Do ML models materially outperform actuarial GLMs?
6. Is ML improvement present for frequency, severity, or both?
7. Does frequency × severity outperform direct Tweedie modeling?
8. Which approach produces the best-calibrated pure premium?
9. Do average-performance winners remain strong in the extreme tail?
10. How concentrated is portfolio risk among high-risk policies?
11. Can large claims be identified before they occur?
12. How much aggregate loss uncertainty comes from extreme claims?

The generated synthesis is available in
[`notebooks/research.ipynb`](notebooks/research.ipynb) and
[`reports/research_findings/report.md`](reports/research_findings/report.md).

## Modeling framework

### Claim frequency

The actuarial candidates are a portfolio-rate baseline, Poisson GLM, and
Negative Binomial GLM. Exposure enters the GLMs through a log offset. Nonlinear
candidates are exposure-weighted HistGradientBoosting and XGBoost Poisson
models. All comparisons use the same policy holdout and report Poisson deviance,
aggregate balance, and calibration by predicted risk.

HistGradientBoosting reduced held-out mean Poisson deviance from 0.3212 for the
Poisson GLM to 0.3048, a 5.09% improvement. A 500-repetition paired bootstrap
placed the ML-minus-GLM improvement between −0.0182 and −0.0146, supporting a
stable nonlinear gain.

Bonus-Malus and vehicle age dominate frequency explanations across coefficient,
permutation, and SHAP views. After controlling for driver, vehicle, and
geographic factors, a 10-point Bonus-Malus increase corresponds to a frequency
relativity of 1.252 (95% interval 1.243–1.260).

### Claim severity

Traditional models include a Gamma GLM and bias-corrected lognormal regression.
ML candidates include Random Forest, Gamma-loss HistGradientBoosting, and
Gamma-objective XGBoost. Splits are grouped by policy to prevent claims from the
same policy appearing in both development and validation samples.

XGBoost reduced held-out MAE from €2,147.69 for the Gamma GLM to €1,777.39, a
17.24% improvement. The paired bootstrap interval for the MAE difference was
−€396.30 to −€347.51. That result does not imply dominance: the lognormal model
had better Gamma deviance than XGBoost, and XGBoost underpredicted aggregate
held-out severity by 20.9%.

Severity drivers are weaker and less consistent than frequency drivers. Driver
age, region, vehicle power, brand, and Bonus-Malus appear under different
explanation methods, but available policy characteristics provide little signal
for truly extreme claims.

### Pure premium

ClaimGuard compares:

- Poisson × Gamma and Poisson × lognormal components;
- nonlinear frequency × nonlinear severity components;
- a direct compound Poisson–Gamma Tweedie GLM;
- direct Tweedie gradient boosting.

GBM components improved held-out mean Tweedie deviance from 70.67 for the
Tweedie GLM to 68.26, a 3.40% gain with bootstrap support. Direct boosting
improved risk ordering: normalized Gini rose from 0.1889 for Tweedie to 0.2294,
and its highest decile captured 28.9% of held-out loss.

Calibration reverses that result. The Tweedie GLM predicted 99.33% of observed
aggregate loss, compared with 116.13% for GBM components and 68.60% for direct
boosting. The project keeps accuracy, ranking, and calibration separate because
optimizing one does not guarantee the others.

### Segmentation and Bonus-Malus

Policies are assigned to interpretable frequency/severity quadrants:

| Segment | Policies | Modeled frequency | Modeled severity | Annual pure premium | Observed loss cost |
|---|---:|---:|---:|---:|---:|
| Standard Risk | 188,447 | 0.0612 | €1,663 | €102 | €60 |
| Frequent Claimant | 150,559 | 0.1619 | €1,681 | €273 | €163 |
| Catastrophic Exposure | 150,559 | 0.0637 | €2,064 | €131 | €113 |
| Critical Risk | 188,448 | 0.1740 | €2,118 | €372 | €362 |

Bonus-Malus adds held-out value for frequency and pure premium after controls,
but it does not improve held-out severity deviance. This supports its use as a
risk separator without treating the relationship as causal.

## Tail risk

### Large-loss classification

Large claims are defined from development-sample thresholds to prevent holdout
leakage. At the 95th-percentile threshold (€4,808), gradient boosting achieved a
PR-AUC of 0.0628 against a 4.92% event rate. At the 99th percentile, both
classifiers performed at or below practical usefulness. The correct conclusion
is data-limited: ordinary policy variables do not identify extreme claims well.

The severity audit reaches the same conclusion. Every severity model captured
less than 2.9% of held-out top-1% aggregate loss. Average-error winners do not
remain reliable in the extreme tail.

### Extreme Value Theory

A Peaks Over Threshold model fits a Generalized Pareto distribution above the
95th-percentile severity threshold (€4,765). The primary fit has shape 0.904 and
scale €3,479. Its fitted mean is finite but variance is infinite. Threshold and
bootstrap analyses show substantial parameter instability, so EVT output is
treated as scenario evidence rather than a precise capital forecast.

| Quantile | Empirical | Gamma | Lognormal | EVT |
|---|---:|---:|---:|---:|
| 99.0% | €16,451 | €12,648 | €13,036 | €17,413 |
| 99.5% | €34,377 | €14,793 | €17,281 | €31,784 |
| 99.9% | €152,223 | €19,826 | €30,904 | €133,152 |
| 99.95% | €210,870 | €22,010 | €38,754 | €248,353 |

In 10,000 aggregate simulations, EVT-tail claims contributed 62.75% of mean
loss and 97.39% of 99% Expected Shortfall. The simulated full-tail 99% VaR was
€313.2 million and 99% Expected Shortfall was €1.53 billion. Those figures are
highly sensitive to the heavy fitted tail and should be read as stress scenarios,
not booked capital estimates.

## Honest model comparison

| Task | Traditional result | ML result | Evidence |
|---|---:|---:|---|
| Frequency Poisson deviance | 0.3212 | 0.3048 | ML improves average fit |
| Severity MAE | €2,147.69 | €1,777.39 | ML improves average error |
| Severity Gamma deviance | 1.6324 | 1.7265 | Traditional distribution fit is better |
| Pure-premium Tweedie deviance | 70.67 | 68.26 | ML components improve fit |
| Normalized Gini | 0.1889 | 0.2294 | Direct boosting ranks better |
| Absolute aggregate calibration error | 0.0067 | 0.1613 | Tweedie GLM calibrates much better |
| Top-5% severity MAE | €19,278.67 | €19,833.07 | Neither controls the tail well |
| Top-1% loss captured | 2.78% | 2.89% | Neither is operationally adequate |

The empirical conclusion is not “ML wins.” A hybrid architecture is more
defensible:

- nonlinear frequency and direct models for discrimination and ranking;
- a calibrated Tweedie benchmark for expected portfolio loss;
- transparent component GLMs for diagnosis and uncertainty;
- separate EVT scenarios for aggregate tail stress.

## ClaimGuard application

The unified engine validates a policy against the fitted category domains and
returns expected annual claims, conditional severity, annual and exposure-period
loss, large-loss probability, portfolio percentile, actuarial risk segment,
primary global drivers, and a zero-to-100 ClaimGuard score.

```text
Browser dashboard
       │
       ▼
FastAPI inference service ────── Precomputed portfolio reports
       │
       ▼
Unified risk engine
   ├── frequency model
   ├── severity model
   ├── large-loss model
   └── portfolio reference distributions
```

The responsive dashboard at `/dashboard/` includes:

- portfolio overview and loss concentration;
- interactive policy risk explorer;
- actuarial-versus-ML benchmark and risk deciles;
- EVT quantiles and tail diagnostics;
- aggregate loss stress metrics and risk segments;
- observed Bonus-Malus risk analysis.

The API is inference-only. Model fitting and report production are offline
research operations.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/policy/score` | Score one validated policy |
| `GET` | `/portfolio/summary` | Portfolio and loss-concentration metrics |
| `GET` | `/portfolio/segments` | Frequency/severity segment results |
| `GET` | `/portfolio/risk-deciles` | Pure-premium risk deciles |
| `GET` | `/portfolio/stress-test` | Aggregate simulation metrics |
| `GET` | `/models/benchmark` | Actuarial-versus-ML comparison |
| `GET` | `/models/calibration` | Cross-task calibration evidence |
| `GET` | `/tail-risk` | EVT diagnostics |
| `GET` | `/bonus-malus` | Controlled Bonus-Malus analysis |

OpenAPI documentation is available at `/docs` while the service is running.

## Repository structure

```text
claimguard/
├── preprocessing/     data acquisition, audit, tables, sensitivity
├── frequency/         baseline, GLM, and nonlinear frequency models
├── severity/          GLM, lognormal, ML, and quantile models
├── pure_premium/      component, Tweedie, and direct nonlinear models
├── tail_risk/         large-loss classification, audit, and EVT
├── calibration/       calibration and statistical uncertainty
├── segmentation/      policy risk deciles and actuarial quadrants
├── portfolio/         EDA, geography, Bonus-Malus, and stress testing
├── explainability/    coefficients, permutation importance, and SHAP
├── integration/       benchmark, findings, and unified risk engine
├── backend/           typed FastAPI inference service
├── frontend/          React and Vite responsive dashboard
├── deployment/        artifact bundling and startup validation
├── notebooks/         reproducible research notebook
└── tests/             deterministic and module-level validation
```

## Reproduce the analysis

ClaimGuard requires Python 3.11 or newer. Python 3.14 is used by the production
container.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m preprocessing.acquire_data
python -m preprocessing.audit_data
python -m preprocessing.build_tables
```

Run the analytical modules in dependency order:

```bash
python -m portfolio.exploratory_analysis
python -m integration.actuarial_baselines
python -m frequency.traditional_models
python -m frequency.ml_models
python -m severity.traditional_models
python -m severity.ml_models
python -m pure_premium.component_models
python -m pure_premium.tweedie_model
python -m pure_premium.ml_model
python -m segmentation.risk_deciles
python -m tail_risk.large_loss_classification
python -m tail_risk.tail_performance
python -m tail_risk.extreme_value
python -m severity.quantile_models
python -m segmentation.risk_segments
python -m portfolio.bonus_malus_analysis
python -m portfolio.geographic_analysis
python -m explainability.model_explainability
python -m calibration.calibration_analysis
python -m portfolio.stress_testing
python -m integration.model_benchmark
python -m calibration.statistical_rigor
python -m preprocessing.data_quality_sensitivity
python -m integration.research_findings
```

The acquisition command is idempotent and validates pinned OpenML schema and
checksums. Generated data and most reports are excluded from version control.

### Validate

```bash
pytest -q
ruff check .
cd frontend && npm run lint && npm run build
```

The test suite includes deterministic synthetic portfolios with mathematically
known frequency, severity, aggregation, pure-premium, calibration, tail,
segmentation, API, and deployment results.

### Run locally

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/dashboard/` for the dashboard or
`http://localhost:8000/docs` for the API schema.

For frontend development with hot reloading, run the API on port 8000 and then:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the React application at `http://localhost:5173` and proxies API
requests to FastAPI.

For a split deployment, set `VITE_API_BASE_URL` in Vercel to the Render service
origin and set `ALLOWED_ORIGINS` in Render to the Vercel site origin. Multiple
allowed origins can be supplied as a comma-separated list.

To score a JSON policy without HTTP:

```bash
python -m integration.risk_engine --policy policy.json
```

## Deployment

Production ships only the fitted inference models and reports consumed by the
application. The generated bundle is 34.5 MiB, compared with roughly 944 MiB of
full research output.

```bash
python -m deployment.build_bundle
python -m deployment.validate_bundle
docker compose up --build
```

The container validates SHA-256 checksums before accepting traffic, runs as a
non-root user, exposes a health check, and performs no model fitting. Port and
worker count are configurable through `PORT` and `WEB_CONCURRENCY`.

## Limitations

- French motor TPL experience may not generalize to other countries, time
  periods, products, or coverage types.
- Available policy characteristics are limited; there are no accident details,
  repair data, telematics, adjuster notes, or longitudinal policy histories.
- Claim amounts are current or ultimate observations, not full claim-development
  triangles; the project is not a reserving model.
- The frequency and severity sources contain documented linkage and count
  inconsistencies. Reconciliation decisions reduce ambiguity but cannot recover
  missing claims.
- Some amounts reflect standardized French claims conventions in the source
  data, which can affect distributional interpretation.
- Extreme-claim classification has weak discrimination, and the fitted EVT tail
  is unstable with infinite variance at the primary threshold.
- Geographic, demographic, Bonus-Malus, and explainability results are
  predictive associations, not causal effects or fairness conclusions.
- ClaimGuard models expected insurance loss, not commercial premiums. Real
  pricing also includes expenses, profit, reinsurance, regulation, taxes,
  competitive constraints, and underwriting judgment.
- The ClaimGuard score is a research summary of portfolio-relative model output;
  it is not a regulatory, underwriting, or consumer credit score.

## Future work

- incorporate telematics and richer accident or vehicle information;
- add claim-development and reserving models;
- validate temporal stability and out-of-time calibration;
- study multi-country and additional coverage portfolios;
- add commercial pricing constraints and reinsurance structures;
- monitor drift, fairness, and post-deployment calibration;
- develop longitudinal policy and claims histories.

## Scope

ClaimGuard intentionally excludes fraud detection, accident prediction, claims
settlement duration, reserving, neural networks, and commercial rate indication.
Keeping those adjacent problems out preserves a coherent actuarial research
question: expected loss, calibration, risk ranking, and tail uncertainty from a
single real motor insurance portfolio.
