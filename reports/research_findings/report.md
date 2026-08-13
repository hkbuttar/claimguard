# ClaimGuard research findings

ClaimGuard compares actuarial GLMs with nonlinear models for French motor third-party liability frequency, severity, pure premium, and extreme-loss risk.

## Findings

### 1. Which characteristics are associated with claim frequency?

**Robust** — Bonus-Malus and vehicle age ranked first and second across GLM, permutation, and SHAP frequency explanations; driver age was also consistently important.

Evidence: Permutation top five: BonusMalus, VehAge, VehBrand, VehPower, DrivAge.

### 2. Which characteristics are associated with claim severity?

**Suggestive** — Severity signals were weaker and less consistent; region, Bonus-Malus, driver age, and vehicle power appeared important under different explanation methods.

Evidence: SHAP top five: DrivAge, Region, VehPower, VehBrand, BonusMalus.

### 3. How different are the drivers of frequency and severity?

**Suggestive** — Frequency is dominated by Bonus-Malus and vehicle age, whereas severity gives more weight to driver age, region, and vehicle power.

Evidence: Only Bonus-Malus appears in every frequency and severity top-five comparison.

### 4. How well does Bonus-Malus separate actual risk?

**Robust** — Bonus-Malus separates frequency and pure premium after controlling for driver, vehicle, and geography, but adds no held-out severity value.

Evidence: Per +10 points: frequency relativity 1.252, pure-premium relativity 1.470; severity deviance change -0.13%.

### 5. Do ML models materially outperform actuarial GLMs?

**Robust** — ML materially improves some predictive metrics, but not calibration, distribution fit, tail protection, or interpretability; no universal winner exists.

Evidence: Frequency deviance improvement 5.09%; severity MAE improvement 17.24%.

### 6. Is ML improvement present for frequency, severity, or both?

**Robust** — Improvements are present for frequency deviance and severity MAE, with paired bootstrap support for both.

Evidence: Stable frequency improvement: True; stable severity improvement: True.

### 7. Does frequency × severity outperform direct Tweedie modeling?

**Suggestive** — Component GBM has lower Tweedie deviance, while direct Tweedie has substantially better aggregate calibration and lower MAE than traditional component models.

Evidence: Component GBM deviance 68.261; Tweedie GLM 70.667.

### 8. Which approach produces the best calibrated pure premium?

**Robust** — The Tweedie GLM is the best aggregate-calibrated held-out pure-premium model.

Evidence: Tweedie predicted/observed ratio: 0.9933.

### 9. Do average-performance winners remain strong in the extreme tail?

**Robust** — No. Every severity model misses more than 97% of held-out top-1% aggregate loss.

Evidence: Best top-1% capture was 2.89% from Random Forest.

### 10. How concentrated is portfolio risk among high-risk policies?

**Suggestive** — Model-ranked high-risk policies concentrate loss, but rankings are noisy because isolated extreme claims dominate observed outcomes.

Evidence: Direct boosting D10 captured 28.91%; GBM components D10 captured 23.88%.

### 11. Can large claims be identified before they occur?

**Data-limited** — Available policy characteristics provide only weak discrimination for large claims and essentially none for the most extreme claims.

Evidence: Q95 boosting PR-AUC 0.0628 at event rate 4.92%.

### 12. How much aggregate loss uncertainty comes from extreme claims?

**Exploratory** — EVT stress scenarios are overwhelmingly tail-driven, but the fitted infinite-variance tail makes capital metrics highly unstable.

Evidence: Tail share of mean loss 62.75%; tail share of 99% ES 97.39%.

## Overall conclusion

ML improves frequency estimation, severity MAE, and risk ranking, while actuarial models retain advantages in calibration, distributional fit, and transparent uncertainty. EVT better represents extreme losses, but its parameters are unstable and cannot compensate for weak policy-level large-loss predictability.

The most defensible architecture is hybrid: nonlinear frequency and ranking, a calibrated Tweedie benchmark for expected loss, interpretable component models for diagnosis, and separate EVT scenarios for tail stress testing.

## Classification definitions

- Robust: supported by held-out results and uncertainty or sensitivity analysis.
- Suggestive: consistent evidence with meaningful metric or sampling trade-offs.
- Exploratory: useful scenario evidence with strong modeling sensitivity.
- Data-limited: available predictors do not support a reliable conclusion or operational model.
