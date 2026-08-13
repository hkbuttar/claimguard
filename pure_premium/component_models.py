"""Combine annual claim-frequency and conditional-severity model components."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)

from frequency.ml_models import (
    fit_hist_gradient_boosting as fit_frequency_hist,
)
from frequency.ml_models import fit_xgboost as fit_frequency_xgb
from frequency.ml_models import prepare_features as prepare_frequency_features
from frequency.traditional_models import MODEL_FORMULA as FREQUENCY_FORMULA
from severity.ml_models import (
    fit_hist_gradient_boosting as fit_severity_hist,
)
from severity.ml_models import fit_xgboost as fit_severity_xgb
from severity.ml_models import prepare_native_features as prepare_severity_features
from severity.traditional_models import (
    LOGNORMAL_FORMULA,
    smearing_factor,
)
from severity.traditional_models import MODEL_FORMULA as SEVERITY_FORMULA

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "component_pure_premium"
PURE_PREMIUM_MODELS: Final = {
    "PoissonGammaAnnualPurePremium": ("PoissonAnnualFrequency", "GammaExpectedSeverity"),
    "PoissonLognormalAnnualPurePremium": (
        "PoissonAnnualFrequency",
        "LognormalExpectedSeverity",
    ),
    "HistGradientBoostingAnnualPurePremium": (
        "HistGradientBoostingAnnualFrequency",
        "HistGradientBoostingExpectedSeverity",
    ),
    "XGBoostAnnualPurePremium": (
        "XGBoostAnnualFrequency",
        "XGBoostExpectedSeverity",
    ),
}


def combine_components(
    policy: pd.DataFrame,
    annual_frequency: dict[str, np.ndarray | pd.Series],
    expected_severity: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Multiply paired annual frequencies and conditional severities by policy."""
    predictions = policy[["IDpol", "Exposure"]].copy()
    for name, values in annual_frequency.items():
        predictions[name] = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    for name, values in expected_severity.items():
        predictions[name] = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    for premium_name, (frequency_name, severity_name) in PURE_PREMIUM_MODELS.items():
        predictions[premium_name] = (
            predictions[frequency_name] * predictions[severity_name]
        )
        predictions[premium_name.replace("AnnualPurePremium", "ExpectedLoss")] = (
            predictions[premium_name] * predictions["Exposure"]
        )
    return predictions


def evaluate_pure_premiums(
    predictions: pd.DataFrame, policy_loss: pd.DataFrame
) -> pd.DataFrame:
    """Return descriptive full-portfolio diagnostics for each component model."""
    evaluated = policy_loss[["IDpol", "TotalLoss"]].merge(
        predictions, on="IDpol", how="inner", validate="one_to_one"
    )
    actual = evaluated["TotalLoss"].to_numpy(dtype=float)
    rows = []
    for premium_name in PURE_PREMIUM_MODELS:
        model = premium_name.replace("AnnualPurePremium", "")
        loss_name = premium_name.replace("AnnualPurePremium", "ExpectedLoss")
        estimate = evaluated[loss_name].to_numpy(dtype=float)
        observed_total = float(actual.sum())
        predicted_total = float(estimate.sum())
        rows.append(
            {
                "Model": model,
                "MAE": float(mean_absolute_error(actual, estimate)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, estimate))),
                "MeanTweedieDeviance": float(
                    mean_tweedie_deviance(actual, estimate, power=1.5)
                ),
                "ObservedAggregateLoss": observed_total,
                "PredictedAggregateLoss": predicted_total,
                "AggregateBias": predicted_total - observed_total,
                "PredictedObservedRatio": predicted_total / observed_total,
            }
        )
    return pd.DataFrame(rows)


def fit_and_predict_components(
    policy: pd.DataFrame, claims: pd.DataFrame, models_dir: Path
) -> tuple[pd.DataFrame, dict]:
    """Refit all component models on the complete analytical portfolio."""
    models_dir.mkdir(parents=True, exist_ok=True)

    poisson = smf.glm(
        formula=FREQUENCY_FORMULA,
        data=policy,
        family=sm.families.Poisson(),
        offset=np.log(policy["Exposure"]),
    ).fit()
    gamma = smf.glm(
        formula=SEVERITY_FORMULA,
        data=claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    log_claims = claims.copy()
    log_claims["LogClaimAmount"] = np.log(log_claims["ClaimAmount"])
    lognormal = smf.ols(formula=LOGNORMAL_FORMULA, data=log_claims).fit()
    smear = smearing_factor(lognormal.resid)

    frequency_hist = fit_frequency_hist(policy)
    frequency_xgb = fit_frequency_xgb(policy)
    severity_hist = fit_severity_hist(claims)
    severity_xgb = fit_severity_xgb(claims)

    policy_frequency_features = prepare_frequency_features(policy)
    policy_severity_features = prepare_severity_features(policy)
    annual_frequency = {
        "PoissonAnnualFrequency": poisson.predict(
            policy, offset=np.zeros(len(policy))
        ),
        "HistGradientBoostingAnnualFrequency": frequency_hist.predict(
            policy_frequency_features
        ),
        "XGBoostAnnualFrequency": frequency_xgb.predict(
            policy_frequency_features, base_margin=np.zeros(len(policy))
        ),
    }
    expected_severity = {
        "GammaExpectedSeverity": gamma.predict(policy),
        "LognormalExpectedSeverity": np.exp(lognormal.predict(policy)) * smear,
        "HistGradientBoostingExpectedSeverity": severity_hist.predict(
            policy_severity_features
        ),
        "XGBoostExpectedSeverity": severity_xgb.predict(policy_severity_features),
    }
    predictions = combine_components(policy, annual_frequency, expected_severity)

    for name, result in (
        ("poisson_glm", poisson),
        ("gamma_glm", gamma),
        ("lognormal", lognormal),
    ):
        pd.DataFrame(
            {"Term": result.params.index, "Coefficient": result.params.to_numpy()}
        ).to_csv(models_dir / f"{name}_coefficients.csv", index=False)
    (models_dir / "actuarial_model_specifications.json").write_text(
        json.dumps(
            {
                "poisson_formula": FREQUENCY_FORMULA,
                "poisson_link": "log",
                "poisson_offset": "log(Exposure)",
                "gamma_formula": SEVERITY_FORMULA,
                "gamma_link": "log",
                "lognormal_formula": LOGNORMAL_FORMULA,
                "lognormal_smearing_factor": smear,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    joblib.dump(frequency_hist, models_dir / "frequency_hist_gradient_boosting.joblib")
    frequency_xgb.save_model(models_dir / "frequency_xgboost.json")
    joblib.dump(severity_hist, models_dir / "severity_hist_gradient_boosting.joblib")
    severity_xgb.save_model(models_dir / "severity_xgboost.json")
    metadata = {
        "frequency_training_policies": len(policy),
        "severity_training_claims": len(claims),
        "lognormal_smearing_factor": smear,
    }
    return predictions, metadata


def _render_report(comparison: pd.DataFrame, predictions: pd.DataFrame) -> str:
    lines = [
        "# Frequency × severity pure premiums",
        "",
        "| Model | Mean annual pure premium | Predicted portfolio loss | Observed ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        annual_name = f"{row.Model}AnnualPurePremium"
        lines.append(
            f"| {row.Model} | €{predictions[annual_name].mean():,.2f} | "
            f"€{row.PredictedAggregateLoss:,.2f} | {row.PredictedObservedRatio:.4f} |"
        )
    lines.extend(
        [
            "",
            "Annual pure premium equals predicted annual frequency multiplied by predicted severity conditional on a claim.",
            "Expected loss for the observed policy period additionally multiplies annual pure premium by exposure.",
            "These full-data refits are production-scoring artifacts; their portfolio diagnostics are descriptive, not held-out model-selection results.",
            "Recorded policy claims without linked severities cause component-model aggregate loss to differ from linked observed loss.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(data_dir: Path, output_dir: Path) -> dict:
    policy = pd.read_parquet(data_dir / "policy_frequency.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    policy_loss = pd.read_parquet(data_dir / "policy_loss.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, metadata = fit_and_predict_components(
        policy, claims, output_dir / "models"
    )
    comparison = evaluate_pure_premiums(predictions, policy_loss)
    predictions.to_parquet(output_dir / "policy_predictions.parquet", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": metadata,
        "models": comparison.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(comparison, predictions)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(args.data_dir.resolve(), args.output_dir.resolve())
