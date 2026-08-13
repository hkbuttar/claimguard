"""Compare component pure premiums with a direct Tweedie GLM."""

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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from frequency.traditional_models import (
    MODEL_FORMULA as FREQUENCY_FORMULA,
)
from frequency.traditional_models import split_portfolio
from severity.ml_models import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from severity.traditional_models import LOGNORMAL_FORMULA, smearing_factor
from severity.traditional_models import MODEL_FORMULA as SEVERITY_FORMULA

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "tweedie_pure_premium"
TWEEDIE_POWER: Final = 1.5


def make_tweedie_pipeline(power: float = TWEEDIE_POWER) -> Pipeline:
    """Create a regularized direct pure-premium GLM with encoded rating factors."""
    if not 1 < power < 2:
        raise ValueError("Compound Poisson-Gamma Tweedie power must be between 1 and 2")
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                TweedieRegressor(
                    power=power,
                    alpha=0.1,
                    link="log",
                    solver="newton-cholesky",
                    max_iter=500,
                    tol=1e-8,
                ),
            ),
        ]
    )


def evaluate_policy_loss(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Evaluate zero-mass continuous policy losses on the original euro scale."""
    observed = actual.to_numpy(dtype=float)
    estimate = np.clip(predicted.to_numpy(dtype=float), 1e-12, None)
    observed_total = float(observed.sum())
    predicted_total = float(estimate.sum())
    return {
        "mae": float(mean_absolute_error(observed, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(observed, estimate))),
        "mean_tweedie_deviance": float(
            mean_tweedie_deviance(observed, estimate, power=TWEEDIE_POWER)
        ),
        "observed_aggregate_loss": observed_total,
        "predicted_aggregate_loss": predicted_total,
        "aggregate_bias": predicted_total - observed_total,
        "predicted_observed_ratio": predicted_total / observed_total,
    }


def premium_calibration(
    actual_loss: pd.Series,
    predicted_loss: pd.Series,
    exposure: pd.Series,
    annual_premium: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ActualLoss": actual_loss.to_numpy(dtype=float),
            "PredictedLoss": predicted_loss.to_numpy(dtype=float),
            "Exposure": exposure.to_numpy(dtype=float),
            "AnnualPremium": annual_premium.to_numpy(dtype=float),
        }
    )
    frame["RiskDecile"] = pd.qcut(
        frame["AnnualPremium"], q=bins, labels=False, duplicates="drop"
    )
    result = (
        frame.groupby("RiskDecile", observed=True)
        .agg(
            Policies=("ActualLoss", "size"),
            Exposure=("Exposure", "sum"),
            ObservedLoss=("ActualLoss", "sum"),
            ExpectedLoss=("PredictedLoss", "sum"),
            MeanAnnualPremium=("AnnualPremium", "mean"),
        )
        .reset_index()
    )
    result["ObservedExpectedRatio"] = result["ObservedLoss"] / result["ExpectedLoss"]
    return result


def fit_holdout_models(
    policy_loss: pd.DataFrame,
    claims: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Fit all pure-premium candidates on one development/holdout partition."""
    train, test = split_portfolio(policy_loss, test_size)
    train_claims = claims.loc[claims["IDpol"].isin(train["IDpol"])].copy()

    poisson = smf.glm(
        formula=FREQUENCY_FORMULA,
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"]),
    ).fit()
    gamma = smf.glm(
        formula=SEVERITY_FORMULA,
        data=train_claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    log_claims = train_claims.copy()
    log_claims["LogClaimAmount"] = np.log(log_claims["ClaimAmount"])
    lognormal = smf.ols(formula=LOGNORMAL_FORMULA, data=log_claims).fit()
    smear = smearing_factor(lognormal.resid)

    tweedie = make_tweedie_pipeline()
    tweedie.fit(
        train[FEATURES],
        train["TotalLoss"] / train["Exposure"],
        model__sample_weight=train["Exposure"],
    )

    predictions = test[["IDpol", "Exposure", "TotalLoss"]].copy()
    predictions["PoissonAnnualFrequency"] = poisson.predict(
        test, offset=np.zeros(len(test))
    )
    predictions["GammaExpectedSeverity"] = gamma.predict(test)
    predictions["LognormalExpectedSeverity"] = (
        np.exp(lognormal.predict(test)) * smear
    )
    predictions["PoissonGammaAnnualPurePremium"] = (
        predictions["PoissonAnnualFrequency"]
        * predictions["GammaExpectedSeverity"]
    )
    predictions["PoissonLognormalAnnualPurePremium"] = (
        predictions["PoissonAnnualFrequency"]
        * predictions["LognormalExpectedSeverity"]
    )
    predictions["TweedieAnnualPurePremium"] = np.clip(
        tweedie.predict(test[FEATURES]), 1e-12, None
    )
    for name in ("PoissonGamma", "PoissonLognormal", "Tweedie"):
        predictions[f"{name}ExpectedLoss"] = (
            predictions[f"{name}AnnualPurePremium"] * predictions["Exposure"]
        )

    metrics = {
        "split": {
            "test_size": test_size,
            "training_policies": len(train),
            "test_policies": len(test),
            "training_claims": len(train_claims),
        },
        "tweedie_power": TWEEDIE_POWER,
        "lognormal_smearing_factor": smear,
    }
    for name, key in (
        ("PoissonGamma", "poisson_gamma"),
        ("PoissonLognormal", "poisson_lognormal"),
        ("Tweedie", "tweedie"),
    ):
        metrics[key] = evaluate_policy_loss(
            predictions["TotalLoss"], predictions[f"{name}ExpectedLoss"]
        )
    tables = {"predictions": predictions}
    for name, key in (
        ("PoissonGamma", "poisson_gamma"),
        ("PoissonLognormal", "poisson_lognormal"),
        ("Tweedie", "tweedie"),
    ):
        tables[f"{key}_calibration"] = premium_calibration(
            predictions["TotalLoss"],
            predictions[f"{name}ExpectedLoss"],
            predictions["Exposure"],
            predictions[f"{name}AnnualPurePremium"],
        )
    return metrics, tables


def fit_full_tweedie(policy_loss: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """Refit direct Tweedie on all policies and return annual premiums."""
    model = make_tweedie_pipeline()
    model.fit(
        policy_loss[FEATURES],
        policy_loss["TotalLoss"] / policy_loss["Exposure"],
        model__sample_weight=policy_loss["Exposure"],
    )
    predictions = policy_loss[["IDpol", "Exposure", "TotalLoss"]].copy()
    predictions["TweedieAnnualPurePremium"] = np.clip(
        model.predict(policy_loss[FEATURES]), 1e-12, None
    )
    predictions["TweedieExpectedLoss"] = (
        predictions["TweedieAnnualPurePremium"] * predictions["Exposure"]
    )
    return model, predictions


def _render_report(metrics: dict) -> str:
    lines = [
        "# Direct Tweedie pure-premium model",
        "",
        f"Training policies: {metrics['split']['training_policies']:,}",
        f"Held-out policies: {metrics['split']['test_policies']:,}",
        f"Tweedie variance power: {metrics['tweedie_power']:.1f}",
        "",
        "| Model | MAE | RMSE | Tweedie deviance | Predicted loss | Observed ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("poisson_gamma", "Poisson × Gamma"),
        ("poisson_lognormal", "Poisson × Lognormal"),
        ("tweedie", "Direct Tweedie"),
    ):
        item = metrics[key]
        lines.append(
            f"| {label} | €{item['mae']:,.2f} | €{item['rmse']:,.2f} | "
            f"{item['mean_tweedie_deviance']:.6f} | "
            f"€{item['predicted_aggregate_loss']:,.2f} | {item['predicted_observed_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All comparisons use one policy-level holdout and exposure-adjusted expected losses.",
            "The direct Tweedie response is annual loss cost, fitted with exposure weights.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(data_dir: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    policy_loss = pd.read_parquet(data_dir / "policy_loss.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    metrics, tables = fit_holdout_models(policy_loss, claims, test_size)
    full_model, full_predictions = fit_full_tweedie(policy_loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_parquet(
        output_dir / "holdout_predictions.parquet", index=False
    )
    for name, table in tables.items():
        if name != "predictions":
            table.to_csv(output_dir / f"{name}.csv", index=False)
    full_predictions.to_parquet(
        output_dir / "portfolio_predictions.parquet", index=False
    )
    joblib.dump(full_model, output_dir / "tweedie_pipeline.joblib")
    feature_names = full_model.named_steps["preprocessor"].get_feature_names_out()
    fitted_glm = full_model.named_steps["model"]
    pd.DataFrame(
        {
            "Term": ["Intercept", *feature_names],
            "Coefficient": [fitted_glm.intercept_, *fitted_glm.coef_],
        }
    ).to_csv(output_dir / "tweedie_coefficients.csv", index=False)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(args.data_dir.resolve(), args.output_dir.resolve(), args.test_size)
