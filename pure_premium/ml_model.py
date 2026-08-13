"""Compare direct and component-based nonlinear pure-premium models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from frequency.ml_models import (
    fit_hist_gradient_boosting as fit_frequency_hist,
)
from frequency.ml_models import prepare_features as prepare_frequency_features
from frequency.traditional_models import RANDOM_STATE, split_portfolio
from pure_premium.tweedie_model import (
    TWEEDIE_POWER,
    evaluate_policy_loss,
    premium_calibration,
)
from severity.ml_models import (
    fit_hist_gradient_boosting as fit_severity_hist,
)
from severity.ml_models import prepare_native_features as prepare_severity_features

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "ml_pure_premium"
DEFAULT_TRADITIONAL_PREDICTIONS: Final = (
    PROJECT_ROOT / "reports" / "tweedie_pure_premium" / "holdout_predictions.parquet"
)


def fit_direct_boosting(train: pd.DataFrame) -> XGBRegressor:
    """Fit direct annual loss cost using a compound Tweedie objective."""
    model = XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=TWEEDIE_POWER,
        eval_metric="tweedie-nloglik@1.5",
        n_estimators=500,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(
        prepare_severity_features(train),
        train["TotalLoss"] / train["Exposure"],
        sample_weight=train["Exposure"],
        verbose=False,
    )
    return model


def normalized_gini(actual: pd.Series, score: pd.Series) -> float:
    """Return normalized concentration Gini, where perfect ranking equals one."""
    observed = actual.to_numpy(dtype=float)
    estimate = score.to_numpy(dtype=float)
    if observed.sum() <= 0 or len(observed) < 2:
        raise ValueError("Normalized Gini requires positive aggregate loss")

    def concentration(values: np.ndarray, ranking: np.ndarray) -> float:
        order = np.argsort(-ranking, kind="stable")
        cumulative_loss = np.concatenate(([0.0], np.cumsum(values[order]) / values.sum()))
        population = np.linspace(0.0, 1.0, len(values) + 1)
        return float(2 * np.trapezoid(cumulative_loss - population, population))

    perfect = concentration(observed, observed)
    if perfect == 0:
        raise ValueError("Normalized Gini is undefined for uniform losses")
    return concentration(observed, estimate) / perfect


def risk_ranking_metrics(actual: pd.Series, annual_premium: pd.Series) -> dict[str, float]:
    """Measure rank discrimination and loss captured by the highest-risk policies."""
    count = max(1, math.ceil(len(actual) * 0.10))
    order = annual_premium.sort_values(ascending=False).index[:count]
    return {
        "normalized_gini": normalized_gini(actual, annual_premium),
        "top_decile_loss_capture": float(actual.loc[order].sum() / actual.sum()),
    }


def fit_ml_holdout(
    policy_loss: pd.DataFrame,
    claims: pd.DataFrame,
    traditional_predictions: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[dict, dict[str, pd.DataFrame], tuple[object, object, XGBRegressor]]:
    """Fit nonlinear candidates and compare all models on the established holdout."""
    train, test = split_portfolio(policy_loss, test_size)
    train_claims = claims.loc[claims["IDpol"].isin(train["IDpol"])].copy()
    frequency_hist = fit_frequency_hist(train)
    severity_hist = fit_severity_hist(train_claims)
    direct_boosting = fit_direct_boosting(train)

    predictions = test[["IDpol", "Exposure", "TotalLoss"]].copy()
    predictions["GBMAnnualFrequency"] = frequency_hist.predict(
        prepare_frequency_features(test)
    )
    predictions["GBMExpectedSeverity"] = severity_hist.predict(
        prepare_severity_features(test)
    )
    predictions["GBMComponentAnnualPurePremium"] = (
        predictions["GBMAnnualFrequency"] * predictions["GBMExpectedSeverity"]
    )
    predictions["DirectBoostingAnnualPurePremium"] = np.clip(
        direct_boosting.predict(prepare_severity_features(test)), 1e-12, None
    )
    for name in ("GBMComponent", "DirectBoosting"):
        predictions[f"{name}ExpectedLoss"] = (
            predictions[f"{name}AnnualPurePremium"] * predictions["Exposure"]
        )

    traditional_columns = [
        "IDpol",
        "PoissonGammaAnnualPurePremium",
        "PoissonGammaExpectedLoss",
        "PoissonLognormalAnnualPurePremium",
        "PoissonLognormalExpectedLoss",
        "TweedieAnnualPurePremium",
        "TweedieExpectedLoss",
    ]
    predictions = predictions.merge(
        traditional_predictions[traditional_columns],
        on="IDpol",
        how="left",
        validate="one_to_one",
    )
    if predictions[traditional_columns[1:]].isna().any(axis=None):
        raise ValueError("Traditional predictions do not match the policy holdout")

    models = {
        "poisson_gamma": "PoissonGamma",
        "poisson_lognormal": "PoissonLognormal",
        "tweedie": "Tweedie",
        "gbm_component": "GBMComponent",
        "direct_boosting": "DirectBoosting",
    }
    metrics: dict = {
        "split": {
            "test_size": test_size,
            "training_policies": len(train),
            "test_policies": len(test),
            "training_claims": len(train_claims),
        },
        "models": {},
    }
    tables = {"predictions": predictions}
    for key, name in models.items():
        model_metrics = evaluate_policy_loss(
            predictions["TotalLoss"], predictions[f"{name}ExpectedLoss"]
        )
        model_metrics.update(
            risk_ranking_metrics(
                predictions["TotalLoss"], predictions[f"{name}AnnualPurePremium"]
            )
        )
        metrics["models"][key] = model_metrics
        tables[f"{key}_calibration"] = premium_calibration(
            predictions["TotalLoss"],
            predictions[f"{name}ExpectedLoss"],
            predictions["Exposure"],
            predictions[f"{name}AnnualPurePremium"],
        )
    return metrics, tables, (frequency_hist, severity_hist, direct_boosting)


def fit_full_direct_boosting(policy_loss: pd.DataFrame) -> tuple[XGBRegressor, pd.DataFrame]:
    model = fit_direct_boosting(policy_loss)
    predictions = policy_loss[["IDpol", "Exposure", "TotalLoss"]].copy()
    predictions["DirectBoostingAnnualPurePremium"] = np.clip(
        model.predict(prepare_severity_features(policy_loss)), 1e-12, None
    )
    predictions["DirectBoostingExpectedLoss"] = (
        predictions["DirectBoostingAnnualPurePremium"] * predictions["Exposure"]
    )
    return model, predictions


def _comparison_table(metrics: dict) -> pd.DataFrame:
    labels = {
        "poisson_gamma": "Poisson × Gamma",
        "poisson_lognormal": "Poisson × Lognormal",
        "tweedie": "Tweedie GLM",
        "gbm_component": "GBM Frequency × GBM Severity",
        "direct_boosting": "Direct Boosting",
    }
    return pd.DataFrame(
        [{"Model": labels[key], **value} for key, value in metrics["models"].items()]
    )


def _render_report(metrics: dict) -> str:
    comparison = _comparison_table(metrics)
    lines = [
        "# Nonlinear pure-premium models",
        "",
        f"Training policies: {metrics['split']['training_policies']:,}",
        f"Held-out policies: {metrics['split']['test_policies']:,}",
        "",
        "| Model | Tweedie deviance | MAE | Aggregate ratio | Normalized Gini | Top-decile loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Model} | {row.mean_tweedie_deviance:.6f} | €{row.mae:,.2f} | "
            f"{row.predicted_observed_ratio:.4f} | {row.normalized_gini:.4f} | "
            f"{row.top_decile_loss_capture:.2%} |"
        )
    lines.extend(
        [
            "",
            "All models use the same policy holdout. Ranking metrics measure discrimination separately from calibration and average error.",
            "Direct boosting uses a Tweedie objective and exposure-weighted annual loss cost.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(
    data_dir: Path,
    output_dir: Path,
    traditional_predictions_path: Path,
    test_size: float = 0.2,
) -> dict:
    if not traditional_predictions_path.exists():
        raise FileNotFoundError(
            "Generate traditional Tweedie holdout predictions before running this comparison"
        )
    policy_loss = pd.read_parquet(data_dir / "policy_loss.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    traditional_predictions = pd.read_parquet(traditional_predictions_path)
    metrics, tables, _ = fit_ml_holdout(
        policy_loss, claims, traditional_predictions, test_size
    )
    full_model, full_predictions = fit_full_direct_boosting(policy_loss)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_parquet(
        output_dir / "holdout_predictions.parquet", index=False
    )
    for name, table in tables.items():
        if name != "predictions":
            table.to_csv(output_dir / f"{name}.csv", index=False)
    comparison = _comparison_table(metrics)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    full_predictions.to_parquet(
        output_dir / "portfolio_predictions.parquet", index=False
    )
    full_model.save_model(output_dir / "direct_boosting.json")
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
    parser.add_argument(
        "--traditional-predictions",
        type=Path,
        default=DEFAULT_TRADITIONAL_PREDICTIONS,
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        args.traditional_predictions.resolve(),
        args.test_size,
    )
