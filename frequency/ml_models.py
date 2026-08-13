"""Train exposure-aware nonlinear claim-frequency models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from frequency.traditional_models import (
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    calibration_table,
    evaluate_frequency,
    split_portfolio,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "ml_frequency"
DEFAULT_TRADITIONAL_METRICS: Final = (
    PROJECT_ROOT / "reports" / "traditional_frequency" / "metrics.json"
)
FEATURES: Final = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]
CATEGORICAL_FEATURES: Final = ["Area", "VehPower", "VehBrand", "VehGas", "Region"]


def prepare_features(policy: pd.DataFrame) -> pd.DataFrame:
    """Create a consistent dataframe with native categorical feature types."""
    features = policy[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("category")
    return features


def fit_hist_gradient_boosting(
    train: pd.DataFrame,
) -> HistGradientBoostingRegressor:
    """Fit a Poisson rate model weighted by policy exposure."""
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )
    model.fit(
        prepare_features(train),
        train["ClaimNb"] / train["Exposure"],
        sample_weight=train["Exposure"],
    )
    return model


def fit_xgboost(train: pd.DataFrame) -> XGBRegressor:
    """Fit a Poisson count model with log exposure supplied as base margin."""
    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        prepare_features(train),
        train["ClaimNb"],
        base_margin=np.log(train["Exposure"]),
        verbose=False,
    )
    return model


def make_ml_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hist_model: HistGradientBoostingRegressor,
    xgb_model: XGBRegressor,
) -> tuple[pd.DataFrame, float]:
    """Generate annual rates and exposure-period counts on the holdout sample."""
    naive_rate = float(train["ClaimNb"].sum() / train["Exposure"].sum())
    test_features = prepare_features(test)
    predictions = test[["IDpol", "Exposure", "ClaimNb"]].copy()
    predictions["NaiveAnnualRate"] = naive_rate
    predictions["NaiveExpectedClaims"] = naive_rate * predictions["Exposure"]
    predictions["HistGradientBoostingAnnualRate"] = np.clip(
        hist_model.predict(test_features), 1e-12, None
    )
    predictions["HistGradientBoostingExpectedClaims"] = (
        predictions["HistGradientBoostingAnnualRate"] * predictions["Exposure"]
    )
    predictions["XGBoostExpectedClaims"] = np.clip(
        xgb_model.predict(
            test_features, base_margin=np.log(test["Exposure"])
        ),
        1e-12,
        None,
    )
    predictions["XGBoostAnnualRate"] = np.clip(
        xgb_model.predict(test_features, base_margin=np.zeros(len(test))),
        1e-12,
        None,
    )
    return predictions, naive_rate


def evaluate_ml_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluate the naive and nonlinear models on the same policy holdout."""
    models = {
        "naive_rate": "Naive",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
    }
    return {
        key: evaluate_frequency(
            predictions["ClaimNb"],
            predictions[f"{prefix}ExpectedClaims"],
            predictions["Exposure"],
        )
        for key, prefix in models.items()
    }


def build_comparison(
    ml_metrics: dict[str, dict[str, float]], traditional_metrics_path: Path
) -> pd.DataFrame:
    """Combine available held-out metrics into the actuarial-to-ML benchmark ladder."""
    rows = [
        {"Model": "Naive Rate", **ml_metrics["naive_rate"]},
    ]
    if traditional_metrics_path.exists():
        traditional = json.loads(traditional_metrics_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {"Model": "Poisson GLM", **traditional["poisson"]},
                {
                    "Model": "Negative Binomial GLM",
                    **traditional["negative_binomial"],
                },
            ]
        )
    rows.extend(
        [
            {
                "Model": "HistGradientBoosting",
                **ml_metrics["hist_gradient_boosting"],
            },
            {"Model": "XGBoost", **ml_metrics["xgboost"]},
        ]
    )
    return pd.DataFrame(rows)


def _render_report(comparison: pd.DataFrame, naive_rate: float, split: dict) -> str:
    lines = [
        "# Nonlinear claim-frequency models",
        "",
        f"Training policies: {split['training_policies']:,}",
        f"Held-out policies: {split['test_policies']:,}",
        f"Training naive annual frequency: {naive_rate:.6f}",
        "",
        "| Model | Mean Poisson deviance | Expected claims | O/E |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Model} | {row.mean_poisson_deviance:.6f} | "
            f"{row.expected_claims:,.2f} | {row.observed_expected_ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "HistGradientBoosting models annual frequency with exposure weights; XGBoost models counts with log exposure as a Poisson base margin.",
            "All performance metrics use the same held-out policies as the traditional GLMs.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(
    data_path: Path,
    output_dir: Path,
    traditional_metrics_path: Path,
    test_size: float = 0.2,
) -> dict:
    policy = pd.read_parquet(data_path)
    train, test = split_portfolio(policy, test_size)
    hist_model = fit_hist_gradient_boosting(train)
    xgb_model = fit_xgboost(train)
    predictions, naive_rate = make_ml_predictions(train, test, hist_model, xgb_model)
    metrics = evaluate_ml_predictions(predictions)
    comparison = build_comparison(metrics, traditional_metrics_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    for key, prefix in (
        ("hist_gradient_boosting", "HistGradientBoosting"),
        ("xgboost", "XGBoost"),
    ):
        calibration_table(
            predictions["ClaimNb"],
            predictions[f"{prefix}ExpectedClaims"],
            predictions["Exposure"],
            predictions[f"{prefix}AnnualRate"],
        ).to_csv(output_dir / f"{key}_calibration.csv", index=False)
    joblib.dump(hist_model, output_dir / "hist_gradient_boosting.joblib")
    xgb_model.save_model(output_dir / "xgboost.json")
    split = {
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "training_policies": len(train),
        "test_policies": len(test),
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "naive_annual_frequency": naive_rate,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(comparison, naive_rate, split)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--traditional-metrics", type=Path, default=DEFAULT_TRADITIONAL_METRICS
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(
        args.data_path.resolve(),
        args.output_dir.resolve(),
        args.traditional_metrics.resolve(),
        args.test_size,
    )
