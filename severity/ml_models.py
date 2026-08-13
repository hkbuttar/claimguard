"""Train and compare nonlinear claim-severity models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from severity.traditional_models import (
    DEFAULT_DATA_PATH,
    RANDOM_STATE,
    evaluate_severity,
    severity_calibration,
    split_claims_by_policy,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "ml_severity"
DEFAULT_TRADITIONAL_METRICS: Final = (
    PROJECT_ROOT / "reports" / "traditional_severity" / "metrics.json"
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
NUMERIC_FEATURES: Final = [
    column for column in FEATURES if column not in CATEGORICAL_FEATURES
]


def prepare_native_features(claims: pd.DataFrame) -> pd.DataFrame:
    """Return features with category dtypes for native categorical learners."""
    features = claims[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("category")
    return features


def fit_random_forest(train: pd.DataFrame) -> Pipeline:
    """Fit a one-hot encoded Random Forest severity benchmark."""
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=20,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train[FEATURES], train["ClaimAmount"])
    return model


def fit_hist_gradient_boosting(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Fit a native-categorical Gamma-loss boosting model."""
    model = HistGradientBoostingRegressor(
        loss="gamma",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=20,
        min_samples_leaf=50,
        l2_regularization=5.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )
    model.fit(prepare_native_features(train), train["ClaimAmount"])
    return model


def fit_xgboost(train: pd.DataFrame) -> XGBRegressor:
    """Fit a native-categorical Gamma-objective XGBoost model."""
    model = XGBRegressor(
        objective="reg:gamma",
        eval_metric="gamma-deviance",
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(prepare_native_features(train), train["ClaimAmount"], verbose=False)
    return model


def make_predictions(
    test: pd.DataFrame,
    random_forest: Pipeline,
    hist_model: HistGradientBoostingRegressor,
    xgb_model: XGBRegressor,
) -> pd.DataFrame:
    """Generate positive euro severity predictions for all nonlinear models."""
    predictions = test[["SourceRow", "IDpol", "ClaimIndex", "ClaimAmount"]].copy()
    native_features = prepare_native_features(test)
    predictions["RandomForestPrediction"] = np.clip(
        random_forest.predict(test[FEATURES]), 1e-12, None
    )
    predictions["HistGradientBoostingPrediction"] = np.clip(
        hist_model.predict(native_features), 1e-12, None
    )
    predictions["XGBoostPrediction"] = np.clip(
        xgb_model.predict(native_features), 1e-12, None
    )
    return predictions


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    models = {
        "random_forest": "RandomForest",
        "hist_gradient_boosting": "HistGradientBoosting",
        "xgboost": "XGBoost",
    }
    return {
        key: evaluate_severity(
            predictions["ClaimAmount"], predictions[f"{prefix}Prediction"]
        )
        for key, prefix in models.items()
    }


def build_comparison(
    ml_metrics: dict[str, dict[str, float]], traditional_metrics_path: Path
) -> pd.DataFrame:
    """Build the complete naive, actuarial, and nonlinear comparison table."""
    rows: list[dict] = []
    if traditional_metrics_path.exists():
        traditional = json.loads(traditional_metrics_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {"Model": "Mean", **traditional["mean"]},
                {"Model": "Gamma GLM", **traditional["gamma"]},
                {"Model": "Lognormal", **traditional["lognormal"]},
            ]
        )
    rows.extend(
        [
            {"Model": "Random Forest", **ml_metrics["random_forest"]},
            {
                "Model": "HistGradientBoosting",
                **ml_metrics["hist_gradient_boosting"],
            },
            {"Model": "XGBoost", **ml_metrics["xgboost"]},
        ]
    )
    return pd.DataFrame(rows)


def _render_report(comparison: pd.DataFrame, split: dict) -> str:
    lines = [
        "# Nonlinear claim-severity models",
        "",
        f"Training claims: {split['training_claims']:,}",
        f"Held-out claims: {split['test_claims']:,}",
        "",
        "| Model | MAE | RMSE | Gamma deviance | Aggregate ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Model} | €{row.mae:,.2f} | €{row.rmse:,.2f} | "
            f"{row.mean_gamma_deviance:.6f} | {row.predicted_observed_ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "All models use identical rating information and the same policy-grouped holdout.",
            "Model selection is deferred until large-loss and tail performance are evaluated.",
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
    claims = pd.read_parquet(data_path)
    train, test = split_claims_by_policy(claims, test_size)
    random_forest = fit_random_forest(train)
    hist_model = fit_hist_gradient_boosting(train)
    xgb_model = fit_xgboost(train)
    predictions = make_predictions(test, random_forest, hist_model, xgb_model)
    metrics = evaluate_predictions(predictions)
    comparison = build_comparison(metrics, traditional_metrics_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    for key, prefix in (
        ("random_forest", "RandomForest"),
        ("hist_gradient_boosting", "HistGradientBoosting"),
        ("xgboost", "XGBoost"),
    ):
        severity_calibration(
            predictions["ClaimAmount"], predictions[f"{prefix}Prediction"]
        ).to_csv(output_dir / f"{key}_calibration.csv", index=False)
    joblib.dump(random_forest, output_dir / "random_forest.joblib")
    joblib.dump(hist_model, output_dir / "hist_gradient_boosting.joblib")
    xgb_model.save_model(output_dir / "xgboost.json")
    split = {
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "training_claims": len(train),
        "test_claims": len(test),
        "training_policies": int(train["IDpol"].nunique()),
        "test_policies": int(test["IDpol"].nunique()),
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(comparison, split)
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
