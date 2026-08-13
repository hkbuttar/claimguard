"""Estimate conditional claim-severity quantiles with gradient boosting."""

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
from sklearn.metrics import mean_pinball_loss

from severity.ml_models import prepare_native_features
from severity.traditional_models import RANDOM_STATE, split_claims_by_policy

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_POLICY_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_frequency.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "quantile_severity"
QUANTILES: Final = (0.50, 0.75, 0.90, 0.95)


def fit_quantile_model(
    claims: pd.DataFrame, quantile: float
) -> HistGradientBoostingRegressor:
    """Fit one native-categorical conditional quantile model."""
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between zero and one")
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=20,
        min_samples_leaf=50,
        l2_regularization=5.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )
    model.fit(prepare_native_features(claims), claims["ClaimAmount"])
    return model


def evaluate_quantile(
    actual: pd.Series,
    predicted: pd.Series,
    quantile: float,
    baseline: float,
) -> dict[str, float]:
    """Evaluate asymmetric loss and marginal quantile coverage."""
    observed = actual.to_numpy(dtype=float)
    estimate = predicted.to_numpy(dtype=float)
    model_loss = float(mean_pinball_loss(observed, estimate, alpha=quantile))
    baseline_loss = float(
        mean_pinball_loss(observed, np.full(len(observed), baseline), alpha=quantile)
    )
    coverage = float(np.mean(observed <= estimate))
    return {
        "pinball_loss": model_loss,
        "baseline_pinball_loss": baseline_loss,
        "pinball_improvement": 1 - model_loss / baseline_loss,
        "empirical_coverage": coverage,
        "coverage_error": coverage - quantile,
        "mean_prediction": float(np.mean(estimate)),
        "median_prediction": float(np.median(estimate)),
    }


def quantile_calibration(
    actual: pd.Series,
    predicted: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Measure coverage and severity by conditional-quantile risk group."""
    frame = pd.DataFrame(
        {"Actual": actual.to_numpy(dtype=float), "Predicted": predicted.to_numpy(dtype=float)}
    )
    frame["RiskGroup"] = pd.qcut(
        frame["Predicted"], q=bins, labels=False, duplicates="drop"
    )
    frame["Covered"] = frame["Actual"].le(frame["Predicted"])
    return (
        frame.groupby("RiskGroup", observed=True)
        .agg(
            Claims=("Actual", "size"),
            MeanObservedSeverity=("Actual", "mean"),
            MeanPredictedQuantile=("Predicted", "mean"),
            EmpiricalCoverage=("Covered", "mean"),
        )
        .reset_index()
    )


def quantile_crossing_rate(predictions: pd.DataFrame) -> float:
    """Return the fraction of rows whose independently fitted quantiles cross."""
    columns = [f"Q{int(quantile * 100)}Prediction" for quantile in QUANTILES]
    values = predictions[columns].to_numpy(dtype=float)
    return float(np.mean(np.any(np.diff(values, axis=1) < 0, axis=1)))


def fit_holdout_models(
    claims: pd.DataFrame, test_size: float = 0.2
) -> tuple[dict, dict[str, pd.DataFrame], dict[str, HistGradientBoostingRegressor]]:
    train, test = split_claims_by_policy(claims, test_size)
    predictions = test[["SourceRow", "IDpol", "ClaimIndex", "ClaimAmount"]].copy()
    models: dict[str, HistGradientBoostingRegressor] = {}
    tables: dict[str, pd.DataFrame] = {}
    metrics: dict = {
        "split": {
            "test_size": test_size,
            "training_claims": len(train),
            "test_claims": len(test),
            "training_policies": int(train["IDpol"].nunique()),
            "test_policies": int(test["IDpol"].nunique()),
        },
        "quantiles": {},
    }
    test_features = prepare_native_features(test)
    for quantile in QUANTILES:
        label = f"q{int(quantile * 100)}"
        model = fit_quantile_model(train, quantile)
        prediction = np.clip(model.predict(test_features), 1e-12, None)
        column = f"Q{int(quantile * 100)}Prediction"
        predictions[column] = prediction
        baseline = float(train["ClaimAmount"].quantile(quantile))
        metrics["quantiles"][label] = {
            "development_unconditional_quantile": baseline,
            **evaluate_quantile(
                predictions["ClaimAmount"], predictions[column], quantile, baseline
            ),
        }
        tables[f"{label}_calibration"] = quantile_calibration(
            predictions["ClaimAmount"], predictions[column]
        )
        models[label] = model
    metrics["quantile_crossing_rate"] = quantile_crossing_rate(predictions)
    tables["predictions"] = predictions
    return metrics, tables, models


def fit_full_models(
    claims: pd.DataFrame, policy: pd.DataFrame
) -> tuple[dict[str, HistGradientBoostingRegressor], pd.DataFrame]:
    """Refit quantile models on every linked claim and score every policy."""
    predictions = policy[["IDpol"]].copy()
    policy_features = prepare_native_features(policy)
    models = {}
    for quantile in QUANTILES:
        label = f"q{int(quantile * 100)}"
        model = fit_quantile_model(claims, quantile)
        predictions[f"Q{int(quantile * 100)}Severity"] = np.clip(
            model.predict(policy_features), 1e-12, None
        )
        models[label] = model
    predictions["QuantileCrossing"] = np.any(
        np.diff(
            predictions[[f"Q{int(q * 100)}Severity" for q in QUANTILES]].to_numpy(),
            axis=1,
        )
        < 0,
        axis=1,
    )
    return models, predictions


def _render_report(metrics: dict) -> str:
    lines = [
        "# Conditional severity quantiles",
        "",
        f"Training claims: {metrics['split']['training_claims']:,}",
        f"Held-out claims: {metrics['split']['test_claims']:,}",
        "",
        "| Quantile | Pinball loss | Baseline loss | Improvement | Coverage | Coverage error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for quantile in QUANTILES:
        item = metrics["quantiles"][f"q{int(quantile * 100)}"]
        lines.append(
            f"| Q{int(quantile * 100)} | €{item['pinball_loss']:,.2f} | "
            f"€{item['baseline_pinball_loss']:,.2f} | {item['pinball_improvement']:.2%} | "
            f"{item['empirical_coverage']:.2%} | {item['coverage_error']:+.2%} |"
        )
    lines.extend(
        [
            "",
            f"Held-out quantile crossing rate: {metrics['quantile_crossing_rate']:.2%}",
            "",
            "Each model uses the same policy-grouped holdout. Baselines are unconditional development-sample quantiles.",
            "Quantiles are retained as independently estimated outputs; crossing is measured rather than silently rearranged.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(
    claims_path: Path,
    policy_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
) -> dict:
    claims = pd.read_parquet(claims_path)
    policy = pd.read_parquet(policy_path)
    metrics, tables, _ = fit_holdout_models(claims, test_size)
    full_models, portfolio_predictions = fit_full_models(claims, policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_parquet(
        output_dir / "holdout_predictions.parquet", index=False
    )
    for name, table in tables.items():
        if name != "predictions":
            table.to_csv(output_dir / f"{name}.csv", index=False)
    portfolio_predictions.to_parquet(
        output_dir / "portfolio_predictions.parquet", index=False
    )
    for name, model in full_models.items():
        joblib.dump(model, output_dir / f"{name}_model.joblib")
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
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(
        args.claims_path.resolve(),
        args.policy_path.resolve(),
        args.output_dir.resolve(),
        args.test_size,
    )
