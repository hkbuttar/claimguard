"""Audit severity model performance across ordinary and tail claims."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_TRADITIONAL_PREDICTIONS: Final = (
    PROJECT_ROOT / "reports" / "traditional_severity" / "predictions.parquet"
)
DEFAULT_ML_PREDICTIONS: Final = PROJECT_ROOT / "reports" / "ml_severity" / "predictions.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "tail_performance"
MODELS: Final = {
    "Mean": "MeanPrediction",
    "Gamma GLM": "GammaPrediction",
    "Lognormal": "LognormalPrediction",
    "Random Forest": "RandomForestPrediction",
    "HistGradientBoosting": "HistGradientBoostingPrediction",
    "XGBoost": "XGBoostPrediction",
}


def development_thresholds(
    claims: pd.DataFrame, holdout_source_rows: pd.Series
) -> dict[str, float]:
    """Calculate tail cutoffs without using held-out claim amounts."""
    development = claims.loc[~claims["SourceRow"].isin(holdout_source_rows)]
    if development.empty:
        raise ValueError("Development claims are required to define tail thresholds")
    return {
        "q90": float(development["ClaimAmount"].quantile(0.90)),
        "q95": float(development["ClaimAmount"].quantile(0.95)),
        "q99": float(development["ClaimAmount"].quantile(0.99)),
    }


def segment_masks(
    amounts: pd.Series, thresholds: dict[str, float]
) -> dict[str, pd.Series]:
    """Return the complete sample and overlapping actuarial tail segments."""
    return {
        "All Claims": pd.Series(True, index=amounts.index),
        "Bottom 90%": amounts.le(thresholds["q90"]),
        "Top 10%": amounts.gt(thresholds["q90"]),
        "Top 5%": amounts.gt(thresholds["q95"]),
        "Top 1%": amounts.gt(thresholds["q99"]),
    }


def evaluate_segment(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """Measure monetary error and aggregate protection within one segment."""
    observed = actual.to_numpy(dtype=float)
    estimate = predicted.to_numpy(dtype=float)
    if observed.size == 0:
        raise ValueError("Cannot evaluate an empty tail segment")
    observed_total = float(observed.sum())
    predicted_total = float(estimate.sum())
    return {
        "claims": len(observed),
        "mae": float(mean_absolute_error(observed, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(observed, estimate))),
        "mean_bias": float(np.mean(estimate - observed)),
        "observed_loss": observed_total,
        "predicted_loss": predicted_total,
        "aggregate_bias": predicted_total - observed_total,
        "predicted_observed_ratio": predicted_total / observed_total,
        "aggregate_underprediction_rate": 1.0 - predicted_total / observed_total,
    }


def audit_tail_performance(
    predictions: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Evaluate every severity model within every requested loss segment."""
    masks = segment_masks(predictions["ClaimAmount"], thresholds)
    rows = []
    for segment, mask in masks.items():
        for model, column in MODELS.items():
            rows.append(
                {
                    "Segment": segment,
                    "Model": model,
                    **evaluate_segment(
                        predictions.loc[mask, "ClaimAmount"],
                        predictions.loc[mask, column],
                    ),
                }
            )
    return pd.DataFrame(rows)


def merge_predictions(
    traditional: pd.DataFrame, ml: pd.DataFrame
) -> pd.DataFrame:
    """Join severity predictions and verify they describe the same holdout claims."""
    traditional_columns = [
        "SourceRow",
        "IDpol",
        "ClaimIndex",
        "ClaimAmount",
        "MeanPrediction",
        "GammaPrediction",
        "LognormalPrediction",
    ]
    ml_columns = [
        "SourceRow",
        "ClaimAmount",
        "RandomForestPrediction",
        "HistGradientBoostingPrediction",
        "XGBoostPrediction",
    ]
    merged = traditional[traditional_columns].merge(
        ml[ml_columns],
        on="SourceRow",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ml"),
    )
    if len(merged) != len(traditional) or len(merged) != len(ml):
        raise ValueError("Severity artifacts do not contain the same holdout claims")
    if not np.allclose(merged["ClaimAmount"], merged["ClaimAmount_ml"]):
        raise ValueError("Observed severity differs between model artifacts")
    return merged.drop(columns="ClaimAmount_ml")


def _save_chart(results: pd.DataFrame, output_dir: Path) -> None:
    segments = ["All Claims", "Bottom 90%", "Top 10%", "Top 5%", "Top 1%"]
    fig, axis = plt.subplots(figsize=(13, 7))
    width = 0.12
    x = np.arange(len(segments))
    for index, model in enumerate(MODELS):
        subset = results.loc[results["Model"].eq(model)].set_index("Segment")
        axis.bar(
            x + (index - 2.5) * width,
            subset.loc[segments, "predicted_observed_ratio"],
            width,
            label=model,
        )
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set_xticks(x, segments)
    axis.set(
        ylabel="Predicted / observed aggregate severity",
        title="Severity model aggregate protection by loss segment",
    )
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "tail_aggregate_performance.png", dpi=150)
    plt.close(fig)


def _render_report(results: pd.DataFrame, thresholds: dict[str, float]) -> str:
    lines = [
        "# Severity tail performance audit",
        "",
        f"Development Q90 threshold: €{thresholds['q90']:,.2f}",
        f"Development Q95 threshold: €{thresholds['q95']:,.2f}",
        f"Development Q99 threshold: €{thresholds['q99']:,.2f}",
        "",
        "| Segment | Model | Claims | MAE | RMSE | Aggregate bias | Predicted/observed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.Segment} | {row.Model} | {row.claims:,} | €{row.mae:,.2f} | "
            f"€{row.rmse:,.2f} | €{row.aggregate_bias:,.2f} | "
            f"{row.predicted_observed_ratio:.4f} |"
        )
    lines.extend(
        [
            "",
            "Tail segments overlap: the top 1% is contained within the top 5% and top 10%.",
            "Thresholds were estimated from development claims only; every metric above uses held-out claims.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    claims_path: Path,
    traditional_predictions_path: Path,
    ml_predictions_path: Path,
    output_dir: Path,
) -> dict:
    claims = pd.read_parquet(claims_path)
    traditional = pd.read_parquet(traditional_predictions_path)
    ml = pd.read_parquet(ml_predictions_path)
    predictions = merge_predictions(traditional, ml)
    thresholds = development_thresholds(claims, predictions["SourceRow"])
    results = audit_tail_performance(predictions, thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "tail_performance.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "holdout_claims": len(predictions),
        "results": results.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(results, thresholds)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_chart(results, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument(
        "--traditional-predictions",
        type=Path,
        default=DEFAULT_TRADITIONAL_PREDICTIONS,
    )
    parser.add_argument("--ml-predictions", type=Path, default=DEFAULT_ML_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(
        args.claims_path.resolve(),
        args.traditional_predictions.resolve(),
        args.ml_predictions.resolve(),
        args.output_dir.resolve(),
    )
