"""Consolidate calibration analysis across insurance modeling tasks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from frequency.traditional_models import calibration_table as frequency_calibration
from severity.traditional_models import severity_calibration
from tail_risk.large_loss_classification import probability_calibration

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPORTS: Final = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_DIR: Final = REPORTS / "calibration"
FREQUENCY_MODELS: Final = {
    "Naive Rate": ("NaiveExpectedClaims", "NaiveAnnualRate"),
    "Poisson GLM": ("PoissonExpectedClaims", "PoissonAnnualRate"),
    "Negative Binomial GLM": (
        "NegativeBinomialExpectedClaims",
        "NegativeBinomialAnnualRate",
    ),
    "HistGradientBoosting": (
        "HistGradientBoostingExpectedClaims",
        "HistGradientBoostingAnnualRate",
    ),
    "XGBoost": ("XGBoostExpectedClaims", "XGBoostAnnualRate"),
}
SEVERITY_MODELS: Final = {
    "Mean": "MeanPrediction",
    "Gamma GLM": "GammaPrediction",
    "Lognormal": "LognormalPrediction",
    "Random Forest": "RandomForestPrediction",
    "HistGradientBoosting": "HistGradientBoostingPrediction",
    "XGBoost": "XGBoostPrediction",
}
PURE_PREMIUM_MODELS: Final = {
    "Poisson × Gamma": "PoissonGammaExpectedLoss",
    "Poisson × Lognormal": "PoissonLognormalExpectedLoss",
    "Tweedie GLM": "TweedieExpectedLoss",
    "GBM Components": "GBMComponentExpectedLoss",
    "Direct Boosting": "DirectBoostingExpectedLoss",
}
LARGE_LOSS_MODELS: Final = {
    "Logistic": "LogisticProbability",
    "Gradient Boosting": "GradientBoostingProbability",
}


def aggregate_calibration(
    actual: pd.Series, predicted: pd.Series
) -> dict[str, float]:
    """Return total calibration, ratio, and mean bias."""
    observed = float(actual.sum())
    expected = float(predicted.sum())
    return {
        "observed": observed,
        "expected": expected,
        "observed_expected_ratio": observed / expected,
        "aggregate_bias": expected - observed,
        "mean_bias": float((predicted - actual).mean()),
    }


def expected_calibration_error(
    table: pd.DataFrame,
    observed_column: str,
    expected_column: str,
    weight_column: str,
) -> float:
    """Calculate weighted absolute calibration error across groups."""
    weights = table[weight_column] / table[weight_column].sum()
    return float(
        np.sum(weights * np.abs(table[observed_column] - table[expected_column]))
    )


def segment_loss_calibration(
    frame: pd.DataFrame, predicted_column: str
) -> pd.DataFrame:
    """Aggregate observed and predicted policy loss by actuarial segment."""
    table = (
        frame.groupby("RiskSegment", observed=True)
        .agg(
            Policies=("IDpol", "size"),
            Exposure=("Exposure", "sum"),
            ObservedLoss=("TotalLoss", "sum"),
            ExpectedLoss=(predicted_column, "sum"),
        )
        .reset_index()
    )
    table["ObservedLossCost"] = table["ObservedLoss"] / table["Exposure"]
    table["ExpectedLossCost"] = table["ExpectedLoss"] / table["Exposure"]
    table["ObservedExpectedRatio"] = table["ObservedLoss"] / table["ExpectedLoss"]
    return table


def load_frequency_predictions() -> pd.DataFrame:
    traditional = pd.read_parquet(REPORTS / "traditional_frequency" / "predictions.parquet")
    ml = pd.read_parquet(REPORTS / "ml_frequency" / "predictions.parquet")
    return ml.merge(
        traditional.drop(columns=["Exposure", "ClaimNb"]),
        on="IDpol",
        how="inner",
        validate="one_to_one",
    )


def load_severity_predictions() -> pd.DataFrame:
    traditional = pd.read_parquet(REPORTS / "traditional_severity" / "predictions.parquet")
    ml = pd.read_parquet(REPORTS / "ml_severity" / "predictions.parquet")
    return traditional.merge(
        ml.drop(columns=["IDpol", "ClaimIndex", "ClaimAmount"]),
        on="SourceRow",
        how="inner",
        validate="one_to_one",
    )


def analyze_frequency(frame: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    metrics = {}
    tables = {}
    for model, (expected, rate) in FREQUENCY_MODELS.items():
        table = frequency_calibration(
            frame["ClaimNb"], frame[expected], frame["Exposure"], frame[rate]
        )
        tables[model] = table
        metrics[model] = {
            **aggregate_calibration(frame["ClaimNb"], frame[expected]),
            "decile_weighted_absolute_frequency_error": expected_calibration_error(
                table, "ObservedFrequency", "ExpectedFrequency", "Exposure"
            ),
        }
    return metrics, tables


def analyze_severity(frame: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    metrics = {}
    tables = {}
    for model, prediction in SEVERITY_MODELS.items():
        table = severity_calibration(frame["ClaimAmount"], frame[prediction])
        tables[model] = table
        metrics[model] = {
            **aggregate_calibration(frame["ClaimAmount"], frame[prediction]),
            "decile_weighted_absolute_severity_error": expected_calibration_error(
                table, "ObservedSeverity", "ExpectedSeverity", "Claims"
            ),
        }
    return metrics, tables


def analyze_pure_premium(
    predictions: pd.DataFrame, segments: pd.DataFrame
) -> tuple[dict, dict[str, pd.DataFrame]]:
    frame = predictions.merge(
        segments[["IDpol", "RiskSegment"]],
        on="IDpol",
        how="left",
        validate="one_to_one",
    )
    metrics = {}
    tables = {}
    for model, prediction in PURE_PREMIUM_MODELS.items():
        table = segment_loss_calibration(frame, prediction)
        tables[model] = table
        metrics[model] = {
            **aggregate_calibration(frame["TotalLoss"], frame[prediction]),
            "segment_weighted_absolute_loss_cost_error": expected_calibration_error(
                table, "ObservedLossCost", "ExpectedLossCost", "Exposure"
            ),
        }
    return metrics, tables


def analyze_large_loss(frame: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    metrics = {}
    tables = {}
    for model, probability in LARGE_LOSS_MODELS.items():
        table = probability_calibration(frame["LargeLoss"], frame[probability])
        tables[model] = table
        metrics[model] = {
            "observed_event_rate": float(frame["LargeLoss"].mean()),
            "mean_predicted_probability": float(frame[probability].mean()),
            "observed_expected_ratio": float(
                frame["LargeLoss"].sum() / frame[probability].sum()
            ),
            "brier_score": float(
                brier_score_loss(frame["LargeLoss"], frame[probability])
            ),
            "reliability_weighted_absolute_error": expected_calibration_error(
                table, "ObservedProbability", "PredictedProbability", "Claims"
            ),
        }
    return metrics, tables


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace(" × ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _save_charts(
    tables: dict[str, dict[str, pd.DataFrame]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for model in ("Poisson GLM", "HistGradientBoosting", "XGBoost"):
        table = tables["frequency"][model]
        axes[0, 0].plot(
            table["ExpectedFrequency"], table["ObservedFrequency"], marker="o", label=model
        )
    axes[0, 0].set(title="Frequency decile calibration", xlabel="Expected", ylabel="Observed")
    for model in ("Gamma GLM", "Lognormal", "HistGradientBoosting", "XGBoost"):
        table = tables["severity"][model]
        axes[0, 1].plot(
            table["ExpectedSeverity"], table["ObservedSeverity"], marker="o", label=model
        )
    axes[0, 1].set(title="Severity decile calibration", xlabel="Expected (€)", ylabel="Observed (€)")
    for model in PURE_PREMIUM_MODELS:
        table = tables["pure_premium"][model]
        axes[1, 0].plot(
            table["ExpectedLossCost"], table["ObservedLossCost"], marker="o", label=model
        )
    axes[1, 0].set(title="Pure-premium segment calibration", xlabel="Expected (€)", ylabel="Observed (€)")
    for model in LARGE_LOSS_MODELS:
        table = tables["large_loss"][model]
        axes[1, 1].plot(
            table["PredictedProbability"], table["ObservedProbability"], marker="o", label=model
        )
    axes[1, 1].set(title="Large-loss reliability", xlabel="Predicted", ylabel="Observed")
    for axis in axes.flat:
        limits = [min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1])]
        axis.plot(limits, limits, "--", color="black", linewidth=1)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_overview.png", dpi=150)
    plt.close(fig)


def _render_report(metrics: dict) -> str:
    lines = ["# Cross-model calibration analysis", ""]
    for task, label in (
        ("frequency", "Frequency"),
        ("severity", "Severity"),
        ("pure_premium", "Pure premium"),
    ):
        lines.extend(
            [
                f"## {label}",
                "",
                "| Model | Observed | Expected | O/E | Aggregate bias |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model, item in metrics[task].items():
            lines.append(
                f"| {model} | {item['observed']:,.2f} | {item['expected']:,.2f} | "
                f"{item['observed_expected_ratio']:.4f} | {item['aggregate_bias']:,.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Large-loss probability",
            "",
            "| Model | Event rate | Mean probability | O/E | Brier score | Reliability error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, item in metrics["large_loss"].items():
        lines.append(
            f"| {model} | {item['observed_event_rate']:.4%} | "
            f"{item['mean_predicted_probability']:.4%} | "
            f"{item['observed_expected_ratio']:.4f} | {item['brier_score']:.6f} | "
            f"{item['reliability_weighted_absolute_error']:.4%} |"
        )
    lines.extend(
        [
            "",
            "All diagnostics use established holdout predictions. Pure-premium calibration is also reported across the actuarial policy segments.",
            "Ranking strength and calibration are separate properties; a model may excel at one while failing at the other.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(output_dir: Path) -> dict:
    frequency = load_frequency_predictions()
    severity = load_severity_predictions()
    pure_premium = pd.read_parquet(REPORTS / "ml_pure_premium" / "holdout_predictions.parquet")
    segments = pd.read_parquet(REPORTS / "risk_segments" / "policy_segments.parquet")
    large_loss = pd.read_parquet(
        REPORTS / "large_loss_classification" / "q95_predictions.parquet"
    )
    frequency_metrics, frequency_tables = analyze_frequency(frequency)
    severity_metrics, severity_tables = analyze_severity(severity)
    premium_metrics, premium_tables = analyze_pure_premium(pure_premium, segments)
    large_metrics, large_tables = analyze_large_loss(large_loss)
    metrics = {
        "frequency": frequency_metrics,
        "severity": severity_metrics,
        "pure_premium": premium_metrics,
        "large_loss": large_metrics,
    }
    tables = {
        "frequency": frequency_tables,
        "severity": severity_tables,
        "pure_premium": premium_tables,
        "large_loss": large_tables,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, task_tables in tables.items():
        for model, table in task_tables.items():
            table.to_csv(output_dir / f"{task}_{_slug(model)}.csv", index=False)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_charts(tables, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.output_dir.resolve())
