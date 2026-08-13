"""Quantify uncertainty in model comparisons, calibration, and EVT estimates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPORTS: Final = PROJECT_ROOT / "reports"
DEFAULT_CLAIMS_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_OUTPUT_DIR: Final = REPORTS / "statistical_rigor"
RANDOM_STATE: Final = 42
DEFAULT_BOOTSTRAPS: Final = 500
EVT_BOOTSTRAPS: Final = 300


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Return percentile interval and probability that an effect is below zero."""
    if values.size == 0 or not 0 < confidence < 1:
        raise ValueError("A nonempty sample and valid confidence level are required")
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "probability_below_zero": float(np.mean(values < 0)),
    }


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    metric_a: Callable[[pd.DataFrame], float],
    metric_b: Callable[[pd.DataFrame], float],
    repetitions: int = DEFAULT_BOOTSTRAPS,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Bootstrap paired metric differences while preserving policy clusters."""
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    group_positions = frame.groupby(cluster_column, sort=False).indices
    groups = [np.asarray(positions) for positions in group_positions.values()]
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        row_positions = np.concatenate([groups[position] for position in selected])
        sample = frame.iloc[row_positions]
        differences[index] = metric_b(sample) - metric_a(sample)
    return differences


def bootstrap_aggregate_ratio(
    frame: pd.DataFrame,
    cluster_column: str,
    actual_column: str,
    predicted_column: str,
    repetitions: int = DEFAULT_BOOTSTRAPS,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    groups = frame.groupby(cluster_column, sort=False)[[actual_column, predicted_column]].sum()
    values = groups.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.integers(0, len(values), size=len(values))
        sample = values[selected]
        ratios[index] = sample[:, 0].sum() / sample[:, 1].sum()
    return ratios


def evt_bootstrap(
    amounts: pd.Series,
    quantile: float,
    repetitions: int = EVT_BOOTSTRAPS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap GPD parameters while re-estimating the threshold each time."""
    values = amounts.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repetitions):
        sample = rng.choice(values, size=len(values), replace=True)
        threshold = float(np.quantile(sample, quantile))
        excess = sample[sample > threshold] - threshold
        shape, _, scale = genpareto.fit(excess, floc=0)
        rows.append(
            {
                "Threshold": threshold,
                "Shape": float(shape),
                "Scale": float(scale),
                "FiniteMean": bool(shape < 1),
                "FiniteVariance": bool(shape < 0.5),
            }
        )
    return pd.DataFrame(rows)


def metric_intervals(
    frequency: pd.DataFrame,
    severity: pd.DataFrame,
    premium: pd.DataFrame,
    repetitions: int,
) -> tuple[pd.DataFrame, dict]:
    comparisons = []
    raw = {}
    frequency_diff = paired_cluster_bootstrap(
        frequency,
        "IDpol",
        lambda sample: mean_poisson_deviance(
            sample["ClaimNb"], sample["PoissonExpectedClaims"]
        ),
        lambda sample: mean_poisson_deviance(
            sample["ClaimNb"], sample["HistGradientBoostingExpectedClaims"]
        ),
        repetitions,
        seed=42,
    )
    severity_diff = paired_cluster_bootstrap(
        severity,
        "IDpol",
        lambda sample: mean_absolute_error(
            sample["ClaimAmount"], sample["GammaPrediction"]
        ),
        lambda sample: mean_absolute_error(
            sample["ClaimAmount"], sample["XGBoostPrediction"]
        ),
        repetitions,
        seed=43,
    )
    premium_diff = paired_cluster_bootstrap(
        premium,
        "IDpol",
        lambda sample: mean_tweedie_deviance(
            sample["TotalLoss"], sample["TweedieExpectedLoss"], power=1.5
        ),
        lambda sample: mean_tweedie_deviance(
            sample["TotalLoss"], sample["GBMComponentExpectedLoss"], power=1.5
        ),
        repetitions,
        seed=44,
    )
    for name, traditional, ml, values in (
        ("Frequency deviance", "Poisson GLM", "HistGradientBoosting", frequency_diff),
        ("Severity MAE", "Gamma GLM", "XGBoost", severity_diff),
        ("Pure-premium deviance", "Tweedie GLM", "GBM Components", premium_diff),
    ):
        interval = percentile_interval(values)
        comparisons.append(
            {
                "Comparison": name,
                "TraditionalModel": traditional,
                "MLModel": ml,
                "DifferenceDefinition": "ML minus traditional",
                **interval,
                "StatisticallyStableMLImprovement": bool(interval["upper"] < 0),
            }
        )
        raw[name] = values
    return pd.DataFrame(comparisons), raw


def _render_report(
    comparisons: pd.DataFrame,
    calibration: pd.DataFrame,
    evt_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Statistical uncertainty analysis",
        "",
        "## Paired model differences",
        "",
        "| Comparison | Difference | 95% interval | P(diff < 0) | Stable ML improvement |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.Comparison} | {row.mean:.6f} | [{row.lower:.6f}, {row.upper:.6f}] | "
            f"{row.probability_below_zero:.2%} | {row.StatisticallyStableMLImprovement} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate calibration intervals",
            "",
            "| Model | Mean O/E | 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for row in calibration.itertuples(index=False):
        lines.append(f"| {row.Model} | {row.mean:.4f} | [{row.lower:.4f}, {row.upper:.4f}] |")
    lines.extend(
        [
            "",
            "## EVT threshold sensitivity and uncertainty",
            "",
            "| Threshold quantile | Shape median | Shape 95% interval | P(finite mean) | P(finite variance) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evt_summary.itertuples(index=False):
        lines.append(
            f"| {row.Quantile:.1%} | {row.ShapeMedian:.4f} | "
            f"[{row.ShapeLower:.4f}, {row.ShapeUpper:.4f}] | "
            f"{row.FiniteMeanProbability:.2%} | {row.FiniteVarianceProbability:.2%} |"
        )
    lines.extend(
        [
            "",
            "Paired bootstraps preserve the policy unit, so multiple claims from one policy remain together.",
            "EVT intervals include both resampling uncertainty and re-estimation of the empirical threshold.",
            "A point-estimate improvement is classified as stable only when its entire paired 95% interval favors ML.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(claims_path: Path, output_dir: Path, repetitions: int) -> dict:
    frequency_traditional = pd.read_parquet(
        REPORTS / "traditional_frequency" / "predictions.parquet"
    )
    frequency_ml = pd.read_parquet(REPORTS / "ml_frequency" / "predictions.parquet")
    frequency = frequency_traditional.merge(
        frequency_ml[["IDpol", "HistGradientBoostingExpectedClaims"]],
        on="IDpol",
        validate="one_to_one",
    )
    severity_traditional = pd.read_parquet(
        REPORTS / "traditional_severity" / "predictions.parquet"
    )
    severity_ml = pd.read_parquet(REPORTS / "ml_severity" / "predictions.parquet")
    severity = severity_traditional.merge(
        severity_ml[["SourceRow", "XGBoostPrediction"]],
        on="SourceRow",
        validate="one_to_one",
    )
    premium = pd.read_parquet(REPORTS / "ml_pure_premium" / "holdout_predictions.parquet")
    comparisons, raw_differences = metric_intervals(
        frequency, severity, premium, repetitions
    )
    calibration_rows = []
    for model, column, seed in (
        ("Tweedie GLM", "TweedieExpectedLoss", 50),
        ("GBM Components", "GBMComponentExpectedLoss", 51),
        ("Direct Boosting", "DirectBoostingExpectedLoss", 52),
    ):
        ratios = bootstrap_aggregate_ratio(
            premium, "IDpol", "TotalLoss", column, repetitions, seed
        )
        calibration_rows.append({"Model": model, **percentile_interval(ratios)})
    calibration = pd.DataFrame(calibration_rows)

    amounts = pd.read_parquet(claims_path, columns=["ClaimAmount"])["ClaimAmount"]
    evt_tables = {}
    evt_rows = []
    for index, quantile in enumerate((0.90, 0.95, 0.975, 0.99)):
        table = evt_bootstrap(amounts, quantile, EVT_BOOTSTRAPS, seed=60 + index)
        evt_tables[quantile] = table
        evt_rows.append(
            {
                "Quantile": quantile,
                "ShapeMedian": float(table["Shape"].median()),
                "ShapeLower": float(table["Shape"].quantile(0.025)),
                "ShapeUpper": float(table["Shape"].quantile(0.975)),
                "ScaleMedian": float(table["Scale"].median()),
                "FiniteMeanProbability": float(table["FiniteMean"].mean()),
                "FiniteVarianceProbability": float(table["FiniteVariance"].mean()),
            }
        )
    evt_summary = pd.DataFrame(evt_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_dir / "paired_model_intervals.csv", index=False)
    calibration.to_csv(output_dir / "calibration_intervals.csv", index=False)
    evt_summary.to_csv(output_dir / "evt_uncertainty_summary.csv", index=False)
    for quantile, table in evt_tables.items():
        table.to_csv(output_dir / f"evt_q{int(quantile * 1000):03d}_bootstrap.csv", index=False)
    for name, values in raw_differences.items():
        pd.DataFrame({"Difference": values}).to_csv(
            output_dir / f"{name.lower().replace(' ', '_')}_bootstrap.csv", index=False
        )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_bootstrap_repetitions": repetitions,
        "evt_bootstrap_repetitions": EVT_BOOTSTRAPS,
        "comparisons": comparisons.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "evt": evt_summary.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(comparisons, calibration, evt_summary)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.claims_path.resolve(), args.output_dir.resolve(), args.bootstraps)
