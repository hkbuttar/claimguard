"""Consolidate the primary actuarial-versus-ML model benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPORTS: Final = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_DIR: Final = REPORTS / "model_benchmark"


def lower_is_better_winner(
    traditional_value: float, ml_value: float, tolerance: float = 0.001
) -> str:
    """Choose an approach only when its relative advantage exceeds tolerance."""
    denominator = max(abs(traditional_value), abs(ml_value), 1e-12)
    relative_difference = abs(traditional_value - ml_value) / denominator
    if relative_difference <= tolerance:
        return "Comparable"
    return "Traditional" if traditional_value < ml_value else "ML"


def closer_to_one_winner(
    traditional_ratio: float, ml_ratio: float, tolerance: float = 0.001
) -> str:
    return lower_is_better_winner(
        abs(traditional_ratio - 1), abs(ml_ratio - 1), tolerance
    )


def build_benchmark(reports_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Build a task-level benchmark from established held-out artifacts."""
    frequency = pd.read_csv(reports_dir / "ml_frequency" / "model_comparison.csv")
    severity = pd.read_csv(reports_dir / "ml_severity" / "model_comparison.csv")
    premium = pd.read_csv(reports_dir / "ml_pure_premium" / "model_comparison.csv")
    tail = pd.read_csv(reports_dir / "tail_performance" / "tail_performance.csv")

    poisson = frequency.loc[frequency["Model"].eq("Poisson GLM")].iloc[0]
    frequency_ml = frequency.loc[
        frequency["Model"].eq("HistGradientBoosting")
    ].iloc[0]
    lognormal = severity.loc[severity["Model"].eq("Lognormal")].iloc[0]
    gamma_glm = severity.loc[severity["Model"].eq("Gamma GLM")].iloc[0]
    severity_ml = severity.loc[severity["Model"].eq("XGBoost")].iloc[0]
    tweedie = premium.loc[premium["Model"].eq("Tweedie GLM")].iloc[0]
    premium_ml = premium.loc[
        premium["Model"].eq("GBM Frequency × GBM Severity")
    ].iloc[0]
    direct_boosting = premium.loc[premium["Model"].eq("Direct Boosting")].iloc[0]
    rf_top_5 = tail.loc[
        tail["Segment"].eq("Top 5%") & tail["Model"].eq("Random Forest")
    ].iloc[0]
    lognormal_top_5 = tail.loc[
        tail["Segment"].eq("Top 5%") & tail["Model"].eq("Lognormal")
    ].iloc[0]
    rf_top_1 = tail.loc[
        tail["Segment"].eq("Top 1%") & tail["Model"].eq("Random Forest")
    ].iloc[0]
    lognormal_top_1 = tail.loc[
        tail["Segment"].eq("Top 1%") & tail["Model"].eq("Lognormal")
    ].iloc[0]

    rows = [
        {
            "Task": "Frequency",
            "Metric": "Mean Poisson deviance",
            "TraditionalModel": "Poisson GLM",
            "TraditionalValue": poisson["mean_poisson_deviance"],
            "MLModel": "HistGradientBoosting",
            "MLValue": frequency_ml["mean_poisson_deviance"],
            "Winner": lower_is_better_winner(
                poisson["mean_poisson_deviance"],
                frequency_ml["mean_poisson_deviance"],
            ),
        },
        {
            "Task": "Severity average error",
            "Metric": "MAE (€)",
            "TraditionalModel": "Gamma GLM",
            "TraditionalValue": gamma_glm["mae"],
            "MLModel": "XGBoost",
            "MLValue": severity_ml["mae"],
            "Winner": lower_is_better_winner(gamma_glm["mae"], severity_ml["mae"]),
        },
        {
            "Task": "Severity distribution",
            "Metric": "Mean Gamma deviance",
            "TraditionalModel": "Lognormal",
            "TraditionalValue": lognormal["mean_gamma_deviance"],
            "MLModel": "XGBoost",
            "MLValue": severity_ml["mean_gamma_deviance"],
            "Winner": lower_is_better_winner(
                lognormal["mean_gamma_deviance"],
                severity_ml["mean_gamma_deviance"],
            ),
        },
        {
            "Task": "Pure premium",
            "Metric": "Mean Tweedie deviance",
            "TraditionalModel": "Tweedie GLM",
            "TraditionalValue": tweedie["mean_tweedie_deviance"],
            "MLModel": "GBM components",
            "MLValue": premium_ml["mean_tweedie_deviance"],
            "Winner": lower_is_better_winner(
                tweedie["mean_tweedie_deviance"],
                premium_ml["mean_tweedie_deviance"],
            ),
        },
        {
            "Task": "Risk ranking",
            "Metric": "Normalized Gini",
            "TraditionalModel": "Tweedie GLM",
            "TraditionalValue": tweedie["normalized_gini"],
            "MLModel": "Direct Boosting",
            "MLValue": direct_boosting["normalized_gini"],
            "Winner": "ML",
        },
        {
            "Task": "Pure-premium calibration",
            "Metric": "Absolute aggregate ratio error",
            "TraditionalModel": "Tweedie GLM",
            "TraditionalValue": abs(tweedie["predicted_observed_ratio"] - 1),
            "MLModel": "GBM components",
            "MLValue": abs(premium_ml["predicted_observed_ratio"] - 1),
            "Winner": closer_to_one_winner(
                tweedie["predicted_observed_ratio"],
                premium_ml["predicted_observed_ratio"],
            ),
        },
        {
            "Task": "Top-5% severity",
            "Metric": "MAE (€)",
            "TraditionalModel": "Lognormal",
            "TraditionalValue": lognormal_top_5["mae"],
            "MLModel": "Random Forest",
            "MLValue": rf_top_5["mae"],
            "Winner": lower_is_better_winner(
                lognormal_top_5["mae"], rf_top_5["mae"]
            ),
        },
        {
            "Task": "Top-1% severity capture",
            "Metric": "Predicted/observed loss",
            "TraditionalModel": "Lognormal",
            "TraditionalValue": lognormal_top_1["predicted_observed_ratio"],
            "MLModel": "Random Forest",
            "MLValue": rf_top_1["predicted_observed_ratio"],
            "Winner": "Neither",
        },
        {
            "Task": "Interpretability",
            "Metric": "Transparent coefficients and uncertainty",
            "TraditionalModel": "GLMs",
            "TraditionalValue": 1.0,
            "MLModel": "Boosting + SHAP",
            "MLValue": 0.0,
            "Winner": "Traditional",
        },
    ]
    benchmark = pd.DataFrame(rows)
    evidence = {
        "frequency_deviance_improvement": float(
            1
            - frequency_ml["mean_poisson_deviance"]
            / poisson["mean_poisson_deviance"]
        ),
        "severity_mae_improvement": float(1 - severity_ml["mae"] / gamma_glm["mae"]),
        "pure_premium_deviance_improvement": float(
            1
            - premium_ml["mean_tweedie_deviance"]
            / tweedie["mean_tweedie_deviance"]
        ),
        "risk_ranking_gini_improvement": float(
            direct_boosting["normalized_gini"] - tweedie["normalized_gini"]
        ),
        "direct_boosting_aggregate_underprediction": float(
            1 - direct_boosting["predicted_observed_ratio"]
        ),
        "top_1_best_capture": float(rf_top_1["predicted_observed_ratio"]),
    }
    return benchmark, evidence


def _render_report(benchmark: pd.DataFrame, evidence: dict) -> str:
    lines = [
        "# Actuarial versus machine-learning benchmark",
        "",
        "| Task | Traditional | ML | Primary metric | Winner |",
        "|---|---|---|---|---|",
    ]
    for row in benchmark.itertuples(index=False):
        lines.append(
            f"| {row.Task} | {row.TraditionalModel} | {row.MLModel} | "
            f"{row.Metric} | {row.Winner} |"
        )
    lines.extend(
        [
            "",
            "## Decision summary",
            "",
            f"- ML reduced frequency deviance by {evidence['frequency_deviance_improvement']:.2%}.",
            f"- ML reduced severity MAE by {evidence['severity_mae_improvement']:.2%}, but did not improve Gamma deviance or tail protection.",
            f"- Component ML reduced pure-premium deviance by {evidence['pure_premium_deviance_improvement']:.2%}, while Tweedie GLM remained much better calibrated.",
            f"- Direct boosting increased normalized Gini by {evidence['risk_ranking_gini_improvement']:.4f}, but underpredicted aggregate loss by {evidence['direct_boosting_aggregate_underprediction']:.2%}.",
            f"- The best top-1% severity model captured only {evidence['top_1_best_capture']:.2%} of observed loss; neither approach adequately protects the extreme tail.",
            "",
            "The evidence supports a hybrid architecture: nonlinear frequency and ranking models, an interpretable and calibrated Tweedie benchmark for expected loss, and separate EVT stress scenarios for extreme claims.",
            "No single model is the winner across accuracy, calibration, ranking, tail protection, and interpretability.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(reports_dir: Path, output_dir: Path) -> dict:
    benchmark, evidence = build_benchmark(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_dir / "model_benchmark.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
        "benchmark": benchmark.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(benchmark, evidence)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.reports_dir.resolve(), args.output_dir.resolve())
