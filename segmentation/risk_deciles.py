"""Evaluate pure-premium models through held-out policy risk deciles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_loss.parquet"
DEFAULT_ML_PREDICTIONS: Final = (
    PROJECT_ROOT / "reports" / "ml_pure_premium" / "holdout_predictions.parquet"
)
DEFAULT_COMPONENT_PREDICTIONS: Final = (
    PROJECT_ROOT / "reports" / "tweedie_pure_premium" / "holdout_predictions.parquet"
)
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "risk_deciles"
MODELS: Final = {
    "poisson_gamma": (
        "Poisson × Gamma",
        "PoissonGammaAnnualPurePremium",
        "PoissonGammaExpectedLoss",
        "PoissonExpectedClaims",
    ),
    "poisson_lognormal": (
        "Poisson × Lognormal",
        "PoissonLognormalAnnualPurePremium",
        "PoissonLognormalExpectedLoss",
        "PoissonExpectedClaims",
    ),
    "tweedie": (
        "Tweedie GLM",
        "TweedieAnnualPurePremium",
        "TweedieExpectedLoss",
        None,
    ),
    "gbm_component": (
        "GBM Frequency × GBM Severity",
        "GBMComponentAnnualPurePremium",
        "GBMComponentExpectedLoss",
        "GBMExpectedClaims",
    ),
    "direct_boosting": (
        "Direct Boosting",
        "DirectBoostingAnnualPurePremium",
        "DirectBoostingExpectedLoss",
        None,
    ),
}


def assign_risk_deciles(
    score: pd.Series, policy_id: pd.Series, bins: int = 10
) -> pd.Series:
    """Assign deterministic, near-equal groups from lowest to highest score."""
    if bins < 2 or len(score) < bins:
        raise ValueError("Risk segmentation requires at least one policy per group")
    if score.isna().any() or not np.isfinite(score).all():
        raise ValueError("Risk scores must be finite and non-missing")
    ordered = pd.DataFrame(
        {"Score": score.to_numpy(), "IDpol": policy_id.to_numpy()}, index=score.index
    ).sort_values(["Score", "IDpol"], kind="stable")
    labels = np.floor(np.arange(len(ordered)) * bins / len(ordered)).astype(int) + 1
    result = pd.Series(index=ordered.index, data=labels, dtype="int64")
    return result.reindex(score.index)


def summarize_deciles(
    frame: pd.DataFrame,
    annual_premium_column: str,
    expected_loss_column: str,
    expected_claims_column: str | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Aggregate insurance outcomes across equal-sized predicted-risk groups."""
    work = frame.copy()
    work["RiskDecile"] = assign_risk_deciles(
        work[annual_premium_column], work["IDpol"], bins
    )
    aggregation: dict = {
        "Policies": ("IDpol", "size"),
        "Exposure": ("Exposure", "sum"),
        "RecordedClaims": ("ClaimNb", "sum"),
        "LinkedClaims": ("ObservedClaimNb", "sum"),
        "PredictedLoss": (expected_loss_column, "sum"),
        "ObservedLoss": ("TotalLoss", "sum"),
        "MeanAnnualPremium": (annual_premium_column, "mean"),
    }
    if expected_claims_column:
        aggregation["PredictedClaims"] = (expected_claims_column, "sum")
    result = (
        work.groupby("RiskDecile", observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    if not expected_claims_column:
        result["PredictedClaims"] = np.nan
    result["AverageObservedSeverity"] = (
        result["ObservedLoss"] / result["LinkedClaims"].replace(0, np.nan)
    )
    result["ObservedLossCost"] = result["ObservedLoss"] / result["Exposure"]
    result["ExpectedLossCost"] = result["PredictedLoss"] / result["Exposure"]
    portfolio_loss_cost = result["ObservedLoss"].sum() / result["Exposure"].sum()
    result["RelativeObservedLossCost"] = (
        result["ObservedLossCost"] / portfolio_loss_cost
    )
    result["ObservedExpectedLossRatio"] = (
        result["ObservedLoss"] / result["PredictedLoss"]
    )
    return result


def decile_diagnostics(table: pd.DataFrame) -> dict[str, float | bool]:
    """Quantify monotonicity, spread, and highest-decile loss capture."""
    observed = table["ObservedLossCost"]
    return {
        "spearman_decile_vs_observed_loss_cost": float(
            table["RiskDecile"].corr(observed, method="spearman")
        ),
        "strictly_increasing_observed_loss_cost": bool(observed.is_monotonic_increasing),
        "highest_to_lowest_loss_cost_ratio": float(observed.iloc[-1] / observed.iloc[0]),
        "highest_decile_loss_capture": float(
            table["ObservedLoss"].iloc[-1] / table["ObservedLoss"].sum()
        ),
        "aggregate_observed_expected_ratio": float(
            table["ObservedLoss"].sum() / table["PredictedLoss"].sum()
        ),
    }


def prepare_analysis_frame(
    policy: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    component_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join held-out predictions to recorded and linked policy outcomes."""
    outcome_columns = [
        "IDpol",
        "ClaimNb",
        "ObservedClaimNb",
        "Exposure",
        "TotalLoss",
    ]
    prediction_columns = [
        "IDpol",
        *{
            column
            for _, annual, loss, _ in MODELS.values()
            for column in (annual, loss)
        },
        "GBMAnnualFrequency",
    ]
    frame = policy[outcome_columns].merge(
        ml_predictions[prediction_columns],
        on="IDpol",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    poisson_frequency = component_predictions[
        ["IDpol", "PoissonAnnualFrequency"]
    ].copy()
    frame = frame.merge(
        poisson_frequency, on="IDpol", how="left", validate="one_to_one"
    )
    if frame["PoissonAnnualFrequency"].isna().any():
        raise ValueError("Component frequency predictions do not match the holdout")
    frame["PoissonExpectedClaims"] = (
        frame["PoissonAnnualFrequency"] * frame["Exposure"]
    )
    frame["GBMExpectedClaims"] = frame["GBMAnnualFrequency"] * frame["Exposure"]
    return frame


def _save_chart(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for key, (label, _, _, _) in MODELS.items():
        table = tables[key]
        axes[0].plot(
            table["RiskDecile"],
            table["ObservedLossCost"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            table["RiskDecile"],
            table["ObservedExpectedLossRatio"],
            marker="o",
            label=label,
        )
    axes[0].set(
        title="Observed loss cost by model-ranked decile",
        xlabel="Risk decile",
        ylabel="Observed loss per policy-year (€)",
    )
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set(
        title="Calibration by model-ranked decile",
        xlabel="Risk decile",
        ylabel="Observed / expected loss",
    )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_deciles.png", dpi=150)
    plt.close(fig)


def _render_report(diagnostics: dict) -> str:
    lines = [
        "# Held-out policy risk deciles",
        "",
        "| Model | Rank correlation | Highest/lowest loss cost | Highest-decile loss | Aggregate O/E |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, (label, _, _, _) in MODELS.items():
        item = diagnostics[key]
        lines.append(
            f"| {label} | {item['spearman_decile_vs_observed_loss_cost']:.4f} | "
            f"{item['highest_to_lowest_loss_cost_ratio']:.2f}× | "
            f"{item['highest_decile_loss_capture']:.2%} | "
            f"{item['aggregate_observed_expected_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "D1 is the lowest predicted-risk group and D10 is the highest. Groups differ in size by at most one policy.",
            "Predicted claim counts are reported only for decomposed frequency×severity models; direct loss models do not identify a separate claim count.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    policy_path: Path,
    ml_predictions_path: Path,
    component_predictions_path: Path,
    output_dir: Path,
) -> dict:
    policy = pd.read_parquet(policy_path)
    ml_predictions = pd.read_parquet(ml_predictions_path)
    component_predictions = pd.read_parquet(component_predictions_path)
    frame = prepare_analysis_frame(policy, ml_predictions, component_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    diagnostics: dict = {}
    for key, (_, annual, loss, claims) in MODELS.items():
        table = summarize_deciles(frame, annual, loss, claims)
        tables[key] = table
        diagnostics[key] = decile_diagnostics(table)
        table.to_csv(output_dir / f"{key}_deciles.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_policies": len(frame),
        "diagnostics": diagnostics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(diagnostics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_chart(tables, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--ml-predictions", type=Path, default=DEFAULT_ML_PREDICTIONS)
    parser.add_argument(
        "--component-predictions", type=Path, default=DEFAULT_COMPONENT_PREDICTIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        args.policy_path.resolve(),
        args.ml_predictions.resolve(),
        args.component_predictions.resolve(),
        args.output_dir.resolve(),
    )
