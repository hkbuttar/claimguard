"""Create interpretable policy risk segments from modeled insurance components."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import matplotlib
import numpy as np
import pandas as pd

from tail_risk.large_loss_classification import (
    make_gradient_boosting,
    prepare_native_features,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_loss.parquet"
DEFAULT_CLAIMS_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_COMPONENT_PREDICTIONS: Final = (
    PROJECT_ROOT / "reports" / "component_pure_premium" / "policy_predictions.parquet"
)
DEFAULT_LARGE_LOSS_METRICS: Final = (
    PROJECT_ROOT / "reports" / "large_loss_classification" / "metrics.json"
)
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "risk_segments"
SEGMENT_ORDER: Final = [
    "Standard Risk",
    "Frequent Claimant",
    "Catastrophic Exposure",
    "Critical Risk",
]


def assign_segments(
    annual_frequency: pd.Series,
    expected_severity: pd.Series,
    frequency_threshold: float,
    severity_threshold: float,
) -> pd.Series:
    """Assign actuarial frequency/severity quadrants."""
    high_frequency = annual_frequency.ge(frequency_threshold)
    high_severity = expected_severity.ge(severity_threshold)
    labels = np.select(
        [
            ~high_frequency & ~high_severity,
            high_frequency & ~high_severity,
            ~high_frequency & high_severity,
            high_frequency & high_severity,
        ],
        SEGMENT_ORDER,
        default="Unknown",
    )
    return pd.Series(labels, index=annual_frequency.index, dtype="object")


def premium_tiers(annual_premium: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Create stable low, medium, and high premium tiers."""
    lower = float(annual_premium.quantile(1 / 3))
    upper = float(annual_premium.quantile(2 / 3))
    tiers = pd.Series(
        np.select(
            [annual_premium.le(lower), annual_premium.le(upper)],
            ["Low", "Medium"],
            default="High",
        ),
        index=annual_premium.index,
        dtype="object",
    )
    return tiers, {"lower": lower, "upper": upper}


def summarize_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize modeled and observed insurance behavior by segment."""
    summary = (
        frame.groupby("RiskSegment", observed=True)
        .agg(
            Policies=("IDpol", "size"),
            Exposure=("Exposure", "sum"),
            PredictedAnnualFrequency=("PredictedAnnualFrequency", "mean"),
            ExpectedSeverity=("ExpectedSeverity", "mean"),
            AnnualPurePremium=("AnnualPurePremium", "mean"),
            LargeLossProbability=("LargeLossProbability", "mean"),
            ElevatedTailRiskShare=("ElevatedTailRisk", "mean"),
            PredictedLoss=("PredictedLoss", "sum"),
            RecordedClaims=("ClaimNb", "sum"),
            LinkedClaims=("ObservedClaimNb", "sum"),
            ObservedLoss=("TotalLoss", "sum"),
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )
    summary["ObservedFrequency"] = summary["RecordedClaims"] / summary["Exposure"]
    summary["ObservedSeverity"] = (
        summary["ObservedLoss"] / summary["LinkedClaims"].replace(0, np.nan)
    )
    summary["ObservedLossCost"] = summary["ObservedLoss"] / summary["Exposure"]
    summary["ObservedExpectedRatio"] = summary["ObservedLoss"] / summary["PredictedLoss"]
    return summary


def build_segments(
    policy: pd.DataFrame,
    claims: pd.DataFrame,
    component_predictions: pd.DataFrame,
    large_loss_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, object]:
    """Refit tail probability and assemble policy-level actuarial segments."""
    tail_target = claims["ClaimAmount"].gt(large_loss_threshold).astype(int)
    tail_model = make_gradient_boosting()
    tail_model.fit(prepare_native_features(claims), tail_target)
    tail_probability = tail_model.predict_proba(prepare_native_features(policy))[:, 1]

    component_columns = [
        "IDpol",
        "HistGradientBoostingAnnualFrequency",
        "HistGradientBoostingExpectedSeverity",
        "HistGradientBoostingAnnualPurePremium",
        "HistGradientBoostingExpectedLoss",
    ]
    frame = policy[
        ["IDpol", "Exposure", "ClaimNb", "ObservedClaimNb", "TotalLoss"]
    ].merge(
        component_predictions[component_columns],
        on="IDpol",
        how="inner",
        validate="one_to_one",
    )
    frame = frame.rename(
        columns={
            "HistGradientBoostingAnnualFrequency": "PredictedAnnualFrequency",
            "HistGradientBoostingExpectedSeverity": "ExpectedSeverity",
            "HistGradientBoostingAnnualPurePremium": "AnnualPurePremium",
            "HistGradientBoostingExpectedLoss": "PredictedLoss",
        }
    )
    if len(frame) != len(policy):
        raise ValueError("Component predictions do not cover the full policy portfolio")
    frame["LargeLossProbability"] = tail_probability
    frequency_threshold = float(frame["PredictedAnnualFrequency"].median())
    severity_threshold = float(frame["ExpectedSeverity"].median())
    tail_threshold = float(frame["LargeLossProbability"].quantile(0.75))
    frame["RiskSegment"] = assign_segments(
        frame["PredictedAnnualFrequency"],
        frame["ExpectedSeverity"],
        frequency_threshold,
        severity_threshold,
    )
    frame["PremiumTier"], premium_thresholds = premium_tiers(
        frame["AnnualPurePremium"]
    )
    frame["ElevatedTailRisk"] = frame["LargeLossProbability"].ge(tail_threshold)
    frame["PriorityReview"] = frame["ElevatedTailRisk"] & (
        frame["PremiumTier"].eq("High")
        | frame["RiskSegment"].isin(["Catastrophic Exposure", "Critical Risk"])
    )
    thresholds = {
        "frequency_median": frequency_threshold,
        "severity_median": severity_threshold,
        "large_loss_probability_q75": tail_threshold,
        "large_claim_amount": large_loss_threshold,
        "premium_tiers": premium_thresholds,
    }
    return frame, summarize_segments(frame), thresholds, tail_model


def _save_chart(frame: pd.DataFrame, thresholds: dict, output_dir: Path) -> None:
    sample = frame.sample(min(30_000, len(frame)), random_state=42)
    fig, axis = plt.subplots(figsize=(10, 7))
    colors = {
        "Standard Risk": "#4c956c",
        "Frequent Claimant": "#2c7da0",
        "Catastrophic Exposure": "#f4a261",
        "Critical Risk": "#c44536",
    }
    for segment in SEGMENT_ORDER:
        subset = sample.loc[sample["RiskSegment"].eq(segment)]
        axis.scatter(
            subset["PredictedAnnualFrequency"],
            subset["ExpectedSeverity"],
            s=6,
            alpha=0.25,
            color=colors[segment],
            label=segment,
        )
    axis.axvline(thresholds["frequency_median"], color="black", linestyle="--")
    axis.axhline(thresholds["severity_median"], color="black", linestyle="--")
    axis.set(
        xlabel="Predicted annual claim frequency",
        ylabel="Expected severity conditional on claim (€)",
        title="Interpretable policy risk segments",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "risk_segments.png", dpi=150)
    plt.close(fig)


def _render_report(summary: pd.DataFrame, thresholds: dict, frame: pd.DataFrame) -> str:
    lines = [
        "# Interpretable policy risk segments",
        "",
        f"High-frequency threshold: {thresholds['frequency_median']:.6f} annual claims",
        f"High-severity threshold: €{thresholds['severity_median']:,.2f}",
        f"Elevated-tail threshold: {thresholds['large_loss_probability_q75']:.2%}",
        "",
        "| Segment | Policies | Frequency | Severity | Pure premium | Tail probability | Observed loss cost | O/E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.RiskSegment} | {row.Policies:,} | {row.PredictedAnnualFrequency:.4f} | "
            f"€{row.ExpectedSeverity:,.2f} | €{row.AnnualPurePremium:,.2f} | "
            f"{row.LargeLossProbability:.2%} | €{row.ObservedLossCost:,.2f} | "
            f"{row.ObservedExpectedRatio:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Policies flagged for priority review: {int(frame['PriorityReview'].sum()):,}",
            "",
            "Segments are deterministic actuarial quadrants, not unsupervised clusters. Premium tier and tail probability are retained as separate decision dimensions.",
            "Large-loss classification showed weak holdout discrimination, so its probability is an enrichment signal rather than a standalone underwriting decision.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    policy_path: Path,
    claims_path: Path,
    component_predictions_path: Path,
    large_loss_metrics_path: Path,
    output_dir: Path,
) -> dict:
    policy = pd.read_parquet(policy_path)
    claims = pd.read_parquet(claims_path)
    component_predictions = pd.read_parquet(component_predictions_path)
    large_loss_metrics = json.loads(large_loss_metrics_path.read_text(encoding="utf-8"))
    amount_threshold = float(
        large_loss_metrics["thresholds"]["q95"]["claim_amount_threshold"]
    )
    frame, summary, thresholds, tail_model = build_segments(
        policy, claims, component_predictions, amount_threshold
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "policy_segments.parquet", index=False)
    summary.to_csv(output_dir / "segment_summary.csv", index=False)
    joblib.dump(tail_model, output_dir / "large_loss_probability_model.joblib")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "policies": len(frame),
        "priority_review_policies": int(frame["PriorityReview"].sum()),
        "segments": summary.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(summary, thresholds, frame)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_chart(frame, thresholds, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument(
        "--component-predictions", type=Path, default=DEFAULT_COMPONENT_PREDICTIONS
    )
    parser.add_argument(
        "--large-loss-metrics", type=Path, default=DEFAULT_LARGE_LOSS_METRICS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        args.policy_path.resolve(),
        args.claims_path.resolve(),
        args.component_predictions.resolve(),
        args.large_loss_metrics.resolve(),
        args.output_dir.resolve(),
    )
