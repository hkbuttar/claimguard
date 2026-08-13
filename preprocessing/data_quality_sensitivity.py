"""Measure sensitivity to reasonable freMTPL2 cleaning choices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from frequency.traditional_models import split_portfolio
from portfolio.exploratory_analysis import loss_concentration
from preprocessing.audit_data import audit_tables
from pure_premium.tweedie_model import (
    evaluate_policy_loss,
    make_tweedie_pipeline,
)
from severity.ml_models import FEATURES

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR: Final = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "data_quality_sensitivity"
BENCHMARK_CLAIM_COUNT_CAP: Final = 4
BENCHMARK_CLAIM_AMOUNT_CAP: Final = 200_000.0


def build_scenarios(
    frequency: pd.DataFrame, severity: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Construct documented minimal, strict, and clipped analytical scenarios."""
    _, frequency_flags, severity_flags = audit_tables(frequency, severity)
    base_frequency = frequency.copy()
    base_frequency["IDpol"] = base_frequency["IDpol"].astype("int64")
    base_frequency["Exposure"] = base_frequency["Exposure"].clip(upper=1.0)
    base_severity = severity.loc[
        severity_flags["disposition"].ne("excluded").to_numpy()
    ].copy()
    base_severity["IDpol"] = base_severity["IDpol"].astype("int64")

    strict_frequency_mask = (
        frequency_flags["disposition"].eq("valid")
        & ~frequency_flags["claim_count_mismatch"]
    )
    strict_frequency = frequency.loc[strict_frequency_mask.to_numpy()].copy()
    strict_frequency["IDpol"] = strict_frequency["IDpol"].astype("int64")
    strict_severity_mask = severity_flags["disposition"].eq("valid")
    strict_severity = severity.loc[strict_severity_mask.to_numpy()].copy()
    strict_severity["IDpol"] = strict_severity["IDpol"].astype("int64")
    strict_severity = strict_severity.loc[
        strict_severity["IDpol"].isin(strict_frequency["IDpol"])
    ]

    benchmark_frequency = base_frequency.copy()
    benchmark_frequency["ClaimNb"] = benchmark_frequency["ClaimNb"].clip(
        upper=BENCHMARK_CLAIM_COUNT_CAP
    )
    benchmark_severity = base_severity.copy()
    benchmark_severity["ClaimAmount"] = benchmark_severity["ClaimAmount"].clip(
        upper=BENCHMARK_CLAIM_AMOUNT_CAP
    )
    return {
        "minimal": (base_frequency, base_severity),
        "strict": (strict_frequency, strict_severity),
        "benchmark_compatible": (benchmark_frequency, benchmark_severity),
    }


def make_policy_loss(
    frequency: pd.DataFrame, severity: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate scenario-specific linked severities to eligible policies."""
    matched = severity.loc[severity["IDpol"].isin(frequency["IDpol"])]
    losses = matched.groupby("IDpol", observed=True)["ClaimAmount"].sum()
    policy = frequency.copy()
    policy["TotalLoss"] = policy["IDpol"].map(losses).fillna(0.0)
    return policy


def scenario_summary(
    frequency: pd.DataFrame, severity: pd.DataFrame, policy_loss: pd.DataFrame
) -> dict[str, float | int]:
    matched = severity.loc[severity["IDpol"].isin(frequency["IDpol"])]
    concentration = loss_concentration(matched["ClaimAmount"])
    top_one = concentration.loc[
        concentration["TopClaimShare"].eq(0.01), "LossShare"
    ].item()
    return {
        "policies": len(frequency),
        "linked_claims": len(matched),
        "exposure": float(frequency["Exposure"].sum()),
        "recorded_claims": int(frequency["ClaimNb"].sum()),
        "annual_frequency": float(
            frequency["ClaimNb"].sum() / frequency["Exposure"].sum()
        ),
        "mean_severity": float(matched["ClaimAmount"].mean()),
        "maximum_severity": float(matched["ClaimAmount"].max()),
        "total_linked_loss": float(matched["ClaimAmount"].sum()),
        "observed_loss_cost": float(
            policy_loss["TotalLoss"].sum() / policy_loss["Exposure"].sum()
        ),
        "top_1_percent_loss_share": float(top_one),
    }


def evaluate_tweedie(policy_loss: pd.DataFrame) -> dict[str, float]:
    """Evaluate one direct Tweedie model using the established deterministic split."""
    train, test = split_portfolio(policy_loss)
    model = make_tweedie_pipeline()
    model.fit(
        train[FEATURES],
        train["TotalLoss"] / train["Exposure"],
        model__sample_weight=train["Exposure"],
    )
    annual = np.clip(model.predict(test[FEATURES]), 1e-12, None)
    expected = pd.Series(annual * test["Exposure"].to_numpy(), index=test.index)
    return evaluate_policy_loss(test["TotalLoss"], expected)


def _render_report(results: pd.DataFrame) -> str:
    lines = [
        "# Data-quality sensitivity analysis",
        "",
        "| Scenario | Policies | Claims | Frequency | Mean severity | Maximum | Linked loss | Top-1% loss share | Tweedie deviance | O/E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.Scenario} | {row.policies:,} | {row.linked_claims:,} | "
            f"{row.annual_frequency:.4f} | €{row.mean_severity:,.2f} | "
            f"€{row.maximum_severity:,.2f} | €{row.total_linked_loss:,.2f} | "
            f"{row.top_1_percent_loss_share:.2%} | {row.mean_tweedie_deviance:.4f} | "
            f"{row.predicted_observed_ratio:.4f} |"
        )
    minimal = results.loc[results["Scenario"].eq("Minimal")].iloc[0]
    benchmark = results.loc[results["Scenario"].eq("Benchmark Compatible")].iloc[0]
    loss_removed = 1 - benchmark["total_linked_loss"] / minimal["total_linked_loss"]
    tail_reduction = 1 - benchmark["top_1_percent_loss_share"] / minimal["top_1_percent_loss_share"]
    lines.extend(
        [
            "",
            f"Clipping claims at €{BENCHMARK_CLAIM_AMOUNT_CAP:,.0f} removes {loss_removed:.2%} of linked loss and reduces measured top-1% concentration by {tail_reduction:.2%}.",
            "Strict cleaning excludes every ambiguous policy or claim; minimal cleaning retains ambiguity flags and applies only deterministic corrections and required link exclusions.",
            "Benchmark-compatible clipping may stabilize average-loss modeling, but it changes the tail-risk question by suppressing the losses ClaimGuard is designed to study.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(raw_dir: Path, output_dir: Path) -> dict:
    frequency = pd.read_parquet(raw_dir / "freMTPL2freq.parquet")
    severity = pd.read_parquet(raw_dir / "freMTPL2sev.parquet")
    scenarios = build_scenarios(frequency, severity)
    rows = []
    for key, (scenario_frequency, scenario_severity) in scenarios.items():
        policy_loss = make_policy_loss(scenario_frequency, scenario_severity)
        rows.append(
            {
                "Scenario": key.replace("_", " ").title(),
                **scenario_summary(
                    scenario_frequency, scenario_severity, policy_loss
                ),
                **evaluate_tweedie(policy_loss),
            }
        )
    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "scenario_comparison.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_caps": {
            "claim_count": BENCHMARK_CLAIM_COUNT_CAP,
            "exposure": 1.0,
            "claim_amount": BENCHMARK_CLAIM_AMOUNT_CAP,
        },
        "scenarios": results.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(results)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.raw_dir.resolve(), args.output_dir.resolve())
