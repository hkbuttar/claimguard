"""Generate exploratory portfolio statistics, tables, charts, and a report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "portfolio_analysis"

NUMERIC_POLICY_COLUMNS: Final = [
    "DrivAge",
    "VehAge",
    "VehPower",
    "BonusMalus",
    "Density",
    "Exposure",
]
CATEGORICAL_POLICY_COLUMNS: Final = ["Area", "VehBrand", "VehGas", "Region"]
FREQUENCY_GROUPS: Final = {
    "driver_age": ("DrivAge", [17, 24, 34, 44, 54, 64, 74, 84, 100]),
    "vehicle_age": ("VehAge", [-1, 1, 3, 5, 10, 15, 20, 40, 100]),
    "bonus_malus": ("BonusMalus", [49, 50, 60, 70, 80, 100, 125, 150, 230]),
    "density": ("Density", [0, 50, 200, 500, 1_000, 5_000, 10_000, 27_000]),
    "vehicle_power": ("VehPower", [3, 4, 5, 6, 7, 8, 10, 12, 15]),
}


def portfolio_frequency(policy: pd.DataFrame) -> float:
    """Return annualized claims per policy-year of exposure."""
    exposure = float(policy["Exposure"].sum())
    if exposure <= 0:
        raise ValueError("Total portfolio exposure must be positive")
    return float(policy["ClaimNb"].sum() / exposure)


def frequency_by_group(policy: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Calculate exposure-weighted frequency for each supplied group."""
    work = policy[["ClaimNb", "Exposure"]].assign(Group=group)
    summary = (
        work.groupby("Group", observed=True, dropna=False)
        .agg(Policies=("ClaimNb", "size"), Claims=("ClaimNb", "sum"), Exposure=("Exposure", "sum"))
        .reset_index()
    )
    summary["Frequency"] = summary["Claims"] / summary["Exposure"]
    summary["Group"] = summary["Group"].astype("string").fillna("Missing")
    return summary


def loss_concentration(amounts: pd.Series, shares: tuple[float, ...] = (0.10, 0.05, 0.01, 0.005)) -> pd.DataFrame:
    """Return the fraction of total loss contributed by the largest claims."""
    clean = amounts.dropna().astype(float)
    if clean.empty or (clean < 0).any() or clean.sum() <= 0:
        raise ValueError("Loss concentration requires nonnegative claims with positive total loss")
    ordered = clean.sort_values(ascending=False).reset_index(drop=True)
    total = float(ordered.sum())
    rows = []
    for share in shares:
        count = max(1, math.ceil(len(ordered) * share))
        rows.append(
            {
                "TopClaimShare": share,
                "ClaimCount": count,
                "Loss": float(ordered.iloc[:count].sum()),
                "LossShare": float(ordered.iloc[:count].sum() / total),
            }
        )
    return pd.DataFrame(rows)


def severity_statistics(amounts: pd.Series) -> dict[str, float | int]:
    clean = amounts.dropna().astype(float)
    if clean.empty or (clean <= 0).any():
        raise ValueError("Severity statistics require positive claim amounts")
    log_amounts = np.log(clean)
    return {
        "count": len(clean),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "p99_5": float(clean.quantile(0.995)),
        "maximum": float(clean.max()),
        "skewness": float(clean.skew()),
        "log_mean": float(log_amounts.mean()),
        "log_std": float(log_amounts.std()),
        "log_skewness": float(log_amounts.skew()),
    }


def _save_policy_chart(policy: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, axis in zip(NUMERIC_POLICY_COLUMNS, axes.flat, strict=True):
        axis.hist(policy[column], bins=40, color="#315b7d", alpha=0.85)
        axis.set(title=column, ylabel="Policies")
    fig.suptitle("Policy portfolio distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "policy_distributions.png", dpi=150)
    plt.close(fig)


def _save_frequency_chart(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    chart_names = ["driver_age", "vehicle_age", "bonus_malus", "region", "density", "vehicle_power", "fuel_type"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    for name, axis in zip(chart_names, axes.flat, strict=False):
        table = tables[name]
        axis.bar(range(len(table)), table["Frequency"], color="#3a7d6b")
        axis.set_xticks(range(len(table)), table["Group"], rotation=60, ha="right")
        axis.set(title=name.replace("_", " ").title(), ylabel="Claims per policy-year")
    axes.flat[-1].axis("off")
    fig.suptitle("Exposure-adjusted claim frequency", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "frequency_by_risk_factor.png", dpi=150)
    plt.close(fig)


def _save_severity_chart(amounts: pd.Series, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    upper = amounts.quantile(0.99)
    axes[0].hist(amounts.clip(upper=upper), bins=60, color="#a4573b")
    axes[0].set(title="Claim severity (capped at p99 for display)", xlabel="Claim amount (€)")
    axes[1].hist(np.log(amounts), bins=60, color="#75528a")
    axes[1].set(title="Log claim severity", xlabel="log(claim amount)")
    fig.tight_layout()
    fig.savefig(output_dir / "severity_distributions.png", dpi=150)
    plt.close(fig)


def _render_report(summary: dict, concentration: pd.DataFrame) -> str:
    severity = summary["severity"]
    lines = [
        "# ClaimGuard portfolio analysis",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Portfolio overview",
        "",
        f"- Policies: {summary['portfolio']['policies']:,}",
        f"- Policy-years of exposure: {summary['portfolio']['exposure']:,.2f}",
        f"- Recorded claims: {summary['portfolio']['recorded_claims']:,}",
        f"- Annualized frequency: {summary['portfolio']['frequency']:.4f}",
        f"- Linked positive claims: {severity['count']:,}",
        f"- Linked incurred loss: €{summary['portfolio']['linked_loss']:,.2f}",
        "",
        "## Severity",
        "",
        f"- Mean: €{severity['mean']:,.2f}",
        f"- Median: €{severity['median']:,.2f}",
        f"- Standard deviation: €{severity['std']:,.2f}",
        f"- 95th percentile: €{severity['p95']:,.2f}",
        f"- 99th percentile: €{severity['p99']:,.2f}",
        f"- Maximum: €{severity['maximum']:,.2f}",
        f"- Raw skewness: {severity['skewness']:,.2f}",
        f"- Log-severity skewness: {severity['log_skewness']:,.2f}",
        "",
        "## Large-loss concentration",
        "",
        "| Largest claims | Claim count | Share of total loss |",
        "|---:|---:|---:|",
    ]
    for row in concentration.itertuples(index=False):
        lines.append(f"| {row.TopClaimShare:.1%} | {row.ClaimCount:,} | {row.LossShare:.2%} |")
    lines.extend(
        [
            "",
            "Detailed policy summaries and frequency tables are available as CSV files alongside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_analysis(data_dir: Path, output_dir: Path) -> dict:
    policy = pd.read_parquet(data_dir / "policy_frequency.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_summary = policy[NUMERIC_POLICY_COLUMNS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    numeric_summary.to_csv(output_dir / "policy_numeric_summary.csv")
    for column in CATEGORICAL_POLICY_COLUMNS:
        policy[column].value_counts(dropna=False).rename("Policies").to_csv(
            output_dir / f"policy_{column.lower()}_counts.csv"
        )

    frequency_tables: dict[str, pd.DataFrame] = {}
    for name, (column, bins) in FREQUENCY_GROUPS.items():
        frequency_tables[name] = frequency_by_group(
            policy, pd.cut(policy[column], bins=bins, include_lowest=True)
        )
    frequency_tables["region"] = frequency_by_group(policy, policy["Region"])
    frequency_tables["fuel_type"] = frequency_by_group(policy, policy["VehGas"])
    for name, table in frequency_tables.items():
        table.to_csv(output_dir / f"frequency_by_{name}.csv", index=False)

    severity = severity_statistics(claims["ClaimAmount"])
    concentration = loss_concentration(claims["ClaimAmount"])
    concentration.to_csv(output_dir / "loss_concentration.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio": {
            "policies": len(policy),
            "exposure": float(policy["Exposure"].sum()),
            "recorded_claims": int(policy["ClaimNb"].sum()),
            "frequency": portfolio_frequency(policy),
            "linked_loss": float(claims["ClaimAmount"].sum()),
        },
        "severity": severity,
        "loss_concentration": concentration.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(summary, concentration), encoding="utf-8")
    _save_policy_chart(policy, output_dir)
    _save_frequency_chart(frequency_tables, output_dir)
    _save_severity_chart(claims["ClaimAmount"], output_dir)
    print(_render_report(summary, concentration))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_analysis(args.data_dir.resolve(), args.output_dir.resolve())
