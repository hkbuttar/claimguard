"""Analyze geographic claim frequency, severity, and pure-premium patterns."""

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
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_loss.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "geographic_risk"
DENSITY_BINS: Final = (0, 50, 200, 500, 1_000, 5_000, 10_000, 27_000)
PROFILE_ORDER: Final = (
    "Low Frequency / Low Severity",
    "High Frequency / Low Severity",
    "Low Frequency / High Severity",
    "High Frequency / High Severity",
)


def portfolio_benchmarks(policy_loss: pd.DataFrame) -> dict[str, float]:
    """Return exposure-weighted portfolio reference outcomes."""
    return {
        "frequency": float(
            policy_loss["ClaimNb"].sum() / policy_loss["Exposure"].sum()
        ),
        "severity": float(
            policy_loss["TotalLoss"].sum() / policy_loss["ObservedClaimNb"].sum()
        ),
        "pure_premium": float(
            policy_loss["TotalLoss"].sum() / policy_loss["Exposure"].sum()
        ),
    }


def risk_profile(
    frequency: pd.Series,
    severity: pd.Series,
    frequency_benchmark: float,
    severity_benchmark: float,
) -> pd.Series:
    high_frequency = frequency.ge(frequency_benchmark)
    high_severity = severity.ge(severity_benchmark)
    labels = np.select(
        [
            ~high_frequency & ~high_severity,
            high_frequency & ~high_severity,
            ~high_frequency & high_severity,
            high_frequency & high_severity,
        ],
        PROFILE_ORDER,
        default="Unknown",
    )
    return pd.Series(labels, index=frequency.index, dtype="object")


def summarize_geography(
    policy_loss: pd.DataFrame,
    group: pd.Series,
    group_name: str,
    benchmarks: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Summarize exposure and linked losses for a geographic grouping."""
    reference = benchmarks or portfolio_benchmarks(policy_loss)
    frame = policy_loss.assign(_Geography=group)
    summary = (
        frame.groupby("_Geography", observed=True, dropna=False)
        .agg(
            Policies=("IDpol", "size"),
            Exposure=("Exposure", "sum"),
            RecordedClaims=("ClaimNb", "sum"),
            LinkedClaims=("ObservedClaimNb", "sum"),
            ObservedLoss=("TotalLoss", "sum"),
            MeanDensity=("Density", "mean"),
        )
        .reset_index()
        .rename(columns={"_Geography": group_name})
    )
    summary[group_name] = summary[group_name].astype("string").fillna("Missing")
    summary["ClaimFrequency"] = summary["RecordedClaims"] / summary["Exposure"]
    summary["ClaimSeverity"] = (
        summary["ObservedLoss"] / summary["LinkedClaims"].replace(0, np.nan)
    )
    summary["PurePremium"] = summary["ObservedLoss"] / summary["Exposure"]
    summary["FrequencyRelativity"] = summary["ClaimFrequency"] / reference["frequency"]
    summary["SeverityRelativity"] = summary["ClaimSeverity"] / reference["severity"]
    summary["PurePremiumRelativity"] = (
        summary["PurePremium"] / reference["pure_premium"]
    )
    summary["RiskProfile"] = risk_profile(
        summary["ClaimFrequency"],
        summary["ClaimSeverity"],
        reference["frequency"],
        reference["severity"],
    )
    return summary


def density_trends(density_summary: pd.DataFrame) -> dict[str, float]:
    """Quantify monotonic association between density and insurance outcomes."""
    return {
        "frequency_spearman": float(
            density_summary["MeanDensity"].corr(
                density_summary["ClaimFrequency"], method="spearman"
            )
        ),
        "severity_spearman": float(
            density_summary["MeanDensity"].corr(
                density_summary["ClaimSeverity"], method="spearman"
            )
        ),
        "pure_premium_spearman": float(
            density_summary["MeanDensity"].corr(
                density_summary["PurePremium"], method="spearman"
            )
        ),
    }


def _save_frequency_severity_chart(
    region: pd.DataFrame, benchmarks: dict[str, float], output_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(11, 8))
    sizes = 30 + 500 * region["Exposure"] / region["Exposure"].max()
    scatter = axis.scatter(
        region["ClaimFrequency"],
        region["ClaimSeverity"],
        s=sizes,
        c=region["PurePremium"],
        cmap="viridis",
        alpha=0.8,
    )
    for row in region.itertuples(index=False):
        axis.annotate(str(row.Region), (row.ClaimFrequency, row.ClaimSeverity), fontsize=7)
    axis.axvline(benchmarks["frequency"], color="black", linestyle="--")
    axis.axhline(benchmarks["severity"], color="black", linestyle="--")
    axis.set(
        xlabel="Recorded claims per policy-year",
        ylabel="Linked-claim severity (€)",
        title="Regional frequency and severity profiles",
    )
    fig.colorbar(scatter, ax=axis, label="Observed pure premium (€)")
    fig.tight_layout()
    fig.savefig(output_dir / "regional_frequency_severity.png", dpi=150)
    plt.close(fig)


def _save_density_chart(density: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, column, label in zip(
        axes,
        ("ClaimFrequency", "ClaimSeverity", "PurePremium"),
        ("Frequency", "Severity (€)", "Pure premium (€)"),
        strict=True,
    ):
        axis.plot(density["MeanDensity"], density[column], marker="o")
        axis.set_xscale("log")
        axis.set(xlabel="Mean population density (log scale)", ylabel=label)
    fig.suptitle("Insurance outcomes across density bands")
    fig.tight_layout()
    fig.savefig(output_dir / "density_trends.png", dpi=150)
    plt.close(fig)


def _extreme_group(table: pd.DataFrame, group: str, metric: str, highest: bool) -> str:
    index = table[metric].idxmax() if highest else table[metric].idxmin()
    row = table.loc[index]
    return f"{row[group]} ({row[metric]:,.4f})"


def _render_report(
    region: pd.DataFrame,
    area: pd.DataFrame,
    density: pd.DataFrame,
    benchmarks: dict[str, float],
    trends: dict[str, float],
) -> str:
    lines = [
        "# Geographic insurance risk analysis",
        "",
        f"Portfolio frequency: {benchmarks['frequency']:.4f} claims per policy-year",
        f"Portfolio linked-claim severity: €{benchmarks['severity']:,.2f}",
        f"Portfolio observed loss cost: €{benchmarks['pure_premium']:,.2f}",
        "",
        "## Regional profiles",
        "",
        "| Region | Policies | Frequency | Severity | Loss cost | Profile |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in region.itertuples(index=False):
        lines.append(
            f"| {row.Region} | {row.Policies:,} | {row.ClaimFrequency:.4f} | "
            f"€{row.ClaimSeverity:,.2f} | €{row.PurePremium:,.2f} | {row.RiskProfile} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Highest regional frequency: {_extreme_group(region, 'Region', 'ClaimFrequency', True)}",
            f"- Highest regional severity: {_extreme_group(region, 'Region', 'ClaimSeverity', True)}",
            f"- Highest area loss cost: {_extreme_group(area, 'Area', 'PurePremium', True)}",
            f"- Density-frequency rank correlation: {trends['frequency_spearman']:.4f}",
            f"- Density-severity rank correlation: {trends['severity_spearman']:.4f}",
            f"- Density-loss-cost rank correlation: {trends['pure_premium_spearman']:.4f}",
            "",
            "Region, area, and density results are descriptive associations. They do not establish that geography causes claim outcomes.",
            "Frequency and severity use different denominators: recorded policy claims for frequency and linked positive claims for severity.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(data_path: Path, output_dir: Path) -> dict:
    policy_loss = pd.read_parquet(data_path)
    benchmarks = portfolio_benchmarks(policy_loss)
    region = summarize_geography(policy_loss, policy_loss["Region"], "Region", benchmarks)
    area = summarize_geography(policy_loss, policy_loss["Area"], "Area", benchmarks)
    density_group = pd.cut(
        policy_loss["Density"], bins=DENSITY_BINS, include_lowest=True
    )
    density = summarize_geography(
        policy_loss, density_group, "DensityBand", benchmarks
    )
    trends = density_trends(density)
    output_dir.mkdir(parents=True, exist_ok=True)
    region.to_csv(output_dir / "risk_by_region.csv", index=False)
    area.to_csv(output_dir / "risk_by_area.csv", index=False)
    density.to_csv(output_dir / "risk_by_density.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
        "density_trends": trends,
        "profile_counts": {
            "region": {
                key: int(value) for key, value in region["RiskProfile"].value_counts().items()
            },
            "area": {
                key: int(value) for key, value in area["RiskProfile"].value_counts().items()
            },
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(region, area, density, benchmarks, trends)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_frequency_severity_chart(region, benchmarks, output_dir)
    _save_density_chart(density, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.data_path.resolve(), args.output_dir.resolve())
