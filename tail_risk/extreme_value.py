"""Estimate extreme claim severity with Peaks Over Threshold EVT."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gamma, genpareto, lognorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "extreme_value"
CANDIDATE_QUANTILES: Final = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99)
PRIMARY_QUANTILE: Final = 0.95
HIGH_QUANTILES: Final = (0.99, 0.995, 0.999, 0.9995)
RETURN_PERIODS: Final = (100, 500, 1_000, 5_000, 10_000)
LOSS_LEVELS: Final = (10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000)


def fit_gpd(amounts: pd.Series, threshold: float) -> dict[str, float | int | None]:
    """Fit a zero-location GPD to losses above a fixed threshold."""
    clean = amounts.dropna().astype(float)
    excess = clean.loc[clean.gt(threshold)] - threshold
    if len(excess) < 50:
        raise ValueError("At least 50 threshold exceedances are required")
    shape, _, scale = genpareto.fit(excess, floc=0)
    expected_excess = float(scale / (1 - shape)) if shape < 1 else None
    return {
        "threshold": float(threshold),
        "exceedances": len(excess),
        "exceedance_probability": float(len(excess) / len(clean)),
        "mean_excess": float(excess.mean()),
        "shape": float(shape),
        "scale": float(scale),
        "finite_mean": bool(shape < 1),
        "finite_variance": bool(shape < 0.5),
        "expected_excess": expected_excess,
        "expected_claim_given_exceedance": (
            float(threshold + expected_excess) if expected_excess is not None else None
        ),
    }


def gpd_survival(loss: float, fit: dict[str, float | int | None]) -> float:
    """Return unconditional probability that a claim exceeds a loss level."""
    threshold = float(fit["threshold"])
    if loss < threshold:
        raise ValueError("GPD survival is defined only at or above the threshold")
    shape = float(fit["shape"])
    scale = float(fit["scale"])
    tail_probability = float(fit["exceedance_probability"])
    excess = loss - threshold
    if abs(shape) < 1e-10:
        conditional = np.exp(-excess / scale)
    else:
        support = 1 + shape * excess / scale
        conditional = support ** (-1 / shape) if support > 0 else 0.0
    return float(tail_probability * conditional)


def gpd_quantile(probability: float, fit: dict[str, float | int | None]) -> float:
    """Return an unconditional high claim quantile from a fitted POT model."""
    if not 0 < probability < 1:
        raise ValueError("Quantile probability must be between zero and one")
    threshold = float(fit["threshold"])
    tail_probability = float(fit["exceedance_probability"])
    if 1 - probability >= tail_probability:
        raise ValueError("Requested quantile lies below the fitted tail threshold")
    shape = float(fit["shape"])
    scale = float(fit["scale"])
    ratio = (1 - probability) / tail_probability
    if abs(shape) < 1e-10:
        excess = -scale * np.log(ratio)
    else:
        excess = scale / shape * (ratio ** (-shape) - 1)
    return float(threshold + excess)


def threshold_diagnostics(amounts: pd.Series) -> pd.DataFrame:
    rows = []
    for quantile in CANDIDATE_QUANTILES:
        threshold = float(amounts.quantile(quantile))
        rows.append({"ThresholdQuantile": quantile, **fit_gpd(amounts, threshold)})
    return pd.DataFrame(rows)


def conventional_tail_comparison(
    amounts: pd.Series, evt_fit: dict[str, float | int | None]
) -> pd.DataFrame:
    """Compare marginal Gamma, lognormal, EVT, and empirical high quantiles."""
    clean = amounts.dropna().astype(float)
    gamma_shape, _, gamma_scale = gamma.fit(clean, floc=0)
    log_shape, _, log_scale = lognorm.fit(clean, floc=0)
    rows = []
    for probability in HIGH_QUANTILES:
        rows.append(
            {
                "Probability": probability,
                "Empirical": float(clean.quantile(probability)),
                "Gamma": float(gamma.ppf(probability, gamma_shape, loc=0, scale=gamma_scale)),
                "Lognormal": float(lognorm.ppf(probability, log_shape, loc=0, scale=log_scale)),
                "EVT": gpd_quantile(probability, evt_fit),
            }
        )
    return pd.DataFrame(rows)


def _save_diagnostic_charts(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(diagnostics["threshold"], diagnostics["mean_excess"], marker="o")
    axes[0].set(title="Mean excess", xlabel="Threshold (€)", ylabel="Mean excess (€)")
    axes[1].plot(diagnostics["threshold"], diagnostics["shape"], marker="o")
    axes[1].axhline(1.0, color="red", linestyle="--", label="Infinite-mean boundary")
    axes[1].set(title="GPD shape stability", xlabel="Threshold (€)", ylabel="Shape ξ")
    axes[1].legend(fontsize=8)
    axes[2].plot(diagnostics["threshold"], diagnostics["scale"], marker="o")
    axes[2].set(title="GPD scale stability", xlabel="Threshold (€)", ylabel="Scale β")
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_diagnostics.png", dpi=150)
    plt.close(fig)


def _save_quantile_chart(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    for model in ("Empirical", "Gamma", "Lognormal", "EVT"):
        axis.plot(comparison["Probability"], comparison[model], marker="o", label=model)
    axis.set_yscale("log")
    axis.set(
        title="High claim quantiles",
        xlabel="Quantile probability",
        ylabel="Claim amount (€; log scale)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "high_quantiles.png", dpi=150)
    plt.close(fig)


def _render_report(
    fit: dict[str, float | int | None],
    quantiles: pd.DataFrame,
    return_levels: pd.DataFrame,
) -> str:
    expected = fit["expected_claim_given_exceedance"]
    expected_text = f"€{expected:,.2f}" if expected is not None else "undefined (ξ ≥ 1)"
    lines = [
        "# Peaks Over Threshold extreme-value analysis",
        "",
        f"Threshold: €{fit['threshold']:,.2f}",
        f"Exceedances: {fit['exceedances']:,} ({fit['exceedance_probability']:.2%} of claims)",
        f"GPD shape ξ: {fit['shape']:.6f}",
        f"GPD scale β: €{fit['scale']:,.2f}",
        f"Finite theoretical mean: {fit['finite_mean']}",
        f"Finite theoretical variance: {fit['finite_variance']}",
        f"Expected claim conditional on exceeding threshold: {expected_text}",
        "",
        "## High quantile comparison",
        "",
        "| Quantile | Empirical | Gamma | Lognormal | EVT |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in quantiles.itertuples(index=False):
        lines.append(
            f"| {row.Probability:.2%} | €{row.Empirical:,.2f} | €{row.Gamma:,.2f} | "
            f"€{row.Lognormal:,.2f} | €{row.EVT:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Return levels",
            "",
            "| Return period (claims) | EVT loss level |",
            "|---:|---:|",
        ]
    )
    for row in return_levels.itertuples(index=False):
        lines.append(f"| {row.ReturnPeriod:,} | €{row.LossLevel:,.2f} |")
    lines.extend(
        [
            "",
            "The selected 95th-percentile threshold balances sample size with tail focus. Shape estimates are inspected across thresholds because the fitted tail is highly sensitive to threshold choice.",
            "Return levels are scenario estimates, not precise forecasts; nearby threshold fits cross the ξ = 1 infinite-mean boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(data_path: Path, output_dir: Path) -> dict:
    amounts = pd.read_parquet(data_path, columns=["ClaimAmount"])["ClaimAmount"]
    diagnostics = threshold_diagnostics(amounts)
    threshold = float(amounts.quantile(PRIMARY_QUANTILE))
    evt_fit = fit_gpd(amounts, threshold)
    quantiles = conventional_tail_comparison(amounts, evt_fit)
    return_levels = pd.DataFrame(
        {
            "ReturnPeriod": RETURN_PERIODS,
            "LossLevel": [gpd_quantile(1 - 1 / period, evt_fit) for period in RETURN_PERIODS],
        }
    )
    exceedance_probabilities = pd.DataFrame(
        {
            "LossLevel": LOSS_LEVELS,
            "ExceedanceProbability": [
                gpd_survival(loss, evt_fit) for loss in LOSS_LEVELS
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "threshold_diagnostics.csv", index=False)
    quantiles.to_csv(output_dir / "high_quantile_comparison.csv", index=False)
    return_levels.to_csv(output_dir / "return_levels.csv", index=False)
    exceedance_probabilities.to_csv(
        output_dir / "exceedance_probabilities.csv", index=False
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_quantile": PRIMARY_QUANTILE,
        "gpd": evt_fit,
        "high_quantiles": quantiles.to_dict(orient="records"),
        "return_levels": return_levels.to_dict(orient="records"),
        "exceedance_probabilities": exceedance_probabilities.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(evt_fit, quantiles, return_levels)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_diagnostic_charts(diagnostics, output_dir)
    _save_quantile_chart(quantiles, output_dir)
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
