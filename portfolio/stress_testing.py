"""Simulate aggregate portfolio loss and quantify extreme-claim volatility."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

from tail_risk.extreme_value import fit_gpd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_frequency.parquet"
DEFAULT_CLAIMS_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "portfolio_stress"
RANDOM_STATE: Final = 42
DEFAULT_SIMULATIONS: Final = 10_000
DEFAULT_THRESHOLD_QUANTILE: Final = 0.95


def sample_gpd_excess(
    rng: np.random.Generator,
    size: int,
    shape: float,
    scale: float,
) -> np.ndarray:
    """Sample Generalized Pareto excesses by inverse transform."""
    if size < 0 or scale <= 0:
        raise ValueError("GPD sample size and scale must be valid")
    uniform = np.clip(rng.random(size), np.finfo(float).eps, 1 - np.finfo(float).eps)
    if abs(shape) < 1e-10:
        return -scale * np.log1p(-uniform)
    return scale / shape * (np.power(1 - uniform, -shape) - 1)


def simulate_portfolio_losses(
    rng: np.random.Generator,
    simulations: int,
    expected_claims: float,
    body_severities: np.ndarray,
    tail_probability: float,
    threshold: float,
    shape: float,
    scale: float,
    include_extreme_tail: bool = True,
) -> pd.DataFrame:
    """Simulate compound-Poisson aggregate loss in memory-bounded batches."""
    if simulations <= 0 or expected_claims <= 0:
        raise ValueError("Simulations and expected claim count must be positive")
    if body_severities.size == 0 or not 0 < tail_probability < 1:
        raise ValueError("Both body and tail severity distributions are required")
    records = []
    batch_size = min(500, simulations)
    completed = 0
    while completed < simulations:
        current = min(batch_size, simulations - completed)
        claim_counts = rng.poisson(expected_claims, size=current)
        tail_counts = rng.binomial(claim_counts, tail_probability)
        body_counts = claim_counts - tail_counts
        body_total = int(body_counts.sum())
        body_values = rng.choice(body_severities, size=body_total, replace=True)
        body_owner = np.repeat(np.arange(current), body_counts)
        body_loss = np.bincount(body_owner, weights=body_values, minlength=current)
        if include_extreme_tail:
            tail_total = int(tail_counts.sum())
            tail_values = threshold + sample_gpd_excess(
                rng, tail_total, shape, scale
            )
            tail_owner = np.repeat(np.arange(current), tail_counts)
            tail_loss = np.bincount(tail_owner, weights=tail_values, minlength=current)
        else:
            tail_loss = np.zeros(current)
        for index in range(current):
            records.append(
                {
                    "Simulation": completed + index,
                    "ClaimCount": int(claim_counts[index]),
                    "TailClaimCount": int(tail_counts[index]),
                    "BodyLoss": float(body_loss[index]),
                    "TailLoss": float(tail_loss[index]),
                    "TotalLoss": float(body_loss[index] + tail_loss[index]),
                }
            )
        completed += current
    return pd.DataFrame(records)


def risk_metrics(losses: pd.Series) -> dict[str, float]:
    """Calculate aggregate distribution, VaR, and Expected Shortfall metrics."""
    values = losses.to_numpy(dtype=float)
    metrics = {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
    }
    for probability in (0.95, 0.99):
        quantile = float(np.quantile(values, probability))
        metrics[f"var_{int(probability * 100)}"] = quantile
        metrics[f"expected_shortfall_{int(probability * 100)}"] = float(
            np.mean(values[values >= quantile])
        )
    return metrics


def _save_chart(
    full: pd.DataFrame, counterfactual: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    upper = full["TotalLoss"].quantile(0.995)
    axes[0].hist(
        full["TotalLoss"].clip(upper=upper) / 1_000_000,
        bins=70,
        alpha=0.65,
        label="Empirical body + EVT tail",
    )
    axes[0].hist(
        counterfactual["TotalLoss"].clip(upper=upper) / 1_000_000,
        bins=70,
        alpha=0.65,
        label="Tail removed",
    )
    axes[0].set(
        xlabel="Aggregate loss (€ millions; capped at p99.5 for display)",
        ylabel="Simulations",
        title="Simulated portfolio loss",
    )
    axes[0].legend()
    axes[1].scatter(
        full["TailClaimCount"],
        full["TotalLoss"] / 1_000_000,
        s=5,
        alpha=0.2,
    )
    axes[1].set(
        xlabel="Simulated tail claim count",
        ylabel="Aggregate loss (€ millions)",
        title="Tail-count contribution to aggregate loss",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "portfolio_loss_simulation.png", dpi=150)
    plt.close(fig)


def _render_report(result: dict) -> str:
    full = result["full_tail"]
    counterfactual = result["tail_removed"]
    impact = result["tail_impact"]
    return "\n".join(
        [
            "# Portfolio loss stress test",
            "",
            f"Simulations: {result['simulations']:,}",
            f"Expected recorded claims: {result['expected_claims']:,.2f}",
            f"EVT threshold: €{result['evt']['threshold']:,.2f}",
            f"EVT shape ξ: {result['evt']['shape']:.6f}",
            "",
            "| Scenario | Mean | Std. dev. | VaR 95 | VaR 99 | ES 95 | ES 99 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Empirical body + EVT tail | €{full['mean']:,.2f} | €{full['standard_deviation']:,.2f} | €{full['var_95']:,.2f} | €{full['var_99']:,.2f} | €{full['expected_shortfall_95']:,.2f} | €{full['expected_shortfall_99']:,.2f} |",
            f"| Extreme tail removed | €{counterfactual['mean']:,.2f} | €{counterfactual['standard_deviation']:,.2f} | €{counterfactual['var_95']:,.2f} | €{counterfactual['var_99']:,.2f} | €{counterfactual['expected_shortfall_95']:,.2f} | €{counterfactual['expected_shortfall_99']:,.2f} |",
            "",
            f"Tail share of expected loss: {impact['mean_loss_share']:.2%}",
            f"Tail share of portfolio variance: {impact['variance_share']:.2%}",
            f"Tail share of 99% Expected Shortfall: {impact['expected_shortfall_99_share']:.2%}",
            f"Monte Carlo standard error of mean loss: €{full['mean_standard_error']:,.2f}",
            "",
            "Frequency is modeled as portfolio-level Poisson using recorded claims. Severities use an empirical body and fitted GPD above the 95th percentile.",
            "The tail-removed scenario is a diagnostic counterfactual, not a forecast of achievable insurance losses.",
            "EVT parameter instability makes the high-tail risk measures scenario estimates rather than precise capital requirements.",
            "Because the fitted GPD has infinite theoretical variance, simulated standard deviation and Expected Shortfall remain intrinsically unstable even with many runs.",
            "",
        ]
    )


def run_simulation(
    policy_path: Path,
    claims_path: Path,
    output_dir: Path,
    simulations: int = DEFAULT_SIMULATIONS,
) -> dict:
    policy = pd.read_parquet(policy_path, columns=["ClaimNb", "Exposure"])
    claims = pd.read_parquet(claims_path, columns=["ClaimAmount"])
    expected_claims = float(
        policy["ClaimNb"].sum() / policy["Exposure"].sum() * policy["Exposure"].sum()
    )
    threshold = float(claims["ClaimAmount"].quantile(DEFAULT_THRESHOLD_QUANTILE))
    evt = fit_gpd(claims["ClaimAmount"], threshold)
    body = claims.loc[claims["ClaimAmount"].le(threshold), "ClaimAmount"].to_numpy()
    tail_probability = float(evt["exceedance_probability"])
    full = simulate_portfolio_losses(
        np.random.default_rng(RANDOM_STATE),
        simulations,
        expected_claims,
        body,
        tail_probability,
        threshold,
        float(evt["shape"]),
        float(evt["scale"]),
        include_extreme_tail=True,
    )
    counterfactual = simulate_portfolio_losses(
        np.random.default_rng(RANDOM_STATE),
        simulations,
        expected_claims,
        body,
        tail_probability,
        threshold,
        float(evt["shape"]),
        float(evt["scale"]),
        include_extreme_tail=False,
    )
    full_metrics = risk_metrics(full["TotalLoss"])
    counterfactual_metrics = risk_metrics(counterfactual["TotalLoss"])
    full_variance = full_metrics["standard_deviation"] ** 2
    counterfactual_variance = counterfactual_metrics["standard_deviation"] ** 2
    impact = {
        "mean_loss_share": 1 - counterfactual_metrics["mean"] / full_metrics["mean"],
        "variance_share": 1 - counterfactual_variance / full_variance,
        "expected_shortfall_99_share": (
            1
            - counterfactual_metrics["expected_shortfall_99"]
            / full_metrics["expected_shortfall_99"]
        ),
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulations": simulations,
        "random_state": RANDOM_STATE,
        "expected_claims": expected_claims,
        "evt": evt,
        "full_tail": full_metrics,
        "tail_removed": counterfactual_metrics,
        "tail_impact": impact,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    full.to_parquet(output_dir / "full_tail_simulations.parquet", index=False)
    counterfactual.to_parquet(
        output_dir / "tail_removed_simulations.parquet", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(result)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_chart(full, counterfactual, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        args.policy_path.resolve(),
        args.claims_path.resolve(),
        args.output_dir.resolve(),
        args.simulations,
    )
