"""Fit and evaluate portfolio-wide actuarial benchmark predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "actuarial_baselines"


@dataclass(frozen=True)
class BaselineRates:
    annual_claim_frequency: float
    mean_claim_severity: float
    median_claim_severity: float
    annual_pure_premium_mean_severity: float
    annual_pure_premium_median_severity: float


def fit_baseline_rates(
    policy_frequency: pd.DataFrame, claim_severity: pd.DataFrame
) -> BaselineRates:
    """Estimate constant portfolio frequency and severity benchmarks."""
    exposure = float(policy_frequency["Exposure"].sum())
    if exposure <= 0:
        raise ValueError("Total exposure must be positive")
    amounts = claim_severity["ClaimAmount"]
    if amounts.empty or amounts.isna().any() or amounts.le(0).any():
        raise ValueError("Severity fitting requires positive, non-missing claims")
    frequency = float(policy_frequency["ClaimNb"].sum() / exposure)
    mean_severity = float(amounts.mean())
    median_severity = float(amounts.median())
    return BaselineRates(
        annual_claim_frequency=frequency,
        mean_claim_severity=mean_severity,
        median_claim_severity=median_severity,
        annual_pure_premium_mean_severity=frequency * mean_severity,
        annual_pure_premium_median_severity=frequency * median_severity,
    )


def make_policy_predictions(
    policy_frequency: pd.DataFrame, rates: BaselineRates
) -> pd.DataFrame:
    """Predict annual and exposure-period frequency and loss per policy."""
    predictions = policy_frequency[["IDpol", "Exposure", "ClaimNb"]].copy()
    predictions["PredictedAnnualFrequency"] = rates.annual_claim_frequency
    predictions["PredictedClaimCount"] = (
        rates.annual_claim_frequency * predictions["Exposure"]
    )
    predictions["PredictedAnnualPurePremiumMean"] = (
        rates.annual_pure_premium_mean_severity
    )
    predictions["PredictedLossMean"] = (
        predictions["PredictedAnnualPurePremiumMean"] * predictions["Exposure"]
    )
    predictions["PredictedAnnualPurePremiumMedian"] = (
        rates.annual_pure_premium_median_severity
    )
    predictions["PredictedLossMedian"] = (
        predictions["PredictedAnnualPurePremiumMedian"] * predictions["Exposure"]
    )
    return predictions


def make_severity_predictions(
    claim_severity: pd.DataFrame, rates: BaselineRates
) -> pd.DataFrame:
    """Predict every observed severity with the portfolio mean and median."""
    predictions = claim_severity[
        ["SourceRow", "IDpol", "ClaimIndex", "ClaimAmount"]
    ].copy()
    predictions["PredictedSeverityMean"] = rates.mean_claim_severity
    predictions["PredictedSeverityMedian"] = rates.median_claim_severity
    return predictions


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def evaluate_baselines(
    policy_predictions: pd.DataFrame,
    severity_predictions: pd.DataFrame,
    policy_loss: pd.DataFrame,
) -> dict:
    """Calculate transparent in-sample benchmark diagnostics."""
    loss = policy_loss[["IDpol", "TotalLoss"]].merge(
        policy_predictions, on="IDpol", how="inner", validate="one_to_one"
    )
    severity_actual = severity_predictions["ClaimAmount"]
    return {
        "frequency": {
            "observed_claims": int(policy_predictions["ClaimNb"].sum()),
            "predicted_claims": float(policy_predictions["PredictedClaimCount"].sum()),
            "aggregate_bias": float(
                policy_predictions["PredictedClaimCount"].sum()
                - policy_predictions["ClaimNb"].sum()
            ),
        },
        "severity_mean": {
            "mae": _mae(severity_actual, severity_predictions["PredictedSeverityMean"]),
            "rmse": _rmse(severity_actual, severity_predictions["PredictedSeverityMean"]),
            "aggregate_bias": float(
                severity_predictions["PredictedSeverityMean"].sum()
                - severity_actual.sum()
            ),
        },
        "severity_median": {
            "mae": _mae(severity_actual, severity_predictions["PredictedSeverityMedian"]),
            "rmse": _rmse(severity_actual, severity_predictions["PredictedSeverityMedian"]),
            "aggregate_bias": float(
                severity_predictions["PredictedSeverityMedian"].sum()
                - severity_actual.sum()
            ),
        },
        "pure_premium_mean": {
            "mae": _mae(loss["TotalLoss"], loss["PredictedLossMean"]),
            "rmse": _rmse(loss["TotalLoss"], loss["PredictedLossMean"]),
            "observed_aggregate_loss": float(loss["TotalLoss"].sum()),
            "predicted_aggregate_loss": float(loss["PredictedLossMean"].sum()),
            "aggregate_bias": float(
                loss["PredictedLossMean"].sum() - loss["TotalLoss"].sum()
            ),
        },
        "pure_premium_median": {
            "mae": _mae(loss["TotalLoss"], loss["PredictedLossMedian"]),
            "rmse": _rmse(loss["TotalLoss"], loss["PredictedLossMedian"]),
            "observed_aggregate_loss": float(loss["TotalLoss"].sum()),
            "predicted_aggregate_loss": float(loss["PredictedLossMedian"].sum()),
            "aggregate_bias": float(
                loss["PredictedLossMedian"].sum() - loss["TotalLoss"].sum()
            ),
        },
    }


def _render_report(rates: BaselineRates, metrics: dict) -> str:
    return "\n".join(
        [
            "# ClaimGuard actuarial baselines",
            "",
            f"Annual claim frequency: {rates.annual_claim_frequency:.6f}",
            f"Mean claim severity: €{rates.mean_claim_severity:,.2f}",
            f"Median claim severity: €{rates.median_claim_severity:,.2f}",
            f"Annual pure premium (mean severity): €{rates.annual_pure_premium_mean_severity:,.2f}",
            f"Annual pure premium (median severity): €{rates.annual_pure_premium_median_severity:,.2f}",
            "",
            "## Aggregate diagnostics",
            "",
            f"- Observed policy claims: {metrics['frequency']['observed_claims']:,}",
            f"- Predicted policy claims: {metrics['frequency']['predicted_claims']:,.2f}",
            f"- Linked observed loss: €{metrics['pure_premium_mean']['observed_aggregate_loss']:,.2f}",
            f"- Mean-based predicted loss: €{metrics['pure_premium_mean']['predicted_aggregate_loss']:,.2f}",
            f"- Median-based predicted loss: €{metrics['pure_premium_median']['predicted_aggregate_loss']:,.2f}",
            "",
            "Recorded policy claim counts include claims without linked severity records, so frequency-based predicted aggregate loss is not expected to equal linked observed loss.",
            "",
            "Metrics are descriptive in-sample benchmarks. Model comparisons must use held-out data.",
            "",
        ]
    )


def run_baselines(data_dir: Path, output_dir: Path) -> dict:
    policy_frequency = pd.read_parquet(data_dir / "policy_frequency.parquet")
    claim_severity = pd.read_parquet(data_dir / "claim_severity.parquet")
    policy_loss = pd.read_parquet(data_dir / "policy_loss.parquet")
    rates = fit_baseline_rates(policy_frequency, claim_severity)
    policy_predictions = make_policy_predictions(policy_frequency, rates)
    severity_predictions = make_severity_predictions(claim_severity, rates)
    metrics = evaluate_baselines(policy_predictions, severity_predictions, policy_loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_predictions.to_parquet(output_dir / "policy_predictions.parquet", index=False)
    severity_predictions.to_parquet(
        output_dir / "severity_predictions.parquet", index=False
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rates": asdict(rates),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(rates, metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baselines(args.data_dir.resolve(), args.output_dir.resolve())
