"""Fit and compare traditional Gamma and lognormal claim-severity models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "traditional_severity"
RANDOM_STATE: Final = 42
MODEL_FORMULA: Final = (
    "ClaimAmount ~ C(Area) + C(VehPower) + VehAge + DrivAge + BonusMalus + "
    "C(VehBrand) + C(VehGas) + np.log1p(Density) + C(Region)"
)
LOGNORMAL_FORMULA: Final = MODEL_FORMULA.replace("ClaimAmount", "LogClaimAmount", 1)


def split_claims_by_policy(
    claims: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by policy so multiple claims from one policy cannot leak across sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=RANDOM_STATE
    )
    train_index, test_index = next(
        splitter.split(claims, groups=claims["IDpol"])
    )
    return claims.iloc[train_index].copy(), claims.iloc[test_index].copy()


def smearing_factor(log_residuals: pd.Series | np.ndarray) -> float:
    """Return Duan's nonparametric factor for retransformation to euro values."""
    residuals = np.asarray(log_residuals, dtype=float)
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Smearing residuals must be finite and nonempty")
    return float(np.mean(np.exp(residuals)))


def evaluate_severity(
    actual: pd.Series, predicted: pd.Series
) -> dict[str, float]:
    """Calculate monetary, distributional, and aggregate severity diagnostics."""
    observed = actual.to_numpy(dtype=float)
    estimate = np.clip(predicted.to_numpy(dtype=float), 1e-12, None)
    observed_total = float(observed.sum())
    predicted_total = float(estimate.sum())
    return {
        "mae": float(mean_absolute_error(observed, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(observed, estimate))),
        "mean_gamma_deviance": float(mean_gamma_deviance(observed, estimate)),
        "mean_bias": float(np.mean(estimate - observed)),
        "observed_aggregate_loss": observed_total,
        "predicted_aggregate_loss": predicted_total,
        "aggregate_bias": predicted_total - observed_total,
        "predicted_observed_ratio": predicted_total / observed_total,
    }


def severity_calibration(
    actual: pd.Series, predicted: pd.Series, bins: int = 10
) -> pd.DataFrame:
    """Aggregate observed and expected severity by predicted-risk quantile."""
    frame = pd.DataFrame(
        {
            "Actual": actual.to_numpy(dtype=float),
            "Predicted": predicted.to_numpy(dtype=float),
        }
    )
    frame["RiskDecile"] = pd.qcut(
        frame["Predicted"], q=bins, labels=False, duplicates="drop"
    )
    result = (
        frame.groupby("RiskDecile", observed=True)
        .agg(
            Claims=("Actual", "size"),
            ObservedSeverity=("Actual", "mean"),
            ExpectedSeverity=("Predicted", "mean"),
            ObservedLoss=("Actual", "sum"),
            ExpectedLoss=("Predicted", "sum"),
        )
        .reset_index()
    )
    result["ObservedExpectedRatio"] = result["ObservedLoss"] / result["ExpectedLoss"]
    return result


def _coefficient_table(result: object) -> pd.DataFrame:
    confidence = result.conf_int()  # type: ignore[attr-defined]
    return pd.DataFrame(
        {
            "Term": result.params.index,  # type: ignore[attr-defined]
            "Coefficient": result.params.to_numpy(),  # type: ignore[attr-defined]
            "StdError": result.bse.to_numpy(),  # type: ignore[attr-defined]
            "PValue": result.pvalues.to_numpy(),  # type: ignore[attr-defined]
            "ConfidenceLow": confidence.iloc[:, 0].to_numpy(),
            "ConfidenceHigh": confidence.iloc[:, 1].to_numpy(),
            "MultiplicativeEffect": np.exp(result.params.to_numpy()),  # type: ignore[attr-defined]
        }
    )


def fit_severity_models(
    claims: pd.DataFrame, test_size: float = 0.2
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Fit mean, Gamma, and bias-corrected lognormal severity models."""
    train, test = split_claims_by_policy(claims, test_size)
    gamma = smf.glm(
        formula=MODEL_FORMULA,
        data=train,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    log_train = train.copy()
    log_train["LogClaimAmount"] = np.log(log_train["ClaimAmount"])
    lognormal = smf.ols(formula=LOGNORMAL_FORMULA, data=log_train).fit()
    smear = smearing_factor(lognormal.resid)

    predictions = test[["SourceRow", "IDpol", "ClaimIndex", "ClaimAmount"]].copy()
    predictions["MeanPrediction"] = float(train["ClaimAmount"].mean())
    predictions["GammaPrediction"] = np.clip(gamma.predict(test), 1e-12, None)
    predictions["LognormalPrediction"] = np.clip(
        np.exp(lognormal.predict(test)) * smear, 1e-12, None
    )
    metrics = {
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": test_size,
            "training_claims": len(train),
            "test_claims": len(test),
            "training_policies": int(train["IDpol"].nunique()),
            "test_policies": int(test["IDpol"].nunique()),
        },
        "formula": MODEL_FORMULA,
        "lognormal_smearing_factor": smear,
        "mean": evaluate_severity(
            predictions["ClaimAmount"], predictions["MeanPrediction"]
        ),
        "gamma": evaluate_severity(
            predictions["ClaimAmount"], predictions["GammaPrediction"]
        ),
        "lognormal": evaluate_severity(
            predictions["ClaimAmount"], predictions["LognormalPrediction"]
        ),
    }
    tables = {
        "predictions": predictions,
        "gamma_coefficients": _coefficient_table(gamma),
        "lognormal_coefficients": _coefficient_table(lognormal),
        "gamma_calibration": severity_calibration(
            predictions["ClaimAmount"], predictions["GammaPrediction"]
        ),
        "lognormal_calibration": severity_calibration(
            predictions["ClaimAmount"], predictions["LognormalPrediction"]
        ),
    }
    return metrics, tables


def _save_calibration_chart(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, color in (("gamma", "#315b7d"), ("lognormal", "#a4573b")):
        table = tables[f"{name}_calibration"]
        axis.plot(
            table["ExpectedSeverity"],
            table["ObservedSeverity"],
            marker="o",
            color=color,
            label=name.title(),
        )
    low = min(axis.get_xlim()[0], axis.get_ylim()[0])
    high = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([low, high], [low, high], "--", color="black", label="Perfect calibration")
    axis.set(
        xlabel="Expected claim severity (€)",
        ylabel="Observed claim severity (€)",
        title="Held-out severity calibration by risk decile",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)


def _render_report(metrics: dict) -> str:
    lines = [
        "# Traditional claim-severity models",
        "",
        f"Training claims: {metrics['split']['training_claims']:,}",
        f"Held-out claims: {metrics['split']['test_claims']:,}",
        f"Lognormal smearing factor: {metrics['lognormal_smearing_factor']:.6f}",
        "",
        "| Model | MAE | RMSE | Gamma deviance | Mean bias | Predicted/observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("mean", "Mean"), ("gamma", "Gamma GLM"), ("lognormal", "Lognormal")):
        item = metrics[key]
        lines.append(
            f"| {label} | €{item['mae']:,.2f} | €{item['rmse']:,.2f} | "
            f"{item['mean_gamma_deviance']:.6f} | €{item['mean_bias']:,.2f} | "
            f"{item['predicted_observed_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Claims were split by policy, preventing claims from one policy appearing in both development and holdout data.",
            "The lognormal predictions use Duan smearing to correct retransformation bias.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    claims = pd.read_parquet(data_path)
    metrics, tables = fit_severity_models(claims, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_parquet(output_dir / "predictions.parquet", index=False)
    for name, table in tables.items():
        if name != "predictions":
            table.to_csv(output_dir / f"{name}.csv", index=False)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_calibration_chart(tables, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(args.data_path.resolve(), args.output_dir.resolve(), args.test_size)
