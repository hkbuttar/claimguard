"""Fit and compare traditional Poisson and Negative Binomial frequency GLMs."""

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
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_frequency.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "traditional_frequency"
RANDOM_STATE: Final = 42
MODEL_FORMULA: Final = (
    "ClaimNb ~ C(Area) + C(VehPower) + VehAge + DrivAge + BonusMalus + "
    "C(VehBrand) + C(VehGas) + np.log1p(Density) + C(Region)"
)


def split_portfolio(
    policy: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible policy-level development and holdout split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    train, test = train_test_split(
        policy, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
    )
    return train.copy(), test.copy()


def estimate_nb_alpha(actual: pd.Series, fitted: pd.Series) -> float:
    """Estimate NB2 alpha from Poisson residual moments."""
    y = actual.to_numpy(dtype=float)
    mu = np.clip(fitted.to_numpy(dtype=float), 1e-12, None)
    alpha = np.sum(np.square(y - mu) - y) / np.sum(np.square(mu))
    return float(max(alpha, 1e-8))


def calibration_table(
    actual: pd.Series,
    predicted_count: pd.Series,
    exposure: pd.Series,
    predicted_rate: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Aggregate observed and predicted frequency by predicted-risk quantile."""
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")
    frame = pd.DataFrame(
        {
            "Actual": actual.to_numpy(dtype=float),
            "Predicted": predicted_count.to_numpy(dtype=float),
            "Exposure": exposure.to_numpy(dtype=float),
            "PredictedRate": predicted_rate.to_numpy(dtype=float),
        }
    )
    frame["RiskDecile"] = pd.qcut(
        frame["PredictedRate"], q=bins, labels=False, duplicates="drop"
    )
    result = (
        frame.groupby("RiskDecile", observed=True)
        .agg(
            Policies=("Actual", "size"),
            Exposure=("Exposure", "sum"),
            ObservedClaims=("Actual", "sum"),
            ExpectedClaims=("Predicted", "sum"),
            MeanPredictedRate=("PredictedRate", "mean"),
        )
        .reset_index()
    )
    result["ObservedFrequency"] = result["ObservedClaims"] / result["Exposure"]
    result["ExpectedFrequency"] = result["ExpectedClaims"] / result["Exposure"]
    result["ObservedExpectedRatio"] = (
        result["ObservedClaims"] / result["ExpectedClaims"]
    )
    return result


def evaluate_frequency(
    actual: pd.Series,
    predicted: pd.Series,
    exposure: pd.Series,
) -> dict[str, float]:
    """Calculate held-out count-model performance and aggregate calibration."""
    prediction = np.clip(predicted.to_numpy(dtype=float), 1e-12, None)
    observed = actual.to_numpy(dtype=float)
    observed_claims = float(observed.sum())
    expected_claims = float(prediction.sum())
    return {
        "poisson_deviance": float(
            mean_poisson_deviance(observed, prediction) * len(observed)
        ),
        "mean_poisson_deviance": float(mean_poisson_deviance(observed, prediction)),
        "observed_claims": observed_claims,
        "expected_claims": expected_claims,
        "observed_expected_ratio": observed_claims / expected_claims,
        "mean_observed_frequency": observed_claims / float(exposure.sum()),
        "mean_expected_frequency": expected_claims / float(exposure.sum()),
    }


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
            "RateRatio": np.exp(result.params.to_numpy()),  # type: ignore[attr-defined]
        }
    )


def fit_frequency_models(
    policy: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Fit both GLMs and return metrics, coefficients, calibration, and predictions."""
    train, test = split_portfolio(policy, test_size)
    train_offset = np.log(train["Exposure"])
    test_offset = np.log(test["Exposure"])

    poisson = smf.glm(
        formula=MODEL_FORMULA,
        data=train,
        family=sm.families.Poisson(),
        offset=train_offset,
    ).fit()
    poisson_train = pd.Series(poisson.predict(train, offset=train_offset), index=train.index)
    alpha = estimate_nb_alpha(train["ClaimNb"], poisson_train)
    negative_binomial = smf.glm(
        formula=MODEL_FORMULA,
        data=train,
        family=sm.families.NegativeBinomial(alpha=alpha),
        offset=train_offset,
    ).fit()

    predictions = test[["IDpol", "Exposure", "ClaimNb"]].copy()
    predictions["PoissonExpectedClaims"] = poisson.predict(test, offset=test_offset)
    predictions["NegativeBinomialExpectedClaims"] = negative_binomial.predict(
        test, offset=test_offset
    )
    predictions["PoissonAnnualRate"] = poisson.predict(
        test, offset=np.zeros(len(test))
    )
    predictions["NegativeBinomialAnnualRate"] = negative_binomial.predict(
        test, offset=np.zeros(len(test))
    )

    poisson_dispersion = float(poisson.pearson_chi2 / poisson.df_resid)
    metrics = {
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": test_size,
            "training_policies": len(train),
            "test_policies": len(test),
        },
        "formula": MODEL_FORMULA,
        "dispersion": {
            "poisson_pearson_dispersion": poisson_dispersion,
            "negative_binomial_alpha": alpha,
            "overdispersion_detected": poisson_dispersion > 1.0,
        },
        "poisson": evaluate_frequency(
            predictions["ClaimNb"],
            predictions["PoissonExpectedClaims"],
            predictions["Exposure"],
        ),
        "negative_binomial": evaluate_frequency(
            predictions["ClaimNb"],
            predictions["NegativeBinomialExpectedClaims"],
            predictions["Exposure"],
        ),
    }
    tables = {
        "predictions": predictions,
        "poisson_coefficients": _coefficient_table(poisson),
        "negative_binomial_coefficients": _coefficient_table(negative_binomial),
        "poisson_calibration": calibration_table(
            predictions["ClaimNb"],
            predictions["PoissonExpectedClaims"],
            predictions["Exposure"],
            predictions["PoissonAnnualRate"],
        ),
        "negative_binomial_calibration": calibration_table(
            predictions["ClaimNb"],
            predictions["NegativeBinomialExpectedClaims"],
            predictions["Exposure"],
            predictions["NegativeBinomialAnnualRate"],
        ),
    }
    return metrics, tables


def _save_calibration_chart(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, color in (("poisson", "#315b7d"), ("negative_binomial", "#a4573b")):
        table = tables[f"{name}_calibration"]
        axis.plot(
            table["ExpectedFrequency"],
            table["ObservedFrequency"],
            marker="o",
            label=name.replace("_", " ").title(),
            color=color,
        )
    limits = axis.get_xlim()
    axis.plot(limits, limits, linestyle="--", color="black", label="Perfect calibration")
    axis.set(
        xlabel="Expected annual frequency",
        ylabel="Observed annual frequency",
        title="Held-out frequency calibration by risk decile",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=150)
    plt.close(fig)


def _render_report(metrics: dict) -> str:
    poisson = metrics["poisson"]
    nb = metrics["negative_binomial"]
    dispersion = metrics["dispersion"]
    return "\n".join(
        [
            "# Traditional claim-frequency models",
            "",
            f"Training policies: {metrics['split']['training_policies']:,}",
            f"Held-out policies: {metrics['split']['test_policies']:,}",
            f"Poisson Pearson dispersion: {dispersion['poisson_pearson_dispersion']:.4f}",
            f"Estimated Negative Binomial alpha: {dispersion['negative_binomial_alpha']:.6f}",
            "",
            "| Model | Mean Poisson deviance | Observed claims | Expected claims | O/E |",
            "|---|---:|---:|---:|---:|",
            f"| Poisson GLM | {poisson['mean_poisson_deviance']:.6f} | {poisson['observed_claims']:,.0f} | {poisson['expected_claims']:,.2f} | {poisson['observed_expected_ratio']:.4f} |",
            f"| Negative Binomial GLM | {nb['mean_poisson_deviance']:.6f} | {nb['observed_claims']:,.0f} | {nb['expected_claims']:,.2f} | {nb['observed_expected_ratio']:.4f} |",
            "",
            "All reported performance metrics use the held-out policy sample.",
            "",
        ]
    )


def run_models(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    policy = pd.read_parquet(data_path)
    metrics, tables = fit_frequency_models(policy, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_parquet(output_dir / "predictions.parquet", index=False)
    for name, table in tables.items():
        if name != "predictions":
            table.to_csv(output_dir / f"{name}.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
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
