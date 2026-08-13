"""Evaluate observed and controlled insurance risk across Bonus-Malus levels."""

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
from sklearn.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)

from frequency.traditional_models import split_portfolio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "bonus_malus"
BONUS_MALUS_BINS: Final = (49, 50, 60, 70, 80, 100, 125, 150, 230)
CONTROL_FORMULA: Final = (
    "C(Area) + C(VehPower) + VehAge + DrivAge + C(VehBrand) + "
    "C(VehGas) + np.log1p(Density) + C(Region)"
)


def summarize_bonus_malus(policy_loss: pd.DataFrame) -> pd.DataFrame:
    """Calculate exposure-aware observed outcomes by Bonus-Malus band."""
    frame = policy_loss.copy()
    frame["BonusMalusBand"] = pd.cut(
        frame["BonusMalus"], bins=BONUS_MALUS_BINS, include_lowest=True
    )
    summary = (
        frame.groupby("BonusMalusBand", observed=True)
        .agg(
            Policies=("IDpol", "size"),
            Exposure=("Exposure", "sum"),
            RecordedClaims=("ClaimNb", "sum"),
            LinkedClaims=("ObservedClaimNb", "sum"),
            ObservedLoss=("TotalLoss", "sum"),
            MeanBonusMalus=("BonusMalus", "mean"),
        )
        .reset_index()
    )
    summary["ClaimFrequency"] = summary["RecordedClaims"] / summary["Exposure"]
    summary["ClaimSeverity"] = (
        summary["ObservedLoss"] / summary["LinkedClaims"].replace(0, np.nan)
    )
    summary["PurePremium"] = summary["ObservedLoss"] / summary["Exposure"]
    summary["BonusMalusBand"] = summary["BonusMalusBand"].astype("string")
    return summary


def _coefficient_summary(result: object) -> dict[str, float]:
    coefficient = float(result.params["BonusMalus"])  # type: ignore[attr-defined]
    confidence = result.conf_int().loc["BonusMalus"]  # type: ignore[attr-defined]
    return {
        "coefficient_per_point": coefficient,
        "p_value": float(result.pvalues["BonusMalus"]),  # type: ignore[attr-defined]
        "relativity_per_10_points": float(np.exp(10 * coefficient)),
        "relativity_95_low": float(np.exp(10 * confidence.iloc[0])),
        "relativity_95_high": float(np.exp(10 * confidence.iloc[1])),
    }


def fit_controlled_models(
    policy_loss: pd.DataFrame,
    claims: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[dict, pd.DataFrame]:
    """Measure held-out incremental value of Bonus-Malus in nested GLMs."""
    train, test = split_portfolio(policy_loss, test_size)
    train_claims = claims.loc[claims["IDpol"].isin(train["IDpol"])].copy()
    test_claims = claims.loc[claims["IDpol"].isin(test["IDpol"])].copy()
    full_formula = f"{{target}} ~ BonusMalus + {CONTROL_FORMULA}"
    reduced_formula = f"{{target}} ~ {CONTROL_FORMULA}"

    poisson_reduced = smf.glm(
        formula=reduced_formula.format(target="ClaimNb"),
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"]),
    ).fit()
    poisson_full = smf.glm(
        formula=full_formula.format(target="ClaimNb"),
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"]),
    ).fit()
    gamma_reduced = smf.glm(
        formula=reduced_formula.format(target="ClaimAmount"),
        data=train_claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    gamma_full = smf.glm(
        formula=full_formula.format(target="ClaimAmount"),
        data=train_claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()

    train = train.copy()
    test = test.copy()
    train["AnnualLossCost"] = train["TotalLoss"] / train["Exposure"]
    test["AnnualLossCost"] = test["TotalLoss"] / test["Exposure"]
    tweedie_family = sm.families.Tweedie(
        var_power=1.5, link=sm.families.links.Log()
    )
    tweedie_reduced = smf.glm(
        formula=reduced_formula.format(target="AnnualLossCost"),
        data=train,
        family=tweedie_family,
        freq_weights=train["Exposure"],
    ).fit()
    tweedie_full = smf.glm(
        formula=full_formula.format(target="AnnualLossCost"),
        data=train,
        family=tweedie_family,
        freq_weights=train["Exposure"],
    ).fit()

    predictions = test[["IDpol", "Exposure", "ClaimNb", "TotalLoss"]].copy()
    predictions["PoissonReduced"] = poisson_reduced.predict(
        test, offset=np.log(test["Exposure"])
    )
    predictions["PoissonFull"] = poisson_full.predict(
        test, offset=np.log(test["Exposure"])
    )
    severity_predictions = test_claims[
        ["SourceRow", "IDpol", "ClaimAmount"]
    ].copy()
    severity_predictions["GammaReduced"] = gamma_reduced.predict(test_claims)
    severity_predictions["GammaFull"] = gamma_full.predict(test_claims)
    predictions["TweedieReduced"] = (
        tweedie_reduced.predict(test) * test["Exposure"]
    )
    predictions["TweedieFull"] = tweedie_full.predict(test) * test["Exposure"]

    metrics = {
        "split": {
            "test_size": test_size,
            "training_policies": len(train),
            "test_policies": len(test),
            "training_claims": len(train_claims),
            "test_claims": len(test_claims),
        },
        "frequency": {
            "reduced_deviance": float(
                mean_poisson_deviance(
                    predictions["ClaimNb"], predictions["PoissonReduced"]
                )
            ),
            "full_deviance": float(
                mean_poisson_deviance(
                    predictions["ClaimNb"], predictions["PoissonFull"]
                )
            ),
            **_coefficient_summary(poisson_full),
        },
        "severity": {
            "reduced_deviance": float(
                mean_gamma_deviance(
                    severity_predictions["ClaimAmount"],
                    severity_predictions["GammaReduced"],
                )
            ),
            "full_deviance": float(
                mean_gamma_deviance(
                    severity_predictions["ClaimAmount"],
                    severity_predictions["GammaFull"],
                )
            ),
            **_coefficient_summary(gamma_full),
        },
        "pure_premium": {
            "reduced_deviance": float(
                mean_tweedie_deviance(
                    predictions["TotalLoss"],
                    predictions["TweedieReduced"],
                    power=1.5,
                )
            ),
            "full_deviance": float(
                mean_tweedie_deviance(
                    predictions["TotalLoss"],
                    predictions["TweedieFull"],
                    power=1.5,
                )
            ),
            **_coefficient_summary(tweedie_full),
        },
    }
    for item in ("frequency", "severity", "pure_premium"):
        values = metrics[item]
        values["deviance_improvement"] = (
            1 - values["full_deviance"] / values["reduced_deviance"]
        )
    return metrics, predictions


def _save_chart(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, column, title in zip(
        axes,
        ("ClaimFrequency", "ClaimSeverity", "PurePremium"),
        ("Observed frequency", "Observed severity", "Observed loss cost"),
        strict=True,
    ):
        axis.plot(summary["MeanBonusMalus"], summary[column], marker="o")
        axis.set(xlabel="Mean Bonus-Malus", ylabel=column, title=title)
    fig.tight_layout()
    fig.savefig(output_dir / "bonus_malus_observed_risk.png", dpi=150)
    plt.close(fig)


def _render_report(summary: pd.DataFrame, metrics: dict) -> str:
    lines = [
        "# Bonus-Malus risk analysis",
        "",
        "## Observed portfolio outcomes",
        "",
        "| Bonus-Malus band | Policies | Frequency | Severity | Loss cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.BonusMalusBand} | {row.Policies:,} | {row.ClaimFrequency:.4f} | "
            f"€{row.ClaimSeverity:,.2f} | €{row.PurePremium:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Incremental controlled value",
            "",
            "| Outcome | Relativity per +10 points | 95% CI | P-value | Held-out deviance improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("frequency", "Frequency"),
        ("severity", "Severity"),
        ("pure_premium", "Pure premium"),
    ):
        item = metrics[key]
        lines.append(
            f"| {label} | {item['relativity_per_10_points']:.4f} | "
            f"[{item['relativity_95_low']:.4f}, {item['relativity_95_high']:.4f}] | "
            f"{item['p_value']:.3g} | {item['deviance_improvement']:.3%} |"
        )
    lines.extend(
        [
            "",
            "Controlled models include driver age, vehicle age and power, brand, fuel, area, density, and region.",
            "Results are predictive associations and should not be interpreted causally.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(data_dir: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    policy_loss = pd.read_parquet(data_dir / "policy_loss.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    summary = summarize_bonus_malus(policy_loss)
    metrics, predictions = fit_controlled_models(policy_loss, claims, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "observed_by_bonus_malus.csv", index=False)
    predictions.to_parquet(output_dir / "holdout_predictions.parquet", index=False)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(summary, metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_chart(summary, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.data_dir.resolve(), args.output_dir.resolve(), args.test_size)
