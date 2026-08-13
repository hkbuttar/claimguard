"""Compare GLM relativities with permutation and SHAP model explanations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
from xgboost import XGBRegressor

from frequency.ml_models import FEATURES, prepare_features
from frequency.traditional_models import split_portfolio
from severity.ml_models import prepare_native_features
from severity.traditional_models import split_claims_by_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
DEFAULT_FREQUENCY_MODEL: Final = PROJECT_ROOT / "reports" / "ml_frequency" / "xgboost.json"
DEFAULT_SEVERITY_MODEL: Final = PROJECT_ROOT / "reports" / "ml_severity" / "xgboost.json"
DEFAULT_FREQUENCY_COEFFICIENTS: Final = (
    PROJECT_ROOT / "reports" / "traditional_frequency" / "poisson_coefficients.csv"
)
DEFAULT_SEVERITY_COEFFICIENTS: Final = (
    PROJECT_ROOT / "reports" / "traditional_severity" / "gamma_coefficients.csv"
)
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "explainability"
PERMUTATION_SAMPLE_SIZE: Final = 20_000
SHAP_SAMPLE_SIZE: Final = 2_000
PERMUTATION_REPEATS: Final = 5


def feature_from_term(term: str) -> str:
    """Map a formula coefficient term back to its source rating feature."""
    if term == "Intercept":
        return "Intercept"
    if term.startswith("C("):
        return term[2:].split(")", maxsplit=1)[0]
    if "Density" in term:
        return "Density"
    for feature in FEATURES:
        if feature in term:
            return feature
    return term


def standardize_glm_coefficients(path: Path, task: str) -> pd.DataFrame:
    """Create a common coefficient and relativity schema for GLM outputs."""
    table = pd.read_csv(path)
    relativity_column = (
        "RateRatio" if "RateRatio" in table.columns else "MultiplicativeEffect"
    )
    result = pd.DataFrame(
        {
            "Task": task,
            "Feature": table["Term"].map(feature_from_term),
            "Term": table["Term"],
            "Coefficient": table["Coefficient"],
            "StdError": table["StdError"],
            "PValue": table["PValue"],
            "ConfidenceLow": table["ConfidenceLow"],
            "ConfidenceHigh": table["ConfidenceHigh"],
            "Relativity": table[relativity_column],
        }
    )
    result["AbsoluteZScore"] = (
        result["Coefficient"] / result["StdError"]
    ).abs()
    return result


def grouped_glm_importance(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Rank each raw feature by its strongest coefficient Wald statistic."""
    return (
        coefficients.loc[coefficients["Feature"].ne("Intercept")]
        .groupby(["Task", "Feature"], observed=True)
        .agg(
            MaxAbsoluteZScore=("AbsoluteZScore", "max"),
            MinimumPValue=("PValue", "min"),
            Terms=("Term", "size"),
        )
        .reset_index()
        .sort_values(["Task", "MaxAbsoluteZScore"], ascending=[True, False])
    )


def raw_permutation_importance(
    features: pd.DataFrame,
    actual: pd.Series,
    predict: Callable[[pd.DataFrame], np.ndarray],
    loss: Callable[[np.ndarray, np.ndarray], float],
    repeats: int = PERMUTATION_REPEATS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure score degradation after permuting one raw feature at a time."""
    observed = actual.to_numpy(dtype=float)
    baseline = float(loss(observed, predict(features)))
    rng = np.random.default_rng(random_state)
    rows = []
    for feature in features.columns:
        losses = []
        for _ in range(repeats):
            permuted = features.copy()
            order = rng.permutation(len(permuted))
            values = features[feature].array.take(order)
            permuted[feature] = pd.Series(
                values,
                index=permuted.index,
                dtype=features[feature].dtype,
            )
            losses.append(float(loss(observed, predict(permuted))))
        rows.append(
            {
                "Feature": feature,
                "BaselineLoss": baseline,
                "PermutedLoss": float(np.mean(losses)),
                "Importance": float(np.mean(losses) - baseline),
                "ImportanceStd": float(np.std(losses, ddof=1)) if repeats > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


def shap_importance(
    model: XGBRegressor, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return global mean absolute SHAP and row-level feature contributions."""
    values = np.asarray(shap.TreeExplainer(model).shap_values(features))
    importance = pd.DataFrame(
        {
            "Feature": features.columns,
            "MeanAbsoluteSHAP": np.mean(np.abs(values), axis=0),
            "MeanSHAP": np.mean(values, axis=0),
        }
    ).sort_values("MeanAbsoluteSHAP", ascending=False)
    contributions = pd.DataFrame(values, columns=features.columns, index=features.index)
    return importance, contributions


def comparison_table(
    glm: pd.DataFrame,
    permutation: dict[str, pd.DataFrame],
    shap_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align feature ranks across tasks and explanation methods."""
    comparison = pd.DataFrame({"Feature": FEATURES})
    for task in ("Frequency", "Severity"):
        glm_task = glm.loc[glm["Task"].eq(task)].copy()
        glm_task["Rank"] = glm_task["MaxAbsoluteZScore"].rank(
            method="min", ascending=False
        )
        perm_task = permutation[task].copy()
        perm_task["Rank"] = perm_task["Importance"].rank(method="min", ascending=False)
        shap_task = shap_tables[task].copy()
        shap_task["Rank"] = shap_task["MeanAbsoluteSHAP"].rank(
            method="min", ascending=False
        )
        comparison = comparison.merge(
            glm_task[["Feature", "Rank"]].rename(columns={"Rank": f"{task}GLMRank"}),
            on="Feature",
            how="left",
        )
        comparison = comparison.merge(
            perm_task[["Feature", "Rank"]].rename(
                columns={"Rank": f"{task}PermutationRank"}
            ),
            on="Feature",
            how="left",
        )
        comparison = comparison.merge(
            shap_task[["Feature", "Rank"]].rename(columns={"Rank": f"{task}SHAPRank"}),
            on="Feature",
            how="left",
        )
    return comparison


def _save_rank_chart(comparison: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in comparison.columns if column.endswith("Rank")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for axis, task in zip(axes, ("Frequency", "Severity"), strict=True):
        task_columns = [column for column in columns if column.startswith(task)]
        width = 0.25
        x = np.arange(len(comparison))
        for index, column in enumerate(task_columns):
            axis.bar(
                x + (index - 1) * width,
                comparison[column],
                width,
                label=column.removeprefix(task).removesuffix("Rank"),
            )
        axis.set_xticks(x, comparison["Feature"], rotation=60, ha="right")
        axis.invert_yaxis()
        axis.set(title=f"{task} feature ranks", ylabel="Rank (1 = most important)")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_rank_comparison.png", dpi=150)
    plt.close(fig)


def _render_report(
    comparison: pd.DataFrame,
    permutation: dict[str, pd.DataFrame],
    shap_tables: dict[str, pd.DataFrame],
) -> str:
    lines = ["# Model explainability comparison", ""]
    for task in ("Frequency", "Severity"):
        top_perm = ", ".join(permutation[task].head(5)["Feature"])
        top_shap = ", ".join(shap_tables[task].head(5)["Feature"])
        lines.extend(
            [
                f"## {task}",
                "",
                f"- Top permutation features: {top_perm}",
                f"- Top SHAP features: {top_shap}",
                "",
            ]
        )
    lines.extend(
        [
            "## Aligned ranks",
            "",
            "| Feature | Freq GLM | Freq permutation | Freq SHAP | Sev GLM | Sev permutation | Sev SHAP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.Feature} | {row.FrequencyGLMRank:.0f} | "
            f"{row.FrequencyPermutationRank:.0f} | {row.FrequencySHAPRank:.0f} | "
            f"{row.SeverityGLMRank:.0f} | {row.SeverityPermutationRank:.0f} | "
            f"{row.SeveritySHAPRank:.0f} |"
        )
    lines.extend(
        [
            "",
            "GLM ranks use the strongest absolute Wald statistic within each raw feature. Permutation importance measures held-out deviance degradation. SHAP ranks summarize absolute log-scale tree contributions.",
            "Feature importance describes predictive model behavior, not causal effects.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    data_dir: Path,
    frequency_model_path: Path,
    severity_model_path: Path,
    frequency_coefficients_path: Path,
    severity_coefficients_path: Path,
    output_dir: Path,
) -> dict:
    policy = pd.read_parquet(data_dir / "policy_frequency.parquet")
    claims = pd.read_parquet(data_dir / "claim_severity.parquet")
    _, frequency_test = split_portfolio(policy)
    _, severity_test = split_claims_by_policy(claims)
    frequency_sample = frequency_test.sample(
        min(PERMUTATION_SAMPLE_SIZE, len(frequency_test)), random_state=42
    )
    severity_sample = severity_test.sample(
        min(PERMUTATION_SAMPLE_SIZE, len(severity_test)), random_state=42
    )
    frequency_model = XGBRegressor()
    frequency_model.load_model(frequency_model_path)
    severity_model = XGBRegressor()
    severity_model.load_model(severity_model_path)

    frequency_features = prepare_features(frequency_sample)
    severity_features = prepare_native_features(severity_sample)
    frequency_permutation = raw_permutation_importance(
        frequency_features,
        frequency_sample["ClaimNb"],
        lambda frame: frequency_model.predict(
            frame, base_margin=np.log(frequency_sample.loc[frame.index, "Exposure"])
        ),
        mean_poisson_deviance,
    )
    severity_permutation = raw_permutation_importance(
        severity_features,
        severity_sample["ClaimAmount"],
        severity_model.predict,
        mean_gamma_deviance,
    )
    shap_frequency_features = frequency_features.sample(
        min(SHAP_SAMPLE_SIZE, len(frequency_features)), random_state=42
    )
    shap_severity_features = severity_features.sample(
        min(SHAP_SAMPLE_SIZE, len(severity_features)), random_state=42
    )
    frequency_shap, frequency_contributions = shap_importance(
        frequency_model, shap_frequency_features
    )
    severity_shap, severity_contributions = shap_importance(
        severity_model, shap_severity_features
    )

    coefficients = pd.concat(
        [
            standardize_glm_coefficients(frequency_coefficients_path, "Frequency"),
            standardize_glm_coefficients(severity_coefficients_path, "Severity"),
        ],
        ignore_index=True,
    )
    glm_importance = grouped_glm_importance(coefficients)
    permutation = {
        "Frequency": frequency_permutation,
        "Severity": severity_permutation,
    }
    shap_tables = {"Frequency": frequency_shap, "Severity": severity_shap}
    comparison = comparison_table(glm_importance, permutation, shap_tables)

    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_dir / "glm_coefficients_and_relativities.csv", index=False)
    glm_importance.to_csv(output_dir / "glm_feature_importance.csv", index=False)
    comparison.to_csv(output_dir / "feature_rank_comparison.csv", index=False)
    for task in ("Frequency", "Severity"):
        name = task.lower()
        permutation[task].to_csv(
            output_dir / f"{name}_permutation_importance.csv", index=False
        )
        shap_tables[task].to_csv(
            output_dir / f"{name}_shap_importance.csv", index=False
        )
    frequency_contributions.assign(IDpol=frequency_sample.loc[frequency_contributions.index, "IDpol"]).to_parquet(
        output_dir / "frequency_shap_values.parquet", index=False
    )
    severity_contributions.assign(SourceRow=severity_sample.loc[severity_contributions.index, "SourceRow"]).to_parquet(
        output_dir / "severity_shap_values.parquet", index=False
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "permutation_sample_sizes": {
            "frequency": len(frequency_sample),
            "severity": len(severity_sample),
        },
        "shap_sample_sizes": {
            "frequency": len(shap_frequency_features),
            "severity": len(shap_severity_features),
        },
        "top_features": {
            task.lower(): {
                "permutation": permutation[task].head(5)["Feature"].tolist(),
                "shap": shap_tables[task].head(5)["Feature"].tolist(),
            }
            for task in ("Frequency", "Severity")
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(comparison, permutation, shap_tables)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    _save_rank_chart(comparison, output_dir)
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--frequency-model", type=Path, default=DEFAULT_FREQUENCY_MODEL)
    parser.add_argument("--severity-model", type=Path, default=DEFAULT_SEVERITY_MODEL)
    parser.add_argument(
        "--frequency-coefficients", type=Path, default=DEFAULT_FREQUENCY_COEFFICIENTS
    )
    parser.add_argument(
        "--severity-coefficients", type=Path, default=DEFAULT_SEVERITY_COEFFICIENTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        args.data_dir.resolve(),
        args.frequency_model.resolve(),
        args.severity_model.resolve(),
        args.frequency_coefficients.resolve(),
        args.severity_coefficients.resolve(),
        args.output_dir.resolve(),
    )
