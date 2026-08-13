"""Synthesize ClaimGuard evidence into classified research findings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
REPORTS: Final = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_DIR: Final = REPORTS / "research_findings"
DEFAULT_NOTEBOOK_PATH: Final = PROJECT_ROOT / "notebooks" / "research.ipynb"
VALID_CLASSIFICATIONS: Final = {"Robust", "Suggestive", "Exploratory", "Data-limited"}


def classify_findings(reports_dir: Path) -> list[dict[str, str]]:
    """Answer the core research questions from persisted, verified metrics."""
    frequency = pd.read_csv(reports_dir / "ml_frequency" / "model_comparison.csv")
    severity = pd.read_csv(reports_dir / "ml_severity" / "model_comparison.csv")
    premium = pd.read_csv(reports_dir / "ml_pure_premium" / "model_comparison.csv")
    tail = pd.read_csv(reports_dir / "tail_performance" / "tail_performance.csv")
    deciles = json.loads((reports_dir / "risk_deciles" / "metrics.json").read_text())
    bonus = json.loads((reports_dir / "bonus_malus" / "metrics.json").read_text())
    stress = json.loads((reports_dir / "portfolio_stress" / "metrics.json").read_text())
    classification = json.loads(
        (reports_dir / "large_loss_classification" / "metrics.json").read_text()
    )
    rigor = json.loads((reports_dir / "statistical_rigor" / "metrics.json").read_text())
    explainability = json.loads((reports_dir / "explainability" / "metrics.json").read_text())

    poisson = frequency.loc[frequency["Model"].eq("Poisson GLM")].iloc[0]
    frequency_ml = frequency.loc[frequency["Model"].eq("HistGradientBoosting")].iloc[0]
    gamma = severity.loc[severity["Model"].eq("Gamma GLM")].iloc[0]
    severity_ml = severity.loc[severity["Model"].eq("XGBoost")].iloc[0]
    tweedie = premium.loc[premium["Model"].eq("Tweedie GLM")].iloc[0]
    component_ml = premium.loc[
        premium["Model"].eq("GBM Frequency × GBM Severity")
    ].iloc[0]
    top_one = tail.loc[tail["Segment"].eq("Top 1%")]
    best_tail = top_one.loc[top_one["predicted_observed_ratio"].idxmax()]
    q95_boosting = classification["thresholds"]["q95"]["models"][
        "gradient_boosting"
    ]
    stable = {
        row["Comparison"]: row["StatisticallyStableMLImprovement"]
        for row in rigor["comparisons"]
    }
    freq_features = explainability["top_features"]["frequency"]
    sev_features = explainability["top_features"]["severity"]

    return [
        {
            "Question": "Which characteristics are associated with claim frequency?",
            "Classification": "Robust",
            "Finding": (
                "Bonus-Malus and vehicle age ranked first and second across GLM, "
                "permutation, and SHAP frequency explanations; driver age was also consistently important."
            ),
            "Evidence": f"Permutation top five: {', '.join(freq_features['permutation'])}.",
        },
        {
            "Question": "Which characteristics are associated with claim severity?",
            "Classification": "Suggestive",
            "Finding": (
                "Severity signals were weaker and less consistent; region, Bonus-Malus, driver age, "
                "and vehicle power appeared important under different explanation methods."
            ),
            "Evidence": f"SHAP top five: {', '.join(sev_features['shap'])}.",
        },
        {
            "Question": "How different are the drivers of frequency and severity?",
            "Classification": "Suggestive",
            "Finding": (
                "Frequency is dominated by Bonus-Malus and vehicle age, whereas severity gives more "
                "weight to driver age, region, and vehicle power."
            ),
            "Evidence": "Only Bonus-Malus appears in every frequency and severity top-five comparison.",
        },
        {
            "Question": "How well does Bonus-Malus separate actual risk?",
            "Classification": "Robust",
            "Finding": (
                "Bonus-Malus separates frequency and pure premium after controlling for driver, vehicle, "
                "and geography, but adds no held-out severity value."
            ),
            "Evidence": (
                f"Per +10 points: frequency relativity {bonus['frequency']['relativity_per_10_points']:.3f}, "
                f"pure-premium relativity {bonus['pure_premium']['relativity_per_10_points']:.3f}; "
                f"severity deviance change {bonus['severity']['deviance_improvement']:.2%}."
            ),
        },
        {
            "Question": "Do ML models materially outperform actuarial GLMs?",
            "Classification": "Robust",
            "Finding": (
                "ML materially improves some predictive metrics, but not calibration, distribution fit, "
                "tail protection, or interpretability; no universal winner exists."
            ),
            "Evidence": (
                f"Frequency deviance improvement {1-frequency_ml['mean_poisson_deviance']/poisson['mean_poisson_deviance']:.2%}; "
                f"severity MAE improvement {1-severity_ml['mae']/gamma['mae']:.2%}."
            ),
        },
        {
            "Question": "Is ML improvement present for frequency, severity, or both?",
            "Classification": "Robust",
            "Finding": "Improvements are present for frequency deviance and severity MAE, with paired bootstrap support for both.",
            "Evidence": (
                f"Stable frequency improvement: {stable['Frequency deviance']}; "
                f"stable severity improvement: {stable['Severity MAE']}."
            ),
        },
        {
            "Question": "Does frequency × severity outperform direct Tweedie modeling?",
            "Classification": "Suggestive",
            "Finding": (
                "Component GBM has lower Tweedie deviance, while direct Tweedie has substantially better "
                "aggregate calibration and lower MAE than traditional component models."
            ),
            "Evidence": (
                f"Component GBM deviance {component_ml['mean_tweedie_deviance']:.3f}; "
                f"Tweedie GLM {tweedie['mean_tweedie_deviance']:.3f}."
            ),
        },
        {
            "Question": "Which approach produces the best calibrated pure premium?",
            "Classification": "Robust",
            "Finding": "The Tweedie GLM is the best aggregate-calibrated held-out pure-premium model.",
            "Evidence": f"Tweedie predicted/observed ratio: {tweedie['predicted_observed_ratio']:.4f}.",
        },
        {
            "Question": "Do average-performance winners remain strong in the extreme tail?",
            "Classification": "Robust",
            "Finding": "No. Every severity model misses more than 97% of held-out top-1% aggregate loss.",
            "Evidence": (
                f"Best top-1% capture was {best_tail['predicted_observed_ratio']:.2%} "
                f"from {best_tail['Model']}."
            ),
        },
        {
            "Question": "How concentrated is portfolio risk among high-risk policies?",
            "Classification": "Suggestive",
            "Finding": (
                "Model-ranked high-risk policies concentrate loss, but rankings are noisy because isolated "
                "extreme claims dominate observed outcomes."
            ),
            "Evidence": (
                f"Direct boosting D10 captured {deciles['diagnostics']['direct_boosting']['highest_decile_loss_capture']:.2%}; "
                f"GBM components D10 captured {deciles['diagnostics']['gbm_component']['highest_decile_loss_capture']:.2%}."
            ),
        },
        {
            "Question": "Can large claims be identified before they occur?",
            "Classification": "Data-limited",
            "Finding": "Available policy characteristics provide only weak discrimination for large claims and essentially none for the most extreme claims.",
            "Evidence": (
                f"Q95 boosting PR-AUC {q95_boosting['pr_auc']:.4f} at event rate "
                f"{q95_boosting['event_rate']:.2%}."
            ),
        },
        {
            "Question": "How much aggregate loss uncertainty comes from extreme claims?",
            "Classification": "Exploratory",
            "Finding": (
                "EVT stress scenarios are overwhelmingly tail-driven, but the fitted infinite-variance tail "
                "makes capital metrics highly unstable."
            ),
            "Evidence": (
                f"Tail share of mean loss {stress['tail_impact']['mean_loss_share']:.2%}; "
                f"tail share of 99% ES {stress['tail_impact']['expected_shortfall_99_share']:.2%}."
            ),
        },
    ]


def validate_findings(findings: list[dict[str, str]]) -> None:
    if len(findings) != 12:
        raise ValueError("The research synthesis must answer all twelve questions")
    invalid = {item["Classification"] for item in findings} - VALID_CLASSIFICATIONS
    if invalid:
        raise ValueError(f"Invalid finding classifications: {sorted(invalid)}")


def render_report(findings: list[dict[str, str]]) -> str:
    lines = [
        "# ClaimGuard research findings",
        "",
        "ClaimGuard compares actuarial GLMs with nonlinear models for French motor third-party liability frequency, severity, pure premium, and extreme-loss risk.",
        "",
        "## Findings",
        "",
    ]
    for index, finding in enumerate(findings, start=1):
        lines.extend(
            [
                f"### {index}. {finding['Question']}",
                "",
                f"**{finding['Classification']}** — {finding['Finding']}",
                "",
                f"Evidence: {finding['Evidence']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Overall conclusion",
            "",
            "ML improves frequency estimation, severity MAE, and risk ranking, while actuarial models retain advantages in calibration, distributional fit, and transparent uncertainty. EVT better represents extreme losses, but its parameters are unstable and cannot compensate for weak policy-level large-loss predictability.",
            "",
            "The most defensible architecture is hybrid: nonlinear frequency and ranking, a calibrated Tweedie benchmark for expected loss, interpretable component models for diagnosis, and separate EVT scenarios for tail stress testing.",
            "",
            "## Classification definitions",
            "",
            "- Robust: supported by held-out results and uncertainty or sensitivity analysis.",
            "- Suggestive: consistent evidence with meaningful metric or sampling trade-offs.",
            "- Exploratory: useful scenario evidence with strong modeling sensitivity.",
            "- Data-limited: available predictors do not support a reliable conclusion or operational model.",
            "",
        ]
    )
    return "\n".join(lines)


def make_notebook(findings: list[dict[str, str]]) -> dict:
    """Create a lightweight notebook that reads canonical generated artifacts."""
    report = render_report(findings)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in report.splitlines()],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Reproducible evidence tables\n",
                "\n",
                "Run the following cell from the repository root after generating the model reports.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "\n",
                "import pandas as pd\n",
                "\n",
                "reports = Path.cwd() / 'reports'\n",
                "benchmark = pd.read_csv(reports / 'model_benchmark' / 'model_benchmark.csv')\n",
                "tail = pd.read_csv(reports / 'tail_performance' / 'tail_performance.csv')\n",
                "sensitivity = pd.read_csv(reports / 'data_quality_sensitivity' / 'scenario_comparison.csv')\n",
                "display(benchmark, tail, sensitivity)\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def run_synthesis(
    reports_dir: Path, output_dir: Path, notebook_path: Path
) -> dict:
    findings = classify_findings(reports_dir)
    validate_findings(findings)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(findings)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    table = pd.DataFrame(findings)
    table.to_csv(output_dir / "findings.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification_counts": {
            key: int(value) for key, value in table["Classification"].value_counts().items()
        },
        "findings": findings,
    }
    (output_dir / "findings.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(make_notebook(findings), indent=2) + "\n", encoding="utf-8"
    )
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notebook-path", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_synthesis(
        args.reports_dir.resolve(),
        args.output_dir.resolve(),
        args.notebook_path.resolve(),
    )
