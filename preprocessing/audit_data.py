"""Audit raw freMTPL2 tables and produce explicit reconciliation decisions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import pandas as pd

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR: Final = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "data" / "audit"

Classification = Literal["valid", "correctable", "ambiguous", "excluded"]


@dataclass(frozen=True)
class AuditResult:
    rule: str
    dataset: str
    classification: Classification
    count: int
    decision: str


def _result(
    rule: str,
    dataset: str,
    classification: Classification,
    mask: pd.Series,
    decision: str,
) -> AuditResult:
    return AuditResult(rule, dataset, classification, int(mask.sum()), decision)


def audit_tables(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[list[AuditResult], pd.DataFrame, pd.DataFrame]:
    """Return rule summaries and non-destructive row-level audit flags."""
    freq_flags = pd.DataFrame({"IDpol": freq["IDpol"].astype("int64")})
    sev_flags = pd.DataFrame(
        {"source_row": range(len(sev)), "IDpol": sev["IDpol"].astype("int64")}
    )

    freq_rules: list[tuple[str, Classification, pd.Series, str]] = [
        ("duplicate_policy_id", "excluded", freq["IDpol"].duplicated(keep=False),
         "Exclude duplicate policy IDs pending source-level resolution."),
        ("missing_value", "excluded", freq.isna().any(axis=1),
         "Exclude rows with missing required policy fields."),
        ("nonpositive_exposure", "excluded", freq["Exposure"] <= 0,
         "Exclude policies with nonpositive exposure."),
        ("exposure_above_one", "correctable", freq["Exposure"] > 1,
         "Cap exposure at one year in the analytical policy table."),
        ("negative_claim_count", "excluded", freq["ClaimNb"] < 0,
         "Exclude policies with impossible negative claim counts."),
        ("vehicle_age_extreme", "ambiguous", freq["VehAge"] > 80,
         "Retain with a flag; investigate before modeling."),
        ("driver_age_extreme", "ambiguous", freq["DrivAge"] > 95,
         "Retain with a flag; investigate before modeling."),
        ("bonus_malus_extreme", "ambiguous", freq["BonusMalus"] > 150,
         "Retain with a flag; investigate or cap during feature engineering."),
    ]

    matched = sev["IDpol"].isin(freq["IDpol"])
    duplicate_claim = sev.duplicated(subset=["IDpol", "ClaimAmount"], keep=False)
    large_loss_threshold = float(sev["ClaimAmount"].quantile(0.995))
    sev_rules: list[tuple[str, Classification, pd.Series, str]] = [
        ("missing_value", "excluded", sev.isna().any(axis=1),
         "Exclude rows with missing required claim fields."),
        ("nonpositive_claim_amount", "excluded", sev["ClaimAmount"] <= 0,
         "Exclude nonpositive severities from positive-severity modeling."),
        ("unmatched_policy", "excluded", ~matched,
         "Exclude from policy-linked models; retain in the raw source."),
        ("possible_duplicate_claim", "ambiguous", duplicate_claim,
         "Retain because no claim identifier exists; review identical policy-amount pairs."),
        ("extreme_claim_amount", "ambiguous", sev["ClaimAmount"] > large_loss_threshold,
         f"Retain and flag for tail analysis (above empirical 99.5th percentile: {large_loss_threshold:.2f})."),
    ]

    results: list[AuditResult] = []
    for rule, classification, mask, decision in freq_rules:
        freq_flags[rule] = mask.to_numpy()
        results.append(_result(rule, "freMTPL2freq", classification, mask, decision))
    for rule, classification, mask, decision in sev_rules:
        sev_flags[rule] = mask.to_numpy()
        results.append(_result(rule, "freMTPL2sev", classification, mask, decision))

    observed = sev.loc[matched].groupby("IDpol", observed=True).size()
    recorded = freq.set_index(freq["IDpol"].astype("int64"))["ClaimNb"]
    observed = observed.reindex(recorded.index, fill_value=0)
    mismatch = recorded.ne(observed)
    freq_flags["claim_count_mismatch"] = mismatch.to_numpy()
    results.append(
        _result(
            "claim_count_mismatch",
            "cross_table",
            "ambiguous",
            mismatch,
            "Retain and report; do not manufacture missing severity records or overwrite ClaimNb.",
        )
    )

    def assign_disposition(
        flags: pd.DataFrame,
        rules: list[tuple[str, Classification, pd.Series, str]],
        extra_ambiguous: tuple[str, ...] = (),
    ) -> pd.Series:
        disposition = pd.Series("valid", index=flags.index, dtype="object")
        for classification in ("correctable", "ambiguous", "excluded"):
            columns = [rule for rule, level, _, _ in rules if level == classification]
            if classification == "ambiguous":
                columns.extend(extra_ambiguous)
            if columns:
                disposition.loc[flags[columns].any(axis=1)] = classification
        return disposition

    freq_flags["disposition"] = assign_disposition(
        freq_flags, freq_rules, ("claim_count_mismatch",)
    )
    sev_flags["disposition"] = assign_disposition(sev_flags, sev_rules)

    return results, freq_flags, sev_flags


def _render_markdown(report: dict) -> str:
    lines = [
        "# freMTPL2 data audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "Raw inputs were not modified. Counts are rule-level and may overlap.",
        "",
        "| Dataset | Rule | Classification | Count | Decision |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['dataset']} | {item['rule']} | {item['classification']} "
            f"| {item['count']:,} | {item['decision']} |"
        )
    lines.extend(["", "## Overall row dispositions", ""])
    for dataset, counts in report["row_dispositions"].items():
        formatted = ", ".join(f"{key}: {value:,}" for key, value in counts.items())
        lines.append(f"- {dataset}: {formatted}")
    lines.extend(
        [
            "",
            "## Classification policy",
            "",
            "- Valid: no rule triggered; retain without adjustment.",
            "- Correctable: deterministic adjustment is documented and deferred to analytical-table construction.",
            "- Ambiguous: retain with a flag because the source does not support a definitive correction.",
            "- Excluded: omit from the affected analytical use while preserving the raw record.",
            "- Precedence for overlapping rules: excluded, ambiguous, correctable, then valid.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(raw_dir: Path, output_dir: Path) -> dict:
    freq = pd.read_parquet(raw_dir / "freMTPL2freq.parquet")
    sev = pd.read_parquet(raw_dir / "freMTPL2sev.parquet")
    results, freq_flags, sev_flags = audit_tables(freq, sev)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": {"freMTPL2freq": len(freq), "freMTPL2sev": len(sev)},
        "row_dispositions": {
            "freMTPL2freq": {
                key: int(value)
                for key, value in freq_flags["disposition"].value_counts().items()
            },
            "freMTPL2sev": {
                key: int(value)
                for key, value in sev_flags["disposition"].value_counts().items()
            },
        },
        "results": [asdict(result) for result in results],
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    freq_flags.to_parquet(output_dir / "frequency_flags.parquet", index=False)
    sev_flags.to_parquet(output_dir / "severity_flags.parquet", index=False)
    print(_render_markdown(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(args.raw_dir.resolve(), args.output_dir.resolve())
