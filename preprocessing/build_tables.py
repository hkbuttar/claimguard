"""Construct the policy-frequency, claim-severity, and policy-loss tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from preprocessing.acquire_data import sha256
from preprocessing.audit_data import audit_tables

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR: Final = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "data" / "processed"

POLICY_FEATURES: Final = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
]


def build_modeling_tables(
    freq: pd.DataFrame, sev: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build modeling tables using the documented audit decisions."""
    _, freq_flags, sev_flags = audit_tables(freq, sev)

    policy_frequency = freq.loc[
        freq_flags["disposition"].ne("excluded").to_numpy()
    ].copy()
    kept_freq_flags = freq_flags.loc[
        freq_flags["disposition"].ne("excluded")
    ].reset_index(drop=True)
    policy_frequency["IDpol"] = policy_frequency["IDpol"].astype("int64")
    policy_frequency.insert(
        policy_frequency.columns.get_loc("Exposure") + 1,
        "ExposureRaw",
        policy_frequency["Exposure"],
    )
    policy_frequency["Exposure"] = policy_frequency["Exposure"].clip(upper=1.0)
    policy_frequency["ExposureCapped"] = kept_freq_flags[
        "exposure_above_one"
    ].to_numpy()
    policy_frequency["PolicyAuditDisposition"] = kept_freq_flags[
        "disposition"
    ].to_numpy()
    policy_frequency["ClaimCountMismatch"] = kept_freq_flags[
        "claim_count_mismatch"
    ].to_numpy()
    policy_frequency.reset_index(drop=True, inplace=True)

    eligible_claim = sev_flags["disposition"].ne("excluded")
    eligible_policy_ids = set(policy_frequency["IDpol"])
    eligible_claim &= sev_flags["IDpol"].isin(eligible_policy_ids)
    claims = sev.loc[eligible_claim.to_numpy()].copy().reset_index(drop=True)
    claim_flags = sev_flags.loc[eligible_claim].reset_index(drop=True)
    claims["IDpol"] = claims["IDpol"].astype("int64")
    claims.insert(0, "SourceRow", claim_flags["source_row"])
    claims.insert(
        2,
        "ClaimIndex",
        claims.groupby("IDpol", observed=True).cumcount() + 1,
    )

    policy_characteristics = policy_frequency[["IDpol", *POLICY_FEATURES]].copy()
    claim_severity = claims.merge(
        policy_characteristics,
        on="IDpol",
        how="left",
        validate="many_to_one",
    )
    claim_severity["PossibleDuplicateClaim"] = claim_flags[
        "possible_duplicate_claim"
    ].to_numpy()
    claim_severity["ExtremeClaimAmount"] = claim_flags[
        "extreme_claim_amount"
    ].to_numpy()
    claim_severity["ClaimAuditDisposition"] = claim_flags["disposition"].to_numpy()

    aggregates = claim_severity.groupby("IDpol", observed=True).agg(
        ObservedClaimNb=("ClaimAmount", "size"),
        TotalLoss=("ClaimAmount", "sum"),
    )
    policy_loss = policy_frequency.merge(
        aggregates, left_on="IDpol", right_index=True, how="left", validate="one_to_one"
    )
    policy_loss["ObservedClaimNb"] = policy_loss["ObservedClaimNb"].fillna(0).astype("int64")
    policy_loss["TotalLoss"] = policy_loss["TotalLoss"].fillna(0.0)
    policy_loss["HasObservedLoss"] = policy_loss["ObservedClaimNb"].gt(0)

    validate_modeling_tables(policy_frequency, claim_severity, policy_loss)
    return policy_frequency, claim_severity, policy_loss


def validate_modeling_tables(
    policy_frequency: pd.DataFrame,
    claim_severity: pd.DataFrame,
    policy_loss: pd.DataFrame,
) -> None:
    """Enforce the core actuarial table invariants."""
    if policy_frequency["IDpol"].duplicated().any():
        raise ValueError("Policy-frequency table must have one row per policy")
    if not policy_frequency["Exposure"].between(0, 1, inclusive="right").all():
        raise ValueError("Analytical exposure must be positive and at most one year")
    if claim_severity["ClaimAmount"].le(0).any():
        raise ValueError("Claim-severity table must contain positive losses only")
    if claim_severity[POLICY_FEATURES].isna().any(axis=None):
        raise ValueError("Every retained claim must have policy characteristics")
    if len(policy_loss) != len(policy_frequency) or policy_loss["IDpol"].duplicated().any():
        raise ValueError("Policy-loss table must have one row per eligible policy")
    if not policy_loss["IDpol"].equals(policy_frequency["IDpol"]):
        raise ValueError("Policy tables must have identical policy ordering")
    if policy_loss["TotalLoss"].lt(0).any():
        raise ValueError("Policy total loss cannot be negative")
    if not pd.api.types.is_numeric_dtype(policy_loss["TotalLoss"]):
        raise ValueError("Policy total loss must be numeric")
    if abs(policy_loss["TotalLoss"].sum() - claim_severity["ClaimAmount"].sum()) > 0.01:
        raise ValueError("Claim and policy aggregate losses do not reconcile")


def write_modeling_tables(raw_dir: Path, output_dir: Path) -> dict:
    freq = pd.read_parquet(raw_dir / "freMTPL2freq.parquet")
    sev = pd.read_parquet(raw_dir / "freMTPL2sev.parquet")
    tables = build_modeling_tables(freq, sev)
    names = ("policy_frequency", "claim_severity", "policy_loss")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {},
    }
    for name, table in zip(names, tables, strict=True):
        destination = output_dir / f"{name}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        table.to_parquet(temporary, index=False)
        temporary.replace(destination)
        manifest["tables"][name] = {
            "file": str(destination.relative_to(PROJECT_ROOT)),
            "rows": len(table),
            "columns": list(table.columns),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
        print(f"{name}: {len(table):,} rows -> {destination}")
    manifest["reconciliation"] = {
        "claim_severity_total_loss": float(tables[1]["ClaimAmount"].sum()),
        "policy_loss_total_loss": float(tables[2]["TotalLoss"].sum()),
        "policies_with_zero_observed_loss": int(tables[2]["TotalLoss"].eq(0).sum()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_modeling_tables(args.raw_dir.resolve(), args.output_dir.resolve())
