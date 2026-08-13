"""Download and validate the raw freMTPL2 datasets from OpenML.

The OpenML dataset IDs are pinned to the same snapshots used by scikit-learn's
French motor third-party liability example. Raw tables are saved as Parquet to
preserve categorical types and keep subsequent reads fast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "data" / "raw"
DEFAULT_CACHE_DIR: Final = PROJECT_ROOT / "data" / ".cache" / "openml"

DATASETS: Final = {
    "freMTPL2freq": {
        "data_id": 41214,
        "filename": "freMTPL2freq.parquet",
        "columns": {
            "IDpol",
            "ClaimNb",
            "Exposure",
            "Area",
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            "VehBrand",
            "VehGas",
            "Density",
            "Region",
        },
    },
    "freMTPL2sev": {
        "data_id": 41215,
        "filename": "freMTPL2sev.parquet",
        "columns": {"IDpol", "ClaimAmount"},
    },
}


def validate_schema(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    """Raise a useful error if a downloaded table does not match its contract."""
    actual = set(frame.columns)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Unexpected schema for {name}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if frame.empty:
        raise ValueError(f"Downloaded dataset {name} is empty")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(output_dir: Path, cache_dir: Path, force: bool = False) -> dict:
    """Fetch both pinned datasets, validate them, and write a data manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": "OpenML",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    for name, spec in DATASETS.items():
        destination = output_dir / str(spec["filename"])
        if destination.exists() and not force:
            frame = pd.read_parquet(destination)
            action = "reused"
        else:
            bunch = fetch_openml(
                data_id=int(spec["data_id"]),
                as_frame=True,
                data_home=cache_dir,
                parser="auto",
            )
            frame = bunch.data
            validate_schema(frame, spec["columns"], name)  # type: ignore[arg-type]
            temporary = destination.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(destination)
            action = "downloaded"

        validate_schema(frame, spec["columns"], name)  # type: ignore[arg-type]
        manifest["datasets"][name] = {
            "openml_data_id": spec["data_id"],
            "file": str(destination.relative_to(PROJECT_ROOT)),
            "rows": len(frame),
            "columns": list(frame.columns),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "action": action,
        }
        print(f"{name}: {action} {len(frame):,} rows -> {destination}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--force", action="store_true", help="replace existing raw Parquet files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    acquire(args.output_dir.resolve(), args.cache_dir.resolve(), args.force)

