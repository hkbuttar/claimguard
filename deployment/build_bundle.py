"""Build a minimal, checksummed inference artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = PROJECT_ROOT / "deployment" / "bundle"
REQUIRED_FILES: Final = (
    "data/processed/policy_loss.parquet",
    "reports/component_pure_premium/models/frequency_hist_gradient_boosting.joblib",
    "reports/component_pure_premium/models/severity_hist_gradient_boosting.joblib",
    "reports/risk_segments/large_loss_probability_model.joblib",
    "reports/risk_segments/metrics.json",
    "reports/risk_segments/policy_segments.parquet",
    "reports/risk_segments/segment_summary.csv",
    "reports/portfolio_analysis/summary.json",
    "reports/portfolio_stress/metrics.json",
    "reports/ml_frequency/metrics.json",
    "reports/ml_severity/metrics.json",
    "reports/ml_pure_premium/metrics.json",
    "reports/calibration/metrics.json",
    "reports/model_benchmark/model_benchmark.csv",
    "reports/risk_deciles/poisson_gamma_deciles.csv",
    "reports/risk_deciles/poisson_lognormal_deciles.csv",
    "reports/risk_deciles/tweedie_deciles.csv",
    "reports/risk_deciles/gbm_component_deciles.csv",
    "reports/risk_deciles/direct_boosting_deciles.csv",
    "reports/extreme_value/metrics.json",
    "reports/extreme_value/high_quantile_comparison.csv",
    "reports/bonus_malus/metrics.json",
    "reports/bonus_malus/observed_by_bonus_malus.csv",
    "reports/explainability/frequency_permutation_importance.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(
    project_root: Path,
    output_dir: Path,
    required_files: tuple[str, ...] = REQUIRED_FILES,
) -> dict:
    """Copy production artifacts atomically and return their manifest."""
    if output_dir.exists():
        raise FileExistsError(
            f"Bundle already exists: {output_dir}. Remove it explicitly before rebuilding."
        )
    missing = [name for name in required_files if not (project_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing production artifacts: {', '.join(missing)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="claimguard-bundle-", dir=output_dir.parent))
    try:
        entries = []
        for relative in required_files:
            source = project_root / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entries.append(
                {
                    "path": relative,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": entries,
            "total_bytes": sum(item["bytes"] for item in entries),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_bundle(PROJECT_ROOT, args.output.resolve())
    print(
        f"Built {len(manifest['files'])} files "
        f"({manifest['total_bytes'] / 1024 / 1024:.1f} MiB) at {args.output}"
    )


if __name__ == "__main__":
    main()
