"""Validate the deployment artifact manifest before serving traffic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from deployment.build_bundle import sha256

DEFAULT_BUNDLE: Final = Path(__file__).resolve().parent / "bundle"


def validate_bundle(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Deployment manifest is unavailable: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("files"):
        raise ValueError("Deployment manifest contains no files")
    for entry in manifest["files"]:
        path = bundle_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Bundled artifact is unavailable: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Bundled artifact size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Bundled artifact checksum mismatch: {entry['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    manifest = validate_bundle(args.bundle)
    print(f"Validated {len(manifest['files'])} deployment artifacts")


if __name__ == "__main__":
    main()
