from pathlib import Path

import pytest

from deployment.build_bundle import build_bundle
from deployment.validate_bundle import validate_bundle


def test_bundle_is_checksummed_and_preserves_relative_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "reports/model").mkdir(parents=True)
    (project / "reports/model/metrics.json").write_text('{"metric": 1}')
    output = tmp_path / "bundle"

    manifest = build_bundle(
        project, output, ("reports/model/metrics.json",)
    )

    assert manifest["total_bytes"] > 0
    assert (output / "reports/model/metrics.json").is_file()
    assert validate_bundle(output)["files"][0]["path"] == (
        "reports/model/metrics.json"
    )


def test_bundle_validation_detects_tampering(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "model.bin").write_bytes(b"original")
    output = tmp_path / "bundle"
    build_bundle(project, output, ("model.bin",))
    (output / "model.bin").write_bytes(b"modified")

    with pytest.raises(ValueError, match="mismatch"):
        validate_bundle(output)


def test_bundle_builder_refuses_missing_or_existing_targets(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing production artifacts"):
        build_bundle(project, tmp_path / "bundle", ("missing",))

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_bundle(project, output, ())
