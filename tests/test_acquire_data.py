from pathlib import Path

import pandas as pd
import pytest

from preprocessing.acquire_data import sha256, validate_schema


def test_validate_schema_accepts_expected_columns() -> None:
    validate_schema(
        pd.DataFrame({"IDpol": [1], "ClaimAmount": [100.0]}),
        {"IDpol", "ClaimAmount"},
        "severity",
    )


def test_validate_schema_rejects_schema_drift() -> None:
    with pytest.raises(ValueError, match="missing=.*ClaimAmount"):
        validate_schema(
            pd.DataFrame({"IDpol": [1]}),
            {"IDpol", "ClaimAmount"},
            "severity",
        )


def test_sha256(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"ClaimGuard")
    assert sha256(sample) == "b3ec73480b077dde619d01d778ddb0798c9c88ae80f714fc41ac97dbeb4c1b1b"
