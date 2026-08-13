import numpy as np
import pandas as pd
import pytest

from severity.traditional_models import (
    evaluate_severity,
    severity_calibration,
    smearing_factor,
    split_claims_by_policy,
)


def test_policy_group_split_is_disjoint_and_reproducible() -> None:
    claims = pd.DataFrame(
        {"IDpol": np.repeat(range(20), 2), "ClaimAmount": np.arange(40) + 1}
    )
    train, test = split_claims_by_policy(claims)
    second_train, second_test = split_claims_by_policy(claims)
    assert set(train["IDpol"]).isdisjoint(test["IDpol"])
    assert train.index.tolist() == second_train.index.tolist()
    assert test.index.tolist() == second_test.index.tolist()


def test_smearing_factor_corrects_log_retransformation() -> None:
    residuals = pd.Series([np.log(0.5), np.log(1.5)])
    assert smearing_factor(residuals) == pytest.approx(1.0)


def test_severity_metrics_include_aggregate_bias() -> None:
    metrics = evaluate_severity(
        pd.Series([100.0, 300.0]), pd.Series([150.0, 250.0])
    )
    assert metrics["mae"] == 50.0
    assert metrics["aggregate_bias"] == 0.0
    assert metrics["predicted_observed_ratio"] == 1.0


def test_severity_calibration_aggregates_deciles() -> None:
    result = severity_calibration(
        pd.Series([100.0, 200.0, 300.0, 400.0]),
        pd.Series([90.0, 210.0, 280.0, 420.0]),
        bins=2,
    )
    assert len(result) == 2
    assert result["Claims"].tolist() == [2, 2]
