import pandas as pd
import pytest

from pure_premium.tweedie_model import (
    evaluate_policy_loss,
    make_tweedie_pipeline,
    premium_calibration,
)


def test_tweedie_power_must_represent_compound_loss() -> None:
    with pytest.raises(ValueError, match="between 1 and 2"):
        make_tweedie_pipeline(power=2.0)


def test_policy_loss_metrics_support_zero_mass() -> None:
    metrics = evaluate_policy_loss(
        pd.Series([0.0, 100.0]), pd.Series([10.0, 90.0])
    )
    assert metrics["mae"] == 10.0
    assert metrics["predicted_observed_ratio"] == 1.0
    assert metrics["mean_tweedie_deviance"] > 0


def test_premium_calibration_uses_aggregate_loss() -> None:
    result = premium_calibration(
        actual_loss=pd.Series([0.0, 100.0, 0.0, 200.0]),
        predicted_loss=pd.Series([10.0, 90.0, 20.0, 180.0]),
        exposure=pd.Series([1.0, 1.0, 1.0, 1.0]),
        annual_premium=pd.Series([10.0, 90.0, 20.0, 180.0]),
        bins=2,
    )
    assert len(result) == 2
    assert result["ObservedLoss"].sum() == 300.0
    assert result["ExpectedLoss"].sum() == 300.0
