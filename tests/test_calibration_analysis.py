import pandas as pd

from calibration.calibration_analysis import (
    aggregate_calibration,
    expected_calibration_error,
    segment_loss_calibration,
)


def test_aggregate_calibration_reports_ratio_and_bias() -> None:
    result = aggregate_calibration(
        pd.Series([10.0, 20.0]), pd.Series([12.0, 18.0])
    )
    assert result["observed_expected_ratio"] == 1.0
    assert result["aggregate_bias"] == 0.0


def test_expected_calibration_error_is_weighted() -> None:
    table = pd.DataFrame(
        {"Observed": [1.0, 3.0], "Expected": [2.0, 1.0], "Weight": [1.0, 3.0]}
    )
    result = expected_calibration_error(table, "Observed", "Expected", "Weight")
    assert result == 1.75


def test_segment_loss_calibration_preserves_totals() -> None:
    frame = pd.DataFrame(
        {
            "IDpol": [1, 2],
            "RiskSegment": ["A", "B"],
            "Exposure": [1.0, 1.0],
            "TotalLoss": [100.0, 200.0],
            "Prediction": [90.0, 210.0],
        }
    )
    result = segment_loss_calibration(frame, "Prediction")
    assert result["ObservedLoss"].sum() == 300.0
    assert result["ExpectedLoss"].sum() == 300.0
