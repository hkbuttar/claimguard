import pandas as pd
import pytest

from severity.quantile_models import (
    evaluate_quantile,
    quantile_calibration,
    quantile_crossing_rate,
)


def test_quantile_evaluation_reports_coverage_and_improvement() -> None:
    metrics = evaluate_quantile(
        actual=pd.Series([10.0, 20.0, 30.0, 40.0]),
        predicted=pd.Series([15.0, 25.0, 35.0, 45.0]),
        quantile=0.5,
        baseline=25.0,
    )
    assert metrics["empirical_coverage"] == 1.0
    assert metrics["coverage_error"] == 0.5
    assert metrics["pinball_loss"] == pytest.approx(2.5)


def test_quantile_crossing_rate_detects_crossed_rows() -> None:
    predictions = pd.DataFrame(
        {
            "Q50Prediction": [10.0, 20.0],
            "Q75Prediction": [15.0, 19.0],
            "Q90Prediction": [20.0, 30.0],
            "Q95Prediction": [25.0, 40.0],
        }
    )
    assert quantile_crossing_rate(predictions) == 0.5


def test_quantile_calibration_preserves_claims() -> None:
    result = quantile_calibration(
        pd.Series([10.0, 20.0, 30.0, 40.0]),
        pd.Series([15.0, 25.0, 35.0, 45.0]),
        bins=2,
    )
    assert result["Claims"].sum() == 4
    assert result["EmpiricalCoverage"].eq(1.0).all()
