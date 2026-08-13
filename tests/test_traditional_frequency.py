import pandas as pd
import pytest

from frequency.traditional_models import (
    calibration_table,
    estimate_nb_alpha,
    evaluate_frequency,
    split_portfolio,
)


def test_split_is_reproducible_and_disjoint() -> None:
    policy = pd.DataFrame({"IDpol": range(100), "ClaimNb": 0, "Exposure": 1.0})
    first_train, first_test = split_portfolio(policy)
    second_train, second_test = split_portfolio(policy)
    assert first_train["IDpol"].tolist() == second_train["IDpol"].tolist()
    assert first_test["IDpol"].tolist() == second_test["IDpol"].tolist()
    assert set(first_train["IDpol"]).isdisjoint(first_test["IDpol"])


def test_nb_alpha_is_positive() -> None:
    actual = pd.Series([0, 0, 0, 5, 6])
    fitted = pd.Series([1.0] * 5)
    assert estimate_nb_alpha(actual, fitted) > 0


def test_frequency_evaluation_reports_observed_expected_ratio() -> None:
    metrics = evaluate_frequency(
        pd.Series([0, 2]), pd.Series([0.5, 1.5]), pd.Series([1.0, 1.0])
    )
    assert metrics["observed_claims"] == 2
    assert metrics["expected_claims"] == 2
    assert metrics["observed_expected_ratio"] == pytest.approx(1.0)


def test_calibration_uses_exposure_weighted_frequency() -> None:
    result = calibration_table(
        actual=pd.Series([0, 1, 0, 2]),
        predicted_count=pd.Series([0.1, 0.2, 0.3, 0.4]),
        exposure=pd.Series([0.5, 0.5, 1.0, 1.0]),
        predicted_rate=pd.Series([0.1, 0.2, 0.3, 0.4]),
        bins=2,
    )
    assert result.loc[0, "ObservedFrequency"] == 1.0
    assert result.loc[1, "ObservedFrequency"] == 1.0
