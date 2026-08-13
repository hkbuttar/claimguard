import pandas as pd
import pytest

from portfolio.exploratory_analysis import (
    frequency_by_group,
    loss_concentration,
    portfolio_frequency,
    severity_statistics,
)


def test_portfolio_frequency_uses_exposure() -> None:
    policy = pd.DataFrame({"ClaimNb": [1, 2], "Exposure": [0.5, 1.0]})
    assert portfolio_frequency(policy) == 2.0


def test_frequency_by_group_aggregates_claims_and_exposure() -> None:
    policy = pd.DataFrame({"ClaimNb": [1, 0, 2], "Exposure": [0.5, 0.5, 1.0]})
    result = frequency_by_group(policy, pd.Series(["A", "A", "B"]))
    assert result["Frequency"].tolist() == [1.0, 2.0]
    assert result["Policies"].tolist() == [2, 1]


def test_loss_concentration_uses_largest_claims() -> None:
    result = loss_concentration(pd.Series([70.0, 20.0, 10.0]), shares=(1 / 3,))
    assert result.loc[0, "ClaimCount"] == 1
    assert result.loc[0, "LossShare"] == pytest.approx(0.7)


def test_severity_statistics_include_log_distribution() -> None:
    result = severity_statistics(pd.Series([10.0, 20.0, 40.0]))
    assert result["count"] == 3
    assert result["median"] == 20.0
    assert result["log_std"] > 0

