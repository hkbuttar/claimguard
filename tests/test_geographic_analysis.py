import pandas as pd

from portfolio.geographic_analysis import (
    portfolio_benchmarks,
    summarize_geography,
)


def test_geographic_summary_decomposes_frequency_and_severity() -> None:
    policy = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "Region": ["A", "A", "B"],
            "Density": [100, 200, 300],
            "Exposure": [0.5, 0.5, 1.0],
            "ClaimNb": [1, 0, 1],
            "ObservedClaimNb": [1, 0, 1],
            "TotalLoss": [100.0, 0.0, 300.0],
        }
    )
    benchmarks = portfolio_benchmarks(policy)
    result = summarize_geography(policy, policy["Region"], "Region", benchmarks)
    first = result.loc[result["Region"].eq("A")].iloc[0]
    assert first["ClaimFrequency"] == 1.0
    assert first["ClaimSeverity"] == 100.0
    assert first["PurePremium"] == 100.0
    assert result["ObservedLoss"].sum() == 400.0


def test_portfolio_benchmarks_use_correct_denominators() -> None:
    policy = pd.DataFrame(
        {
            "ClaimNb": [2],
            "ObservedClaimNb": [1],
            "Exposure": [0.5],
            "TotalLoss": [100.0],
        }
    )
    result = portfolio_benchmarks(policy)
    assert result == {"frequency": 4.0, "severity": 100.0, "pure_premium": 200.0}
