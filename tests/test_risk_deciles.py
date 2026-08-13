import pandas as pd

from segmentation.risk_deciles import (
    assign_risk_deciles,
    decile_diagnostics,
    summarize_deciles,
)


def test_deciles_are_deterministic_and_nearly_equal() -> None:
    score = pd.Series([1.0] * 23)
    policy_id = pd.Series(range(23, 0, -1))
    result = assign_risk_deciles(score, policy_id)
    counts = result.value_counts()
    assert set(result.unique()) == set(range(1, 11))
    assert counts.max() - counts.min() <= 1
    assert result.loc[policy_id.eq(1)].item() == 1


def test_decile_summary_calculates_insurance_metrics() -> None:
    frame = pd.DataFrame(
        {
            "IDpol": range(10),
            "Exposure": [1.0] * 10,
            "ClaimNb": [0] * 9 + [1],
            "ObservedClaimNb": [0] * 9 + [1],
            "TotalLoss": [0.0] * 9 + [100.0],
            "Premium": range(1, 11),
            "ExpectedLoss": [10.0] * 10,
            "ExpectedClaims": [0.1] * 10,
        }
    )
    result = summarize_deciles(
        frame, "Premium", "ExpectedLoss", "ExpectedClaims", bins=5
    )
    assert result["Policies"].tolist() == [2] * 5
    assert result["ObservedLoss"].sum() == 100.0
    assert result["PredictedClaims"].sum() == 1.0
    assert result.iloc[-1]["RelativeObservedLossCost"] == 5.0


def test_decile_diagnostics_identify_strong_ranking() -> None:
    table = pd.DataFrame(
        {
            "RiskDecile": [1, 2, 3],
            "ObservedLossCost": [10.0, 20.0, 30.0],
            "ObservedLoss": [10.0, 20.0, 30.0],
            "PredictedLoss": [10.0, 20.0, 30.0],
        }
    )
    metrics = decile_diagnostics(table)
    assert metrics["strictly_increasing_observed_loss_cost"]
    assert metrics["highest_to_lowest_loss_cost_ratio"] == 3.0
    assert metrics["aggregate_observed_expected_ratio"] == 1.0
