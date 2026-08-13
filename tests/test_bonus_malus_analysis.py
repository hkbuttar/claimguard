import pandas as pd

from portfolio.bonus_malus_analysis import summarize_bonus_malus


def test_bonus_malus_summary_uses_exposure_and_linked_claims() -> None:
    policy = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "BonusMalus": [50, 50, 80],
            "Exposure": [0.5, 0.5, 1.0],
            "ClaimNb": [1, 0, 1],
            "ObservedClaimNb": [1, 0, 1],
            "TotalLoss": [100.0, 0.0, 300.0],
        }
    )
    result = summarize_bonus_malus(policy)
    first = result.iloc[0]
    assert first["Policies"] == 2
    assert first["ClaimFrequency"] == 1.0
    assert first["ClaimSeverity"] == 100.0
    assert first["PurePremium"] == 100.0
    assert result["ObservedLoss"].sum() == 400.0
