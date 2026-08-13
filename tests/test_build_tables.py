import pandas as pd

from preprocessing.build_tables import build_modeling_tables


def sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = pd.DataFrame(
        {
            "IDpol": [1.0, 2.0, 3.0],
            "ClaimNb": [2, 0, 1],
            "Exposure": [1.2, 0.5, 1.0],
            "Area": ["A", "B", "C"],
            "VehPower": [5, 6, 7],
            "VehAge": [2, 3, 4],
            "DrivAge": [30, 40, 50],
            "BonusMalus": [50, 60, 70],
            "VehBrand": ["B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular"],
            "Density": [100, 200, 300],
            "Region": ["R1", "R2", "R3"],
        }
    )
    sev = pd.DataFrame(
        {
            "IDpol": [1, 1, 99],
            "ClaimAmount": [100.0, 250.0, 500.0],
        }
    )
    return freq, sev


def test_builds_three_reconciled_tables() -> None:
    freq, sev = sample_tables()
    policy_frequency, claim_severity, policy_loss = build_modeling_tables(freq, sev)

    assert len(policy_frequency) == 3
    assert len(claim_severity) == 2
    assert len(policy_loss) == 3
    assert policy_frequency.loc[0, "Exposure"] == 1.0
    assert policy_frequency.loc[0, "ExposureRaw"] == 1.2
    assert policy_frequency.loc[0, "ExposureCapped"]
    assert claim_severity["ClaimIndex"].tolist() == [1, 2]
    assert claim_severity["DrivAge"].tolist() == [30, 30]
    assert policy_loss["TotalLoss"].tolist() == [350.0, 0.0, 0.0]
    assert policy_loss["ObservedClaimNb"].tolist() == [2, 0, 0]


def test_policy_loss_reconciles_to_retained_claims() -> None:
    freq, sev = sample_tables()
    _, claim_severity, policy_loss = build_modeling_tables(freq, sev)

    assert policy_loss["TotalLoss"].sum() == claim_severity["ClaimAmount"].sum()
    assert policy_loss.loc[policy_loss["IDpol"].eq(2), "TotalLoss"].item() == 0
