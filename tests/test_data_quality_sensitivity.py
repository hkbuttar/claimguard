import pandas as pd

from preprocessing.data_quality_sensitivity import (
    build_scenarios,
    make_policy_loss,
)


def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frequency = pd.DataFrame(
        {
            "IDpol": [1.0, 2.0],
            "ClaimNb": [5, 0],
            "Exposure": [1.2, 1.0],
            "VehAge": [2, 3],
            "DrivAge": [30, 40],
            "BonusMalus": [50, 50],
        }
    )
    severity = pd.DataFrame(
        {"IDpol": [1, 1, 99], "ClaimAmount": [100.0, 300_000.0, 10.0]}
    )
    return frequency, severity


def test_benchmark_scenario_clips_without_changing_raw_inputs() -> None:
    frequency, severity = sample_data()
    scenarios = build_scenarios(frequency, severity)
    benchmark_frequency, benchmark_severity = scenarios["benchmark_compatible"]
    assert benchmark_frequency["ClaimNb"].max() == 4
    assert benchmark_frequency["Exposure"].max() == 1.0
    assert benchmark_severity["ClaimAmount"].max() == 200_000.0
    assert frequency["ClaimNb"].max() == 5
    assert severity["ClaimAmount"].max() == 300_000.0


def test_policy_loss_assigns_zero_to_no_claim_policy() -> None:
    frequency, severity = sample_data()
    minimal_frequency, minimal_severity = build_scenarios(frequency, severity)["minimal"]
    result = make_policy_loss(minimal_frequency, minimal_severity)
    assert result.loc[result["IDpol"].eq(1), "TotalLoss"].item() == 300_100.0
    assert result.loc[result["IDpol"].eq(2), "TotalLoss"].item() == 0.0
