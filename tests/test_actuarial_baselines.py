import pandas as pd
import pytest

from integration.actuarial_baselines import (
    evaluate_baselines,
    fit_baseline_rates,
    make_policy_predictions,
    make_severity_predictions,
)


def test_rates_use_total_exposure_and_claim_severity() -> None:
    policy = pd.DataFrame(
        {"IDpol": [1, 2], "ClaimNb": [1, 2], "Exposure": [0.5, 1.0]}
    )
    claims = pd.DataFrame({"ClaimAmount": [100.0, 300.0]})
    rates = fit_baseline_rates(policy, claims)
    assert rates.annual_claim_frequency == 2.0
    assert rates.mean_claim_severity == 200.0
    assert rates.median_claim_severity == 200.0
    assert rates.annual_pure_premium_mean_severity == 400.0


def test_policy_predictions_respect_exposure() -> None:
    policy = pd.DataFrame(
        {"IDpol": [1, 2], "ClaimNb": [1, 2], "Exposure": [0.5, 1.0]}
    )
    claims = pd.DataFrame({"ClaimAmount": [100.0, 300.0]})
    rates = fit_baseline_rates(policy, claims)
    predictions = make_policy_predictions(policy, rates)
    assert predictions["PredictedClaimCount"].tolist() == [1.0, 2.0]
    assert predictions["PredictedLossMean"].tolist() == [200.0, 400.0]


def test_evaluation_reports_fitted_frequency_balance() -> None:
    policy = pd.DataFrame(
        {"IDpol": [1, 2], "ClaimNb": [1, 2], "Exposure": [0.5, 1.0]}
    )
    claims = pd.DataFrame(
        {
            "SourceRow": [0, 1],
            "IDpol": [1, 2],
            "ClaimIndex": [1, 1],
            "ClaimAmount": [100.0, 300.0],
        }
    )
    loss = pd.DataFrame({"IDpol": [1, 2], "TotalLoss": [100.0, 300.0]})
    rates = fit_baseline_rates(policy, claims)
    metrics = evaluate_baselines(
        make_policy_predictions(policy, rates),
        make_severity_predictions(claims, rates),
        loss,
    )
    assert metrics["frequency"]["aggregate_bias"] == pytest.approx(0.0)
    assert metrics["severity_mean"]["aggregate_bias"] == pytest.approx(0.0)
