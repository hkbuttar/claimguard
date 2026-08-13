import pandas as pd

from pure_premium.component_models import (
    combine_components,
    evaluate_pure_premiums,
)


def test_combines_annual_frequency_and_conditional_severity() -> None:
    policy = pd.DataFrame({"IDpol": [1, 2], "Exposure": [0.5, 1.0]})
    frequency = {
        "PoissonAnnualFrequency": [0.1, 0.2],
        "HistGradientBoostingAnnualFrequency": [0.1, 0.2],
        "XGBoostAnnualFrequency": [0.1, 0.2],
    }
    severity = {
        "GammaExpectedSeverity": [1_000, 2_000],
        "LognormalExpectedSeverity": [900, 1_800],
        "HistGradientBoostingExpectedSeverity": [800, 1_600],
        "XGBoostExpectedSeverity": [700, 1_400],
    }
    result = combine_components(policy, frequency, severity)
    assert result["PoissonGammaAnnualPurePremium"].tolist() == [100, 400]
    assert result["PoissonGammaExpectedLoss"].tolist() == [50, 400]


def test_pure_premium_evaluation_uses_exposure_period_loss() -> None:
    policy = pd.DataFrame({"IDpol": [1, 2], "Exposure": [0.5, 1.0]})
    frequency = {
        "PoissonAnnualFrequency": [0.1, 0.2],
        "HistGradientBoostingAnnualFrequency": [0.1, 0.2],
        "XGBoostAnnualFrequency": [0.1, 0.2],
    }
    severity = {
        "GammaExpectedSeverity": [1_000, 2_000],
        "LognormalExpectedSeverity": [1_000, 2_000],
        "HistGradientBoostingExpectedSeverity": [1_000, 2_000],
        "XGBoostExpectedSeverity": [1_000, 2_000],
    }
    predictions = combine_components(policy, frequency, severity)
    loss = pd.DataFrame({"IDpol": [1, 2], "TotalLoss": [50.0, 400.0]})
    comparison = evaluate_pure_premiums(predictions, loss)
    assert comparison["MAE"].eq(0).all()
    assert comparison["PredictedObservedRatio"].eq(1).all()
