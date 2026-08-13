from pathlib import Path

import pandas as pd

from frequency.ml_models import (
    build_comparison,
    evaluate_ml_predictions,
    prepare_features,
)


def test_prepare_features_marks_categories() -> None:
    policy = pd.DataFrame(
        {
            "Area": ["A"],
            "VehPower": [5],
            "VehAge": [2],
            "DrivAge": [30],
            "BonusMalus": [50],
            "VehBrand": ["B1"],
            "VehGas": ["Regular"],
            "Density": [100],
            "Region": ["R1"],
        }
    )
    result = prepare_features(policy)
    assert str(result["Area"].dtype) == "category"
    assert str(result["VehPower"].dtype) == "category"
    assert result.columns.tolist() == list(policy.columns)


def test_evaluate_ml_predictions_compares_all_models() -> None:
    predictions = pd.DataFrame(
        {
            "ClaimNb": [0, 1],
            "Exposure": [1.0, 1.0],
            "NaiveExpectedClaims": [0.5, 0.5],
            "HistGradientBoostingExpectedClaims": [0.25, 0.75],
            "XGBoostExpectedClaims": [0.2, 0.8],
        }
    )
    metrics = evaluate_ml_predictions(predictions)
    assert set(metrics) == {"naive_rate", "hist_gradient_boosting", "xgboost"}
    assert metrics["xgboost"]["observed_expected_ratio"] == 1.0


def test_comparison_works_without_optional_traditional_artifact() -> None:
    metric = {
        "poisson_deviance": 1.0,
        "mean_poisson_deviance": 0.5,
        "observed_claims": 1.0,
        "expected_claims": 1.0,
        "observed_expected_ratio": 1.0,
        "mean_observed_frequency": 0.5,
        "mean_expected_frequency": 0.5,
    }
    metrics = {
        "naive_rate": metric,
        "hist_gradient_boosting": metric,
        "xgboost": metric,
    }
    result = build_comparison(metrics, Path("does-not-exist.json"))
    assert result["Model"].tolist() == [
        "Naive Rate",
        "HistGradientBoosting",
        "XGBoost",
    ]
