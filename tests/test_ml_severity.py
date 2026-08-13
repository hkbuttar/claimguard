from pathlib import Path

import pandas as pd

from severity.ml_models import (
    build_comparison,
    evaluate_predictions,
    prepare_native_features,
)


def test_prepare_native_features_marks_categories() -> None:
    claims = pd.DataFrame(
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
    result = prepare_native_features(claims)
    assert str(result["Area"].dtype) == "category"
    assert str(result["VehPower"].dtype) == "category"


def test_evaluate_predictions_returns_all_nonlinear_models() -> None:
    predictions = pd.DataFrame(
        {
            "ClaimAmount": [100.0, 200.0],
            "RandomForestPrediction": [110.0, 190.0],
            "HistGradientBoostingPrediction": [100.0, 200.0],
            "XGBoostPrediction": [90.0, 210.0],
        }
    )
    metrics = evaluate_predictions(predictions)
    assert set(metrics) == {"random_forest", "hist_gradient_boosting", "xgboost"}
    assert metrics["hist_gradient_boosting"]["mae"] == 0


def test_comparison_works_without_traditional_artifact() -> None:
    metric = {
        "mae": 1.0,
        "rmse": 1.0,
        "mean_gamma_deviance": 1.0,
        "mean_bias": 0.0,
        "observed_aggregate_loss": 10.0,
        "predicted_aggregate_loss": 10.0,
        "aggregate_bias": 0.0,
        "predicted_observed_ratio": 1.0,
    }
    metrics = {
        "random_forest": metric,
        "hist_gradient_boosting": metric,
        "xgboost": metric,
    }
    result = build_comparison(metrics, Path("missing.json"))
    assert result["Model"].tolist() == [
        "Random Forest",
        "HistGradientBoosting",
        "XGBoost",
    ]
