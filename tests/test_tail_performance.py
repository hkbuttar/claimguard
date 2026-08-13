import pandas as pd

from tail_risk.tail_performance import (
    audit_tail_performance,
    evaluate_segment,
    segment_masks,
)


def test_segment_masks_use_development_thresholds() -> None:
    amounts = pd.Series([10.0, 20.0, 30.0, 40.0])
    masks = segment_masks(amounts, {"q90": 20.0, "q95": 30.0, "q99": 35.0})
    assert masks["Bottom 90%"].sum() == 2
    assert masks["Top 10%"].sum() == 2
    assert masks["Top 5%"].sum() == 1
    assert masks["Top 1%"].sum() == 1


def test_segment_metrics_report_aggregate_underprediction() -> None:
    metrics = evaluate_segment(
        pd.Series([100.0, 300.0]), pd.Series([50.0, 150.0])
    )
    assert metrics["mae"] == 100.0
    assert metrics["aggregate_bias"] == -200.0
    assert metrics["predicted_observed_ratio"] == 0.5
    assert metrics["aggregate_underprediction_rate"] == 0.5


def test_audit_evaluates_every_model_and_segment() -> None:
    predictions = pd.DataFrame({"ClaimAmount": [10.0, 20.0, 30.0, 40.0]})
    for column in (
        "MeanPrediction",
        "GammaPrediction",
        "LognormalPrediction",
        "RandomForestPrediction",
        "HistGradientBoostingPrediction",
        "XGBoostPrediction",
    ):
        predictions[column] = 20.0
    result = audit_tail_performance(
        predictions, {"q90": 20.0, "q95": 30.0, "q99": 35.0}
    )
    assert len(result) == 30
    assert result["Model"].nunique() == 6
    assert result["Segment"].nunique() == 5
