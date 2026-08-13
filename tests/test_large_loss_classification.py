import numpy as np
import pandas as pd
import pytest

from tail_risk.large_loss_classification import (
    evaluate_classifier,
    probability_calibration,
    select_f1_threshold,
)


def test_select_f1_threshold_uses_probability_scores() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.9])
    assert select_f1_threshold(actual, probability) == pytest.approx(0.7)


def test_classifier_metrics_emphasize_rare_event_quality() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.9])
    metrics = evaluate_classifier(actual, probability, 0.5)
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert "accuracy" not in metrics


def test_probability_calibration_preserves_event_counts() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.9])
    result = probability_calibration(actual, probability, bins=2)
    assert result["Claims"].sum() == 4
    assert result["LargeClaims"].sum() == 2
