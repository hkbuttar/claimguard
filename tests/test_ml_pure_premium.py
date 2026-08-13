import pandas as pd
import pytest

from pure_premium.ml_model import normalized_gini, risk_ranking_metrics


def test_normalized_gini_is_one_for_perfect_ranking() -> None:
    actual = pd.Series([0.0, 10.0, 30.0, 60.0])
    assert normalized_gini(actual, actual) == pytest.approx(1.0)


def test_normalized_gini_penalizes_reversed_ranking() -> None:
    actual = pd.Series([0.0, 10.0, 30.0, 60.0])
    reversed_score = pd.Series([60.0, 30.0, 10.0, 0.0])
    assert normalized_gini(actual, reversed_score) < 0


def test_ranking_reports_top_decile_loss_capture() -> None:
    actual = pd.Series([100.0, *([0.0] * 9)])
    score = pd.Series([10.0, *([1.0] * 9)])
    metrics = risk_ranking_metrics(actual, score)
    assert metrics["top_decile_loss_capture"] == 1.0
