import numpy as np
import pandas as pd
import pytest

from integration.risk_engine import ClaimGuardRiskEngine, percentile_rank, risk_label
from severity.ml_models import CATEGORICAL_FEATURES


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([self.value] * len(features))


class ConstantClassifier:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([0.8, 0.2], (len(features), 1))


@pytest.fixture
def engine() -> ClaimGuardRiskEngine:
    return ClaimGuardRiskEngine(
        ConstantRegressor(0.1),
        ConstantRegressor(2_000.0),
        ConstantClassifier(),
        {name: ["known"] for name in CATEGORICAL_FEATURES},
        {
            "frequency": np.array([0.05, 0.10, 0.20]),
            "severity": np.array([1_000.0, 2_000.0, 3_000.0]),
            "premium": np.array([100.0, 200.0, 300.0]),
            "tail": np.array([0.05, 0.10, 0.20]),
        },
        {"frequency_median": 0.08, "severity_median": 1_500.0},
        ("BonusMalus", "DrivAge"),
    )


def policy() -> dict:
    values = {name: "known" for name in CATEGORICAL_FEATURES}
    values.update({"VehAge": 5, "DrivAge": 40, "BonusMalus": 60, "Density": 500, "Exposure": 0.5})
    return values


def test_percentile_and_labels_have_explicit_boundaries() -> None:
    assert percentile_rank(np.array([1.0, 2.0, 3.0]), 2.0) == pytest.approx(2 / 3)
    assert risk_label(0.2) == "LOW"
    assert risk_label(0.5) == "MEDIUM"
    assert risk_label(0.9) == "HIGH"


def test_engine_combines_predictions_into_risk_profile(engine: ClaimGuardRiskEngine) -> None:
    profile = engine.score_policy(policy())
    assert profile.expected_claims_per_year == 0.1
    assert profile.expected_claim_severity == 2_000.0
    assert profile.expected_annual_loss == 200.0
    assert profile.expected_loss_for_exposure == 100.0
    assert profile.large_loss_probability == 0.2
    assert profile.tail_risk_percentile == pytest.approx(100.0)
    assert profile.risk_segment == "Critical Risk"
    assert 0 <= profile.claimguard_score <= 100


def test_engine_rejects_incomplete_or_unknown_policies(engine: ClaimGuardRiskEngine) -> None:
    incomplete = policy()
    incomplete.pop("Exposure")
    with pytest.raises(ValueError, match="Missing policy fields"):
        engine.score_policy(incomplete)

    unknown = policy()
    unknown["Region"] = "unseen"
    with pytest.raises(ValueError, match="Unknown Region category"):
        engine.score_policy(unknown)

    invalid_exposure = policy()
    invalid_exposure["Exposure"] = 0
    with pytest.raises(ValueError, match="Exposure"):
        engine.score_policy(invalid_exposure)
