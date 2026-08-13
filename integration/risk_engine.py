"""Unified policy scoring across frequency, severity, premium, and tail risk."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol

import joblib
import numpy as np
import pandas as pd

from severity.ml_models import CATEGORICAL_FEATURES, FEATURES

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "policy_loss.parquet"
DEFAULT_REPORTS: Final = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_DIR: Final = DEFAULT_REPORTS / "risk_engine"
NUMERIC_FEATURES: Final = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]


class Regressor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class Classifier(Protocol):
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class RiskProfile:
    expected_claims_per_year: float
    expected_claim_severity: float
    expected_annual_loss: float
    expected_loss_for_exposure: float
    large_loss_probability: float
    tail_risk_percentile: float
    frequency_risk: str
    severity_risk: str
    overall_risk: str
    risk_segment: str
    claimguard_score: float
    primary_risk_drivers: tuple[str, ...]


def percentile_rank(reference: np.ndarray, value: float) -> float:
    """Return an inclusive empirical percentile on a zero-to-one scale."""
    clean = np.sort(np.asarray(reference, dtype=float))
    if clean.size == 0 or not np.isfinite(clean).all():
        raise ValueError("Reference distributions must be finite and non-empty")
    return float(np.searchsorted(clean, value, side="right") / clean.size)


def risk_label(percentile: float) -> str:
    if percentile >= 2 / 3:
        return "HIGH"
    if percentile >= 1 / 3:
        return "MEDIUM"
    return "LOW"


class ClaimGuardRiskEngine:
    """Score individual policies with fitted ClaimGuard artifacts."""

    def __init__(
        self,
        frequency_model: Regressor,
        severity_model: Regressor,
        large_loss_model: Classifier,
        category_levels: dict[str, list],
        reference_scores: dict[str, np.ndarray],
        thresholds: dict,
        primary_drivers: tuple[str, ...],
    ) -> None:
        self.frequency_model = frequency_model
        self.severity_model = severity_model
        self.large_loss_model = large_loss_model
        self.category_levels = category_levels
        self.reference_scores = reference_scores
        self.thresholds = thresholds
        self.primary_drivers = primary_drivers

    @classmethod
    def from_artifacts(
        cls,
        data_path: Path = DEFAULT_DATA_PATH,
        reports_dir: Path = DEFAULT_REPORTS,
    ) -> ClaimGuardRiskEngine:
        """Load fitted models and portfolio reference distributions."""
        component_dir = reports_dir / "component_pure_premium"
        segment_dir = reports_dir / "risk_segments"
        required = [
            data_path,
            component_dir / "models" / "frequency_hist_gradient_boosting.joblib",
            component_dir / "models" / "severity_hist_gradient_boosting.joblib",
            segment_dir / "large_loss_probability_model.joblib",
            segment_dir / "metrics.json",
            segment_dir / "policy_segments.parquet",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing risk-engine artifacts: {', '.join(missing)}")

        policy = pd.read_parquet(data_path, columns=FEATURES)
        segments = pd.read_parquet(segment_dir / "policy_segments.parquet")
        thresholds = json.loads((segment_dir / "metrics.json").read_text())["thresholds"]
        importance_path = reports_dir / "explainability" / "frequency_permutation_importance.csv"
        drivers = tuple(
            pd.read_csv(importance_path).sort_values("Importance", ascending=False)["Feature"].head(4)
        ) if importance_path.exists() else ("BonusMalus", "VehAge", "DrivAge", "Density")

        return cls(
            joblib.load(component_dir / "models" / "frequency_hist_gradient_boosting.joblib"),
            joblib.load(component_dir / "models" / "severity_hist_gradient_boosting.joblib"),
            joblib.load(segment_dir / "large_loss_probability_model.joblib"),
            {
                column: sorted(policy[column].dropna().unique().tolist())
                for column in CATEGORICAL_FEATURES
            },
            {
                "frequency": segments["PredictedAnnualFrequency"].to_numpy(),
                "severity": segments["ExpectedSeverity"].to_numpy(),
                "premium": segments["AnnualPurePremium"].to_numpy(),
                "tail": segments["LargeLossProbability"].to_numpy(),
            },
            thresholds,
            drivers,
        )

    def _prepare_policy(self, policy: dict) -> tuple[pd.DataFrame, float]:
        missing = [name for name in [*FEATURES, "Exposure"] if name not in policy]
        if missing:
            raise ValueError(f"Missing policy fields: {', '.join(missing)}")
        exposure = float(policy["Exposure"])
        if not 0 < exposure <= 1:
            raise ValueError("Exposure must be greater than zero and at most one year")
        frame = pd.DataFrame([{name: policy[name] for name in FEATURES}])
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(frame[column].iloc[0]):
                raise ValueError(f"{column} must be a finite number")
        for column in CATEGORICAL_FEATURES:
            value = frame[column].iloc[0]
            if value not in self.category_levels[column]:
                raise ValueError(f"Unknown {column} category: {value}")
            frame[column] = pd.Categorical(frame[column], categories=self.category_levels[column])
        return frame, exposure

    def score_policy(self, policy: dict) -> RiskProfile:
        """Return a complete, JSON-safe risk profile for one policy."""
        features, exposure = self._prepare_policy(policy)
        frequency = max(float(self.frequency_model.predict(features)[0]), 0.0)
        severity = max(float(self.severity_model.predict(features)[0]), 0.0)
        tail_probability = float(self.large_loss_model.predict_proba(features)[0, 1])
        tail_probability = float(np.clip(tail_probability, 0.0, 1.0))
        premium = frequency * severity

        percentiles = {
            name: percentile_rank(self.reference_scores[name], value)
            for name, value in {
                "frequency": frequency,
                "severity": severity,
                "premium": premium,
                "tail": tail_probability,
            }.items()
        }
        score = 100 * (
            0.30 * percentiles["frequency"]
            + 0.25 * percentiles["severity"]
            + 0.30 * percentiles["premium"]
            + 0.15 * percentiles["tail"]
        )
        high_frequency = frequency >= float(self.thresholds["frequency_median"])
        high_severity = severity >= float(self.thresholds["severity_median"])
        segment = {
            (False, False): "Standard Risk",
            (True, False): "Frequent Claimant",
            (False, True): "Catastrophic Exposure",
            (True, True): "Critical Risk",
        }[(high_frequency, high_severity)]
        return RiskProfile(
            expected_claims_per_year=frequency,
            expected_claim_severity=severity,
            expected_annual_loss=premium,
            expected_loss_for_exposure=premium * exposure,
            large_loss_probability=tail_probability,
            tail_risk_percentile=100 * percentiles["tail"],
            frequency_risk=risk_label(percentiles["frequency"]),
            severity_risk=risk_label(percentiles["severity"]),
            overall_risk=risk_label(score / 100),
            risk_segment=segment,
            claimguard_score=score,
            primary_risk_drivers=self.primary_drivers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="JSON file containing one policy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    engine = ClaimGuardRiskEngine.from_artifacts()
    if args.policy:
        policy = json.loads(args.policy.read_text())
    else:
        policy = pd.read_parquet(DEFAULT_DATA_PATH).iloc[len(pd.read_parquet(DEFAULT_DATA_PATH)) // 2].to_dict()
    profile = asdict(engine.score_policy(policy))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "example_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
