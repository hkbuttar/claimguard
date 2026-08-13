import pandas as pd

from segmentation.risk_segments import (
    SEGMENT_ORDER,
    assign_segments,
    premium_tiers,
    summarize_segments,
)


def test_assigns_actuarial_quadrants() -> None:
    result = assign_segments(
        annual_frequency=pd.Series([0.1, 0.3, 0.1, 0.3]),
        expected_severity=pd.Series([1_000, 1_000, 3_000, 3_000]),
        frequency_threshold=0.2,
        severity_threshold=2_000,
    )
    assert result.tolist() == SEGMENT_ORDER


def test_premium_tiers_are_ordered() -> None:
    tiers, thresholds = premium_tiers(pd.Series(range(1, 10)))
    assert tiers.tolist() == ["Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High"]
    assert thresholds["lower"] < thresholds["upper"]


def test_segment_summary_reports_observed_metrics() -> None:
    frame = pd.DataFrame(
        {
            "RiskSegment": SEGMENT_ORDER,
            "IDpol": range(4),
            "Exposure": [1.0] * 4,
            "PredictedAnnualFrequency": [0.1, 0.2, 0.1, 0.2],
            "ExpectedSeverity": [100.0, 100.0, 200.0, 200.0],
            "AnnualPurePremium": [10.0, 20.0, 20.0, 40.0],
            "LargeLossProbability": [0.01, 0.01, 0.02, 0.02],
            "ElevatedTailRisk": [False, False, True, True],
            "PredictedLoss": [10.0, 20.0, 20.0, 40.0],
            "ClaimNb": [0, 1, 0, 1],
            "ObservedClaimNb": [0, 1, 0, 1],
            "TotalLoss": [0.0, 20.0, 0.0, 40.0],
        }
    )
    result = summarize_segments(frame)
    assert result["Policies"].sum() == 4
    assert result["ObservedLoss"].sum() == 60.0
    assert result.loc[result["RiskSegment"].eq("Critical Risk"), "ObservedExpectedRatio"].item() == 1.0
