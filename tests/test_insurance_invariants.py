"""Cross-module checks using a portfolio with mathematically known results."""

import pandas as pd
import pytest

from calibration.calibration_analysis import aggregate_calibration
from integration.actuarial_baselines import fit_baseline_rates
from preprocessing.audit_data import audit_tables
from preprocessing.build_tables import build_modeling_tables
from pure_premium.component_models import combine_components
from segmentation.risk_deciles import assign_risk_deciles


def test_known_portfolio_audit_and_join(
    known_insurance_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    policies, claims = known_insurance_tables
    results, policy_flags, claim_flags = audit_tables(policies, claims)
    counts = {(item.dataset, item.rule): item.count for item in results}

    assert policies["IDpol"].is_unique
    assert counts[("freMTPL2sev", "unmatched_policy")] == 1
    assert counts[("cross_table", "claim_count_mismatch")] == 0
    assert not policy_flags["claim_count_mismatch"].any()
    assert claim_flags["unmatched_policy"].tolist() == [False, False, False, True]


def test_known_portfolio_aggregation_preserves_claims_and_zero_losses(
    known_insurance_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    policies, claims = known_insurance_tables
    frequency, severity, loss = build_modeling_tables(policies, claims)

    assert len(frequency) == 4
    assert len(severity) == 3
    assert severity["ClaimAmount"].sum() == 600.0
    assert loss["TotalLoss"].tolist() == [400.0, 0.0, 200.0, 0.0]
    assert loss["ObservedClaimNb"].tolist() == [2, 0, 1, 0]


def test_known_portfolio_frequency_and_pure_premium_arithmetic(
    known_insurance_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    policies, claims = known_insurance_tables
    _, severity, _ = build_modeling_tables(policies, claims)
    rates = fit_baseline_rates(policies, severity)

    assert rates.annual_claim_frequency == pytest.approx(3 / 2.75)
    assert rates.mean_claim_severity == 200.0
    assert rates.annual_pure_premium_mean_severity == pytest.approx(600 / 2.75)

    combined = combine_components(
        policies,
        {
            "PoissonAnnualFrequency": pd.Series([2.0] * 4),
            "HistGradientBoostingAnnualFrequency": pd.Series([2.0] * 4),
            "XGBoostAnnualFrequency": pd.Series([2.0] * 4),
        },
        {
            "GammaExpectedSeverity": pd.Series([100.0] * 4),
            "LognormalExpectedSeverity": pd.Series([150.0] * 4),
            "HistGradientBoostingExpectedSeverity": pd.Series([100.0] * 4),
            "XGBoostExpectedSeverity": pd.Series([100.0] * 4),
        },
    )
    assert combined["PoissonGammaAnnualPurePremium"].eq(200.0).all()
    assert combined["PoissonGammaExpectedLoss"].tolist() == [200.0, 100.0, 50.0, 200.0]


def test_known_calibration_and_risk_groups_are_exact() -> None:
    calibration = aggregate_calibration(
        pd.Series([100.0, 200.0]), pd.Series([120.0, 180.0])
    )
    assert calibration["observed_expected_ratio"] == 1.0
    assert calibration["aggregate_bias"] == 0.0

    scores = pd.Series(range(20), dtype=float)
    groups = assign_risk_deciles(scores, pd.Series(range(20)))
    assert groups.value_counts().sort_index().tolist() == [2] * 10
    assert groups.is_monotonic_increasing
