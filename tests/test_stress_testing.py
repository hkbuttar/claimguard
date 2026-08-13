import numpy as np
import pandas as pd
import pytest

from portfolio.stress_testing import (
    risk_metrics,
    sample_gpd_excess,
    simulate_portfolio_losses,
)


def test_gpd_excess_samples_are_nonnegative() -> None:
    samples = sample_gpd_excess(np.random.default_rng(1), 1_000, 0.2, 50.0)
    assert len(samples) == 1_000
    assert np.all(samples >= 0)


def test_portfolio_simulation_is_reproducible() -> None:
    arguments = (100, 5.0, np.array([10.0, 20.0]), 0.1, 100.0, 0.2, 50.0)
    first = simulate_portfolio_losses(np.random.default_rng(42), *arguments)
    second = simulate_portfolio_losses(np.random.default_rng(42), *arguments)
    pd.testing.assert_frame_equal(first, second)
    assert (first["TotalLoss"] == first["BodyLoss"] + first["TailLoss"]).all()


def test_risk_metrics_calculate_expected_shortfall() -> None:
    metrics = risk_metrics(pd.Series(range(1, 101), dtype=float))
    assert metrics["mean"] == 50.5
    assert metrics["var_95"] == pytest.approx(95.05)
    assert metrics["expected_shortfall_95"] == 98.0
    assert metrics["expected_shortfall_99"] == 100.0
