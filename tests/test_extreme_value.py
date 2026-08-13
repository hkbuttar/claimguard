import pandas as pd
import pytest

from tail_risk.extreme_value import fit_gpd, gpd_quantile, gpd_survival


def test_gpd_fit_requires_sufficient_exceedances() -> None:
    with pytest.raises(ValueError, match="50"):
        fit_gpd(pd.Series(range(100)), threshold=90)


def test_gpd_survival_equals_tail_probability_at_threshold() -> None:
    fit = {
        "threshold": 100.0,
        "shape": 0.2,
        "scale": 50.0,
        "exceedance_probability": 0.05,
    }
    assert gpd_survival(100.0, fit) == pytest.approx(0.05)
    assert gpd_survival(200.0, fit) < 0.05


def test_gpd_quantile_inverts_survival_probability() -> None:
    fit = {
        "threshold": 100.0,
        "shape": 0.2,
        "scale": 50.0,
        "exceedance_probability": 0.05,
    }
    quantile = gpd_quantile(0.99, fit)
    assert gpd_survival(quantile, fit) == pytest.approx(0.01)


def test_gpd_quantile_rejects_body_probability() -> None:
    fit = {
        "threshold": 100.0,
        "shape": 0.2,
        "scale": 50.0,
        "exceedance_probability": 0.05,
    }
    with pytest.raises(ValueError, match="below"):
        gpd_quantile(0.90, fit)
