"""Shared deterministic fixtures for insurance validation tests."""

import pandas as pd
import pytest


@pytest.fixture
def known_insurance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a small portfolio with exact, hand-checkable outcomes."""
    policies = pd.DataFrame(
        {
            "IDpol": [101, 102, 103, 104],
            "ClaimNb": [2, 0, 1, 0],
            "Exposure": [1.0, 0.5, 0.25, 1.0],
            "Area": ["A", "B", "C", "D"],
            "VehPower": [4, 5, 6, 7],
            "VehAge": [1, 2, 3, 4],
            "DrivAge": [25, 35, 45, 55],
            "BonusMalus": [50, 60, 70, 80],
            "VehBrand": ["B1", "B2", "B3", "B4"],
            "VehGas": ["Regular", "Diesel", "Regular", "Diesel"],
            "Density": [100, 200, 300, 400],
            "Region": ["R1", "R1", "R2", "R2"],
        }
    )
    claims = pd.DataFrame(
        {
            "IDpol": [101, 101, 103, 999],
            "ClaimAmount": [100.0, 300.0, 200.0, 900.0],
        }
    )
    return policies, claims
