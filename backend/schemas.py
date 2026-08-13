"""Typed HTTP contracts for policy scoring."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exposure: float = Field(gt=0, le=1)
    area: str
    vehicle_power: int = Field(ge=1)
    vehicle_age: float = Field(ge=0)
    driver_age: float = Field(gt=0)
    bonus_malus: float = Field(ge=0)
    vehicle_brand: str
    fuel_type: str
    density: float = Field(ge=0)
    region: str

    def to_engine_policy(self) -> dict:
        return {
            "Exposure": self.exposure,
            "Area": self.area,
            "VehPower": self.vehicle_power,
            "VehAge": self.vehicle_age,
            "DrivAge": self.driver_age,
            "BonusMalus": self.bonus_malus,
            "VehBrand": self.vehicle_brand,
            "VehGas": self.fuel_type,
            "Density": self.density,
            "Region": self.region,
        }


class RiskProfileResponse(BaseModel):
    expected_claims_per_year: float = Field(ge=0)
    expected_claim_severity: float = Field(ge=0)
    expected_annual_loss: float = Field(ge=0)
    expected_loss_for_exposure: float = Field(ge=0)
    large_loss_probability: float = Field(ge=0, le=1)
    tail_risk_percentile: float = Field(ge=0, le=100)
    frequency_risk: Literal["LOW", "MEDIUM", "HIGH"]
    severity_risk: Literal["LOW", "MEDIUM", "HIGH"]
    overall_risk: Literal["LOW", "MEDIUM", "HIGH"]
    risk_segment: Literal[
        "Standard Risk",
        "Frequent Claimant",
        "Catastrophic Exposure",
        "Critical Risk",
    ]
    claimguard_score: float = Field(ge=0, le=100)
    primary_risk_drivers: tuple[str, ...]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    inference_only: Literal[True]
