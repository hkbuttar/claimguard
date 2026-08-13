from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from integration.risk_engine import RiskProfile


class StubEngine:
    def score_policy(self, policy: dict) -> RiskProfile:
        if policy["Region"] == "unknown":
            raise ValueError("Unknown Region category: unknown")
        return RiskProfile(
            expected_claims_per_year=0.1,
            expected_claim_severity=2_000.0,
            expected_annual_loss=200.0,
            expected_loss_for_exposure=200.0 * policy["Exposure"],
            large_loss_probability=0.05,
            tail_risk_percentile=70.0,
            frequency_risk="MEDIUM",
            severity_risk="HIGH",
            overall_risk="HIGH",
            risk_segment="Critical Risk",
            claimguard_score=72.0,
            primary_risk_drivers=("BonusMalus",),
        )


def valid_payload() -> dict:
    return {
        "exposure": 0.5,
        "area": "A",
        "vehicle_power": 5,
        "vehicle_age": 4,
        "driver_age": 40,
        "bonus_malus": 60,
        "vehicle_brand": "B1",
        "fuel_type": "Regular",
        "density": 500,
        "region": "R1",
    }


def test_health_and_policy_scoring_contract(tmp_path: Path) -> None:
    with TestClient(create_app(StubEngine(), tmp_path)) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "inference_only": True,
        }
        response = client.post("/policy/score", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["expected_annual_loss"] == 200.0
    assert body["expected_loss_for_exposure"] == 100.0
    assert body["claimguard_score"] == 72.0


def test_policy_schema_and_engine_errors_return_422(tmp_path: Path) -> None:
    with TestClient(create_app(StubEngine(), tmp_path)) as client:
        invalid = valid_payload()
        invalid["exposure"] = 0
        assert client.post("/policy/score", json=invalid).status_code == 422

        unknown = valid_payload()
        unknown["region"] = "unknown"
        response = client.post("/policy/score", json=unknown)
        assert response.status_code == 422
        assert "Unknown Region" in response.json()["detail"]


def test_report_endpoints_are_read_only_artifact_views(tmp_path: Path) -> None:
    paths = {
        "portfolio_analysis/summary.json": '{"policies": 4}',
        "portfolio_stress/metrics.json": '{"var_99": 1000}',
        "ml_frequency/metrics.json": '{"winner": "gbm"}',
        "ml_severity/metrics.json": '{"winner": "gamma"}',
        "ml_pure_premium/metrics.json": '{"winner": "tweedie"}',
        "calibration/metrics.json": '{"ratio": 1.0}',
        "extreme_value/metrics.json": '{"threshold": 5000}',
        "bonus_malus/metrics.json": '{"levels": 3}',
        "risk_segments/segment_summary.csv": "RiskSegment,Policies\nStandard Risk,4\n",
        "model_benchmark/model_benchmark.csv": "Task,Model\nFrequency,Poisson\n",
        "extreme_value/high_quantile_comparison.csv": "Probability,EVT\n0.99,1000\n",
        "bonus_malus/observed_by_bonus_malus.csv": "BonusMalus,PurePremium\n50,100\n",
    }
    for model in (
        "poisson_gamma",
        "poisson_lognormal",
        "tweedie",
        "gbm_component",
        "direct_boosting",
    ):
        paths[f"risk_deciles/{model}_deciles.csv"] = (
            "RiskDecile,ObservedLossCost\n1,100\n"
        )
    for relative, content in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    with TestClient(create_app(StubEngine(), tmp_path)) as client:
        assert client.get("/portfolio/summary").json()["policies"] == 4
        assert client.get("/portfolio/segments").json()[0]["Policies"] == 4
        assert client.get("/portfolio/stress-test").json()["var_99"] == 1000
        assert client.get("/models/frequency").json()["winner"] == "gbm"
        assert client.get("/models/severity").json()["winner"] == "gamma"
        assert client.get("/models/pure-premium").json()["winner"] == "tweedie"
        assert client.get("/models/calibration").json()["ratio"] == 1.0
        assert client.get("/models/benchmark").json()[0]["Model"] == "Poisson"
        assert client.get("/tail-risk").json()["threshold"] == 5000
        assert client.get("/tail-risk/quantiles").json()[0]["EVT"] == 1000
        assert client.get("/bonus-malus").json()["levels"] == 3
        assert client.get("/bonus-malus/observed").json()[0]["PurePremium"] == 100
        assert client.get("/portfolio/risk-deciles").json()["tweedie"][0][
            "RiskDecile"
        ] == 1


def test_dashboard_assets_are_served(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>ClaimGuard</h1>")
    with TestClient(create_app(StubEngine(), tmp_path, frontend)) as client:
        redirect = client.get("/", follow_redirects=False)
        dashboard = client.get("/dashboard/")
    assert redirect.status_code == 307
    assert dashboard.status_code == 200
    assert "ClaimGuard" in dashboard.text


def test_missing_report_returns_service_unavailable(tmp_path: Path) -> None:
    with TestClient(create_app(StubEngine(), tmp_path)) as client:
        response = client.get("/models/frequency")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
