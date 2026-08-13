"""FastAPI application serving pre-trained ClaimGuard models and reports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.artifacts import ArtifactRepository
from backend.schemas import HealthResponse, PolicyRequest, RiskProfileResponse
from integration.risk_engine import DEFAULT_REPORTS, ClaimGuardRiskEngine

DEFAULT_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def create_app(
    engine: ClaimGuardRiskEngine | None = None,
    reports_dir: Path = DEFAULT_REPORTS,
    frontend_dir: Path = DEFAULT_FRONTEND,
) -> FastAPI:
    """Create an inference-only API, optionally with an injected engine."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if engine is None:
            application.state.risk_engine = ClaimGuardRiskEngine.from_artifacts(
                reports_dir=reports_dir
            )
        yield

    application = FastAPI(
        title="ClaimGuard API",
        version="1.0.0",
        description="Inference and precomputed analytics for French motor TPL risk.",
        lifespan=lifespan,
    )
    if engine is not None:
        application.state.risk_engine = engine
    repository = ArtifactRepository(reports_dir)

    def report(directory: str, filename: str = "metrics.json") -> dict:
        try:
            return repository.json_report(directory, filename)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/health", response_model=HealthResponse)
    def health() -> dict:
        return {"status": "ok", "inference_only": True}

    @application.post("/policy/score", response_model=RiskProfileResponse)
    def score_policy(payload: PolicyRequest, request: Request) -> dict:
        try:
            profile = request.app.state.risk_engine.score_policy(
                payload.to_engine_policy()
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return asdict(profile)

    @application.get("/portfolio/summary")
    def portfolio_summary() -> dict:
        return report("portfolio_analysis", "summary.json")

    @application.get("/portfolio/segments")
    def portfolio_segments() -> list[dict]:
        try:
            return repository.csv_records("risk_segments", "segment_summary.csv")
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/portfolio/stress-test")
    def portfolio_stress_test() -> dict:
        return report("portfolio_stress")

    @application.get("/portfolio/risk-deciles")
    def portfolio_risk_deciles() -> dict[str, list[dict]]:
        models = (
            "poisson_gamma",
            "poisson_lognormal",
            "tweedie",
            "gbm_component",
            "direct_boosting",
        )
        try:
            return {
                model: repository.csv_records(
                    "risk_deciles", f"{model}_deciles.csv"
                )
                for model in models
            }
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/models/frequency")
    def frequency_models() -> dict:
        return report("ml_frequency")

    @application.get("/models/severity")
    def severity_models() -> dict:
        return report("ml_severity")

    @application.get("/models/pure-premium")
    def pure_premium_models() -> dict:
        return report("ml_pure_premium")

    @application.get("/models/calibration")
    def model_calibration() -> dict:
        return report("calibration")

    @application.get("/models/benchmark")
    def model_benchmark() -> list[dict]:
        try:
            return repository.csv_records("model_benchmark", "model_benchmark.csv")
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/tail-risk")
    def tail_risk() -> dict:
        return report("extreme_value")

    @application.get("/tail-risk/quantiles")
    def tail_quantiles() -> list[dict]:
        try:
            return repository.csv_records(
                "extreme_value", "high_quantile_comparison.csv"
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/bonus-malus")
    def bonus_malus() -> dict:
        return report("bonus_malus")

    @application.get("/bonus-malus/observed")
    def observed_bonus_malus() -> list[dict]:
        try:
            return repository.csv_records(
                "bonus_malus", "observed_by_bonus_malus.csv"
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    if frontend_dir.exists():
        application.mount(
            "/dashboard",
            StaticFiles(directory=frontend_dir, html=True),
            name="dashboard",
        )

        @application.get("/", include_in_schema=False)
        def dashboard_redirect() -> RedirectResponse:
            return RedirectResponse("/dashboard/")

    return application


app = create_app()
