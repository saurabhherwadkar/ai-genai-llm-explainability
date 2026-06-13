# Health check endpoints
"""Provides liveness and readiness probes for the application."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_explainability.api.dependencies import get_config
from llm_explainability.api.schemas.responses import HealthResponse
from llm_explainability.config.settings import AppSettings

# Create the router for health endpoints
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    config: AppSettings = Depends(get_config),
) -> HealthResponse:
    """Return basic application health status (liveness probe)."""
    # Return healthy status with app version and environment
    return HealthResponse(
        status="healthy",
        version=config.app.version,
        environment=config.app.environment,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(
    config: AppSettings = Depends(get_config),
) -> HealthResponse:
    """Return application readiness status (checks all dependencies)."""
    # For now, if the app is running and config loaded, it's ready
    return HealthResponse(
        status="ready",
        version=config.app.version,
        environment=config.app.environment,
    )
