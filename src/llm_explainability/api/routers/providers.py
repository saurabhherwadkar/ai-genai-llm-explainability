# Provider management endpoints
"""API endpoints for listing and checking LLM provider status."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_explainability.api.dependencies import get_config, get_provider_factory
from llm_explainability.api.schemas.responses import (
    ProviderInfo,
    ProvidersListResponse,
)
from llm_explainability.config.settings import AppSettings
from llm_explainability.providers.factory import ProviderFactory
from llm_explainability.providers.models import ProviderHealthStatus

# Create the router for provider endpoints
router = APIRouter(prefix="/api/v1/providers", tags=["providers"])


@router.get("", response_model=ProvidersListResponse)
async def list_providers(
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
) -> ProvidersListResponse:
    """List all registered LLM providers and their capabilities."""
    # Get list of registered provider names
    registered = factory.list_registered()
    # Build provider info for each registered provider
    providers: list[ProviderInfo] = []
    for name in registered:
        try:
            # Create provider instance to query its features
            provider = factory.create(name, config)
            # Get the supported features list
            features = [f.value for f in provider.get_supported_features()]
            # Build the provider info object
            providers.append(ProviderInfo(
                name=name,
                is_healthy=True,  # Assume healthy until checked
                supported_features=features,
                model=provider.model_name,
            ))
        except Exception:
            # Include provider with unhealthy status if creation fails
            providers.append(ProviderInfo(
                name=name,
                is_healthy=False,
                supported_features=[],
                model="",
            ))
    # Return the providers list response
    return ProvidersListResponse(
        providers=providers,
        default_provider=config.providers.default,
    )


@router.get("/{provider_name}/health")
async def check_provider_health(
    provider_name: str,
    config: AppSettings = Depends(get_config),
    factory: ProviderFactory = Depends(get_provider_factory),
) -> ProviderHealthStatus:
    """Check the health of a specific LLM provider."""
    # Create the provider instance
    provider = factory.create(provider_name, config)
    # Run the health check
    health = await provider.health_check()
    # Return the health status
    return health
