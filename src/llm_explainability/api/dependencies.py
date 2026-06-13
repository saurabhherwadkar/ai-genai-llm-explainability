# FastAPI dependency injection providers
"""Defines dependencies that are injected into route handlers via FastAPI's Depends()."""

from __future__ import annotations

from functools import lru_cache

from llm_explainability.aggregator.aggregator import ExplanationAggregator
from llm_explainability.config.loader import ConfigLoader
from llm_explainability.config.settings import AppSettings
from llm_explainability.explainers.registry import ExplainerRegistry, create_default_registry
from llm_explainability.providers.factory import ProviderFactory, create_default_factory
from llm_explainability.security.input_sanitizer import InputSanitizer
from llm_explainability.security.rate_limiter import RateLimiter


@lru_cache(maxsize=1)
def get_config() -> AppSettings:
    """Load and return the application configuration (singleton via lru_cache)."""
    # Create a config loader and load the settings
    loader = ConfigLoader()
    # Return the loaded and validated settings
    return loader.load()


@lru_cache(maxsize=1)
def get_provider_factory() -> ProviderFactory:
    """Create and return the provider factory with all providers registered."""
    # Create the default factory with all built-in providers
    return create_default_factory()


@lru_cache(maxsize=1)
def get_explainer_registry() -> ExplainerRegistry:
    """Create and return the explainer registry with all techniques registered."""
    # Create the default registry with all built-in explainers
    return create_default_registry()


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """Create and return the rate limiter configured from settings."""
    # Get the configuration to read rate limit settings
    config = get_config()
    # Create the rate limiter with configured requests per minute
    return RateLimiter(requests_per_minute=config.security.rate_limit_per_minute)


@lru_cache(maxsize=1)
def get_input_sanitizer() -> InputSanitizer:
    """Create and return the input sanitizer configured from settings."""
    # Get the configuration for security settings
    config = get_config()
    # Create the sanitizer with security configuration
    return InputSanitizer(config.security)


def get_aggregator() -> ExplanationAggregator:
    """Create and return the explanation aggregator."""
    # Get the registry and config dependencies
    registry = get_explainer_registry()
    config = get_config()
    # Create the aggregator with registry and explainer config
    return ExplanationAggregator(registry=registry, config=config.explainers)
