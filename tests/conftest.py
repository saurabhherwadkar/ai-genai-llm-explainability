# Shared test fixtures and configuration
"""Provides common fixtures, mock objects, and test utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_explainability.config.settings import (
    AppSettings,
    ExplainersConfig,
    LoggingConfig,
    SecurityConfig,
    TokenAttributionConfig,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationResponseWithMetadata,
    ProviderFeature,
    ProviderHealthStatus,
    TokenLogProbabilities,
    UsageStats,
)


@pytest.fixture
def test_config() -> AppSettings:
    """Provide a test configuration with safe defaults."""
    # Create minimal config for testing without file I/O
    return AppSettings()


@pytest.fixture
def test_logging_config() -> LoggingConfig:
    """Provide a test logging configuration."""
    return LoggingConfig(level="debug", format="console")


@pytest.fixture
def test_security_config() -> SecurityConfig:
    """Provide a test security configuration."""
    return SecurityConfig(
        rate_limit_per_minute=100,
        max_prompt_length=5000,
    )


@pytest.fixture
def test_explainers_config() -> ExplainersConfig:
    """Provide a test explainers configuration."""
    return ExplainersConfig(
        enabled=["token_attribution"],
        token_attribution=TokenAttributionConfig(top_k_tokens=5),
    )


@pytest.fixture
def sample_generation_request() -> GenerationRequest:
    """Provide a sample generation request for testing."""
    return GenerationRequest(
        prompt="What is the capital of France?",
        system_message="You are a helpful assistant.",
        max_tokens=100,
        temperature=0.0,
        include_logprobs=True,
    )


@pytest.fixture
def sample_generation_response() -> GenerationResponse:
    """Provide a sample basic generation response."""
    return GenerationResponse(
        text="The capital of France is Paris.",
        finish_reason="stop",
        usage=UsageStats(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        model="test-model",
    )


@pytest.fixture
def sample_response_with_metadata() -> GenerationResponseWithMetadata:
    """Provide a sample response with metadata for explainability testing."""
    return GenerationResponseWithMetadata(
        text="The capital of France is Paris.",
        finish_reason="stop",
        usage=UsageStats(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        model="test-model",
        token_logprobs=TokenLogProbabilities(
            tokens=["The", " capital", " of", " France", " is", " Paris", "."],
            logprobs=[-0.1, -0.2, -0.05, -0.3, -0.1, -0.01, -0.05],
            top_alternatives=[
                {"The": -0.1, "A": -2.0},
                {" capital": -0.2, " city": -1.5},
                {" of": -0.05, " in": -3.0},
                {" France": -0.3, " Germany": -2.5},
                {" is": -0.1, " was": -2.0},
                {" Paris": -0.01, " Lyon": -4.0},
                {".": -0.05, "!": -3.0},
            ],
        ),
        attention_weights=None,
        token_info=[],
        raw_response={},
    )


@pytest.fixture
def mock_provider() -> BaseLLMProvider:
    """Provide a mock LLM provider with canned responses."""
    # Create a mock that implements the BaseLLMProvider interface
    provider = MagicMock(spec=BaseLLMProvider)
    # Configure the provider name property
    provider.provider_name = "mock"
    provider.model_name = "mock-model"
    # Configure the generate method to return a canned response
    provider.generate = AsyncMock(return_value=GenerationResponse(
        text="The capital of France is Paris.",
        finish_reason="stop",
        usage=UsageStats(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        model="mock-model",
    ))
    # Configure generate_with_metadata
    provider.generate_with_metadata = AsyncMock(return_value=GenerationResponseWithMetadata(
        text="The capital of France is Paris.",
        finish_reason="stop",
        usage=UsageStats(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        model="mock-model",
        token_logprobs=TokenLogProbabilities(
            tokens=["The", " capital", " of", " France", " is", " Paris", "."],
            logprobs=[-0.1, -0.2, -0.05, -0.3, -0.1, -0.01, -0.05],
            top_alternatives=[],
        ),
    ))
    # Configure health check
    provider.health_check = AsyncMock(return_value=ProviderHealthStatus(
        is_healthy=True, latency_ms=50.0, provider_name="mock", model_name="mock-model"
    ))
    # Configure feature support
    provider.supports_feature = MagicMock(side_effect=lambda f: f == ProviderFeature.LOGPROBS)
    provider.get_supported_features = MagicMock(return_value=[ProviderFeature.LOGPROBS])
    # Configure is_available check for explainers
    provider.is_available_for_provider = MagicMock(return_value=True)
    # Return the configured mock
    return provider
