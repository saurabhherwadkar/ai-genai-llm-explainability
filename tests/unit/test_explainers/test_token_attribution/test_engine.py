# Unit tests for the token attribution engine
"""Tests for the token attribution orchestrator and method selection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_explainability.config.settings import TokenAttributionConfig
from llm_explainability.explainers.token_attribution.engine import TokenAttributionEngine
from llm_explainability.providers.models import (
    GenerationResponseWithMetadata,
    ProviderFeature,
    TokenLogProbabilities,
    UsageStats,
)


@pytest.fixture
def engine() -> TokenAttributionEngine:
    """Provide a token attribution engine instance."""
    config = TokenAttributionConfig(method="logprob", top_k_tokens=5)
    return TokenAttributionEngine(config)


@pytest.fixture
def mock_provider_with_logprobs() -> MagicMock:
    """Provide a mock provider that supports logprobs."""
    provider = MagicMock()
    provider.provider_name = "mock"
    provider.supports_feature = MagicMock(
        side_effect=lambda f: f == ProviderFeature.LOGPROBS
    )
    return provider


class TestTokenAttributionEngine:
    """Tests for the TokenAttributionEngine class."""

    def test_get_technique_name(self, engine: TokenAttributionEngine) -> None:
        """Verify the technique name is correct."""
        assert engine.get_technique_name() == "Token Attribution"

    def test_supported_providers_includes_all(self, engine: TokenAttributionEngine) -> None:
        """Verify that the engine works with all providers."""
        assert "all" in engine.get_supported_providers()

    def test_select_method_prefers_logprob(self, engine: TokenAttributionEngine) -> None:
        """Verify logprob method is selected when provider supports it."""
        provider = MagicMock()
        provider.supports_feature = MagicMock(
            side_effect=lambda f: f == ProviderFeature.LOGPROBS
        )
        method = engine._select_method(provider)
        assert method == "logprob"

    def test_select_method_fallback_to_perturbation(
        self, engine: TokenAttributionEngine
    ) -> None:
        """Verify fallback to perturbation when no features supported."""
        provider = MagicMock()
        provider.supports_feature = MagicMock(return_value=False)
        method = engine._select_method(provider)
        assert method == "logprob_perturbation"

    @pytest.mark.asyncio
    async def test_explain_returns_result(
        self, engine: TokenAttributionEngine, mock_provider
    ) -> None:
        """Verify that explain returns a valid ExplanationResult."""
        response = GenerationResponseWithMetadata(
            text="Paris is the capital.",
            finish_reason="stop",
            usage=UsageStats(),
            model="test",
            token_logprobs=TokenLogProbabilities(
                tokens=["Paris", " is", " the", " capital"],
                logprobs=[-0.1, -0.2, -0.05, -0.3],
                top_alternatives=[],
            ),
        )
        result = await engine.explain(
            "What is the capital of France?", response, mock_provider
        )
        # Verify result structure
        assert result.technique_name == "token_attribution"
        assert result.success is True
        assert "tokens" in result.data
        assert "scores" in result.data
