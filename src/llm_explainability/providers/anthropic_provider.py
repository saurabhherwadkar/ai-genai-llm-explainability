# Anthropic Claude LLM provider implementation
"""Implements the BaseLLMProvider interface for Anthropic's Claude models."""

from __future__ import annotations

import time

import anthropic

from llm_explainability.config.settings import AnthropicProviderConfig
from llm_explainability.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
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
from llm_explainability.utils.async_helpers import create_retry_decorator


class AnthropicProvider(BaseLLMProvider):
    """LLM provider implementation for Anthropic's Claude API."""

    # Set of features that Anthropic Claude supports
    _SUPPORTED_FEATURES = frozenset({
        ProviderFeature.CHAIN_OF_THOUGHT,
        ProviderFeature.STREAMING,
        ProviderFeature.SYSTEM_MESSAGE,
    })

    def __init__(self, config: AnthropicProviderConfig) -> None:
        """Initialize the Anthropic provider with its configuration."""
        # Store the provider configuration
        self._config = config
        # Create the async Anthropic client
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,  # We handle retries ourselves
        )
        # Create the retry decorator with config values
        self._retry = create_retry_decorator(max_attempts=config.max_retries)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier string."""
        return "anthropic"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._config.model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a text completion using the Anthropic API."""
        try:
            # Call the Anthropic messages endpoint
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=request.max_tokens,
                system=request.system_message or "",
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                stop_sequences=request.stop_sequences or [],
            )
            # Extract the text content from the response
            content = self._extract_text_content(response)
            # Build and return the generation response
            return GenerationResponse(
                text=content,
                finish_reason=response.stop_reason or "end_turn",
                usage=self._extract_usage(response),
                model=response.model,
            )
        except anthropic.AuthenticationError as exc:
            # Handle invalid API key errors
            raise ProviderAuthError(
                "Authentication failed", "anthropic", details=str(exc)
            ) from exc
        except anthropic.RateLimitError as exc:
            # Handle rate limit exceeded errors
            raise ProviderRateLimitError("anthropic", details=str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            # Handle request timeout errors
            raise ProviderTimeoutError(
                "Request timed out", "anthropic", details=str(exc)
            ) from exc
        except anthropic.APIConnectionError as exc:
            # Handle connection failure errors
            raise ProviderConnectionError(
                "Connection failed", "anthropic", details=str(exc)
            ) from exc
        except anthropic.APIError as exc:
            # Handle all other API errors
            raise ProviderResponseError(
                f"API error: {str(exc)}", "anthropic", details=str(exc)
            ) from exc

    async def generate_with_metadata(
        self, request: GenerationRequest
    ) -> GenerationResponseWithMetadata:
        """Generate a completion with metadata for explainability analysis."""
        try:
            # Call Anthropic API (no native logprobs support)
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=request.max_tokens,
                system=request.system_message or "",
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                stop_sequences=request.stop_sequences or [],
            )
            # Extract text content from response blocks
            content = self._extract_text_content(response)
            # Build the metadata-enriched response
            return GenerationResponseWithMetadata(
                text=content,
                finish_reason=response.stop_reason or "end_turn",
                usage=self._extract_usage(response),
                model=response.model,
                token_logprobs=None,  # Anthropic does not provide logprobs
                attention_weights=None,  # Anthropic does not expose attention
                token_info=[],
                raw_response={},
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(
                "Authentication failed", "anthropic", details=str(exc)
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError("anthropic", details=str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(
                "Request timed out", "anthropic", details=str(exc)
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderConnectionError(
                "Connection failed", "anthropic", details=str(exc)
            ) from exc
        except anthropic.APIError as exc:
            raise ProviderResponseError(
                f"API error: {str(exc)}", "anthropic", details=str(exc)
            ) from exc

    async def get_token_logprobs(
        self, request: GenerationRequest
    ) -> TokenLogProbabilities:
        """Retrieve per-token log probabilities (not natively supported by Anthropic)."""
        # Anthropic does not provide native logprob access
        # Return empty structure; explainers will use perturbation-based methods
        return TokenLogProbabilities()

    async def health_check(self) -> ProviderHealthStatus:
        """Check if the Anthropic API is reachable and responding."""
        # Record start time for latency calculation
        start_time = time.perf_counter()
        try:
            # Send a minimal request to verify connectivity
            await self._client.messages.create(
                model=self._config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            # Calculate response latency
            latency = (time.perf_counter() - start_time) * 1000
            # Return healthy status
            return ProviderHealthStatus(
                is_healthy=True,
                latency_ms=latency,
                provider_name="anthropic",
                model_name=self._config.model,
            )
        except Exception as exc:
            # Calculate latency for failed request
            latency = (time.perf_counter() - start_time) * 1000
            # Return unhealthy status with error message
            return ProviderHealthStatus(
                is_healthy=False,
                latency_ms=latency,
                provider_name="anthropic",
                model_name=self._config.model,
                error_message=str(exc),
            )

    def supports_feature(self, feature: ProviderFeature) -> bool:
        """Check if Anthropic supports the given feature."""
        # Check membership in the supported features set
        return feature in self._SUPPORTED_FEATURES

    def _extract_text_content(self, response: object) -> str:
        """Extract text content from Anthropic's content blocks."""
        # Get the content list from the response
        content_blocks = getattr(response, "content", [])
        # Collect text from all text-type content blocks
        text_parts: list[str] = []
        for block in content_blocks:
            # Check if this is a text content block
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        # Join all text parts with newlines
        return "\n".join(text_parts)

    def _extract_usage(self, response: object) -> UsageStats:
        """Extract token usage statistics from the Anthropic response."""
        # Access the usage attribute from the response
        usage = getattr(response, "usage", None)
        # Return empty stats if usage not available
        if usage is None:
            return UsageStats()
        # Map Anthropic's usage fields to our UsageStats model
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        return UsageStats(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
