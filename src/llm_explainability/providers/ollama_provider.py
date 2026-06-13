# Ollama local LLM provider implementation
"""Implements the BaseLLMProvider interface for locally-hosted models via Ollama."""

from __future__ import annotations

import time

import ollama

from llm_explainability.config.settings import OllamaProviderConfig
from llm_explainability.exceptions import (
    ProviderConnectionError,
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


class OllamaProvider(BaseLLMProvider):
    """LLM provider implementation for locally-hosted models via Ollama server."""

    # Set of features that Ollama supports
    _SUPPORTED_FEATURES = frozenset({
        ProviderFeature.STREAMING,
        ProviderFeature.SYSTEM_MESSAGE,
    })

    def __init__(self, config: OllamaProviderConfig) -> None:
        """Initialize the Ollama provider with its configuration."""
        # Store the provider configuration
        self._config = config
        # Create the async Ollama client pointing to the local server
        self._client = ollama.AsyncClient(host=config.base_url)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier string."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._config.model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a text completion using the local Ollama server."""
        try:
            # Build the messages for the Ollama chat endpoint
            messages = self._build_messages(request)
            # Call the Ollama chat completion endpoint
            response = await self._client.chat(
                model=self._config.model,
                messages=messages,
                options={"temperature": request.temperature},
            )
            # Extract the generated text from the response
            content = response.get("message", {}).get("content", "")
            # Extract token usage metrics
            usage = self._extract_usage(response)
            # Build and return the generation response
            return GenerationResponse(
                text=content,
                finish_reason="stop",
                usage=usage,
                model=self._config.model,
            )
        except ollama.ResponseError as exc:
            # Handle Ollama server response errors
            raise ProviderResponseError(
                f"Ollama error: {str(exc)}", "ollama", details=str(exc)
            ) from exc
        except Exception as exc:
            # Handle connection and timeout errors
            error_str = str(exc).lower()
            if "timeout" in error_str:
                raise ProviderTimeoutError(
                    "Request timed out", "ollama", details=str(exc)
                ) from exc
            raise ProviderConnectionError(
                "Failed to connect to Ollama server", "ollama", details=str(exc)
            ) from exc

    async def generate_with_metadata(
        self, request: GenerationRequest
    ) -> GenerationResponseWithMetadata:
        """Generate a completion with available metadata from Ollama."""
        try:
            # Build messages for the chat request
            messages = self._build_messages(request)
            # Call Ollama with any available metadata options
            response = await self._client.chat(
                model=self._config.model,
                messages=messages,
                options={"temperature": request.temperature},
            )
            # Extract the generated text
            content = response.get("message", {}).get("content", "")
            # Extract usage statistics
            usage = self._extract_usage(response)
            # Return response with metadata (Ollama has limited metadata)
            return GenerationResponseWithMetadata(
                text=content,
                finish_reason="stop",
                usage=usage,
                model=self._config.model,
                token_logprobs=None,  # Ollama does not expose logprobs via API
                attention_weights=None,  # Not available via Ollama
                token_info=[],
                raw_response=response if isinstance(response, dict) else {},
            )
        except ollama.ResponseError as exc:
            raise ProviderResponseError(
                f"Ollama error: {str(exc)}", "ollama", details=str(exc)
            ) from exc
        except Exception as exc:
            error_str = str(exc).lower()
            if "timeout" in error_str:
                raise ProviderTimeoutError(
                    "Request timed out", "ollama", details=str(exc)
                ) from exc
            raise ProviderConnectionError(
                "Failed to connect to Ollama server", "ollama", details=str(exc)
            ) from exc

    async def get_token_logprobs(
        self, request: GenerationRequest
    ) -> TokenLogProbabilities:
        """Retrieve per-token log probabilities (limited support in Ollama)."""
        # Ollama does not reliably expose logprobs via its API
        # Return empty structure; explainers will use perturbation methods
        return TokenLogProbabilities()

    async def health_check(self) -> ProviderHealthStatus:
        """Check if the Ollama server is running and accessible."""
        # Record start time for latency measurement
        start_time = time.perf_counter()
        try:
            # List models to verify server connectivity
            await self._client.list()
            # Calculate response latency
            latency = (time.perf_counter() - start_time) * 1000
            # Return healthy status
            return ProviderHealthStatus(
                is_healthy=True,
                latency_ms=latency,
                provider_name="ollama",
                model_name=self._config.model,
            )
        except Exception as exc:
            # Calculate latency for failed health check
            latency = (time.perf_counter() - start_time) * 1000
            # Return unhealthy status with error details
            return ProviderHealthStatus(
                is_healthy=False,
                latency_ms=latency,
                provider_name="ollama",
                model_name=self._config.model,
                error_message=str(exc),
            )

    def supports_feature(self, feature: ProviderFeature) -> bool:
        """Check if Ollama supports the given feature."""
        # Check membership in the supported features set
        return feature in self._SUPPORTED_FEATURES

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        """Construct the messages list for the Ollama chat request."""
        # Initialize empty messages list
        messages: list[dict[str, str]] = []
        # Add system message if provided
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        # Add the user prompt
        messages.append({"role": "user", "content": request.prompt})
        # Return the messages list
        return messages

    def _extract_usage(self, response: dict) -> UsageStats:
        """Extract token usage statistics from the Ollama response."""
        # Ollama provides eval_count and prompt_eval_count
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        # Build usage stats from available data
        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
