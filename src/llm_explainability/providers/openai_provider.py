# OpenAI LLM provider implementation
"""Implements the BaseLLMProvider interface for OpenAI's GPT models."""

from __future__ import annotations

import time

import openai

from llm_explainability.config.settings import OpenAIProviderConfig
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
    TokenInfo,
    TokenLogProbabilities,
    UsageStats,
)
from llm_explainability.utils.async_helpers import create_retry_decorator


class OpenAIProvider(BaseLLMProvider):
    """LLM provider implementation for OpenAI's API (GPT-4o, GPT-4, etc.)."""

    # Set of features that OpenAI supports
    _SUPPORTED_FEATURES = frozenset({
        ProviderFeature.LOGPROBS,
        ProviderFeature.STREAMING,
        ProviderFeature.SYSTEM_MESSAGE,
    })

    def __init__(self, config: OpenAIProviderConfig) -> None:
        """Initialize the OpenAI provider with its configuration."""
        # Store the provider configuration
        self._config = config
        # Create the async OpenAI client
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,  # We handle retries ourselves
        )
        # Create the retry decorator with config values
        self._retry = create_retry_decorator(max_attempts=config.max_retries)

    @property
    def provider_name(self) -> str:
        """Return the provider identifier string."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._config.model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a text completion using the OpenAI API."""
        # Build the messages list from the request
        messages = self._build_messages(request)
        try:
            # Call the OpenAI chat completions endpoint
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences or None,
            )
            # Extract the generated text from the response
            content = response.choices[0].message.content or ""
            # Build and return the generation response
            return GenerationResponse(
                text=content,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=self._extract_usage(response),
                model=response.model,
            )
        except openai.AuthenticationError as exc:
            # Handle invalid API key errors
            raise ProviderAuthError(
                "Authentication failed", "openai", details=str(exc)
            ) from exc
        except openai.RateLimitError as exc:
            # Handle rate limit exceeded errors
            raise ProviderRateLimitError("openai", details=str(exc)) from exc
        except openai.APITimeoutError as exc:
            # Handle request timeout errors
            raise ProviderTimeoutError(
                "Request timed out", "openai", details=str(exc)
            ) from exc
        except openai.APIConnectionError as exc:
            # Handle connection failure errors
            raise ProviderConnectionError(
                "Connection failed", "openai", details=str(exc)
            ) from exc
        except openai.APIError as exc:
            # Handle all other API errors
            raise ProviderResponseError(
                f"API error: {exc.message}", "openai", details=str(exc)
            ) from exc

    async def generate_with_metadata(
        self, request: GenerationRequest
    ) -> GenerationResponseWithMetadata:
        """Generate a completion with log probabilities and token details."""
        # Build the messages list from the request
        messages = self._build_messages(request)
        try:
            # Call OpenAI with logprobs enabled for explainability
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences or None,
                logprobs=True,
                top_logprobs=request.top_logprobs,
            )
            # Extract the generated text content
            content = response.choices[0].message.content or ""
            # Extract token-level log probability data
            token_logprobs = self._extract_logprobs(response)
            # Build token info list from logprobs data
            token_info = self._extract_token_info(response)
            # Build and return the metadata-enriched response
            return GenerationResponseWithMetadata(
                text=content,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=self._extract_usage(response),
                model=response.model,
                token_logprobs=token_logprobs,
                attention_weights=None,  # OpenAI does not expose attention
                token_info=token_info,
                raw_response={},
            )
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(
                "Authentication failed", "openai", details=str(exc)
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimitError("openai", details=str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(
                "Request timed out", "openai", details=str(exc)
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderConnectionError(
                "Connection failed", "openai", details=str(exc)
            ) from exc
        except openai.APIError as exc:
            raise ProviderResponseError(
                f"API error: {exc.message}", "openai", details=str(exc)
            ) from exc

    async def get_token_logprobs(
        self, request: GenerationRequest
    ) -> TokenLogProbabilities:
        """Retrieve per-token log probabilities for attribution analysis."""
        # Force logprobs flag on for this request
        request.include_logprobs = True
        # Generate with metadata to get logprobs
        response = await self.generate_with_metadata(request)
        # Return the token logprobs or empty structure
        return response.token_logprobs or TokenLogProbabilities()

    async def health_check(self) -> ProviderHealthStatus:
        """Check if the OpenAI API is reachable and responding."""
        # Record the start time for latency measurement
        start_time = time.perf_counter()
        try:
            # Send a minimal request to verify connectivity
            await self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            # Calculate response latency in milliseconds
            latency = (time.perf_counter() - start_time) * 1000
            # Return healthy status with latency
            return ProviderHealthStatus(
                is_healthy=True,
                latency_ms=latency,
                provider_name="openai",
                model_name=self._config.model,
            )
        except Exception as exc:
            # Calculate latency even for failed requests
            latency = (time.perf_counter() - start_time) * 1000
            # Return unhealthy status with error details
            return ProviderHealthStatus(
                is_healthy=False,
                latency_ms=latency,
                provider_name="openai",
                model_name=self._config.model,
                error_message=str(exc),
            )

    def supports_feature(self, feature: ProviderFeature) -> bool:
        """Check if OpenAI supports the given feature."""
        # Check membership in the supported features set
        return feature in self._SUPPORTED_FEATURES

    def _build_messages(
        self, request: GenerationRequest
    ) -> list[dict[str, str]]:
        """Construct the messages array for the OpenAI API call."""
        # Start with an empty messages list
        messages: list[dict[str, str]] = []
        # Add system message if provided
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        # Add the user prompt as a user message
        messages.append({"role": "user", "content": request.prompt})
        # Return the constructed messages list
        return messages

    def _extract_usage(self, response: object) -> UsageStats:
        """Extract token usage statistics from the OpenAI response."""
        # Access the usage attribute from the response
        usage = getattr(response, "usage", None)
        # Return empty stats if usage data is not available
        if usage is None:
            return UsageStats()
        # Map OpenAI's usage fields to our UsageStats model
        return UsageStats(
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def _extract_logprobs(self, response: object) -> TokenLogProbabilities:
        """Extract per-token log probabilities from the OpenAI response."""
        # Navigate to the logprobs content in the response structure
        choices = getattr(response, "choices", [])
        if not choices:
            return TokenLogProbabilities()
        # Get logprobs from the first choice
        logprobs_data = getattr(choices[0], "logprobs", None)
        if logprobs_data is None:
            return TokenLogProbabilities()
        # Get the content-level logprobs list
        content_logprobs = getattr(logprobs_data, "content", [])
        if not content_logprobs:
            return TokenLogProbabilities()
        # Build the token and logprob lists
        tokens: list[str] = []
        logprobs: list[float] = []
        top_alternatives: list[dict[str, float]] = []
        # Iterate over each token's logprob data
        for token_data in content_logprobs:
            # Extract the token string
            tokens.append(getattr(token_data, "token", ""))
            # Extract the log probability value
            logprobs.append(getattr(token_data, "logprob", 0.0))
            # Extract top alternative tokens and their logprobs
            alternatives: dict[str, float] = {}
            for alt in getattr(token_data, "top_logprobs", []):
                alternatives[getattr(alt, "token", "")] = getattr(alt, "logprob", 0.0)
            top_alternatives.append(alternatives)
        # Return the structured log probability data
        return TokenLogProbabilities(
            tokens=tokens,
            logprobs=logprobs,
            top_alternatives=top_alternatives,
        )

    def _extract_token_info(self, response: object) -> list[TokenInfo]:
        """Extract individual token information from the OpenAI response."""
        # Navigate to logprobs content
        choices = getattr(response, "choices", [])
        if not choices:
            return []
        logprobs_data = getattr(choices[0], "logprobs", None)
        if logprobs_data is None:
            return []
        content_logprobs = getattr(logprobs_data, "content", [])
        # Build token info list from the logprobs data
        token_info_list: list[TokenInfo] = []
        for token_data in content_logprobs:
            # Create alternatives dictionary from top_logprobs
            alternatives: dict[str, float] = {}
            for alt in getattr(token_data, "top_logprobs", []):
                alternatives[getattr(alt, "token", "")] = getattr(alt, "logprob", 0.0)
            # Create the TokenInfo object for this token
            info = TokenInfo(
                text=getattr(token_data, "token", ""),
                token_id=None,  # OpenAI does not expose token IDs
                logprob=getattr(token_data, "logprob", 0.0),
                top_alternatives=alternatives,
            )
            token_info_list.append(info)
        # Return the complete token info list
        return token_info_list
