# Abstract base class for all LLM providers
"""Defines the interface contract that all LLM provider implementations must fulfill."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationResponseWithMetadata,
    ProviderFeature,
    ProviderHealthStatus,
    TokenLogProbabilities,
)


class BaseLLMProvider(ABC):
    """Abstract base class defining the contract all LLM providers must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique identifier name of this provider."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model currently configured for this provider."""
        ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a text completion from the LLM.

        Args:
            request: The generation request containing prompt and parameters.

        Returns:
            A basic response with generated text and usage stats.
        """
        ...

    @abstractmethod
    async def generate_with_metadata(
        self, request: GenerationRequest
    ) -> GenerationResponseWithMetadata:
        """Generate a completion with full metadata for explainability analysis.

        Args:
            request: The generation request with metadata flags enabled.

        Returns:
            An extended response including logprobs, attention weights, and token info.
        """
        ...

    @abstractmethod
    async def get_token_logprobs(
        self, request: GenerationRequest
    ) -> TokenLogProbabilities:
        """Retrieve per-token log probabilities for the generated response.

        Args:
            request: The generation request to analyze.

        Returns:
            Token-level log probability data for attribution analysis.
        """
        ...

    @abstractmethod
    async def health_check(self) -> ProviderHealthStatus:
        """Verify that the provider is reachable and responding.

        Returns:
            A health status object with connectivity and latency information.
        """
        ...

    @abstractmethod
    def supports_feature(self, feature: ProviderFeature) -> bool:
        """Check if this provider supports a specific capability.

        Args:
            feature: The feature to check support for.

        Returns:
            True if the feature is supported, False otherwise.
        """
        ...

    def get_supported_features(self) -> list[ProviderFeature]:
        """Return a list of all features supported by this provider."""
        # Check each feature and collect the supported ones
        supported = []
        for feature in ProviderFeature:
            # Test each feature against the provider's implementation
            if self.supports_feature(feature):
                supported.append(feature)
        # Return the list of supported features
        return supported
