# Abstract base class for all explainability engines
"""Defines the interface contract that all explainer implementations must fulfill."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import GenerationResponseWithMetadata


@dataclass
class ExplanationResult:
    """Container for the output of any explainability technique."""

    # Name of the technique that produced this result
    technique_name: str
    # Whether the explanation was successfully computed
    success: bool = True
    # Human-readable summary of the explanation
    summary: str = ""
    # Structured data specific to the technique
    data: dict[str, Any] = field(default_factory=dict)
    # Visualization-ready data for the dashboard
    visualization_data: dict[str, Any] = field(default_factory=dict)
    # Confidence score for the explanation [0.0, 1.0]
    confidence: float = 0.0
    # Error message if the explanation failed
    error_message: str | None = None
    # Metadata about the computation (timing, samples used, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseExplainer(ABC):
    """Abstract base class that all explainability techniques must implement."""

    @abstractmethod
    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ExplanationResult:
        """Produce an explanation for the given prompt-response pair.

        Args:
            prompt: The original user prompt that was sent to the LLM.
            response: The LLM's response with metadata (logprobs, attention, etc.).
            provider: The provider instance for making additional API calls if needed.

        Returns:
            An ExplanationResult containing the analysis output.
        """
        ...

    @abstractmethod
    def get_technique_name(self) -> str:
        """Return the human-readable name of this explainability technique."""
        ...

    @abstractmethod
    def get_supported_providers(self) -> list[str]:
        """Return list of provider names this technique works with."""
        ...

    def is_available_for_provider(self, provider: BaseLLMProvider) -> bool:
        """Check if this technique can work with the given provider."""
        # Get the list of supported providers for this technique
        supported = self.get_supported_providers()
        # Check if 'all' is in the list (works with any provider)
        if "all" in supported:
            return True
        # Check if the specific provider is in the supported list
        return provider.provider_name in supported
