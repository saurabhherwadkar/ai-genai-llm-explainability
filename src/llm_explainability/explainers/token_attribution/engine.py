# Token attribution engine orchestrator
"""Selects and runs the best attribution method based on provider capabilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm_explainability.config.settings import TokenAttributionConfig
from llm_explainability.exceptions import ExplainerComputationError
from llm_explainability.explainers.base import BaseExplainer, ExplanationResult
from llm_explainability.explainers.token_attribution.attention_analyzer import (
    AttentionAnalyzer,
)
from llm_explainability.explainers.token_attribution.logprob_analyzer import (
    LogProbAnalyzer,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationResponseWithMetadata,
    ProviderFeature,
)


@dataclass
class TokenScore:
    """A single token with its attribution score."""

    # The text of the token
    token: str
    # Attribution score indicating influence [0.0, 1.0]
    score: float
    # Position index in the original token sequence
    position: int


@dataclass
class TokenAttributionResult:
    """Complete result of token attribution analysis."""

    # All input tokens in order
    tokens: list[str] = field(default_factory=list)
    # Attribution score per token [0.0, 1.0]
    scores: list[float] = field(default_factory=list)
    # Which method was used (logprob, attention, gradient)
    method_used: str = ""
    # Top-K most influential tokens sorted by score
    top_k_influential: list[TokenScore] = field(default_factory=list)
    # Data formatted for heatmap visualization
    heatmap_data: list[dict[str, object]] = field(default_factory=list)


class TokenAttributionEngine(BaseExplainer):
    """Orchestrates token attribution analysis using the best available method."""

    def __init__(self, config: TokenAttributionConfig | None = None) -> None:
        """Initialize the engine with its configuration."""
        # Use default config if none provided
        self._config = config or TokenAttributionConfig()
        # Initialize the log probability analyzer
        self._logprob_analyzer = LogProbAnalyzer()
        # Initialize the attention weight analyzer
        self._attention_analyzer = AttentionAnalyzer()

    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ExplanationResult:
        """Produce token attribution explanation for the prompt-response pair."""
        # Record start time for performance metadata
        start_time = time.perf_counter()
        try:
            # Select the best available attribution method for this provider
            method = self._select_method(provider)
            # Run the selected attribution method
            attribution = await self._run_attribution(
                method, prompt, response, provider
            )
            # Calculate execution duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            # Build the explanation result from attribution data
            return ExplanationResult(
                technique_name="token_attribution",
                success=True,
                summary=self._generate_summary(attribution),
                data={
                    "tokens": attribution.tokens,
                    "scores": attribution.scores,
                    "method_used": attribution.method_used,
                    "top_k_influential": [
                        {"token": ts.token, "score": ts.score, "position": ts.position}
                        for ts in attribution.top_k_influential
                    ],
                },
                visualization_data={"heatmap": attribution.heatmap_data},
                confidence=self._calculate_confidence(attribution),
                metadata={"duration_ms": duration_ms, "method": method},
            )
        except Exception as exc:
            # Return a failed result with the error message
            return ExplanationResult(
                technique_name="token_attribution",
                success=False,
                error_message=str(exc),
                metadata={"duration_ms": (time.perf_counter() - start_time) * 1000},
            )

    def get_technique_name(self) -> str:
        """Return the human-readable name of this technique."""
        return "Token Attribution"

    def get_supported_providers(self) -> list[str]:
        """Return list of providers this technique works with."""
        # Works with all providers (falls back gracefully)
        return ["all"]

    def _select_method(self, provider: BaseLLMProvider) -> str:
        """Select the best attribution method based on provider capabilities."""
        # Prefer the configured method if the provider supports it
        if self._config.method == "attention":
            if provider.supports_feature(ProviderFeature.ATTENTION_WEIGHTS):
                return "attention"
        if self._config.method == "gradient":
            if provider.supports_feature(ProviderFeature.GRADIENT_ATTRIBUTION):
                return "gradient"
        # Fall back to logprob if available
        if provider.supports_feature(ProviderFeature.LOGPROBS):
            return "logprob"
        # Fall back to attention if available
        if provider.supports_feature(ProviderFeature.ATTENTION_WEIGHTS):
            return "attention"
        # Default to perturbation-based logprob approximation
        return "logprob_perturbation"

    async def _run_attribution(
        self,
        method: str,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> TokenAttributionResult:
        """Execute the selected attribution method and return results."""
        # Route to the appropriate analyzer based on selected method
        if method == "attention":
            return self._attention_analyzer.analyze(prompt, response)
        elif method in ("logprob", "logprob_perturbation"):
            return await self._logprob_analyzer.analyze(
                prompt, response, provider, use_perturbation=(method == "logprob_perturbation")
            )
        else:
            # Unsupported method should not reach here
            raise ExplainerComputationError(
                f"Unsupported attribution method: {method}",
                "token_attribution",
            )

    def _generate_summary(self, attribution: TokenAttributionResult) -> str:
        """Generate a human-readable summary of the attribution results."""
        # Return empty summary if no results
        if not attribution.top_k_influential:
            return "No significant token attributions found."
        # Build summary from top influential tokens
        top_tokens = attribution.top_k_influential[:5]
        token_list = ", ".join(
            f"'{ts.token}' ({ts.score:.2f})" for ts in top_tokens
        )
        # Return the formatted summary string
        return (
            f"The most influential input tokens were: {token_list}. "
            f"Method used: {attribution.method_used}."
        )

    def _calculate_confidence(self, attribution: TokenAttributionResult) -> float:
        """Calculate a confidence score for the attribution results."""
        # No confidence if no scores
        if not attribution.scores:
            return 0.0
        # Higher confidence when scores have clear differentiation
        max_score = max(attribution.scores)
        min_score = min(attribution.scores)
        # Score spread indicates clear signal vs noise
        spread = max_score - min_score
        # Normalize to [0, 1] range
        return min(1.0, spread * 2.0)
