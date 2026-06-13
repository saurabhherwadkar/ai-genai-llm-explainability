# SHAP/LIME explainability engine orchestrator
"""Coordinates perturbation-based explanation using SHAP or LIME methods."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm_explainability.config.settings import ShapLimeConfig
from llm_explainability.explainers.base import BaseExplainer, ExplanationResult
from llm_explainability.explainers.shap_lime.lime_explainer import LimeTextExplainer
from llm_explainability.explainers.shap_lime.shap_explainer import ShapTextExplainer
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import GenerationResponseWithMetadata


@dataclass
class ShapLimeResult:
    """Combined result from SHAP or LIME analysis."""

    # Method used (shap, lime, or both)
    method: str = ""
    # Names/labels of the analyzed features (segments)
    feature_names: list[str] = field(default_factory=list)
    # Signed importance score for each feature
    importance_scores: list[float] = field(default_factory=list)
    # Expected output value without features (base value)
    base_value: float = 0.0
    # Data for rendering SHAP force/waterfall plots
    explanation_plot_data: dict[str, object] = field(default_factory=dict)
    # Number of perturbation samples used
    num_samples_used: int = 0


class ShapLimeEngine(BaseExplainer):
    """Orchestrates SHAP and/or LIME based prompt explanation."""

    def __init__(self, config: ShapLimeConfig | None = None) -> None:
        """Initialize the engine with its configuration."""
        # Use default config if none provided
        self._config = config or ShapLimeConfig()
        # Initialize the SHAP text explainer
        self._shap_explainer = ShapTextExplainer(
            num_samples=self._config.num_samples,
            max_features=self._config.max_features,
        )
        # Initialize the LIME text explainer
        self._lime_explainer = LimeTextExplainer(
            num_samples=self._config.num_samples,
            max_features=self._config.max_features,
        )

    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ExplanationResult:
        """Produce SHAP/LIME explanation for the prompt-response pair."""
        # Record start time for performance tracking
        start_time = time.perf_counter()
        try:
            # Select which method(s) to run based on configuration
            method = self._config.method
            # Run the selected method
            if method == "shap":
                result = await self._shap_explainer.explain(prompt, response, provider)
            elif method == "lime":
                result = await self._lime_explainer.explain(prompt, response, provider)
            elif method == "both":
                # Run both and combine results
                result = await self._run_both(prompt, response, provider)
            else:
                # Default to SHAP
                result = await self._shap_explainer.explain(prompt, response, provider)
            # Calculate execution duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            # Build the explanation result
            return ExplanationResult(
                technique_name="shap_lime",
                success=True,
                summary=self._generate_summary(result),
                data={
                    "method": result.method,
                    "feature_names": result.feature_names,
                    "importance_scores": result.importance_scores,
                    "base_value": result.base_value,
                    "num_samples_used": result.num_samples_used,
                },
                visualization_data=result.explanation_plot_data,
                confidence=self._calculate_confidence(result),
                metadata={"duration_ms": duration_ms, "method": method},
            )
        except Exception as exc:
            # Return a failed result with error details
            return ExplanationResult(
                technique_name="shap_lime",
                success=False,
                error_message=str(exc),
                metadata={"duration_ms": (time.perf_counter() - start_time) * 1000},
            )

    def get_technique_name(self) -> str:
        """Return the human-readable name of this technique."""
        return "SHAP/LIME Prompt Analysis"

    def get_supported_providers(self) -> list[str]:
        """Return list of providers this technique works with."""
        # Works with all providers (uses perturbation which only needs generate)
        return ["all"]

    async def _run_both(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ShapLimeResult:
        """Run both SHAP and LIME and combine their results."""
        # Run SHAP analysis
        shap_result = await self._shap_explainer.explain(prompt, response, provider)
        # Run LIME analysis
        lime_result = await self._lime_explainer.explain(prompt, response, provider)
        # Average the importance scores from both methods
        combined_scores = self._average_scores(
            shap_result.importance_scores,
            lime_result.importance_scores,
        )
        # Return combined result
        return ShapLimeResult(
            method="both",
            feature_names=shap_result.feature_names,
            importance_scores=combined_scores,
            base_value=(shap_result.base_value + lime_result.base_value) / 2.0,
            explanation_plot_data={
                "shap": shap_result.explanation_plot_data,
                "lime": lime_result.explanation_plot_data,
            },
            num_samples_used=shap_result.num_samples_used + lime_result.num_samples_used,
        )

    def _average_scores(
        self, scores_a: list[float], scores_b: list[float]
    ) -> list[float]:
        """Average two lists of scores element-wise."""
        # Use the shorter length to avoid index errors
        min_len = min(len(scores_a), len(scores_b))
        # Compute element-wise average
        averaged = [(scores_a[i] + scores_b[i]) / 2.0 for i in range(min_len)]
        # Return the averaged scores
        return averaged

    def _generate_summary(self, result: ShapLimeResult) -> str:
        """Generate a human-readable summary of the SHAP/LIME results."""
        # Handle empty results
        if not result.feature_names:
            return "No significant features identified."
        # Find the most influential features (absolute value)
        scored_features = list(zip(result.feature_names, result.importance_scores))
        scored_features.sort(key=lambda x: abs(x[1]), reverse=True)
        # Take top 5 most influential
        top_features = scored_features[:5]
        # Build the summary string
        parts = [f"'{name}' ({score:+.3f})" for name, score in top_features]
        feature_list = ", ".join(parts)
        # Return formatted summary
        return (
            f"Most influential prompt segments ({result.method}): {feature_list}. "
            f"Based on {result.num_samples_used} perturbation samples."
        )

    def _calculate_confidence(self, result: ShapLimeResult) -> float:
        """Calculate confidence based on sample size and score spread."""
        # More samples = higher confidence (diminishing returns)
        sample_confidence = min(1.0, result.num_samples_used / 200.0)
        # Higher score spread = clearer signal
        if result.importance_scores:
            spread = max(result.importance_scores) - min(result.importance_scores)
            spread_confidence = min(1.0, spread * 5.0)
        else:
            spread_confidence = 0.0
        # Combine both confidence factors
        return (sample_confidence + spread_confidence) / 2.0
