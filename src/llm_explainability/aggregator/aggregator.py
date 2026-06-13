# Explanation aggregator that combines results from multiple techniques
"""Orchestrates multiple explainability engines and merges their outputs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from llm_explainability.config.settings import ExplainersConfig
from llm_explainability.explainers.base import BaseExplainer, ExplanationResult
from llm_explainability.explainers.registry import ExplainerRegistry
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import GenerationResponseWithMetadata


@dataclass
class AggregatedExplanation:
    """Unified result combining all requested explainability technique outputs."""

    # Natural language summary of the combined explanation
    summary: str = ""
    # Individual results keyed by technique name
    technique_results: dict[str, ExplanationResult] = field(default_factory=dict)
    # Overall confidence score (average of technique confidences)
    overall_confidence: float = 0.0
    # Cross-technique correlations found
    correlations: list[str] = field(default_factory=list)
    # Total computation time in milliseconds
    total_duration_ms: float = 0.0
    # Metadata about the aggregation process
    metadata: dict[str, Any] = field(default_factory=dict)


class ExplanationAggregator:
    """Orchestrates and combines results from multiple explainability techniques."""

    def __init__(self, registry: ExplainerRegistry, config: ExplainersConfig) -> None:
        """Initialize the aggregator with the explainer registry and config."""
        # Store reference to the explainer registry
        self._registry = registry
        # Store the explainers configuration
        self._config = config

    async def explain_all(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
        techniques: list[str] | None = None,
    ) -> AggregatedExplanation:
        """Run all requested techniques and aggregate their results."""
        # Record start time for total duration
        start_time = time.perf_counter()
        # Determine which techniques to run
        technique_names = techniques or self._config.enabled
        # Get explainer instances for the requested techniques
        explainers = self._get_explainers(technique_names)
        # Filter to only those available for this provider
        available = [
            (name, exp) for name, exp in explainers
            if exp.is_available_for_provider(provider)
        ]
        # Run all available explainers concurrently
        results = await self._run_concurrent(available, prompt, response, provider)
        # Calculate overall confidence from individual results
        confidence = self._calculate_overall_confidence(results)
        # Find cross-technique correlations
        correlations = self._find_correlations(results)
        # Generate a combined summary
        summary = self._generate_combined_summary(results, correlations)
        # Calculate total duration
        total_duration = (time.perf_counter() - start_time) * 1000
        # Build and return the aggregated explanation
        return AggregatedExplanation(
            summary=summary,
            technique_results=results,
            overall_confidence=confidence,
            correlations=correlations,
            total_duration_ms=total_duration,
            metadata={
                "techniques_requested": technique_names,
                "techniques_run": list(results.keys()),
                "provider": provider.provider_name,
                "model": provider.model_name,
            },
        )

    def _get_explainers(
        self, technique_names: list[str]
    ) -> list[tuple[str, BaseExplainer]]:
        """Get explainer instances for the given technique names."""
        # Collect name-instance pairs
        explainers: list[tuple[str, BaseExplainer]] = []
        for name in technique_names:
            try:
                # Get the explainer from the registry
                instance = self._registry.get(name, self._config)
                explainers.append((name, instance))
            except Exception:
                # Skip techniques that fail to load
                continue
        # Return the collected explainers
        return explainers

    async def _run_concurrent(
        self,
        explainers: list[tuple[str, BaseExplainer]],
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> dict[str, ExplanationResult]:
        """Run multiple explainers concurrently and collect their results."""
        # Create async tasks for each explainer
        async def run_single(
            name: str, explainer: BaseExplainer
        ) -> tuple[str, ExplanationResult]:
            """Run a single explainer and return its result with name."""
            result = await explainer.explain(prompt, response, provider)
            return (name, result)

        # Execute all tasks concurrently
        tasks = [run_single(name, exp) for name, exp in explainers]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        # Collect successful results into a dictionary
        results: dict[str, ExplanationResult] = {}
        for item in completed:
            if isinstance(item, Exception):
                continue
            name, result = item
            results[name] = result
        # Return the results dictionary
        return results

    def _calculate_overall_confidence(
        self, results: dict[str, ExplanationResult]
    ) -> float:
        """Calculate average confidence across all successful technique results."""
        # Collect confidence values from successful results
        confidences = [
            r.confidence for r in results.values() if r.success
        ]
        # Return average or 0 if no successful results
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def _find_correlations(
        self, results: dict[str, ExplanationResult]
    ) -> list[str]:
        """Identify agreements between different technique results."""
        # Collect correlations between techniques
        correlations: list[str] = []
        # Check if token attribution and SHAP agree on top features
        if "token_attribution" in results and "shap_lime" in results:
            attr_result = results["token_attribution"]
            shap_result = results["shap_lime"]
            if attr_result.success and shap_result.success:
                # Get top tokens from attribution
                attr_top = set(
                    item.get("token", "")
                    for item in attr_result.data.get("top_k_influential", [])[:5]
                )
                # Get top features from SHAP
                shap_features = shap_result.data.get("feature_names", [])[:5]
                # Check for overlap
                overlap = attr_top & set(shap_features)
                if overlap:
                    correlations.append(
                        f"Token attribution and SHAP agree on: {', '.join(overlap)}"
                    )
        # Return found correlations
        return correlations

    def _generate_combined_summary(
        self,
        results: dict[str, ExplanationResult],
        correlations: list[str],
    ) -> str:
        """Generate a natural language summary combining all technique results."""
        # Collect individual summaries
        parts: list[str] = []
        for name, result in results.items():
            if result.success and result.summary:
                parts.append(result.summary)
        # Add correlation notes
        if correlations:
            parts.append("Cross-technique agreement: " + "; ".join(correlations))
        # Join with newlines for readability
        return " ".join(parts) if parts else "No explanations were generated."
