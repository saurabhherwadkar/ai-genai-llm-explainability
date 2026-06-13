# SHAP-based text explainer for LLM prompts
"""Applies SHAP (Shapley Additive Explanations) to explain prompt influence."""

from __future__ import annotations

import asyncio
from typing import Any

from llm_explainability.explainers.shap_lime.engine import ShapLimeResult
from llm_explainability.explainers.shap_lime.perturbation import (
    PerturbationStrategy,
    SegmentPerturbation,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponseWithMetadata,
)


class ShapTextExplainer:
    """Applies SHAP values to explain which prompt segments influence the output."""

    def __init__(self, num_samples: int = 100, max_features: int = 15) -> None:
        """Initialize the SHAP explainer with sampling parameters."""
        # Number of perturbation samples to generate
        self._num_samples = num_samples
        # Maximum number of features to include in the explanation
        self._max_features = max_features
        # Perturbation strategy for creating masked inputs
        self._perturbation = SegmentPerturbation()

    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ShapLimeResult:
        """Compute SHAP values for prompt segments explaining the output."""
        # Split the prompt into segments (features) for analysis
        segments = self._perturbation.segment_text(prompt)
        # Limit to max_features segments
        segments = segments[: self._max_features]
        # Get the baseline output (all segments present)
        baseline_output = response.text
        # Compute baseline score (text similarity to original output)
        baseline_score = 1.0
        # Generate perturbation masks (binary coalitions)
        masks = self._generate_coalition_masks(len(segments), self._num_samples)
        # Evaluate each perturbation by calling the LLM
        scores = await self._evaluate_perturbations(
            segments, masks, baseline_output, provider
        )
        # Compute marginal contributions (simplified SHAP estimation)
        shap_values = self._compute_shap_values(segments, masks, scores, baseline_score)
        # Build visualization data for force plot
        plot_data = self._build_force_plot_data(segments, shap_values, baseline_score)
        # Return the SHAP result
        return ShapLimeResult(
            method="shap",
            feature_names=segments,
            importance_scores=shap_values,
            base_value=baseline_score,
            explanation_plot_data=plot_data,
            num_samples_used=len(masks),
        )

    def _generate_coalition_masks(
        self, num_features: int, num_samples: int
    ) -> list[list[bool]]:
        """Generate binary masks representing feature coalitions for SHAP."""
        # Generate diverse coalition masks using systematic sampling
        masks: list[list[bool]] = []
        # Always include all-features-present mask
        masks.append([True] * num_features)
        # Always include empty mask (no features)
        masks.append([False] * num_features)
        # Generate intermediate masks by toggling individual features
        for i in range(num_features):
            # Create mask with only feature i removed (leave-one-out)
            mask = [True] * num_features
            mask[i] = False
            masks.append(mask)
        # Generate additional random coalition masks up to num_samples
        import random

        random.seed(42)  # Reproducible masks for consistency
        while len(masks) < num_samples:
            # Create a random binary mask
            mask = [random.random() > 0.5 for _ in range(num_features)]
            masks.append(mask)
        # Limit to the requested number of samples
        return masks[:num_samples]

    async def _evaluate_perturbations(
        self,
        segments: list[str],
        masks: list[list[bool]],
        baseline_output: str,
        provider: BaseLLMProvider,
    ) -> list[float]:
        """Evaluate each perturbed prompt and score output similarity."""
        # Build perturbed prompts from masks
        perturbed_prompts = [
            self._perturbation.apply_mask(segments, mask) for mask in masks
        ]
        # Evaluate all prompts (batch with concurrency limit)
        scores: list[float] = []
        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(perturbed_prompts), batch_size):
            batch = perturbed_prompts[i : i + batch_size]
            # Create async tasks for this batch
            tasks = [
                self._evaluate_single(prompt_text, baseline_output, provider)
                for prompt_text in batch
            ]
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Collect results, using 0.0 for failures
            for result in batch_results:
                if isinstance(result, Exception):
                    scores.append(0.0)
                else:
                    scores.append(result)
        # Return all scores
        return scores

    async def _evaluate_single(
        self, prompt_text: str, baseline_output: str, provider: BaseLLMProvider
    ) -> float:
        """Evaluate a single perturbed prompt and return similarity score."""
        # Skip empty prompts
        if not prompt_text.strip():
            return 0.0
        # Generate response for the perturbed prompt
        request = GenerationRequest(prompt=prompt_text, max_tokens=100, temperature=0.0)
        response = await provider.generate(request)
        # Score similarity between perturbed output and baseline
        return self._text_similarity(response.text, baseline_output)

    def _compute_shap_values(
        self,
        segments: list[str],
        masks: list[list[bool]],
        scores: list[float],
        baseline: float,
    ) -> list[float]:
        """Estimate SHAP values using marginal contribution averaging."""
        # Initialize SHAP values to zero
        shap_values = [0.0] * len(segments)
        # Count contributions for averaging
        contribution_counts = [0] * len(segments)
        # For each mask, compute marginal contribution of each feature
        for mask_idx, mask in enumerate(masks):
            if mask_idx >= len(scores):
                break
            score = scores[mask_idx]
            for feature_idx in range(len(segments)):
                if mask[feature_idx]:
                    # Feature is present; its contribution is score - baseline_without_it
                    shap_values[feature_idx] += score
                    contribution_counts[feature_idx] += 1
        # Average the contributions
        for i in range(len(shap_values)):
            if contribution_counts[i] > 0:
                shap_values[i] = (shap_values[i] / contribution_counts[i]) - baseline
        # Return the estimated SHAP values
        return shap_values

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute a simple word-overlap similarity score between two texts."""
        # Handle empty texts
        if not text_a or not text_b:
            return 0.0
        # Split into word sets
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        # Compute Jaccard similarity
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        # Return the similarity ratio
        return intersection / union if union > 0 else 0.0

    def _build_force_plot_data(
        self,
        segments: list[str],
        shap_values: list[float],
        base_value: float,
    ) -> dict[str, Any]:
        """Build data structure for rendering a SHAP force plot."""
        # Create the force plot data dictionary
        return {
            "type": "force_plot",
            "base_value": base_value,
            "features": [
                {"name": seg, "value": val}
                for seg, val in zip(segments, shap_values)
            ],
            "output_value": base_value + sum(shap_values),
        }
