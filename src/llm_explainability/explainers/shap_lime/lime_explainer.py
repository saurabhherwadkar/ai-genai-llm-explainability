# LIME-based text explainer for LLM prompts
"""Applies LIME (Local Interpretable Model-agnostic Explanations) to explain prompts."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from llm_explainability.explainers.shap_lime.engine import ShapLimeResult
from llm_explainability.explainers.shap_lime.perturbation import SegmentPerturbation
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponseWithMetadata,
)


class LimeTextExplainer:
    """Applies LIME to explain which prompt parts influence LLM output."""

    def __init__(self, num_samples: int = 100, max_features: int = 15) -> None:
        """Initialize the LIME explainer with sampling parameters."""
        # Number of perturbation samples for the local model
        self._num_samples = num_samples
        # Maximum features to include in the linear explanation
        self._max_features = max_features
        # Perturbation strategy for generating neighborhood samples
        self._perturbation = SegmentPerturbation()

    async def explain(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
    ) -> ShapLimeResult:
        """Compute LIME explanation for the prompt's influence on the output."""
        # Segment the prompt into interpretable features
        segments = self._perturbation.segment_text(prompt)
        # Limit to max_features
        segments = segments[: self._max_features]
        num_features = len(segments)
        # Store baseline output for similarity scoring
        baseline_output = response.text
        # Generate random perturbation samples around the input
        perturbation_masks, distances = self._generate_neighborhood(
            num_features, self._num_samples
        )
        # Evaluate each perturbation against the LLM
        similarity_scores = await self._evaluate_neighborhood(
            segments, perturbation_masks, baseline_output, provider
        )
        # Fit a weighted linear model to approximate local behavior
        feature_weights = self._fit_linear_model(
            perturbation_masks, similarity_scores, distances
        )
        # Build waterfall plot visualization data
        plot_data = self._build_waterfall_data(segments, feature_weights)
        # Return the LIME result
        return ShapLimeResult(
            method="lime",
            feature_names=segments,
            importance_scores=feature_weights.tolist(),
            base_value=0.0,
            explanation_plot_data=plot_data,
            num_samples_used=len(perturbation_masks),
        )

    def _generate_neighborhood(
        self, num_features: int, num_samples: int
    ) -> tuple[list[list[bool]], list[float]]:
        """Generate random perturbation samples in the neighborhood of the input."""
        # Initialize random state for reproducibility
        rng = np.random.default_rng(seed=42)
        # Generate binary masks (each sample randomly includes/excludes features)
        masks: list[list[bool]] = []
        distances: list[float] = []
        for _ in range(num_samples):
            # Each feature has ~50% chance of being included
            mask = rng.random(num_features) > 0.5
            masks.append(mask.tolist())
            # Compute cosine distance from original (all-True) vector
            num_active = mask.sum()
            distance = 1.0 - (num_active / num_features)
            distances.append(distance)
        # Always include the original input (distance = 0)
        masks.insert(0, [True] * num_features)
        distances.insert(0, 0.0)
        # Return masks and their distances from original
        return masks, distances

    async def _evaluate_neighborhood(
        self,
        segments: list[str],
        masks: list[list[bool]],
        baseline_output: str,
        provider: BaseLLMProvider,
    ) -> list[float]:
        """Evaluate all neighborhood perturbations and return similarity scores."""
        # Build perturbed prompts from masks
        scores: list[float] = []
        # Process in batches for rate limiting
        batch_size = 10
        for i in range(0, len(masks), batch_size):
            batch_masks = masks[i : i + batch_size]
            # Create perturbed prompts for this batch
            batch_prompts = [
                self._perturbation.apply_mask(segments, mask) for mask in batch_masks
            ]
            # Create async evaluation tasks
            tasks = [
                self._score_perturbation(prompt_text, baseline_output, provider)
                for prompt_text in batch_prompts
            ]
            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Collect scores
            for result in batch_results:
                if isinstance(result, Exception):
                    scores.append(0.0)
                else:
                    scores.append(result)
        # Return all similarity scores
        return scores

    async def _score_perturbation(
        self, prompt_text: str, baseline_output: str, provider: BaseLLMProvider
    ) -> float:
        """Score a single perturbation by its output similarity to baseline."""
        # Handle empty perturbed prompts
        if not prompt_text.strip():
            return 0.0
        # Generate LLM response for the perturbed input
        request = GenerationRequest(prompt=prompt_text, max_tokens=100, temperature=0.0)
        response = await provider.generate(request)
        # Compute word-overlap similarity to baseline output
        return self._compute_similarity(response.text, baseline_output)

    def _fit_linear_model(
        self,
        masks: list[list[bool]],
        scores: list[float],
        distances: list[float],
    ) -> np.ndarray:
        """Fit a weighted linear regression to extract feature importance."""
        # Convert masks to numpy array (features matrix)
        x_matrix = np.array(masks, dtype=np.float64)
        # Convert scores to numpy array (target vector)
        y_vector = np.array(scores, dtype=np.float64)
        # Compute sample weights (closer samples get higher weight)
        kernel_width = 0.75
        weights = np.exp(-np.array(distances) ** 2 / kernel_width ** 2)
        # Apply weights to features and target
        sqrt_weights = np.sqrt(weights)
        x_weighted = x_matrix * sqrt_weights[:, np.newaxis]
        y_weighted = y_vector * sqrt_weights
        # Solve weighted least squares using pseudo-inverse
        try:
            # Use numpy's least squares solver
            coefficients, _, _, _ = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)
        except np.linalg.LinAlgError:
            # Fall back to zeros if solver fails
            coefficients = np.zeros(x_matrix.shape[1])
        # Return the feature importance coefficients
        return coefficients

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute word-level overlap similarity between two texts."""
        # Handle empty inputs
        if not text_a or not text_b:
            return 0.0
        # Split into word sets for Jaccard similarity
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        # Calculate intersection over union
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        # Return the Jaccard coefficient
        return intersection / union if union > 0 else 0.0

    def _build_waterfall_data(
        self, segments: list[str], weights: np.ndarray
    ) -> dict[str, Any]:
        """Build data for rendering a LIME waterfall chart."""
        # Sort features by absolute importance for the waterfall
        feature_data = sorted(
            [{"name": seg, "value": float(w)} for seg, w in zip(segments, weights)],
            key=lambda x: abs(x["value"]),
            reverse=True,
        )
        # Return structured visualization data
        return {
            "type": "waterfall",
            "features": feature_data,
            "total_effect": float(weights.sum()),
        }
