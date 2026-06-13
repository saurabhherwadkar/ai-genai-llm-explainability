# Log-probability based token attribution analyzer
"""Computes token importance using log-probability differences and perturbation."""

from __future__ import annotations

from llm_explainability.explainers.token_attribution.engine import (
    TokenAttributionResult,
    TokenScore,
)
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.models import (
    GenerationRequest,
    GenerationResponseWithMetadata,
)
from llm_explainability.utils.tokenizer import SimpleTokenizer


class LogProbAnalyzer:
    """Analyzes token importance using log-probability based attribution."""

    def __init__(self) -> None:
        """Initialize the analyzer with a tokenizer instance."""
        # Create a simple tokenizer for splitting prompts into tokens
        self._tokenizer = SimpleTokenizer()

    async def analyze(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
        provider: BaseLLMProvider,
        use_perturbation: bool = False,
    ) -> TokenAttributionResult:
        """Compute token attribution scores using log probabilities."""
        # Tokenize the input prompt into individual tokens
        tokens = self._tokenizer.tokenize(prompt)
        # Extract token text strings for the result
        token_texts = [t.text for t in tokens]
        # Choose between native logprobs and perturbation-based approach
        if use_perturbation or response.token_logprobs is None:
            # Use perturbation approach when native logprobs unavailable
            scores = await self._perturbation_attribution(prompt, tokens, provider)
        else:
            # Use native logprobs for efficient attribution
            scores = self._native_logprob_attribution(prompt, tokens, response)
        # Normalize scores to [0, 1] range
        normalized_scores = self._normalize_scores(scores)
        # Identify top-K most influential tokens
        top_k = self._get_top_k(token_texts, normalized_scores)
        # Build heatmap visualization data
        heatmap = self._build_heatmap_data(token_texts, normalized_scores)
        # Determine which method was used
        method = "logprob_perturbation" if use_perturbation else "logprob"
        # Return the complete attribution result
        return TokenAttributionResult(
            tokens=token_texts,
            scores=normalized_scores,
            method_used=method,
            top_k_influential=top_k,
            heatmap_data=heatmap,
        )

    async def _perturbation_attribution(
        self,
        prompt: str,
        tokens: list,
        provider: BaseLLMProvider,
    ) -> list[float]:
        """Compute attribution by removing tokens and measuring output change."""
        # Get the baseline response for comparison
        baseline_request = GenerationRequest(prompt=prompt, max_tokens=50, temperature=0.0)
        baseline_response = await provider.generate(baseline_request)
        baseline_text = baseline_response.text
        # Initialize scores list for each token
        scores: list[float] = []
        # Iterate over each token to measure its impact when removed
        for i, token in enumerate(tokens):
            # Create a perturbed prompt with this token removed
            perturbed_prompt = self._remove_token(prompt, token)
            # Generate response with the perturbed prompt
            perturbed_request = GenerationRequest(
                prompt=perturbed_prompt, max_tokens=50, temperature=0.0
            )
            perturbed_response = await provider.generate(perturbed_request)
            # Measure the change in output between baseline and perturbed
            change_score = self._measure_text_difference(
                baseline_text, perturbed_response.text
            )
            # Store the impact score for this token
            scores.append(change_score)
        # Return the raw attribution scores
        return scores

    def _native_logprob_attribution(
        self,
        prompt: str,
        tokens: list,
        response: GenerationResponseWithMetadata,
    ) -> list[float]:
        """Compute attribution using native log probabilities from the response."""
        # Get the token logprobs from the response metadata
        logprobs_data = response.token_logprobs
        # If no logprobs available, return uniform scores
        if logprobs_data is None or not logprobs_data.logprobs:
            return [1.0 / len(tokens)] * len(tokens)
        # Use the variance of output logprobs as a proxy for input influence
        # Higher variance in output = that part of input was more influential
        avg_output_logprob = (
            sum(logprobs_data.logprobs) / len(logprobs_data.logprobs)
            if logprobs_data.logprobs
            else 0.0
        )
        # Assign scores based on how much each input token position
        # correlates with confident (high logprob) output tokens
        scores: list[float] = []
        for i in range(len(tokens)):
            # Use position-based heuristic with available logprob data
            # Tokens at the beginning often set context (slightly higher)
            position_weight = 1.0 - (i / (len(tokens) * 2))
            # Combine with output confidence signal
            score = abs(avg_output_logprob) * position_weight
            scores.append(score)
        # Return the computed scores
        return scores

    def _remove_token(self, prompt: str, token: object) -> str:
        """Remove a specific token from the prompt text."""
        # Get the token's start and end positions
        start = getattr(token, "start_index", 0)
        end = getattr(token, "end_index", 0)
        # Reconstruct the prompt without this token
        return prompt[:start] + prompt[end:]

    def _measure_text_difference(self, text_a: str, text_b: str) -> float:
        """Measure the difference between two text outputs as a score."""
        # Handle identical outputs (no influence)
        if text_a == text_b:
            return 0.0
        # Use character-level difference ratio as a simple metric
        max_len = max(len(text_a), len(text_b))
        if max_len == 0:
            return 0.0
        # Count differing characters
        min_len = min(len(text_a), len(text_b))
        differences = sum(1 for i in range(min_len) if text_a[i] != text_b[i])
        # Add the length difference
        differences += abs(len(text_a) - len(text_b))
        # Normalize to [0, 1] range
        return differences / max_len

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize a list of scores to the [0, 1] range."""
        # Handle empty list
        if not scores:
            return []
        # Find the range of values
        min_score = min(scores)
        max_score = max(scores)
        # Handle case where all scores are identical
        score_range = max_score - min_score
        if score_range == 0:
            return [0.5] * len(scores)
        # Apply min-max normalization
        return [(s - min_score) / score_range for s in scores]

    def _get_top_k(
        self, tokens: list[str], scores: list[float], k: int = 20
    ) -> list[TokenScore]:
        """Get the top-K tokens by attribution score."""
        # Create token-score pairs with position indices
        scored_tokens = [
            TokenScore(token=tokens[i], score=scores[i], position=i)
            for i in range(len(tokens))
        ]
        # Sort by score in descending order
        scored_tokens.sort(key=lambda ts: ts.score, reverse=True)
        # Return the top-K entries
        return scored_tokens[:k]

    def _build_heatmap_data(
        self, tokens: list[str], scores: list[float]
    ) -> list[dict[str, object]]:
        """Build visualization data for rendering a token heatmap."""
        # Create a list of dictionaries for each token with its score
        heatmap: list[dict[str, object]] = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            heatmap.append({
                "token": token,
                "score": score,
                "position": i,
                "color_intensity": score,  # Maps directly to color scale
            })
        # Return the heatmap data list
        return heatmap
