# Attention weight based token attribution analyzer
"""Computes token importance by analyzing transformer attention patterns."""

from __future__ import annotations

from llm_explainability.explainers.token_attribution.engine import (
    TokenAttributionResult,
    TokenScore,
)
from llm_explainability.providers.models import GenerationResponseWithMetadata
from llm_explainability.utils.tokenizer import SimpleTokenizer


class AttentionAnalyzer:
    """Analyzes token importance using transformer attention weight matrices."""

    def __init__(self) -> None:
        """Initialize the analyzer with a tokenizer instance."""
        # Create a tokenizer for splitting input text
        self._tokenizer = SimpleTokenizer()

    def analyze(
        self,
        prompt: str,
        response: GenerationResponseWithMetadata,
    ) -> TokenAttributionResult:
        """Compute token attribution scores from attention weights."""
        # Tokenize the input prompt
        tokens = self._tokenizer.tokenize(prompt)
        token_texts = [t.text for t in tokens]
        # Check if attention data is available in the response
        if response.attention_weights is None:
            # Return uniform scores when attention data is unavailable
            uniform_score = 1.0 / max(len(tokens), 1)
            return TokenAttributionResult(
                tokens=token_texts,
                scores=[uniform_score] * len(tokens),
                method_used="attention_unavailable",
                top_k_influential=[],
                heatmap_data=[],
            )
        # Extract and aggregate attention weights across layers
        aggregated_scores = self._aggregate_attention_layers(
            response.attention_weights.layer_weights, len(tokens)
        )
        # Normalize scores to [0, 1] range
        normalized = self._normalize_scores(aggregated_scores)
        # Get top-K most attended tokens
        top_k = self._get_top_k(token_texts, normalized)
        # Build heatmap visualization data
        heatmap = self._build_heatmap(token_texts, normalized)
        # Return the complete attribution result
        return TokenAttributionResult(
            tokens=token_texts,
            scores=normalized,
            method_used="attention",
            top_k_influential=top_k,
            heatmap_data=heatmap,
        )

    def _aggregate_attention_layers(
        self, layer_weights: list[list[list[list[float]]]], num_input_tokens: int
    ) -> list[float]:
        """Aggregate attention weights across all layers and heads."""
        # Initialize aggregated scores to zero for each input position
        scores = [0.0] * num_input_tokens
        # Count total attention distributions for averaging
        total_distributions = 0
        # Iterate over each layer's attention data
        for layer in layer_weights:
            # Iterate over each attention head in this layer
            for head in layer:
                # Each head contains a matrix [seq_len x seq_len]
                # Sum the attention each position receives from all other positions
                for row in head:
                    # Limit to input token positions only
                    for i in range(min(len(row), num_input_tokens)):
                        scores[i] += row[i]
                total_distributions += 1
        # Average across all heads and layers
        if total_distributions > 0:
            scores = [s / total_distributions for s in scores]
        # Return the averaged attention scores
        return scores

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        # Handle empty list
        if not scores:
            return []
        # Calculate the score range
        min_val = min(scores)
        max_val = max(scores)
        score_range = max_val - min_val
        # Return uniform scores if all values are identical
        if score_range == 0:
            return [0.5] * len(scores)
        # Apply min-max normalization
        return [(s - min_val) / score_range for s in scores]

    def _get_top_k(
        self, tokens: list[str], scores: list[float], k: int = 20
    ) -> list[TokenScore]:
        """Get the top-K tokens by attention score."""
        # Create scored token objects with position information
        scored = [
            TokenScore(token=tokens[i], score=scores[i], position=i)
            for i in range(len(tokens))
        ]
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        # Return top-K entries
        return scored[:k]

    def _build_heatmap(
        self, tokens: list[str], scores: list[float]
    ) -> list[dict[str, object]]:
        """Build visualization data for an attention heatmap."""
        # Create a dictionary entry for each token
        heatmap: list[dict[str, object]] = []
        for i, (token, score) in enumerate(zip(tokens, scores)):
            heatmap.append({
                "token": token,
                "score": score,
                "position": i,
                "color_intensity": score,
            })
        # Return the heatmap data
        return heatmap
