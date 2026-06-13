# Gradient-based token attribution analyzer
"""Computes token importance using gradient saliency for local models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GradientAttributionResult:
    """Result of gradient-based attribution analysis."""

    # Token strings from the input
    tokens: list[str] = field(default_factory=list)
    # Gradient magnitude per token (higher = more influential)
    gradient_scores: list[float] = field(default_factory=list)
    # Method used (integrated_gradients, saliency, etc.)
    method: str = "saliency"


class GradientAnalyzer:
    """Computes token importance using gradient-based saliency maps."""

    def __init__(self) -> None:
        """Initialize the gradient analyzer."""
        # Flag indicating if captum library is available
        self._captum_available = self._check_captum_available()

    def analyze(
        self,
        input_ids: list[int],
        model: object,
        tokenizer: object,
    ) -> GradientAttributionResult:
        """Compute gradient-based attribution scores for input tokens.

        This method requires a locally-loaded model with gradient access.
        It computes the gradient of the output with respect to input embeddings.
        """
        # Check if required libraries are available
        if not self._captum_available:
            return GradientAttributionResult(
                tokens=[],
                gradient_scores=[],
                method="unavailable",
            )
        try:
            # Import torch for gradient computation
            import torch

            # Convert input IDs to tensor
            input_tensor = torch.tensor([input_ids])
            # Get the model's embedding layer
            embeddings = self._get_embeddings(model, input_tensor)
            # Enable gradient tracking on embeddings
            embeddings.requires_grad_(True)
            # Forward pass to get output logits
            outputs = self._forward_with_embeddings(model, embeddings)
            # Compute gradients of the max logit w.r.t. input embeddings
            max_logit = outputs.logits[0, -1].max()
            max_logit.backward()
            # Extract gradient magnitudes per token
            gradients = embeddings.grad[0]  # Shape: [seq_len, hidden_dim]
            # Compute L2 norm across embedding dimension for each token
            token_scores = torch.norm(gradients, dim=-1).tolist()
            # Decode tokens for labels
            token_texts = [tokenizer.decode([tid]) for tid in input_ids]
            # Return the gradient attribution result
            return GradientAttributionResult(
                tokens=token_texts,
                gradient_scores=token_scores,
                method="saliency",
            )
        except Exception:
            # Return empty result on failure
            return GradientAttributionResult(
                tokens=[],
                gradient_scores=[],
                method="failed",
            )

    def _check_captum_available(self) -> bool:
        """Check if the captum library is installed and importable."""
        try:
            # Attempt to import captum
            import captum  # noqa: F401
            return True
        except ImportError:
            # Captum not available
            return False

    def _get_embeddings(self, model: object, input_ids: object) -> object:
        """Extract input embeddings from the model's embedding layer."""
        # Access the model's embedding layer
        import torch

        embed_layer = getattr(model, "get_input_embeddings", None)
        if embed_layer is None:
            raise RuntimeError("Model does not have get_input_embeddings method")
        # Get embeddings for the input IDs
        embeddings = embed_layer()(input_ids)
        return embeddings

    def _forward_with_embeddings(self, model: object, embeddings: object) -> object:
        """Run forward pass using embeddings instead of token IDs."""
        # Call model with inputs_embeds parameter
        outputs = model(inputs_embeds=embeddings)
        return outputs
