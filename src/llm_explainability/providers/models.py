# Data models for LLM provider requests and responses
"""Defines the data structures exchanged between providers and the rest of the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProviderFeature(Enum):
    """Enumeration of capabilities that a provider may or may not support."""

    # Provider can return per-token log probabilities
    LOGPROBS = "logprobs"
    # Provider can return attention weight matrices
    ATTENTION_WEIGHTS = "attention_weights"
    # Provider supports streaming token-by-token responses
    STREAMING = "streaming"
    # Provider supports chain-of-thought / extended thinking
    CHAIN_OF_THOUGHT = "chain_of_thought"
    # Provider supports gradient-based attribution
    GRADIENT_ATTRIBUTION = "gradient_attribution"
    # Provider can return token-level embeddings
    EMBEDDINGS = "embeddings"
    # Provider supports system message / system prompt
    SYSTEM_MESSAGE = "system_message"


@dataclass
class GenerationRequest:
    """Request parameters for generating a completion from an LLM."""

    # The user prompt to send to the LLM
    prompt: str
    # Optional system message to set context
    system_message: str | None = None
    # Maximum tokens to generate in the response
    max_tokens: int = 4096
    # Sampling temperature (0.0 = deterministic)
    temperature: float = 0.0
    # Whether to request log probabilities in the response
    include_logprobs: bool = False
    # Number of top logprob alternatives per token
    top_logprobs: int = 5
    # Whether to request attention weights (local models only)
    include_attention: bool = False
    # Optional stop sequences to halt generation
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class TokenInfo:
    """Information about a single token in the response."""

    # The text representation of this token
    text: str
    # The token's ID in the model's vocabulary
    token_id: int | None = None
    # Log probability of this token being generated
    logprob: float | None = None
    # Alternative tokens and their log probabilities
    top_alternatives: dict[str, float] = field(default_factory=dict)


@dataclass
class UsageStats:
    """Token usage statistics for a generation request."""

    # Number of tokens in the input prompt
    prompt_tokens: int = 0
    # Number of tokens in the generated response
    completion_tokens: int = 0
    # Total tokens consumed (prompt + completion)
    total_tokens: int = 0


@dataclass
class GenerationResponse:
    """Basic response from an LLM generation request."""

    # The generated text content
    text: str
    # Reason the generation stopped (stop, length, etc.)
    finish_reason: str = "stop"
    # Token usage statistics
    usage: UsageStats = field(default_factory=UsageStats)
    # Name of the model that generated this response
    model: str = ""


@dataclass
class TokenLogProbabilities:
    """Per-token log probability data for attribution analysis."""

    # List of tokens in the response
    tokens: list[str] = field(default_factory=list)
    # Log probability for each token
    logprobs: list[float] = field(default_factory=list)
    # Top alternative tokens and logprobs at each position
    top_alternatives: list[dict[str, float]] = field(default_factory=list)


@dataclass
class AttentionWeights:
    """Attention weight matrices from transformer model layers."""

    # Attention matrices per layer: shape [num_heads, seq_len, seq_len]
    layer_weights: list[list[list[list[float]]]] = field(default_factory=list)
    # Number of attention heads in the model
    num_heads: int = 0
    # Number of layers in the model
    num_layers: int = 0
    # Token strings corresponding to matrix positions
    token_labels: list[str] = field(default_factory=list)


@dataclass
class GenerationResponseWithMetadata(GenerationResponse):
    """Extended response including metadata needed for explainability analysis."""

    # Per-token log probabilities (if requested and supported)
    token_logprobs: TokenLogProbabilities | None = None
    # Attention weight matrices (if requested and supported)
    attention_weights: AttentionWeights | None = None
    # List of individual token information objects
    token_info: list[TokenInfo] = field(default_factory=list)
    # Raw provider-specific response data for debugging
    raw_response: dict[str, object] = field(default_factory=dict)


@dataclass
class ProviderHealthStatus:
    """Health check result for an LLM provider."""

    # Whether the provider is currently reachable and responding
    is_healthy: bool
    # Response latency in milliseconds
    latency_ms: float = 0.0
    # Name of the provider being checked
    provider_name: str = ""
    # Model currently configured for this provider
    model_name: str = ""
    # Optional error message if unhealthy
    error_message: str | None = None
