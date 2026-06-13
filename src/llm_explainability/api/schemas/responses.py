# API response models
"""Pydantic models for structuring API response bodies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """Metadata about the explanation computation."""

    # Total time to compute the explanation in milliseconds
    duration_ms: float = 0.0
    # Which techniques were requested
    techniques_requested: list[str] = Field(default_factory=list)
    # Which techniques actually produced results
    techniques_completed: list[str] = Field(default_factory=list)
    # Token usage from the LLM provider
    prompt_tokens: int = 0
    # Tokens in the generated completion
    completion_tokens: int = 0


class TechniqueResult(BaseModel):
    """Result from a single explainability technique."""

    # Whether the technique completed successfully
    success: bool = True
    # Human-readable summary of this technique's findings
    summary: str = ""
    # Confidence score [0.0, 1.0]
    confidence: float = 0.0
    # Technique-specific structured data
    data: dict[str, Any] = Field(default_factory=dict)
    # Visualization-ready data for the dashboard
    visualization_data: dict[str, Any] = Field(default_factory=dict)
    # Error message if the technique failed
    error_message: str | None = None


class ExplainResponse(BaseModel):
    """Response body for the /explain endpoint."""

    # Unique identifier for this request
    request_id: str = ""
    # Echo of the input prompt
    prompt: str = ""
    # The LLM's actual response text
    llm_response: str = ""
    # Which provider generated the response
    provider_used: str = ""
    # Which model generated the response
    model_used: str = ""
    # Combined summary across all techniques
    summary: str = ""
    # Overall confidence score
    overall_confidence: float = 0.0
    # Cross-technique correlations found
    correlations: list[str] = Field(default_factory=list)
    # Individual technique results
    explanations: dict[str, TechniqueResult] = Field(default_factory=dict)
    # Request metadata (timing, tokens, etc.)
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ProviderInfo(BaseModel):
    """Information about an available LLM provider."""

    # Provider identifier name
    name: str
    # Whether the provider is currently healthy
    is_healthy: bool = False
    # List of supported features
    supported_features: list[str] = Field(default_factory=list)
    # Configured model name
    model: str = ""


class ProvidersListResponse(BaseModel):
    """Response body for the /providers endpoint."""

    # List of available provider information
    providers: list[ProviderInfo] = Field(default_factory=list)
    # Default provider name from configuration
    default_provider: str = ""


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    # Overall application health status
    status: str = "healthy"
    # Application version
    version: str = ""
    # Current environment
    environment: str = ""


class ErrorResponse(BaseModel):
    """Standard error response body."""

    # Error type identifier
    error: str = ""
    # Human-readable error message
    message: str = ""
    # Additional error details (only in development)
    details: str | None = None
