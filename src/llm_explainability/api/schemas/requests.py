# API request models
"""Pydantic models for validating incoming API request bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExplainOptions(BaseModel):
    """Optional configuration parameters for an explanation request."""

    # Maximum tokens for the LLM response generation
    max_tokens: int = Field(default=1024, ge=1, le=100000)
    # Sampling temperature for the LLM (0.0 = deterministic)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Number of top tokens to include in attribution results
    top_k_attribution: int = Field(default=20, ge=1, le=100)
    # Number of perturbation samples for SHAP/LIME
    shap_samples: int = Field(default=100, ge=10, le=10000)
    # Maximum reasoning steps to parse in chain-of-thought
    cot_max_steps: int = Field(default=20, ge=1, le=100)


class ExplainRequest(BaseModel):
    """Request body for the main /explain endpoint."""

    # The user prompt to explain
    prompt: str = Field(..., min_length=1, max_length=10000)
    # Optional system message to include with the prompt
    system_message: str | None = Field(default=None, max_length=5000)
    # Which LLM provider to use for generation
    provider: str = Field(default="openai")
    # Optional model override (uses provider default if not specified)
    model: str | None = None
    # Which explainability techniques to run (["all"] for all enabled)
    techniques: list[str] = Field(default=["all"])
    # Optional advanced configuration
    options: ExplainOptions = Field(default_factory=ExplainOptions)


class SingleTechniqueRequest(BaseModel):
    """Request body for single-technique endpoints."""

    # The user prompt to explain
    prompt: str = Field(..., min_length=1, max_length=10000)
    # Optional system message
    system_message: str | None = Field(default=None, max_length=5000)
    # Which LLM provider to use
    provider: str = Field(default="openai")
    # Optional model override
    model: str | None = None
    # Technique-specific options
    options: ExplainOptions = Field(default_factory=ExplainOptions)
