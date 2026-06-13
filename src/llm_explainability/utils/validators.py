# Common input validators used across the application
"""Provides validation functions for user inputs and configuration values."""

from __future__ import annotations

from llm_explainability.config.constants import (
    SUPPORTED_EXPLAINERS,
    SUPPORTED_PROVIDERS,
)
from llm_explainability.exceptions import ValidationError


def validate_provider_name(provider_name: str) -> str:
    """Validate that the provider name is supported and return normalized form."""
    # Normalize the provider name to lowercase
    normalized = provider_name.lower().strip()
    # Check if the normalized name is in the supported list
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValidationError(
            f"Unsupported provider. Must be one of: {SUPPORTED_PROVIDERS}",
            field_name="provider",
        )
    # Return the validated and normalized name
    return normalized


def validate_technique_names(techniques: list[str]) -> list[str]:
    """Validate that all technique names are supported and return normalized list."""
    # Handle the special 'all' keyword
    if techniques == ["all"]:
        return list(SUPPORTED_EXPLAINERS)
    # Validate each technique name individually
    validated: list[str] = []
    for technique in techniques:
        # Normalize to lowercase
        normalized = technique.lower().strip()
        # Check against supported explainers
        if normalized not in SUPPORTED_EXPLAINERS:
            raise ValidationError(
                f"Unsupported technique '{technique}'. "
                f"Must be one of: {SUPPORTED_EXPLAINERS}",
                field_name="techniques",
            )
        # Add to validated list if not already present
        if normalized not in validated:
            validated.append(normalized)
    # Return the validated list of technique names
    return validated


def validate_temperature(temperature: float) -> float:
    """Validate that temperature is within the acceptable range [0.0, 2.0]."""
    # Check lower bound
    if temperature < 0.0:
        raise ValidationError(
            "Temperature must be >= 0.0",
            field_name="temperature",
        )
    # Check upper bound
    if temperature > 2.0:
        raise ValidationError(
            "Temperature must be <= 2.0",
            field_name="temperature",
        )
    # Return the validated temperature value
    return temperature


def validate_max_tokens(max_tokens: int) -> int:
    """Validate that max_tokens is a positive integer within bounds."""
    # Check that tokens is positive
    if max_tokens <= 0:
        raise ValidationError(
            "max_tokens must be a positive integer",
            field_name="max_tokens",
        )
    # Check reasonable upper bound
    if max_tokens > 100000:
        raise ValidationError(
            "max_tokens cannot exceed 100000",
            field_name="max_tokens",
        )
    # Return the validated value
    return max_tokens


def validate_num_samples(num_samples: int) -> int:
    """Validate the number of SHAP/LIME perturbation samples."""
    # Check minimum samples required for meaningful results
    if num_samples < 10:
        raise ValidationError(
            "num_samples must be at least 10 for meaningful results",
            field_name="num_samples",
        )
    # Check upper bound to prevent excessive API costs
    if num_samples > 10000:
        raise ValidationError(
            "num_samples cannot exceed 10000 to prevent excessive API costs",
            field_name="num_samples",
        )
    # Return the validated value
    return num_samples
