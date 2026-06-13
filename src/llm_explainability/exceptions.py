# Custom exception hierarchy for the application
"""Defines all application-specific exceptions with clear categorization."""

from __future__ import annotations


class LLMExplainabilityError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize with a human-readable message and optional details."""
        # Store the primary error message
        self.message = message
        # Store additional context about the error
        self.details = details
        # Call parent constructor with the message
        super().__init__(message)


class ConfigError(LLMExplainabilityError):
    """Raised when configuration loading or validation fails."""

    pass


class ProviderError(LLMExplainabilityError):
    """Base exception for all LLM provider-related errors."""

    def __init__(
        self, message: str, provider_name: str, details: str | None = None
    ) -> None:
        """Initialize with the provider name that caused the error."""
        # Store which provider encountered the error
        self.provider_name = provider_name
        # Call parent with the formatted message
        super().__init__(f"[{provider_name}] {message}", details)


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM provider request exceeds the timeout threshold."""

    pass


class ProviderAuthError(ProviderError):
    """Raised when authentication with an LLM provider fails."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when an LLM provider returns a rate limit exceeded response."""

    def __init__(
        self,
        provider_name: str,
        retry_after_seconds: float | None = None,
        details: str | None = None,
    ) -> None:
        """Initialize with optional retry-after duration."""
        # Store the suggested wait time before retrying
        self.retry_after_seconds = retry_after_seconds
        # Call parent with descriptive message
        super().__init__("Rate limit exceeded", provider_name, details)


class ProviderConnectionError(ProviderError):
    """Raised when connection to an LLM provider cannot be established."""

    pass


class ProviderResponseError(ProviderError):
    """Raised when an LLM provider returns an unexpected or malformed response."""

    pass


class ExplainerError(LLMExplainabilityError):
    """Base exception for all explainability engine errors."""

    def __init__(
        self, message: str, technique_name: str, details: str | None = None
    ) -> None:
        """Initialize with the technique name that caused the error."""
        # Store which explainer technique encountered the error
        self.technique_name = technique_name
        # Call parent with formatted message
        super().__init__(f"[{technique_name}] {message}", details)


class ExplainerNotAvailableError(ExplainerError):
    """Raised when a requested explainability technique is not available for the provider."""

    pass


class ExplainerComputationError(ExplainerError):
    """Raised when an explainability computation fails during execution."""

    pass


class ValidationError(LLMExplainabilityError):
    """Raised when input validation fails for user-provided data."""

    def __init__(self, message: str, field_name: str, details: str | None = None) -> None:
        """Initialize with the field name that failed validation."""
        # Store which input field was invalid
        self.field_name = field_name
        # Call parent with formatted message
        super().__init__(f"Validation failed for '{field_name}': {message}", details)


class SecurityError(LLMExplainabilityError):
    """Raised when a security check fails (rate limit, auth, injection detection)."""

    pass
