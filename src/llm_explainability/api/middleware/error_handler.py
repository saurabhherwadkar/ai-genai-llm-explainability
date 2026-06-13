# Global exception handler for the FastAPI application
"""Maps application exceptions to appropriate HTTP error responses."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_explainability.exceptions import (
    ConfigError,
    ExplainerError,
    ExplainerNotAvailableError,
    LLMExplainabilityError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SecurityError,
    ValidationError,
)


async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle all unhandled exceptions and return safe error responses."""
    # Determine if we're in debug mode from app state
    debug = getattr(request.app.state, "debug", False)
    # Map known exceptions to HTTP status codes and messages
    if isinstance(exc, ValidationError):
        return _error_response(422, "validation_error", exc.message, exc, debug)
    if isinstance(exc, ProviderAuthError):
        return _error_response(401, "auth_error", "Authentication failed", exc, debug)
    if isinstance(exc, ProviderRateLimitError):
        return _error_response(429, "rate_limit", "Rate limit exceeded", exc, debug)
    if isinstance(exc, ProviderTimeoutError):
        return _error_response(504, "timeout", "Request timed out", exc, debug)
    if isinstance(exc, ProviderConnectionError):
        return _error_response(502, "connection_error", "Provider unavailable", exc, debug)
    if isinstance(exc, ExplainerNotAvailableError):
        return _error_response(
            400, "explainer_unavailable", "Technique not available for this provider", exc, debug
        )
    if isinstance(exc, ExplainerError):
        return _error_response(500, "explainer_error", "Explanation failed", exc, debug)
    if isinstance(exc, SecurityError):
        return _error_response(429, "security_error", str(exc), exc, debug)
    if isinstance(exc, ConfigError):
        return _error_response(500, "config_error", "Configuration error", exc, debug)
    if isinstance(exc, LLMExplainabilityError):
        return _error_response(500, "internal_error", exc.message, exc, debug)
    # Handle completely unexpected exceptions
    return _error_response(
        500, "internal_error", "An unexpected error occurred", exc, debug
    )


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    exc: Exception,
    debug: bool,
) -> JSONResponse:
    """Build a standardized JSON error response."""
    # Build the response body
    body: dict[str, str | None] = {
        "error": error_type,
        "message": message,
    }
    # Include details only in debug mode to prevent information leakage
    if debug:
        body["details"] = str(exc)
    else:
        body["details"] = None
    # Return the JSON response with appropriate status code
    return JSONResponse(status_code=status_code, content=body)
