# Custom structlog processors for log enrichment and sanitization
"""Processors that add context and redact sensitive information from log entries."""

from __future__ import annotations

import re
from typing import Any

# Pattern to match common API key formats in log values
_API_KEY_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,})",
    re.IGNORECASE,
)

# Set of dictionary keys that likely contain sensitive values
_SENSITIVE_KEYS = frozenset({
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "password",
    "token",
    "authorization",
    "auth",
    "credentials",
    "private_key",
})

# Redaction placeholder string
_REDACTED = "***REDACTED***"


def add_app_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add application-level context fields to every log entry."""
    # Add the application name to the log entry
    event_dict.setdefault("app", "llm-explainability")
    # Return the enriched event dictionary
    return event_dict


def sanitize_sensitive_data(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Remove or redact sensitive values from log entries before output."""
    # Iterate over all keys in the event dictionary
    sanitized = {}
    for key, value in event_dict.items():
        # Check if this key is in the set of known sensitive field names
        if key.lower() in _SENSITIVE_KEYS:
            # Replace the value with a redaction placeholder
            sanitized[key] = _REDACTED
        elif isinstance(value, str):
            # Check string values for embedded API key patterns
            sanitized[key] = _redact_embedded_keys(value)
        else:
            # Pass through non-sensitive, non-string values unchanged
            sanitized[key] = value
    # Return the sanitized event dictionary
    return sanitized


def _redact_embedded_keys(value: str) -> str:
    """Replace any API key patterns found within a string value."""
    # Apply the regex substitution to mask key-like patterns
    return _API_KEY_PATTERN.sub(_REDACTED, value)
