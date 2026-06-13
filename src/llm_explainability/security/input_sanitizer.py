# Input sanitization for user-provided data
"""Validates and cleans user inputs to prevent injection and abuse."""

from __future__ import annotations

import html
import re

from llm_explainability.config.settings import SecurityConfig
from llm_explainability.exceptions import ValidationError

# Pattern to detect potential script injection attempts
_SCRIPT_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)

# Pattern to detect SQL injection keywords in suspicious contexts
_SQL_INJECTION_PATTERN = re.compile(
    r"(\b(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|EXECUTE)\b\s+(TABLE|FROM|INTO|DATABASE))",
    re.IGNORECASE,
)

# Pattern to detect null bytes which can bypass security checks
_NULL_BYTE_PATTERN = re.compile(r"\x00")


class InputSanitizer:
    """Validates and sanitizes user input to prevent security vulnerabilities."""

    def __init__(self, config: SecurityConfig) -> None:
        """Initialize the sanitizer with security configuration."""
        # Store the maximum allowed prompt length from config
        self._max_prompt_length = config.max_prompt_length
        # Store the maximum allowed response length from config
        self._max_response_length = config.max_response_length

    def sanitize_prompt(self, prompt: str) -> str:
        """Validate and clean a user-provided prompt string."""
        # Check that the prompt is not empty or whitespace-only
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt cannot be empty", "prompt")
        # Check that the prompt does not exceed maximum length
        if len(prompt) > self._max_prompt_length:
            raise ValidationError(
                f"Prompt exceeds maximum length of {self._max_prompt_length} characters",
                "prompt",
            )
        # Remove null bytes that could bypass downstream checks
        cleaned = _NULL_BYTE_PATTERN.sub("", prompt)
        # Strip leading and trailing whitespace
        cleaned = cleaned.strip()
        # Return the cleaned prompt (we don't HTML-escape prompts sent to LLMs)
        return cleaned

    def sanitize_display_text(self, text: str) -> str:
        """Sanitize text that will be displayed in HTML contexts."""
        # Escape HTML special characters to prevent XSS
        escaped = html.escape(text)
        # Remove any script tags that survived escaping
        escaped = _SCRIPT_PATTERN.sub("", escaped)
        # Return the sanitized display-safe text
        return escaped

    def validate_provider_name(self, provider_name: str) -> str:
        """Validate that a provider name contains only safe characters."""
        # Check that the name uses only alphanumeric characters and underscores
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", provider_name):
            raise ValidationError(
                "Provider name must contain only letters, numbers, and underscores",
                "provider",
            )
        # Convert to lowercase for consistent matching
        return provider_name.lower()

    def check_for_injection(self, text: str) -> bool:
        """Check if text contains potential injection patterns. Returns True if suspicious."""
        # Check for SQL injection patterns
        if _SQL_INJECTION_PATTERN.search(text):
            return True
        # Check for null bytes
        if _NULL_BYTE_PATTERN.search(text):
            return True
        # No suspicious patterns detected
        return False
