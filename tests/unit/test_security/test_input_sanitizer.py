# Unit tests for the input sanitizer
"""Tests for prompt validation and sanitization."""

from __future__ import annotations

import pytest

from llm_explainability.config.settings import SecurityConfig
from llm_explainability.exceptions import ValidationError
from llm_explainability.security.input_sanitizer import InputSanitizer


@pytest.fixture
def sanitizer() -> InputSanitizer:
    """Provide an InputSanitizer instance with test config."""
    config = SecurityConfig(max_prompt_length=100, max_response_length=200)
    return InputSanitizer(config)


class TestInputSanitizer:
    """Tests for the InputSanitizer class."""

    def test_valid_prompt_passes(self, sanitizer: InputSanitizer) -> None:
        """Verify that a normal prompt passes validation."""
        result = sanitizer.sanitize_prompt("What is the capital of France?")
        assert result == "What is the capital of France?"

    def test_empty_prompt_raises(self, sanitizer: InputSanitizer) -> None:
        """Verify that empty prompts are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            sanitizer.sanitize_prompt("")

    def test_whitespace_only_prompt_raises(self, sanitizer: InputSanitizer) -> None:
        """Verify that whitespace-only prompts are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            sanitizer.sanitize_prompt("   \n\t  ")

    def test_oversized_prompt_raises(self, sanitizer: InputSanitizer) -> None:
        """Verify that prompts exceeding max length are rejected."""
        long_prompt = "x" * 101
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            sanitizer.sanitize_prompt(long_prompt)

    def test_null_bytes_removed(self, sanitizer: InputSanitizer) -> None:
        """Verify that null bytes are stripped from prompts."""
        result = sanitizer.sanitize_prompt("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_leading_trailing_whitespace_stripped(self, sanitizer: InputSanitizer) -> None:
        """Verify that whitespace is trimmed from prompts."""
        result = sanitizer.sanitize_prompt("  hello world  ")
        assert result == "hello world"

    def test_html_escape_in_display_text(self, sanitizer: InputSanitizer) -> None:
        """Verify that HTML is escaped for display text."""
        result = sanitizer.sanitize_display_text("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;" in result

    def test_valid_provider_name(self, sanitizer: InputSanitizer) -> None:
        """Verify that valid provider names pass validation."""
        result = sanitizer.validate_provider_name("openai")
        assert result == "openai"

    def test_invalid_provider_name_raises(self, sanitizer: InputSanitizer) -> None:
        """Verify that provider names with special chars are rejected."""
        with pytest.raises(ValidationError, match="must contain only"):
            sanitizer.validate_provider_name("open-ai!")

    def test_injection_detection_sql(self, sanitizer: InputSanitizer) -> None:
        """Verify that SQL injection patterns are detected."""
        assert sanitizer.check_for_injection("DROP TABLE users") is True

    def test_injection_detection_clean(self, sanitizer: InputSanitizer) -> None:
        """Verify that clean text is not flagged as injection."""
        assert sanitizer.check_for_injection("What is machine learning?") is False
