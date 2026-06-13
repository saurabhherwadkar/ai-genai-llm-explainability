# Unit tests for input validators
"""Tests for validation utility functions."""

from __future__ import annotations

import pytest

from llm_explainability.exceptions import ValidationError
from llm_explainability.utils.validators import (
    validate_max_tokens,
    validate_num_samples,
    validate_provider_name,
    validate_technique_names,
    validate_temperature,
)


class TestValidateProviderName:
    """Tests for provider name validation."""

    def test_valid_providers(self) -> None:
        """Verify all valid provider names pass."""
        for name in ("openai", "anthropic", "ollama", "huggingface"):
            assert validate_provider_name(name) == name

    def test_case_insensitive(self) -> None:
        """Verify validation is case-insensitive."""
        assert validate_provider_name("OpenAI") == "openai"

    def test_invalid_provider_raises(self) -> None:
        """Verify unsupported providers raise error."""
        with pytest.raises(ValidationError, match="Unsupported provider"):
            validate_provider_name("google")


class TestValidateTechniqueNames:
    """Tests for technique name validation."""

    def test_all_keyword(self) -> None:
        """Verify 'all' returns all supported techniques."""
        result = validate_technique_names(["all"])
        assert "token_attribution" in result
        assert "shap_lime" in result
        assert "chain_of_thought" in result

    def test_valid_techniques(self) -> None:
        """Verify valid technique names pass."""
        result = validate_technique_names(["token_attribution", "shap_lime"])
        assert result == ["token_attribution", "shap_lime"]

    def test_invalid_technique_raises(self) -> None:
        """Verify unsupported techniques raise error."""
        with pytest.raises(ValidationError, match="Unsupported technique"):
            validate_technique_names(["invalid_technique"])

    def test_deduplication(self) -> None:
        """Verify duplicate techniques are removed."""
        result = validate_technique_names(["shap_lime", "shap_lime"])
        assert result == ["shap_lime"]


class TestValidateTemperature:
    """Tests for temperature validation."""

    def test_valid_range(self) -> None:
        """Verify values in range pass."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

    def test_negative_raises(self) -> None:
        """Verify negative temperature raises error."""
        with pytest.raises(ValidationError, match="must be >= 0.0"):
            validate_temperature(-0.1)

    def test_over_max_raises(self) -> None:
        """Verify temperature over 2.0 raises error."""
        with pytest.raises(ValidationError, match="must be <= 2.0"):
            validate_temperature(2.1)


class TestValidateMaxTokens:
    """Tests for max_tokens validation."""

    def test_valid_value(self) -> None:
        """Verify valid token count passes."""
        assert validate_max_tokens(1024) == 1024

    def test_zero_raises(self) -> None:
        """Verify zero tokens raises error."""
        with pytest.raises(ValidationError, match="positive integer"):
            validate_max_tokens(0)

    def test_over_limit_raises(self) -> None:
        """Verify exceeding limit raises error."""
        with pytest.raises(ValidationError, match="cannot exceed"):
            validate_max_tokens(200000)


class TestValidateNumSamples:
    """Tests for num_samples validation."""

    def test_valid_value(self) -> None:
        """Verify valid sample count passes."""
        assert validate_num_samples(100) == 100

    def test_too_few_raises(self) -> None:
        """Verify too few samples raises error."""
        with pytest.raises(ValidationError, match="at least 10"):
            validate_num_samples(5)

    def test_too_many_raises(self) -> None:
        """Verify too many samples raises error."""
        with pytest.raises(ValidationError, match="cannot exceed"):
            validate_num_samples(20000)
