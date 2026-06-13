# Unit tests for the settings module
"""Tests for Pydantic settings model validation and defaults."""

from __future__ import annotations

import pytest

from llm_explainability.config.settings import (
    AppConfig,
    AppSettings,
    LoggingConfig,
    SecurityConfig,
)


class TestAppConfig:
    """Tests for the AppConfig model."""

    def test_default_values(self) -> None:
        """Verify that AppConfig has correct default values."""
        # Create config with defaults
        config = AppConfig()
        # Assert default values are set correctly
        assert config.name == "LLM Explainability Tool"
        assert config.environment == "development"
        assert config.debug is False

    def test_valid_environment(self) -> None:
        """Verify that valid environment names are accepted."""
        # Test all valid environment values
        for env in ("development", "production", "testing"):
            config = AppConfig(environment=env)
            assert config.environment == env

    def test_invalid_environment_raises(self) -> None:
        """Verify that invalid environment names raise ValidationError."""
        # Attempt to create config with invalid environment
        with pytest.raises(ValueError, match="Environment must be one of"):
            AppConfig(environment="staging")

    def test_environment_case_insensitive(self) -> None:
        """Verify that environment matching is case-insensitive."""
        # Test uppercase input
        config = AppConfig(environment="PRODUCTION")
        assert config.environment == "production"


class TestLoggingConfig:
    """Tests for the LoggingConfig model."""

    def test_default_log_level(self) -> None:
        """Verify default log level is info."""
        config = LoggingConfig()
        assert config.level == "info"

    def test_valid_log_levels(self) -> None:
        """Verify all valid log levels are accepted."""
        for level in ("debug", "info", "warn", "error"):
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_invalid_log_level_raises(self) -> None:
        """Verify that invalid log levels raise ValidationError."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            LoggingConfig(level="trace")

    def test_default_format(self) -> None:
        """Verify default format is json."""
        config = LoggingConfig()
        assert config.format == "json"


class TestSecurityConfig:
    """Tests for the SecurityConfig model."""

    def test_default_rate_limit(self) -> None:
        """Verify default rate limit value."""
        config = SecurityConfig()
        assert config.rate_limit_per_minute == 60

    def test_default_api_key_not_required(self) -> None:
        """Verify API key is not required by default."""
        config = SecurityConfig()
        assert config.api_key_required is False


class TestAppSettings:
    """Tests for the root AppSettings model."""

    def test_creates_with_all_defaults(self) -> None:
        """Verify AppSettings can be created with all default values."""
        settings = AppSettings()
        # Check that all sub-configs are populated
        assert settings.app is not None
        assert settings.logging is not None
        assert settings.providers is not None
        assert settings.explainers is not None
        assert settings.security is not None
        assert settings.api is not None
        assert settings.dashboard is not None

    def test_get_provider_config(self) -> None:
        """Verify provider config can be retrieved by name."""
        settings = AppSettings()
        # Get OpenAI config
        openai_config = settings.get_provider_config("openai")
        assert openai_config is not None
        assert openai_config.model == "gpt-4o"

    def test_get_nonexistent_provider_returns_none(self) -> None:
        """Verify nonexistent provider returns None."""
        settings = AppSettings()
        result = settings.get_provider_config("nonexistent")
        assert result is None
