# Unit tests for the provider factory
"""Tests for provider registration, creation, and caching."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_explainability.config.settings import AppSettings
from llm_explainability.exceptions import ConfigError
from llm_explainability.providers.base import BaseLLMProvider
from llm_explainability.providers.factory import ProviderFactory


class TestProviderFactory:
    """Tests for the ProviderFactory class."""

    def test_register_and_list(self) -> None:
        """Verify that providers can be registered and listed."""
        factory = ProviderFactory()
        mock_class = MagicMock(spec=type)
        # Register a mock provider
        factory.register("test_provider", mock_class)
        # Verify it appears in the registry
        assert "test_provider" in factory.list_registered()

    def test_create_unregistered_raises(self) -> None:
        """Verify that creating an unregistered provider raises ConfigError."""
        factory = ProviderFactory()
        config = AppSettings()
        with pytest.raises(ConfigError, match="not registered"):
            factory.create("nonexistent", config)

    def test_create_caches_instance(self) -> None:
        """Verify that factory caches and reuses provider instances."""
        factory = ProviderFactory()
        # Create a mock provider class
        mock_instance = MagicMock(spec=BaseLLMProvider)
        mock_class = MagicMock(return_value=mock_instance)
        factory.register("cached", mock_class)
        config = AppSettings()
        # Create twice
        first = factory.create("cached", config)
        second = factory.create("cached", config)
        # Should be same instance
        assert first is second
        # Constructor should only be called once
        assert mock_class.call_count == 1

    def test_clear_instances(self) -> None:
        """Verify that clearing instances forces recreation."""
        factory = ProviderFactory()
        mock_instance = MagicMock(spec=BaseLLMProvider)
        mock_class = MagicMock(return_value=mock_instance)
        factory.register("clearable", mock_class)
        config = AppSettings()
        # Create once
        factory.create("clearable", config)
        # Clear
        factory.clear_instances()
        # Create again - should call constructor again
        factory.create("clearable", config)
        assert mock_class.call_count == 2

    def test_name_normalization(self) -> None:
        """Verify that provider names are normalized to lowercase."""
        factory = ProviderFactory()
        mock_class = MagicMock(return_value=MagicMock(spec=BaseLLMProvider))
        factory.register("MyProvider", mock_class)
        config = AppSettings()
        # Should be findable in lowercase
        assert "myprovider" in factory.list_registered()
