# Unit tests for the configuration loader
"""Tests for YAML loading, environment overlay, and variable substitution."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_explainability.config.loader import ConfigLoader


class TestConfigLoader:
    """Tests for the ConfigLoader class."""

    def test_load_default_config(self, tmp_path: Path) -> None:
        """Verify that loading works with a minimal settings file."""
        # Create a minimal settings.yaml
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            "app:\n  name: Test App\n  environment: testing\n"
        )
        # Load configuration from the temp directory
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="testing")
        # Verify the loaded values
        assert config.app.name == "Test App"
        assert config.app.environment == "testing"

    def test_environment_overlay(self, tmp_path: Path) -> None:
        """Verify that environment-specific files override base settings."""
        # Create base settings
        base = tmp_path / "settings.yaml"
        base.write_text("app:\n  name: Base App\n  debug: false\n")
        # Create development overlay
        dev = tmp_path / "settings.development.yaml"
        dev.write_text("app:\n  debug: true\n")
        # Load with development environment
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="development")
        # Verify overlay was applied
        assert config.app.debug is True
        # Verify base value was preserved
        assert config.app.name == "Base App"

    def test_env_var_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that ${ENV_VAR} placeholders are resolved."""
        # Set an environment variable
        monkeypatch.setenv("TEST_API_KEY", "sk-test-12345")
        # Create settings with env var reference
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            "app:\n  name: Test\nproviders:\n  openai:\n    api_key: ${TEST_API_KEY}\n"
        )
        # Load and verify substitution
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="testing")
        assert config.providers.openai.api_key == "sk-test-12345"

    def test_missing_env_var_resolves_to_empty(self, tmp_path: Path) -> None:
        """Verify that missing env vars resolve to empty string."""
        # Create settings referencing a non-existent env var
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            "providers:\n  openai:\n    api_key: ${NONEXISTENT_VAR}\n"
        )
        # Load and verify empty string
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="testing")
        assert config.providers.openai.api_key == ""

    def test_missing_config_file_returns_defaults(self, tmp_path: Path) -> None:
        """Verify that missing config files result in default values."""
        # Load from empty directory (no settings.yaml)
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="testing")
        # Should have all defaults
        assert config.app.name == "LLM Explainability Tool"

    def test_deep_merge_preserves_nested_values(self, tmp_path: Path) -> None:
        """Verify that deep merge preserves non-overridden nested values."""
        # Create base with nested structure
        base = tmp_path / "settings.yaml"
        base.write_text(
            "providers:\n  openai:\n    model: gpt-4o\n    max_tokens: 4096\n"
        )
        # Overlay changes only one nested field
        dev = tmp_path / "settings.development.yaml"
        dev.write_text("providers:\n  openai:\n    max_tokens: 1024\n")
        # Load and verify
        loader = ConfigLoader(config_dir=tmp_path)
        config = loader.load(environment="development")
        # Overridden value should be new
        assert config.providers.openai.max_tokens == 1024
        # Non-overridden value should be preserved
        assert config.providers.openai.model == "gpt-4o"
