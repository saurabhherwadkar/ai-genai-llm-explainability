# Configuration loader with YAML parsing and environment overlay
"""Loads and merges configuration from YAML files with environment variable substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from llm_explainability.config.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_ENVIRONMENT,
    ENV_ENVIRONMENT_KEY,
)
from llm_explainability.config.settings import AppSettings


class ConfigLoader:
    """Loads configuration from YAML files and applies environment-specific overrides."""

    def __init__(self, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> None:
        """Initialize the config loader with the configuration directory path."""
        # Store the path to the configuration directory
        self._config_dir = Path(config_dir)
        # Regex pattern to match ${ENV_VAR} placeholders in YAML values
        self._env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    def load(self, environment: str | None = None) -> AppSettings:
        """Load and return the fully-merged application settings."""
        # Determine active environment from argument or environment variable
        active_env = environment or os.getenv(ENV_ENVIRONMENT_KEY, DEFAULT_ENVIRONMENT)
        # Load the base configuration file
        base_config = self._load_yaml_file(DEFAULT_CONFIG_FILE)
        # Load the environment-specific override file
        env_config = self._load_yaml_file(f"settings.{active_env}.yaml")
        # Deep merge environment overrides onto base configuration
        merged_config = self._deep_merge(base_config, env_config)
        # Resolve all ${ENV_VAR} placeholders with actual environment values
        resolved_config = self._resolve_env_vars(merged_config)
        # Validate and return as a structured Pydantic settings model
        return AppSettings(**resolved_config)

    def _load_yaml_file(self, filename: str) -> dict[str, Any]:
        """Load a single YAML file and return its contents as a dictionary."""
        # Construct the full path to the YAML file
        file_path = self._config_dir / filename
        # Return empty dict if the file does not exist
        if not file_path.exists():
            return {}
        # Open and parse the YAML file safely
        with open(file_path, "r", encoding="utf-8") as file_handle:
            # Use safe_load to prevent arbitrary code execution
            content = yaml.safe_load(file_handle)
        # Return empty dict if file is empty or contains no data
        return content if content is not None else {}

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override dictionary into base dictionary."""
        # Create a copy of base to avoid mutating the original
        merged = base.copy()
        # Iterate over each key in the override dictionary
        for key, override_value in override.items():
            # If both base and override have dicts for this key, recurse
            if key in merged and isinstance(merged[key], dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(merged[key], override_value)
            else:
                # Otherwise override the value directly
                merged[key] = override_value
        # Return the merged result
        return merged

    def _resolve_env_vars(self, config: Any) -> Any:
        """Recursively resolve ${ENV_VAR} placeholders in configuration values."""
        # Handle dictionary values by recursing into each value
        if isinstance(config, dict):
            return {key: self._resolve_env_vars(value) for key, value in config.items()}
        # Handle list values by recursing into each element
        if isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        # Handle string values by replacing environment variable placeholders
        if isinstance(config, str):
            return self._substitute_env_vars(config)
        # Return non-string primitives unchanged
        return config

    def _substitute_env_vars(self, value: str) -> str:
        """Replace all ${ENV_VAR} patterns in a string with their environment values."""

        def replace_match(match: re.Match[str]) -> str:
            """Replace a single regex match with the corresponding environment value."""
            # Extract the variable name from the match group
            var_name = match.group(1)
            # Look up the environment variable, defaulting to empty string
            return os.getenv(var_name, "")

        # Apply the substitution pattern to the input string
        return self._env_var_pattern.sub(replace_match, value)
