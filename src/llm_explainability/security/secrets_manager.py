# Secure management of API keys and secrets
"""Handles loading, caching, and secure access to sensitive credentials."""

from __future__ import annotations

import os

from llm_explainability.exceptions import ConfigError


class SecretsManager:
    """Securely loads and provides access to API keys and other secrets."""

    def __init__(self) -> None:
        """Initialize the secrets manager with an empty cache."""
        # Cache for resolved secret values to avoid repeated env lookups
        self._cache: dict[str, str] = {}

    def get_secret(self, env_var_name: str, required: bool = True) -> str:
        """Retrieve a secret value from environment variables."""
        # Check cache first to avoid repeated OS calls
        if env_var_name in self._cache:
            return self._cache[env_var_name]
        # Look up the environment variable
        value = os.getenv(env_var_name, "")
        # Raise an error if the secret is required but not set
        if required and not value:
            raise ConfigError(
                f"Required secret '{env_var_name}' is not set in environment",
                details="Set this environment variable or add it to your .env file",
            )
        # Cache the resolved value for future lookups
        self._cache[env_var_name] = value
        # Return the secret value
        return value

    def get_api_key(self, provider_name: str) -> str:
        """Retrieve the API key for a specific LLM provider."""
        # Map provider names to their environment variable names
        env_var_map: dict[str, str] = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "huggingface": "HUGGINGFACE_TOKEN",
        }
        # Look up the environment variable name for this provider
        env_var = env_var_map.get(provider_name)
        # Raise error if provider name is not recognized
        if env_var is None:
            raise ConfigError(
                f"No API key mapping for provider '{provider_name}'",
                details=f"Supported providers: {list(env_var_map.keys())}",
            )
        # Retrieve and return the API key value
        return self.get_secret(env_var, required=True)

    def clear_cache(self) -> None:
        """Clear the cached secret values from memory."""
        # Overwrite cached values before clearing for security
        for key in self._cache:
            self._cache[key] = ""
        # Remove all entries from the cache
        self._cache.clear()

    def mask_secret(self, value: str) -> str:
        """Return a masked version of a secret for safe logging."""
        # Show only the last 4 characters for identification
        if len(value) <= 4:
            return "****"
        # Mask all but the last 4 characters
        return "*" * (len(value) - 4) + value[-4:]
