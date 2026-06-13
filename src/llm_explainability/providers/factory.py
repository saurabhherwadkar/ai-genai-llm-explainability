# Provider factory for instantiating LLM providers by name
"""Creates provider instances based on configuration using the factory pattern."""

from __future__ import annotations

from typing import Any

from llm_explainability.config.settings import AppSettings
from llm_explainability.exceptions import ConfigError
from llm_explainability.providers.base import BaseLLMProvider


class ProviderFactory:
    """Factory that creates and manages LLM provider instances by name."""

    def __init__(self) -> None:
        """Initialize the factory with an empty provider registry."""
        # Registry mapping provider names to their class types
        self._registry: dict[str, type[BaseLLMProvider]] = {}
        # Cache of instantiated provider instances
        self._instances: dict[str, BaseLLMProvider] = {}

    def register(self, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a provider class with a given name in the factory."""
        # Store the class type in the registry under the normalized name
        self._registry[name.lower()] = provider_class

    def create(self, provider_name: str, config: AppSettings) -> BaseLLMProvider:
        """Create or retrieve a cached provider instance by name."""
        # Normalize the provider name to lowercase
        normalized_name = provider_name.lower()
        # Return cached instance if already created
        if normalized_name in self._instances:
            return self._instances[normalized_name]
        # Check that the provider is registered
        if normalized_name not in self._registry:
            raise ConfigError(
                f"Provider '{provider_name}' is not registered",
                details=f"Available providers: {list(self._registry.keys())}",
            )
        # Get the provider class from the registry
        provider_class = self._registry[normalized_name]
        # Get the provider-specific configuration
        provider_config = config.get_provider_config(normalized_name)
        # Raise error if no config found for this provider
        if provider_config is None:
            raise ConfigError(
                f"No configuration found for provider '{provider_name}'",
                details="Check your settings.yaml file",
            )
        # Instantiate the provider with its configuration
        instance = provider_class(provider_config)
        # Cache the instance for reuse
        self._instances[normalized_name] = instance
        # Return the created provider instance
        return instance

    def get_default_provider(self, config: AppSettings) -> BaseLLMProvider:
        """Create or retrieve the default provider as specified in configuration."""
        # Get the default provider name from settings
        default_name = config.providers.default
        # Create and return the default provider instance
        return self.create(default_name, config)

    def list_registered(self) -> list[str]:
        """Return a list of all registered provider names."""
        # Return the registry keys as a sorted list
        return sorted(self._registry.keys())

    def clear_instances(self) -> None:
        """Clear all cached provider instances (useful for testing)."""
        # Remove all cached instances
        self._instances.clear()


def create_default_factory() -> ProviderFactory:
    """Create a factory with all built-in providers registered."""
    # Import provider implementations here to avoid circular imports
    from llm_explainability.providers.anthropic_provider import AnthropicProvider
    from llm_explainability.providers.huggingface_provider import HuggingFaceProvider
    from llm_explainability.providers.ollama_provider import OllamaProvider
    from llm_explainability.providers.openai_provider import OpenAIProvider

    # Create a new factory instance
    factory = ProviderFactory()
    # Register all built-in provider implementations
    factory.register("openai", OpenAIProvider)
    factory.register("anthropic", AnthropicProvider)
    factory.register("ollama", OllamaProvider)
    factory.register("huggingface", HuggingFaceProvider)
    # Return the configured factory
    return factory
