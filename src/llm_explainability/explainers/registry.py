# Explainer registry for plugin-style technique discovery
"""Central registry that manages available explainability techniques."""

from __future__ import annotations

from llm_explainability.config.settings import ExplainersConfig
from llm_explainability.exceptions import ConfigError
from llm_explainability.explainers.base import BaseExplainer


class ExplainerRegistry:
    """Registry that manages and provides access to explainer technique instances."""

    def __init__(self) -> None:
        """Initialize the registry with empty class and instance maps."""
        # Mapping from technique names to their class types
        self._registry: dict[str, type[BaseExplainer]] = {}
        # Mapping from technique names to instantiated explainer objects
        self._instances: dict[str, BaseExplainer] = {}

    def register(self, name: str, explainer_class: type[BaseExplainer]) -> None:
        """Register an explainer class with a given technique name."""
        # Store the class type under the normalized name
        self._registry[name.lower()] = explainer_class

    def get(self, name: str, config: ExplainersConfig) -> BaseExplainer:
        """Get or create an explainer instance by technique name."""
        # Normalize the name for consistent lookup
        normalized = name.lower()
        # Return cached instance if already created
        if normalized in self._instances:
            return self._instances[normalized]
        # Check that the technique is registered
        if normalized not in self._registry:
            raise ConfigError(
                f"Explainer technique '{name}' is not registered",
                details=f"Available techniques: {list(self._registry.keys())}",
            )
        # Get the explainer class
        explainer_class = self._registry[normalized]
        # Get technique-specific config
        technique_config = getattr(config, normalized, None)
        # Instantiate the explainer with its config
        if technique_config is not None:
            instance = explainer_class(technique_config)
        else:
            instance = explainer_class()
        # Cache the instance for reuse
        self._instances[normalized] = instance
        # Return the explainer instance
        return instance

    def get_enabled(self, config: ExplainersConfig) -> list[BaseExplainer]:
        """Get all enabled explainer instances based on configuration."""
        # Collect enabled explainer instances
        enabled_explainers: list[BaseExplainer] = []
        # Iterate over the enabled technique names from config
        for technique_name in config.enabled:
            # Get or create the explainer instance
            explainer = self.get(technique_name, config)
            # Add to the enabled list
            enabled_explainers.append(explainer)
        # Return all enabled explainers
        return enabled_explainers

    def list_registered(self) -> list[str]:
        """Return a list of all registered technique names."""
        # Return sorted list of registry keys
        return sorted(self._registry.keys())

    def clear_instances(self) -> None:
        """Clear all cached instances (useful for testing)."""
        # Remove all cached explainer instances
        self._instances.clear()


def create_default_registry() -> ExplainerRegistry:
    """Create a registry with all built-in explainer techniques registered."""
    # Import explainer implementations here to avoid circular imports
    from llm_explainability.explainers.chain_of_thought.engine import (
        ChainOfThoughtEngine,
    )
    from llm_explainability.explainers.shap_lime.engine import ShapLimeEngine
    from llm_explainability.explainers.token_attribution.engine import (
        TokenAttributionEngine,
    )

    # Create a new registry instance
    registry = ExplainerRegistry()
    # Register all built-in explainer techniques
    registry.register("token_attribution", TokenAttributionEngine)
    registry.register("shap_lime", ShapLimeEngine)
    registry.register("chain_of_thought", ChainOfThoughtEngine)
    # Return the configured registry
    return registry
