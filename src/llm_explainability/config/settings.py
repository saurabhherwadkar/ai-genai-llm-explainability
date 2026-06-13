# Pydantic settings models for application configuration
"""Defines all configuration data models with validation and defaults."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from llm_explainability.config.constants import (
    DEFAULT_COHERENCE_THRESHOLD,
    DEFAULT_ENVIRONMENT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PROVIDER_TIMEOUT,
    DEFAULT_RATE_LIMIT_RPM,
    DEFAULT_TOP_K_TOKENS,
    VALID_ENVIRONMENTS,
    VALID_LOG_LEVELS,
)


class AppConfig(BaseModel):
    """Top-level application identity and environment settings."""

    # Human-readable application name
    name: str = "LLM Explainability Tool"
    # Semantic version string
    version: str = "1.0.0"
    # Active environment (development, production, testing)
    environment: str = DEFAULT_ENVIRONMENT
    # Enable debug mode for verbose output
    debug: bool = False

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Ensure environment is one of the allowed values."""
        # Convert to lowercase for case-insensitive matching
        normalized = value.lower()
        # Reject invalid environment names
        if normalized not in VALID_ENVIRONMENTS:
            raise ValueError(f"Environment must be one of {VALID_ENVIRONMENTS}, got '{value}'")
        # Return the normalized lowercase value
        return normalized


class LoggingConfig(BaseModel):
    """Logging subsystem configuration controlling levels and output."""

    # Global log level threshold
    level: str = "info"
    # Output format (json for structured, console for human-readable)
    format: str = "json"
    # Path to log file output
    file_output: str = "logs/app.log"
    # Maximum log file size before rotation in megabytes
    max_file_size_mb: int = 50
    # Rotation strategy (daily, size-based)
    rotation: str = "daily"
    # Per-module log level overrides
    modules: dict[str, str] = Field(default_factory=dict)

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure the log level is a recognized value."""
        # Normalize to lowercase
        normalized = value.lower()
        # Check against allowed log levels
        if normalized not in VALID_LOG_LEVELS:
            raise ValueError(f"Log level must be one of {VALID_LOG_LEVELS}, got '{value}'")
        # Return normalized value
        return normalized


class OpenAIProviderConfig(BaseModel):
    """Configuration specific to the OpenAI LLM provider."""

    # API authentication key (loaded from environment)
    api_key: str = ""
    # Model identifier to use for generation
    model: str = "gpt-4o"
    # Maximum tokens in generated response
    max_tokens: int = 4096
    # Sampling temperature (0.0 = deterministic)
    temperature: float = 0.0
    # Request timeout in seconds
    timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT
    # Number of retry attempts on transient failure
    max_retries: int = DEFAULT_MAX_RETRIES


class AnthropicProviderConfig(BaseModel):
    """Configuration specific to the Anthropic Claude provider."""

    # API authentication key (loaded from environment)
    api_key: str = ""
    # Model identifier to use for generation
    model: str = "claude-sonnet-4-20250514"
    # Maximum tokens in generated response
    max_tokens: int = 4096
    # Sampling temperature (0.0 = deterministic)
    temperature: float = 0.0
    # Request timeout in seconds
    timeout_seconds: int = DEFAULT_PROVIDER_TIMEOUT
    # Number of retry attempts on transient failure
    max_retries: int = DEFAULT_MAX_RETRIES


class OllamaProviderConfig(BaseModel):
    """Configuration specific to the Ollama local LLM provider."""

    # Base URL of the Ollama server
    base_url: str = "http://localhost:11434"
    # Model name to use from Ollama's local registry
    model: str = "llama3"
    # Request timeout in seconds (longer for local inference)
    timeout_seconds: int = 60
    # Number of retry attempts on transient failure
    max_retries: int = 2


class HuggingFaceProviderConfig(BaseModel):
    """Configuration specific to the HuggingFace transformers provider."""

    # HuggingFace model identifier (repository/model-name)
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    # Device to load model onto (auto, cpu, cuda)
    device: str = "auto"
    # Enable 8-bit quantization for memory efficiency
    load_in_8bit: bool = True
    # Maximum tokens in generated response
    max_tokens: int = 2048
    # Timeout for model loading and inference in seconds
    timeout_seconds: int = 120


class ProvidersConfig(BaseModel):
    """Aggregated configuration for all LLM providers."""

    # Name of the default provider to use when none specified
    default: str = "openai"
    # OpenAI-specific settings
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    # Anthropic-specific settings
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)
    # Ollama-specific settings
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    # HuggingFace-specific settings
    huggingface: HuggingFaceProviderConfig = Field(default_factory=HuggingFaceProviderConfig)


class TokenAttributionConfig(BaseModel):
    """Configuration for the token attribution explainability engine."""

    # Attribution method to use (logprob, attention, gradient)
    method: str = "logprob"
    # Number of top influential tokens to return
    top_k_tokens: int = DEFAULT_TOP_K_TOKENS
    # Whether to normalize scores to [0, 1] range
    normalize_scores: bool = True


class ShapLimeConfig(BaseModel):
    """Configuration for the SHAP/LIME explainability engine."""

    # Which method to apply (shap, lime, both)
    method: str = "shap"
    # Number of perturbation samples to generate
    num_samples: int = DEFAULT_NUM_SAMPLES
    # Maximum number of features to include in explanation
    max_features: int = 15
    # Batch size for concurrent API calls during perturbation
    batch_size: int = 10


class ChainOfThoughtConfig(BaseModel):
    """Configuration for the chain-of-thought analysis engine."""

    # Maximum reasoning steps to parse
    max_steps: int = 20
    # Minimum coherence score to consider reasoning valid
    coherence_threshold: float = DEFAULT_COHERENCE_THRESHOLD
    # Whether to use an LLM as judge for coherence scoring
    use_llm_judge: bool = True


class ExplainersConfig(BaseModel):
    """Aggregated configuration for all explainability engines."""

    # List of enabled explainer technique names
    enabled: list[str] = Field(
        default_factory=lambda: ["token_attribution", "shap_lime", "chain_of_thought"]
    )
    # Token attribution engine settings
    token_attribution: TokenAttributionConfig = Field(default_factory=TokenAttributionConfig)
    # SHAP/LIME engine settings
    shap_lime: ShapLimeConfig = Field(default_factory=ShapLimeConfig)
    # Chain-of-thought engine settings
    chain_of_thought: ChainOfThoughtConfig = Field(default_factory=ChainOfThoughtConfig)


class SecurityConfig(BaseModel):
    """Security-related configuration for rate limiting and input validation."""

    # Maximum requests per minute per client
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_RPM
    # Maximum allowed prompt character length
    max_prompt_length: int = 10000
    # Maximum allowed response character length
    max_response_length: int = 50000
    # List of allowed CORS origins
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501"]
    )
    # Whether API key authentication is required
    api_key_required: bool = False
    # HTTP header name for API key
    api_key_header: str = "X-API-Key"


class ApiConfig(BaseModel):
    """FastAPI server configuration."""

    # Host address to bind the server to
    host: str = "0.0.0.0"
    # Port number for the server
    port: int = 8000
    # Number of uvicorn worker processes
    workers: int = 4
    # Whether to enable CORS middleware
    cors_enabled: bool = True


class DashboardConfig(BaseModel):
    """Streamlit dashboard configuration."""

    # Base URL of the FastAPI backend
    api_base_url: str = "http://localhost:8000"
    # Browser tab title for the dashboard
    page_title: str = "LLM Explainability Dashboard"
    # Maximum number of explanation history items to retain
    max_history_items: int = 50


class AppSettings(BaseModel):
    """Root configuration model containing all application settings."""

    # Application identity and environment
    app: AppConfig = Field(default_factory=AppConfig)
    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # LLM provider settings
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    # Explainability engine settings
    explainers: ExplainersConfig = Field(default_factory=ExplainersConfig)
    # Security configuration
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    # API server settings
    api: ApiConfig = Field(default_factory=ApiConfig)
    # Dashboard settings
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    def get_provider_config(self, provider_name: str) -> Any:
        """Retrieve configuration for a specific provider by name."""
        # Access the provider config attribute dynamically
        return getattr(self.providers, provider_name, None)
