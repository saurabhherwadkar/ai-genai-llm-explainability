# Application-wide constants
"""Immutable values used throughout the application."""

# Default path to the configuration directory
DEFAULT_CONFIG_DIR = "config"

# Default base configuration filename
DEFAULT_CONFIG_FILE = "settings.yaml"

# Environment variable prefix for all app settings
ENV_PREFIX = "LLMX_"

# Environment variable that determines the active environment
ENV_ENVIRONMENT_KEY = "LLMX_ENVIRONMENT"

# Default environment when none is specified
DEFAULT_ENVIRONMENT = "development"

# Supported environment names
VALID_ENVIRONMENTS = ("development", "production", "testing")

# API version prefix for all endpoints
API_VERSION_PREFIX = "/api/v1"

# Maximum allowed prompt length in characters
MAX_PROMPT_LENGTH = 10000

# Maximum allowed response length in characters
MAX_RESPONSE_LENGTH = 50000

# Default timeout for LLM provider API calls in seconds
DEFAULT_PROVIDER_TIMEOUT = 30

# Default maximum retry attempts for transient errors
DEFAULT_MAX_RETRIES = 3

# Default number of SHAP/LIME perturbation samples
DEFAULT_NUM_SAMPLES = 100

# Default top-k tokens to return in attribution results
DEFAULT_TOP_K_TOKENS = 20

# Minimum coherence score threshold for chain-of-thought
DEFAULT_COHERENCE_THRESHOLD = 0.7

# Rate limiting defaults (requests per minute)
DEFAULT_RATE_LIMIT_RPM = 60

# Supported LLM provider names
SUPPORTED_PROVIDERS = ("openai", "anthropic", "ollama", "huggingface")

# Supported explainability technique names
SUPPORTED_EXPLAINERS = ("token_attribution", "shap_lime", "chain_of_thought")

# Log level options
VALID_LOG_LEVELS = ("debug", "info", "warn", "error")
