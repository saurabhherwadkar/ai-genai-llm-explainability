# LLM Explainability Tool

A Python tool that explains **why** Large Language Models (LLMs) produce their outputs. It uses multiple explainability techniques — token attribution, SHAP/LIME, and chain-of-thought analysis — to provide insights into LLM behavior across multiple providers.

## Features

- **Provider-Agnostic**: Supports OpenAI, Anthropic, Ollama, and HuggingFace models
- **Multiple Explainability Techniques**:
  - **Token Attribution**: Identifies which input tokens most influenced the output
  - **SHAP/LIME**: Perturbation-based explanation of prompt segment influence
  - **Chain-of-Thought Analysis**: Parses and scores reasoning quality
- **REST API**: FastAPI-based API for programmatic access
- **Interactive Dashboard**: Streamlit UI with heatmaps, plots, and flow diagrams
- **Configurable**: YAML-based configuration with environment-specific overrides

## Project Structure

```
ai-genai-llm-explainability/
├── config/                          # Configuration files
│   ├── settings.yaml                # Main configuration
│   ├── settings.development.yaml    # Development overrides
│   ├── settings.production.yaml     # Production overrides
│   └── .env.example                 # Environment variable template
├── src/llm_explainability/          # Main source package
│   ├── config/                      # Configuration loading & validation
│   ├── providers/                   # LLM provider abstraction layer
│   │   ├── base.py                  # Abstract provider interface
│   │   ├── openai_provider.py       # OpenAI GPT implementation
│   │   ├── anthropic_provider.py    # Anthropic Claude implementation
│   │   ├── ollama_provider.py       # Ollama local model implementation
│   │   ├── huggingface_provider.py  # HuggingFace transformers implementation
│   │   └── factory.py              # Provider factory
│   ├── explainers/                  # Explainability engines
│   │   ├── token_attribution/       # Token-level attribution analysis
│   │   ├── shap_lime/              # SHAP and LIME implementations
│   │   ├── chain_of_thought/       # CoT reasoning analysis
│   │   └── registry.py            # Technique plugin registry
│   ├── aggregator/                  # Results combination & formatting
│   ├── api/                         # FastAPI REST endpoints
│   │   ├── routers/                # Endpoint route handlers
│   │   ├── middleware/             # Security, logging, error handling
│   │   └── schemas/               # Request/response models
│   ├── dashboard/                   # Streamlit interactive UI
│   │   ├── components/             # Visualization widgets
│   │   └── pages/                  # Dashboard pages
│   ├── logging/                     # Structured logging setup
│   ├── security/                    # Input sanitization & rate limiting
│   └── utils/                       # Shared utilities
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests (mocked dependencies)
│   ├── integration/                 # API integration tests
│   └── e2e/                         # End-to-end tests
├── pyproject.toml                   # Dependencies and tool configuration
├── Makefile                         # Common development commands
├── Dockerfile                       # Container build
└── docker-compose.yml               # Multi-service orchestration
```

## Dependencies

| Category | Libraries |
|----------|-----------|
| Core | pydantic, pydantic-settings, pyyaml, python-dotenv, structlog |
| Providers | openai, anthropic, ollama, transformers, torch |
| Explainability | shap, lime, captum, numpy |
| API | fastapi, uvicorn, httpx, slowapi |
| Dashboard | streamlit, plotly |
| Dev | pytest, pytest-asyncio, pytest-cov, ruff, mypy |

## Deployment

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager
- (Optional) Docker and Docker Compose

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-genai-llm-explainability
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   make install-dev
   # Or manually:
   pip install -e ".[dev]"
   ```

4. **Configure environment variables:**
   ```bash
   cp config/.env.example .env
   # Edit .env and add your API keys
   ```

5. **Start the API server:**
   ```bash
   make run-api
   # Server starts at http://localhost:8000
   # API docs at http://localhost:8000/docs
   ```

6. **Start the dashboard (separate terminal):**
   ```bash
   make run-dashboard
   # Dashboard at http://localhost:8501
   ```

### Docker Deployment

```bash
docker-compose up --build
```

This starts both the API (port 8000) and dashboard (port 8501).

### Production Deployment

1. Set `LLMX_ENVIRONMENT=production` environment variable
2. Set all required API keys as environment variables
3. Configure `settings.production.yaml` with your allowed origins
4. Run with multiple workers:
   ```bash
   uvicorn llm_explainability.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
   ```

## Configuration

All configuration is managed via YAML files in the `config/` directory:

- `settings.yaml` — Base configuration (all defaults)
- `settings.development.yaml` — Development overrides (debug mode, console logging)
- `settings.production.yaml` — Production overrides (JSON logging, rate limits)

**Environment variables** override YAML values using the `LLMX_` prefix:
- `LLMX_ENVIRONMENT` — Active environment
- `LLMX_LOG_LEVEL` — Override log level
- `OPENAI_API_KEY` — OpenAI API key
- `ANTHROPIC_API_KEY` — Anthropic API key

## Usage

### API Usage

```bash
# Full explanation (all techniques)
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Why is the sky blue?", "provider": "openai"}'

# Single technique
curl -X POST http://localhost:8000/api/v1/explain/token-attribution \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Why is the sky blue?", "provider": "openai"}'

# List providers
curl http://localhost:8000/api/v1/providers

# Health check
curl http://localhost:8000/health
```

### Development Commands

```bash
make help          # Show all available commands
make lint          # Run linter and type checker
make format        # Auto-format code
make test          # Run unit tests
make test-cov      # Run tests with coverage report
make clean         # Remove build artifacts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/explain` | Full multi-technique explanation |
| POST | `/api/v1/explain/token-attribution` | Token attribution only |
| POST | `/api/v1/explain/shap-lime` | SHAP/LIME analysis only |
| POST | `/api/v1/explain/chain-of-thought` | Chain-of-thought analysis only |
| GET | `/api/v1/providers` | List available providers |
| GET | `/api/v1/providers/{name}/health` | Check provider health |
| GET | `/health` | Application liveness probe |
| GET | `/ready` | Application readiness probe |

## Architecture

The project uses several design patterns for modularity:

- **Strategy Pattern**: Providers and explainers implement a common interface
- **Factory Pattern**: Providers are created by name from configuration
- **Registry Pattern**: Explainer techniques are discoverable via a central registry
- **Dependency Injection**: FastAPI's `Depends()` for loose coupling

All LLM API calls are async for optimal performance when running multiple explainability techniques concurrently.

## License

MIT
