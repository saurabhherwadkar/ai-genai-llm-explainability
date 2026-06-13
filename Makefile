.PHONY: install lint format test test-cov run-api run-dashboard clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install project dependencies"
	@echo "  make install-dev    - Install project with dev dependencies"
	@echo "  make lint           - Run ruff linter and mypy type checker"
	@echo "  make format         - Auto-format code with ruff"
	@echo "  make test           - Run unit tests"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make test-all       - Run all tests including e2e"
	@echo "  make run-api        - Start the FastAPI server"
	@echo "  make run-dashboard  - Start the Streamlit dashboard"
	@echo "  make clean          - Remove build artifacts"

# Install dependencies
install:
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install -e ".[dev]"

# Linting and type checking
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

# Auto-format code
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Run unit tests only
test:
	pytest tests/unit/ -v

# Run tests with coverage
test-cov:
	pytest tests/unit/ tests/integration/ --cov --cov-report=html --cov-report=term-missing

# Run all tests including e2e
test-all:
	pytest tests/ -v --run-e2e

# Start FastAPI server
run-api:
	uvicorn llm_explainability.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

# Start Streamlit dashboard
run-dashboard:
	streamlit run src/llm_explainability/dashboard/app.py --server.port 8501

# Remove build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
