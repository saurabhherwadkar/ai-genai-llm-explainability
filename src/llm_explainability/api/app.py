# FastAPI application factory
"""Creates and configures the FastAPI application with all middleware and routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from llm_explainability.api.dependencies import get_config
from llm_explainability.api.middleware.error_handler import global_exception_handler
from llm_explainability.api.middleware.logging import RequestLoggingMiddleware
from llm_explainability.api.middleware.security import configure_cors
from llm_explainability.api.routers import explain, health, providers
from llm_explainability.config.settings import AppSettings
from llm_explainability.exceptions import LLMExplainabilityError
from llm_explainability.logging.setup import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Load configuration at startup
    config = get_config()
    # Configure the logging system
    setup_logging(config.logging)
    # Store debug flag in app state for error handler access
    app.state.debug = config.app.debug
    # Yield control to the application (runs until shutdown)
    yield
    # Cleanup on shutdown (close connections, etc.)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    # Load configuration for app metadata
    config = get_config()
    # Create the FastAPI application with metadata
    app = FastAPI(
        title=config.app.name,
        version=config.app.version,
        description="API for explaining LLM outputs using multiple techniques",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    # Configure CORS middleware
    configure_cors(app, config.security)
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    # Register global exception handler
    app.add_exception_handler(LLMExplainabilityError, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
    # Include all route modules
    app.include_router(health.router)
    app.include_router(providers.router)
    app.include_router(explain.router)
    # Return the configured application
    return app
