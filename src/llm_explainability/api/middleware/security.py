# Security middleware for CORS and request validation
"""Provides CORS configuration and request-level security checks."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_explainability.config.settings import SecurityConfig


def configure_cors(app: FastAPI, config: SecurityConfig) -> None:
    """Configure CORS middleware on the FastAPI application."""
    # Add CORS middleware with settings from configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )
