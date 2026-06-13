# Integration tests for API endpoints
"""Tests the FastAPI endpoints with mocked providers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_explainability.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Provide a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Verify the /health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self, client: TestClient) -> None:
        """Verify the /health response includes the app version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] != ""

    def test_ready_returns_200(self, client: TestClient) -> None:
        """Verify the /ready endpoint returns 200 OK."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestProvidersEndpoints:
    """Integration tests for provider listing endpoints."""

    def test_list_providers_returns_200(self, client: TestClient) -> None:
        """Verify the /api/v1/providers endpoint returns a list."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "default_provider" in data
