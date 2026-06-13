# HTTP client for communicating with the FastAPI backend
"""Typed client that calls the FastAPI API endpoints from the dashboard."""

from __future__ import annotations

from typing import Any

import httpx


class ApiClient:
    """HTTP client for calling the LLM Explainability API from the dashboard."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client with the API base URL."""
        # Store the base URL for all API calls
        self._base_url = base_url.rstrip("/")
        # Create a reusable httpx client with timeout
        self._client = httpx.Client(base_url=self._base_url, timeout=120.0)

    def explain(
        self,
        prompt: str,
        provider: str = "openai",
        techniques: list[str] | None = None,
        system_message: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Submit a prompt for full explanation and return the results."""
        # Build the request payload
        payload: dict[str, Any] = {
            "prompt": prompt,
            "provider": provider,
            "techniques": techniques or ["all"],
            "options": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        # Add system message if provided
        if system_message:
            payload["system_message"] = system_message
        # Send POST request to the explain endpoint
        response = self._client.post("/api/v1/explain", json=payload)
        # Raise an error for non-2xx responses
        response.raise_for_status()
        # Return the parsed JSON response
        return response.json()

    def explain_single_technique(
        self,
        prompt: str,
        technique: str,
        provider: str = "openai",
    ) -> dict[str, Any]:
        """Submit a prompt for a single technique explanation."""
        # Map technique names to endpoint paths
        endpoint_map = {
            "token_attribution": "/api/v1/explain/token-attribution",
            "shap_lime": "/api/v1/explain/shap-lime",
            "chain_of_thought": "/api/v1/explain/chain-of-thought",
        }
        # Get the endpoint path for this technique
        endpoint = endpoint_map.get(technique, "/api/v1/explain")
        # Build the request payload
        payload = {"prompt": prompt, "provider": provider}
        # Send the POST request
        response = self._client.post(endpoint, json=payload)
        response.raise_for_status()
        # Return parsed response
        return response.json()

    def list_providers(self) -> dict[str, Any]:
        """Get the list of available providers from the API."""
        # Send GET request to providers endpoint
        response = self._client.get("/api/v1/providers")
        response.raise_for_status()
        return response.json()

    def check_health(self) -> dict[str, Any]:
        """Check if the API backend is healthy."""
        # Send GET request to health endpoint
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()
