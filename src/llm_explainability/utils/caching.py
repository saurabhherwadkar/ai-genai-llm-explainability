# In-memory caching with TTL support
"""Provides LRU caches with time-to-live expiration for expensive computations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cachetools import TTLCache


class ResponseCache:
    """LRU cache with TTL for storing LLM responses and computation results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """Initialize the cache with maximum size and TTL in seconds."""
        # Create the TTL-based LRU cache with configured limits
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=max_size, ttl=ttl_seconds)

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key, returning None if not found or expired."""
        # Attempt to retrieve the value from the cache
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache with the configured TTL."""
        # Insert or update the cache entry
        self._cache[key] = value

    def has(self, key: str) -> bool:
        """Check whether a key exists in the cache and has not expired."""
        # Use the in operator which respects TTL expiration
        return key in self._cache

    def delete(self, key: str) -> None:
        """Remove a specific entry from the cache."""
        # Pop the key if it exists, ignoring KeyError
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        # Clear the entire cache contents
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return the current number of items in the cache."""
        # Return the current cache length
        return len(self._cache)

    @staticmethod
    def generate_cache_key(prompt: str, provider: str, model: str, **kwargs: Any) -> str:
        """Generate a deterministic cache key from request parameters."""
        # Build a dictionary of all key components
        key_components = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            **kwargs,
        }
        # Serialize to JSON with sorted keys for deterministic ordering
        serialized = json.dumps(key_components, sort_keys=True)
        # Hash the serialized string to produce a fixed-length key
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
