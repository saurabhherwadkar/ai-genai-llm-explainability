# Unit tests for the rate limiter
"""Tests for token-bucket rate limiting behavior."""

from __future__ import annotations

import pytest

from llm_explainability.exceptions import SecurityError
from llm_explainability.security.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_allows_requests_within_limit(self) -> None:
        """Verify that requests within the limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10)
        # Should not raise for first 10 requests
        for _ in range(10):
            limiter.check_rate_limit("client1")

    def test_blocks_requests_over_limit(self) -> None:
        """Verify that requests exceeding the limit are blocked."""
        limiter = RateLimiter(requests_per_minute=5)
        # Use up all tokens
        for _ in range(5):
            limiter.check_rate_limit("client1")
        # Next request should be blocked
        with pytest.raises(SecurityError, match="Rate limit exceeded"):
            limiter.check_rate_limit("client1")

    def test_different_clients_independent(self) -> None:
        """Verify that rate limits are independent per client."""
        limiter = RateLimiter(requests_per_minute=2)
        # Client A uses all tokens
        limiter.check_rate_limit("clientA")
        limiter.check_rate_limit("clientA")
        # Client B should still have tokens
        limiter.check_rate_limit("clientB")  # Should not raise

    def test_get_remaining_tokens(self) -> None:
        """Verify remaining tokens are reported correctly."""
        limiter = RateLimiter(requests_per_minute=10)
        # Initial remaining should be 10
        remaining = limiter.get_remaining_tokens("client1")
        assert remaining == 10.0
        # After one request, should be 9
        limiter.check_rate_limit("client1")
        remaining = limiter.get_remaining_tokens("client1")
        assert remaining == pytest.approx(9.0, abs=0.1)

    def test_reset_client(self) -> None:
        """Verify that resetting a client restores their tokens."""
        limiter = RateLimiter(requests_per_minute=2)
        # Use up all tokens
        limiter.check_rate_limit("client1")
        limiter.check_rate_limit("client1")
        # Reset the client
        limiter.reset_client("client1")
        # Should have full tokens again
        limiter.check_rate_limit("client1")  # Should not raise
