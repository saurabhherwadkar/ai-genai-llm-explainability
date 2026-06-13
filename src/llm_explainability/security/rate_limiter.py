# Token-bucket rate limiter implementation
"""Implements rate limiting to protect against abuse and excessive API usage."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from llm_explainability.exceptions import SecurityError


@dataclass
class TokenBucket:
    """Represents a single token bucket for rate limiting a client."""

    # Maximum number of tokens the bucket can hold
    capacity: float
    # Current number of available tokens
    tokens: float
    # Rate at which tokens refill (tokens per second)
    refill_rate: float
    # Timestamp of the last token refill calculation
    last_refill_time: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Token-bucket rate limiter that tracks requests per client identifier."""

    def __init__(self, requests_per_minute: int) -> None:
        """Initialize the rate limiter with the allowed requests per minute."""
        # Calculate the token refill rate (tokens per second)
        self._refill_rate = requests_per_minute / 60.0
        # Store the bucket capacity (burst allowance equals per-minute limit)
        self._capacity = float(requests_per_minute)
        # Dictionary mapping client identifiers to their token buckets
        self._buckets: dict[str, TokenBucket] = defaultdict(self._create_bucket)

    def check_rate_limit(self, client_id: str) -> None:
        """Check if the client has available rate limit tokens. Raises if exceeded."""
        # Get or create the token bucket for this client
        bucket = self._buckets[client_id]
        # Refill tokens based on time elapsed since last check
        self._refill_bucket(bucket)
        # Check if there are tokens available for this request
        if bucket.tokens < 1.0:
            raise SecurityError(
                f"Rate limit exceeded for client '{client_id}'. "
                f"Try again in {self._seconds_until_next_token(bucket):.1f} seconds.",
            )
        # Consume one token for this request
        bucket.tokens -= 1.0

    def get_remaining_tokens(self, client_id: str) -> float:
        """Return the number of remaining tokens for a client."""
        # Get the bucket for this client
        bucket = self._buckets[client_id]
        # Refill based on elapsed time before checking
        self._refill_bucket(bucket)
        # Return the current token count
        return bucket.tokens

    def reset_client(self, client_id: str) -> None:
        """Reset the rate limit bucket for a specific client."""
        # Remove the client's bucket, allowing it to be recreated fresh
        if client_id in self._buckets:
            del self._buckets[client_id]

    def _create_bucket(self) -> TokenBucket:
        """Create a new token bucket with full capacity."""
        # Initialize a bucket with maximum tokens available
        return TokenBucket(
            capacity=self._capacity,
            tokens=self._capacity,
            refill_rate=self._refill_rate,
        )

    def _refill_bucket(self, bucket: TokenBucket) -> None:
        """Add tokens to the bucket based on elapsed time."""
        # Get the current monotonic timestamp
        now = time.monotonic()
        # Calculate seconds elapsed since last refill
        elapsed = now - bucket.last_refill_time
        # Calculate how many tokens to add based on elapsed time
        tokens_to_add = elapsed * bucket.refill_rate
        # Add tokens but do not exceed bucket capacity
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        # Update the last refill timestamp
        bucket.last_refill_time = now

    def _seconds_until_next_token(self, bucket: TokenBucket) -> float:
        """Calculate seconds until the next token becomes available."""
        # If tokens are available, no wait needed
        if bucket.tokens >= 1.0:
            return 0.0
        # Calculate the deficit and divide by refill rate
        deficit = 1.0 - bucket.tokens
        # Return the time needed to refill one token
        return deficit / bucket.refill_rate
