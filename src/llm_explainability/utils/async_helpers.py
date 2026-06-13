# Async utility functions for retry logic and timeout handling
"""Provides decorators and helpers for resilient asynchronous operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from llm_explainability.exceptions import (
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Type variable for generic return type preservation
T = TypeVar("T")


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 30.0,
) -> Any:
    """Create a tenacity retry decorator with exponential backoff and jitter."""
    # Build the retry decorator with configured parameters
    return retry(
        # Retry on timeout and rate limit errors (transient failures)
        retry=retry_if_exception_type((ProviderTimeoutError, ProviderRateLimitError)),
        # Stop after the configured number of attempts
        stop=stop_after_attempt(max_attempts),
        # Use exponential backoff with jitter to spread retries
        wait=wait_exponential_jitter(
            initial=min_wait_seconds,
            max=max_wait_seconds,
            jitter=2,
        ),
        # Re-raise the last exception if all retries are exhausted
        reraise=True,
    )


async def run_with_timeout(
    coroutine: Awaitable[T],
    timeout_seconds: float,
    error_message: str = "Operation timed out",
) -> T:
    """Execute a coroutine with a timeout, raising ProviderTimeoutError on expiry."""
    try:
        # Wrap the coroutine with asyncio's timeout mechanism
        result = await asyncio.wait_for(coroutine, timeout=timeout_seconds)
        # Return the successful result
        return result
    except asyncio.TimeoutError:
        # Convert asyncio timeout into our custom exception type
        raise ProviderTimeoutError(
            message=error_message,
            provider_name="unknown",
            details=f"Operation exceeded {timeout_seconds}s timeout",
        )


async def run_concurrent(
    tasks: list[Awaitable[T]],
    max_concurrency: int = 10,
) -> list[T | None]:
    """Run multiple async tasks concurrently with a concurrency limit."""
    # Create a semaphore to limit concurrent execution
    semaphore = asyncio.Semaphore(max_concurrency)
    # List to collect results in order
    results: list[T | None] = []

    async def limited_task(task: Awaitable[T]) -> T | None:
        """Execute a single task while respecting the concurrency semaphore."""
        # Acquire the semaphore before running
        async with semaphore:
            try:
                # Execute the task and return its result
                return await task
            except Exception:
                # Return None for failed tasks to maintain result ordering
                return None

    # Create limited versions of all tasks
    limited_tasks = [limited_task(task) for task in tasks]
    # Execute all tasks concurrently and gather results
    results = await asyncio.gather(*limited_tasks)
    # Return the list of results (some may be None on failure)
    return list(results)


def async_retry(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 30.0,
) -> Callable[..., Any]:
    """Decorator that adds retry logic to async functions."""
    # Create the tenacity retry configuration
    retry_decorator = create_retry_decorator(max_attempts, min_wait_seconds, max_wait_seconds)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Apply the retry decorator to the target async function."""
        # Wrap the function with retry logic while preserving its signature
        @wraps(func)
        @retry_decorator
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute the wrapped function with retry on transient errors."""
            return await func(*args, **kwargs)

        # Return the wrapped function
        return wrapper

    # Return the decorator
    return decorator
