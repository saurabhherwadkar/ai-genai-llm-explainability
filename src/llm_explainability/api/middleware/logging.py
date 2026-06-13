# Request/response logging middleware
"""Logs incoming requests and outgoing responses for observability."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Module-level logger instance
logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every HTTP request and response with timing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request, log it, call handler, then log the response."""
        # Generate a unique request ID for correlation
        request_id = str(uuid.uuid4())[:8]
        # Attach request ID to request state for downstream access
        request.state.request_id = request_id
        # Record request start time
        start_time = time.perf_counter()
        # Log the incoming request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )
        # Call the next middleware or route handler
        response = await call_next(request)
        # Calculate request duration in milliseconds
        duration_ms = (time.perf_counter() - start_time) * 1000
        # Log the completed response
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        # Add request ID header to response for client correlation
        response.headers["X-Request-ID"] = request_id
        # Return the response to the client
        return response
