"""
Request logging middleware with structured logging.
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from nexus.infrastructure.logging.structured import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging.

    Logs:
    - Request start with method, path, client info
    - Request completion with status code and latency
    - Errors with exception details
    """

    def __init__(self, app, logger=None):
        """Initialize logging middleware.

        Args:
            app: ASGI application
            logger: Optional logger instance
        """
        super().__init__(app)
        self.logger = logger or get_logger("http")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with logging."""
        # Generate request ID if not present
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Add request ID to request state
        request.state.request_id = request_id

        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = self._get_client_ip(request)

        # Log request start
        self.logger.info(
            "request_started",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            query_params=str(request.query_params) if request.query_params else None,
        )

        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log completion
            log_fn = self.logger.info if response.status_code < 400 else self.logger.warning

            log_fn(
                "request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.logger.error(
                "request_error",
                request_id=request_id,
                method=method,
                path=path,
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=round(latency_ms, 2),
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check forwarded headers
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"
