"""
Token bucket rate limiting middleware.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    tokens: float = field(default=0.0, init=False)
    last_update: float = field(default_factory=time.time)
    refill_rate: float = 1.0  # tokens per second

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        now = time.time()

        # Refill tokens based on time passed
        time_passed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.refill_rate,
        )
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    @property
    def available_tokens(self) -> int:
        """Get currently available tokens."""
        return int(self.tokens)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm.

    Limits requests per client based on IP address or API key.

    Example:
        >>> app.add_middleware(
        ...     RateLimitMiddleware,
        ...     requests_per_minute=60,
        ...     burst_size=10,
        ... )
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        key_func=None,
    ):
        """Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst size
            key_func: Optional function to extract rate limit key from request
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.key_func = key_func or self._default_key_func
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=burst_size,
                refill_rate=requests_per_minute / 60.0,
            )
        )

    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        # Check for X-Forwarded-For header
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/healthz", "/ready"):
            return await call_next(request)

        # Get rate limit key
        key = self.key_func(request)
        bucket = self._buckets[key]

        # Try to consume a token
        if not bucket.consume():
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Max {self.requests_per_minute} requests per minute.",
                    "retry_after_seconds": 60.0 / self.requests_per_minute,
                },
                headers={
                    "Retry-After": str(int(60.0 / self.requests_per_minute)),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(bucket.available_tokens),
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(bucket.available_tokens)

        return response
