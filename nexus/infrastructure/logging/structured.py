"""
Structured logging configuration using structlog.

Provides JSON logging for production and pretty printing for development.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    add_timestamp: bool = True,
) -> None:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production)
        add_timestamp: Add timestamp to logs
    """
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger.

    Args:
        name: Logger name (optional)

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    return logger


class RequestLogger:
    """Logger for HTTP requests with automatic context."""

    def __init__(self, logger: structlog.BoundLogger | None = None):
        self.logger = logger or get_logger("http")

    def log_request(
        self,
        method: str,
        path: str,
        request_id: str,
        **extra: Any,
    ) -> None:
        """Log an incoming request."""
        self.logger.info(
            "request_started",
            method=method,
            path=path,
            request_id=request_id,
            **extra,
        )

    def log_response(
        self,
        method: str,
        path: str,
        request_id: str,
        status_code: int,
        latency_ms: float,
        **extra: Any,
    ) -> None:
        """Log a response."""
        log_fn = self.logger.info if status_code < 400 else self.logger.warning

        log_fn(
            "request_completed",
            method=method,
            path=path,
            request_id=request_id,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            **extra,
        )

    def log_error(
        self,
        method: str,
        path: str,
        request_id: str,
        error: Exception,
        **extra: Any,
    ) -> None:
        """Log a request error."""
        self.logger.error(
            "request_error",
            method=method,
            path=path,
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **extra,
        )
