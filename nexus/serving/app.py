"""
FastAPI application factory and server entry point.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nexus import __version__
from nexus.core.config import NexusConfig, ServingConfig
from nexus.core.exceptions import NexusError, RateLimitError
from nexus.serving.routers import inference_router, models_router
from nexus.serving.middleware.rate_limiting import RateLimitMiddleware
from nexus.serving.middleware.request_logging import RequestLoggingMiddleware
from nexus.serving.dependencies import get_app_state
from nexus.serving.schemas.common import HealthResponse, ErrorResponse
from nexus.infrastructure.logging.structured import setup_logging


def create_app(config: NexusConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional configuration (loads from env if not provided)

    Returns:
        Configured FastAPI application
    """
    config = config or NexusConfig()

    # Setup logging
    setup_logging(
        level=config.log_level,
        json_format=False,
    )

    # Track startup time
    startup_time = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager."""
        # Startup
        state = get_app_state()
        await state.initialize(config)

        yield

        # Shutdown
        await state.shutdown()

    # Create app
    app = FastAPI(
        title="Nexus-LLM API",
        description="End-to-end LLM post-training, evaluation, and serving platform",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if config.serving.enable_docs else None,
        redoc_url="/redoc" if config.serving.enable_docs else None,
    )

    # Store config in app state
    app.state.config = config
    app.state.startup_time = startup_time

    # Add middleware (order matters - first added is outermost)
    _configure_middleware(app, config.serving)

    # Add routers
    app.include_router(inference_router)
    app.include_router(models_router)

    # Add root routes
    _add_root_routes(app)

    # Add exception handlers
    _configure_exception_handlers(app)

    return app


def _configure_middleware(app: FastAPI, config: ServingConfig) -> None:
    """Configure application middleware."""

    # Request logging (innermost)
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.rate_limit_requests,
        burst_size=min(config.rate_limit_requests // 6, 20),
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _add_root_routes(app: FastAPI) -> None:
    """Add root-level routes."""

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint."""
        return {
            "name": "Nexus-LLM API",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check(request: Request):
        """Health check endpoint."""
        state = get_app_state()
        startup_time = getattr(request.app.state, "startup_time", time.time())

        return HealthResponse(
            status="healthy" if state.is_initialized else "starting",
            version=__version__,
            model_loaded=state.inference_engine.is_ready if state.inference_engine else False,
            redis_connected=state.redis_client.is_connected if state.redis_client else False,
            uptime_seconds=time.time() - startup_time,
        )

    @app.get("/ready", tags=["system"])
    async def readiness_check():
        """Kubernetes readiness probe."""
        state = get_app_state()

        if not state.is_initialized:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "message": "Application initializing"},
            )

        if state.inference_engine and not state.inference_engine.is_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "message": "Model loading"},
            )

        return {"status": "ready"}


def _configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers."""

    @app.exception_handler(NexusError)
    async def nexus_error_handler(request: Request, exc: NexusError):
        """Handle Nexus-specific errors."""
        status_code = 500

        if isinstance(exc, RateLimitError):
            status_code = 429

        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.message,
                details=exc.details,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """Run the server using uvicorn.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    import uvicorn

    uvicorn.run(
        "nexus.serving.app:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    run_server()
