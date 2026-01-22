"""
Common schemas used across API endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    redis_connected: bool = Field(default=False, description="Redis connection status")
    uptime_seconds: float = Field(default=0.0, description="Service uptime")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional details")
    request_id: str | None = Field(default=None, description="Request ID if available")


class ModelInfo(BaseModel):
    """Model information response."""

    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    backend: str = Field(..., description="Inference backend")
    max_context_length: int = Field(..., description="Maximum context length")
    capabilities: list[str] = Field(default_factory=list, description="Model capabilities")
    loaded_at: datetime | None = Field(default=None, description="When model was loaded")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens in completion")
    total_tokens: int = Field(..., description="Total tokens used")


class PaginatedResponse(BaseModel):
    """Base for paginated responses."""

    total: int = Field(..., description="Total items")
    page: int = Field(default=1, description="Current page")
    per_page: int = Field(default=20, description="Items per page")
    has_next: bool = Field(default=False, description="Has next page")
    has_prev: bool = Field(default=False, description="Has previous page")
