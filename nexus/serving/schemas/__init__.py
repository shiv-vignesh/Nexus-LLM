"""API request/response schemas."""

from nexus.serving.schemas.inference import (
    CompletionRequest,
    CompletionResponse,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    StreamChoice,
)
from nexus.serving.schemas.common import (
    HealthResponse,
    ErrorResponse,
    ModelInfo,
)

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "StreamChoice",
    "HealthResponse",
    "ErrorResponse",
    "ModelInfo",
]
