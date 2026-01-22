"""
Inference request/response schemas.

OpenAI-compatible API schemas for completions and chat.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.serving.schemas.common import UsageInfo


class FinishReason(str, Enum):
    """Reasons for completion finishing."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


# Completion API (OpenAI-compatible)

class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str = Field(default="default", description="Model to use")
    prompt: str | list[str] = Field(..., description="Prompt(s) to complete")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    n: int = Field(default=1, ge=1, le=8, description="Number of completions")
    stream: bool = Field(default=False, description="Stream responses")
    stop: str | list[str] | None = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = Field(default=None, description="User identifier")

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(..., description="Choice index")
    text: str = Field(..., description="Generated text")
    finish_reason: FinishReason = Field(..., description="Why generation stopped")
    logprobs: dict | None = Field(default=None, description="Log probabilities")


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(..., description="Completion ID")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[CompletionChoice] = Field(..., description="Completion choices")
    usage: UsageInfo = Field(..., description="Token usage")


# Chat API (OpenAI-compatible)

class ChatRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A chat message."""

    role: ChatRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(default=None, description="Optional name")


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str = Field(default="default", description="Model to use")
    messages: list[ChatMessage] = Field(..., min_length=1, description="Chat messages")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=8)
    stream: bool = Field(default=False)
    stop: str | list[str] | None = Field(default=None)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = Field(default=None)

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    def to_prompt(self) -> str:
        """Convert messages to a prompt string.

        Uses a simple format compatible with most models.
        """
        parts = []
        for msg in self.messages:
            if msg.role == ChatRole.SYSTEM:
                parts.append(f"System: {msg.content}")
            elif msg.role == ChatRole.USER:
                parts.append(f"Human: {msg.content}")
            elif msg.role == ChatRole.ASSISTANT:
                parts.append(f"Assistant: {msg.content}")

        # Add assistant prefix for generation
        parts.append("Assistant:")

        return "\n\n".join(parts)


class ChatChoice(BaseModel):
    """A single chat completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: FinishReason = Field(..., description="Why generation stopped")


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(..., description="Completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatChoice] = Field(..., description="Completion choices")
    usage: UsageInfo = Field(..., description="Token usage")


# Streaming schemas

class StreamChoice(BaseModel):
    """A streaming chunk choice."""

    index: int = Field(..., description="Choice index")
    delta: dict[str, str] = Field(..., description="Content delta")
    finish_reason: FinishReason | None = Field(default=None)


class StreamResponse(BaseModel):
    """Streaming response chunk."""

    id: str = Field(..., description="Completion ID")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: list[StreamChoice] = Field(..., description="Stream choices")
