"""Inference engine interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class GenerationRequest:
    """Request for text generation."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False

    # Optional metadata
    user_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""

    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str  # "stop", "length", "error"

    # Performance metrics
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Optional detailed info
    logprobs: list[dict] | None = None

    @property
    def is_complete(self) -> bool:
        """Check if generation completed successfully."""
        return self.finish_reason in ("stop", "length")


@dataclass
class StreamingChunk:
    """A streaming chunk of generated text."""

    request_id: str
    text: str
    is_final: bool = False
    finish_reason: str | None = None


class IInferenceEngine(ABC):
    """Interface for inference engines.

    Implementations can use different backends:
    - vLLM for high-performance serving
    - HuggingFace Transformers for flexibility
    - Custom engines for specialized use cases
    """

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a single request.

        Args:
            request: Generation request with prompt and parameters

        Returns:
            GenerationResponse with generated text and metrics
        """
        ...

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text with streaming output.

        Args:
            request: Generation request (stream=True expected)

        Yields:
            StreamingChunk objects as generation progresses
        """
        ...

    @abstractmethod
    async def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResponse]:
        """Generate text for multiple requests.

        Args:
            requests: List of generation requests

        Returns:
            List of responses in same order as requests
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if engine is healthy and ready.

        Returns:
            True if engine is ready to serve requests
        """
        ...

    @abstractmethod
    async def get_model_info(self) -> dict:
        """Get information about loaded model.

        Returns:
            Dictionary with model metadata (name, parameters, etc.)
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if engine is initialized and ready."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get the backend name (e.g., 'vllm', 'transformers')."""
        ...
