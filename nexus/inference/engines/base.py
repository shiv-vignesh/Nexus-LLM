"""
Base inference engine with common functionality.
"""

import asyncio
import time
from abc import abstractmethod
from typing import AsyncIterator

from nexus.core.interfaces.inference_engine import (
    IInferenceEngine,
    GenerationRequest,
    GenerationResponse,
    StreamingChunk,
)
from nexus.core.config import InferenceConfig
from nexus.inference.generation.sampling import SamplingConfig


class BaseInferenceEngine(IInferenceEngine):
    """Base class for inference engines with common functionality.

    Provides:
    - Request/response handling
    - Metrics tracking
    - Default implementations where possible
    """

    def __init__(self, config: InferenceConfig):
        """Initialize base engine.

        Args:
            config: Inference configuration
        """
        self.config = config
        self._is_ready = False
        self._request_count = 0
        self._total_tokens_generated = 0
        self._total_latency_ms = 0.0

    def _create_sampling_config(self, request: GenerationRequest) -> SamplingConfig:
        """Create sampling config from request."""
        return SamplingConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop_sequences=request.stop_sequences,
        )

    def _create_response(
        self,
        request: GenerationRequest,
        generated_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
        latency_ms: float,
        logprobs: list[dict] | None = None,
    ) -> GenerationResponse:
        """Create a standardized response."""
        tokens_per_second = (
            completion_tokens / (latency_ms / 1000)
            if latency_ms > 0 else 0.0
        )

        return GenerationResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            logprobs=logprobs,
        )

    def _update_metrics(
        self,
        tokens_generated: int,
        latency_ms: float,
    ) -> None:
        """Update internal metrics."""
        self._request_count += 1
        self._total_tokens_generated += tokens_generated
        self._total_latency_ms += latency_ms

    async def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResponse]:
        """Default batch implementation using concurrent generate calls.

        Override for more efficient batched processing.
        """
        tasks = [self.generate(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingChunk]:
        """Default streaming implementation.

        Override for true streaming support.
        """
        # Default: generate full response and yield as single chunk
        response = await self.generate(request)

        yield StreamingChunk(
            request_id=request.request_id,
            text=response.generated_text,
            is_final=True,
            finish_reason=response.finish_reason,
        )

    def get_metrics(self) -> dict:
        """Get engine metrics."""
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0 else 0.0
        )

        return {
            "request_count": self._request_count,
            "total_tokens_generated": self._total_tokens_generated,
            "average_latency_ms": avg_latency,
            "is_ready": self._is_ready,
        }

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready."""
        return self._is_ready

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text - must be implemented by subclasses."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Health check - must be implemented by subclasses."""
        ...

    @abstractmethod
    async def get_model_info(self) -> dict:
        """Get model info - must be implemented by subclasses."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown - must be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Backend name - must be implemented by subclasses."""
        ...
