"""
vLLM-based inference engine for high-performance serving.

Provides:
- Continuous batching
- PagedAttention for efficient memory
- Prefix caching for repeated prompts
- Async streaming generation
"""

import asyncio
import time
from typing import AsyncIterator

from nexus.core.config import InferenceConfig
from nexus.core.interfaces.inference_engine import (
    GenerationRequest,
    GenerationResponse,
    StreamingChunk,
)
from nexus.core.exceptions import InferenceError, EngineNotReadyError, ModelLoadError
from nexus.inference.engines.base import BaseInferenceEngine


class VLLMInferenceEngine(BaseInferenceEngine):
    """High-performance inference engine using vLLM.

    Features:
    - Continuous batching for high throughput
    - PagedAttention for memory efficiency
    - Automatic prefix caching
    - Tensor parallelism support
    - Async streaming generation

    Example:
        >>> config = InferenceConfig(model_name_or_path="meta-llama/Llama-2-7b-hf")
        >>> engine = VLLMInferenceEngine(config)
        >>> await engine.initialize()
        >>> response = await engine.generate(request)
    """

    def __init__(self, config: InferenceConfig):
        """Initialize vLLM engine.

        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self._engine = None
        self._tokenizer = None
        self._model_info: dict = {}

    async def initialize(self) -> None:
        """Initialize the vLLM engine.

        Loads the model and prepares for inference.
        """
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs

            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.config.model_name_or_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                enable_prefix_caching=self.config.enable_prefix_caching,
                enforce_eager=self.config.enforce_eager,
                trust_remote_code=True,
            )

            # Create async engine
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)

            # Get tokenizer for token counting
            self._tokenizer = await self._engine.get_tokenizer()

            # Store model info
            self._model_info = {
                "model_name": self.config.model_name_or_path,
                "max_model_len": self.config.max_model_len,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "prefix_caching": self.config.enable_prefix_caching,
                "backend": "vllm",
            }

            self._is_ready = True
            print(f"vLLM engine initialized with {self.config.model_name_or_path}")

        except ImportError:
            raise ModelLoadError(
                self.config.model_name_or_path,
                "vLLM is not installed. Install with: pip install vllm"
            )
        except Exception as e:
            raise ModelLoadError(self.config.model_name_or_path, str(e))

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using vLLM.

        Args:
            request: Generation request

        Returns:
            GenerationResponse with generated text
        """
        if not self._is_ready or self._engine is None:
            raise EngineNotReadyError("vllm")

        start_time = time.perf_counter()

        try:
            # Create sampling params
            sampling_config = self._create_sampling_config(request)
            sampling_params = sampling_config.to_vllm_params()

            # Count prompt tokens
            prompt_tokens = len(self._tokenizer.encode(request.prompt))

            # Generate
            generated_text = ""
            completion_tokens = 0
            finish_reason = "stop"

            async for output in self._engine.generate(
                request.prompt,
                sampling_params,
                request_id=request.request_id,
            ):
                # Get the final output
                if output.finished:
                    if output.outputs:
                        result = output.outputs[0]
                        generated_text = result.text
                        completion_tokens = len(result.token_ids)
                        finish_reason = result.finish_reason or "stop"

            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_metrics(completion_tokens, latency_ms)

            return self._create_response(
                request=request,
                generated_text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise InferenceError(f"Generation failed: {e}")

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamingChunk]:
        """Stream generation results.

        Args:
            request: Generation request with stream=True

        Yields:
            StreamingChunk objects as generation progresses
        """
        if not self._is_ready or self._engine is None:
            raise EngineNotReadyError("vllm")

        try:
            sampling_config = self._create_sampling_config(request)
            sampling_params = sampling_config.to_vllm_params()

            previous_text = ""

            async for output in self._engine.generate(
                request.prompt,
                sampling_params,
                request_id=request.request_id,
            ):
                if output.outputs:
                    result = output.outputs[0]
                    current_text = result.text

                    # Yield only the new text (delta)
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text

                        yield StreamingChunk(
                            request_id=request.request_id,
                            text=delta,
                            is_final=output.finished,
                            finish_reason=result.finish_reason if output.finished else None,
                        )

        except Exception as e:
            yield StreamingChunk(
                request_id=request.request_id,
                text="",
                is_final=True,
                finish_reason="error",
            )
            raise InferenceError(f"Streaming generation failed: {e}")

    async def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResponse]:
        """Batch generation with vLLM's continuous batching.

        vLLM handles batching internally, so we submit all requests
        and collect results efficiently.

        Args:
            requests: List of generation requests

        Returns:
            List of responses in request order
        """
        if not self._is_ready or self._engine is None:
            raise EngineNotReadyError("vllm")

        # Submit all requests concurrently
        tasks = [self.generate(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """Check if engine is healthy.

        Returns:
            True if engine is ready and responsive
        """
        if not self._is_ready or self._engine is None:
            return False

        try:
            # Try a minimal generation to verify engine is working
            test_request = GenerationRequest(
                request_id="health_check",
                prompt="Hello",
                max_tokens=1,
            )

            response = await asyncio.wait_for(
                self.generate(test_request),
                timeout=10.0,
            )

            return response.is_complete

        except Exception:
            return False

    async def get_model_info(self) -> dict:
        """Get information about loaded model.

        Returns:
            Dictionary with model metadata
        """
        return self._model_info.copy()

    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        if self._engine is not None:
            # vLLM AsyncLLMEngine cleanup
            self._engine = None
            self._tokenizer = None
            self._is_ready = False
            print("vLLM engine shutdown complete")

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "vllm"

    async def abort_request(self, request_id: str) -> bool:
        """Abort an in-progress request.

        Args:
            request_id: ID of request to abort

        Returns:
            True if abort was successful
        """
        if self._engine is not None:
            try:
                await self._engine.abort(request_id)
                return True
            except Exception:
                return False
        return False
