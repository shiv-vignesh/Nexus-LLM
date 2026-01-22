"""
Inference worker for processing requests from Redis queue.

Runs as a separate process to handle inference requests distributed
across multiple workers for horizontal scaling.
"""

import asyncio
import signal
import uuid
from typing import Any

from nexus.core.config import NexusConfig, InferenceConfig
from nexus.core.interfaces.inference_engine import (
    IInferenceEngine,
    GenerationRequest,
    GenerationResponse,
)
from nexus.inference.engines.vllm_engine import VLLMInferenceEngine
from nexus.inference.engines.transformers_engine import TransformersInferenceEngine
from nexus.infrastructure.redis.client import RedisClient
from nexus.infrastructure.redis.streams import RedisStreamQueue, StreamMessage
from nexus.infrastructure.logging.structured import setup_logging, get_logger


class InferenceWorker:
    """Worker that processes inference requests from Redis queue.

    Consumes requests from a Redis stream, processes them through
    the inference engine, and stores results back in Redis.

    Supports:
    - Horizontal scaling (multiple workers)
    - Automatic request claiming for failed workers
    - Graceful shutdown

    Example:
        >>> config = NexusConfig()
        >>> worker = InferenceWorker(config)
        >>> await worker.start()
    """

    def __init__(
        self,
        config: NexusConfig,
        worker_id: str | None = None,
    ):
        """Initialize worker.

        Args:
            config: Application configuration
            worker_id: Unique worker identifier
        """
        self.config = config
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

        self.logger = get_logger(f"worker.{self.worker_id}")
        self._engine: IInferenceEngine | None = None
        self._redis: RedisClient | None = None
        self._queue: RedisStreamQueue | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize worker resources."""
        self.logger.info("initializing_worker", worker_id=self.worker_id)

        # Initialize inference engine
        inference_config = self.config.serving.inference

        if inference_config.backend.value == "vllm":
            self._engine = VLLMInferenceEngine(inference_config)
        else:
            self._engine = TransformersInferenceEngine(inference_config)

        await self._engine.initialize()

        # Initialize Redis
        self._redis = RedisClient(self.config.serving.redis)
        await self._redis.connect()

        # Initialize stream queue
        self._queue = RedisStreamQueue(
            self._redis.client,
            self.config.serving.redis,
            consumer_name=self.worker_id,
        )
        await self._queue.initialize()

        self.logger.info("worker_initialized", worker_id=self.worker_id)

    async def start(self) -> None:
        """Start processing requests."""
        await self.initialize()

        self._running = True

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self.shutdown()),
            )

        self.logger.info("worker_started", worker_id=self.worker_id)

        # Process loop
        try:
            await self._process_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    async def _process_loop(self) -> None:
        """Main processing loop."""
        async for message in self._queue.consume(batch_size=1, block_ms=1000):
            if not self._running:
                break

            try:
                await self._process_message(message)
                await self._queue.acknowledge(message)

            except Exception as e:
                self.logger.error(
                    "message_processing_failed",
                    message_id=message.message_id,
                    error=str(e),
                )
                # Message will be redelivered to another worker

    async def _process_message(self, message: StreamMessage) -> None:
        """Process a single message.

        Args:
            message: Stream message to process
        """
        data = message.data
        request_id = data.get("request_id", message.message_id)

        self.logger.info(
            "processing_request",
            request_id=request_id,
            message_id=message.message_id,
        )

        # Build generation request
        gen_request = GenerationRequest(
            request_id=request_id,
            prompt=data.get("prompt", ""),
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 50),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            stop_sequences=data.get("stop", []),
            user_id=data.get("user"),
        )

        # Generate
        response = await self._engine.generate(gen_request)

        # Store result
        result_key = f"result:{request_id}"
        result_data = {
            "request_id": request_id,
            "status": "completed",
            "generated_text": response.generated_text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
        }

        await self._redis.set(
            result_key,
            result_data,
            ttl_seconds=self.config.serving.redis.result_ttl_seconds,
        )

        self.logger.info(
            "request_completed",
            request_id=request_id,
            tokens=response.completion_tokens,
            latency_ms=round(response.latency_ms, 2),
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown worker."""
        self.logger.info("worker_shutting_down", worker_id=self.worker_id)
        self._running = False
        self._shutdown_event.set()

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._engine:
            await self._engine.shutdown()

        if self._redis:
            await self._redis.disconnect()

        self.logger.info("worker_stopped", worker_id=self.worker_id)


async def run_worker(config_path: str | None = None) -> None:
    """Run an inference worker.

    Args:
        config_path: Optional path to configuration file
    """
    setup_logging(level="INFO")

    config = NexusConfig()
    if config_path:
        config = NexusConfig.from_yaml(config_path)

    worker = InferenceWorker(config)
    await worker.start()


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_worker(config_path))
