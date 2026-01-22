"""
Dynamic request batching for inference.

Provides intelligent batching with size and time-based triggers
for efficient GPU utilization.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from nexus.core.interfaces.inference_engine import (
    GenerationRequest,
    GenerationResponse,
)


@dataclass
class BatcherConfig:
    """Configuration for the dynamic batcher."""

    max_batch_size: int = 32
    max_wait_ms: int = 50
    max_queue_size: int = 1000


@dataclass
class PendingRequest:
    """A request waiting to be batched."""

    request: GenerationRequest
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.perf_counter)


class DynamicBatcher:
    """Dynamic batching for inference requests.

    Accumulates requests and flushes them as batches based on:
    1. Batch size threshold (max_batch_size)
    2. Time threshold (max_wait_ms)

    This maximizes GPU utilization while maintaining low latency
    for individual requests.

    Example:
        >>> async def process_batch(requests):
        ...     return await engine.generate_batch(requests)
        >>>
        >>> batcher = DynamicBatcher(config, process_batch)
        >>> await batcher.start()
        >>> response = await batcher.submit(request)
    """

    def __init__(
        self,
        config: BatcherConfig,
        batch_processor: Callable[
            [list[GenerationRequest]], Awaitable[list[GenerationResponse]]
        ],
    ):
        """Initialize the batcher.

        Args:
            config: Batcher configuration
            batch_processor: Async function to process a batch of requests
        """
        self.config = config
        self.batch_processor = batch_processor

        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self._pending: list[PendingRequest] = []
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Metrics
        self._batches_processed = 0
        self._total_requests = 0
        self._total_batch_size = 0

    async def start(self) -> None:
        """Start the batcher worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        print("Dynamic batcher started")

    async def stop(self) -> None:
        """Stop the batcher worker."""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Process remaining requests
        if self._pending:
            await self._flush_batch()

        print("Dynamic batcher stopped")

    async def submit(self, request: GenerationRequest) -> GenerationResponse:
        """Submit a request for batched processing.

        Args:
            request: Generation request

        Returns:
            GenerationResponse when processing completes
        """
        if not self._running:
            raise RuntimeError("Batcher is not running")

        # Create future for result
        future: asyncio.Future[GenerationResponse] = asyncio.Future()

        pending = PendingRequest(
            request=request,
            future=future,
        )

        await self._queue.put(pending)
        return await future

    async def _worker_loop(self) -> None:
        """Main worker loop for batching."""
        while self._running:
            try:
                # Wait for first request or check timeout
                if not self._pending:
                    try:
                        pending = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=0.01,  # 10ms polling
                        )
                        self._pending.append(pending)
                    except asyncio.TimeoutError:
                        continue

                # Collect more requests up to batch size
                while len(self._pending) < self.config.max_batch_size:
                    try:
                        pending = self._queue.get_nowait()
                        self._pending.append(pending)
                    except asyncio.QueueEmpty:
                        break

                # Check if should flush
                if self._should_flush():
                    await self._flush_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Batcher worker error: {e}")
                # Fail all pending requests
                for pending in self._pending:
                    if not pending.future.done():
                        pending.future.set_exception(e)
                self._pending.clear()

    def _should_flush(self) -> bool:
        """Check if batch should be flushed."""
        if not self._pending:
            return False

        # Flush if batch is full
        if len(self._pending) >= self.config.max_batch_size:
            return True

        # Flush if oldest request has waited long enough
        oldest = self._pending[0]
        wait_time_ms = (time.perf_counter() - oldest.enqueue_time) * 1000
        if wait_time_ms >= self.config.max_wait_ms:
            return True

        return False

    async def _flush_batch(self) -> None:
        """Process the current batch."""
        if not self._pending:
            return

        async with self._lock:
            batch = self._pending.copy()
            self._pending.clear()

        requests = [p.request for p in batch]

        try:
            # Process batch
            responses = await self.batch_processor(requests)

            # Update metrics
            self._batches_processed += 1
            self._total_requests += len(batch)
            self._total_batch_size += len(batch)

            # Deliver results
            for pending, response in zip(batch, responses):
                if not pending.future.done():
                    pending.future.set_result(response)

        except Exception as e:
            # Fail all requests in batch
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)

    def get_metrics(self) -> dict:
        """Get batcher metrics."""
        avg_batch_size = (
            self._total_batch_size / self._batches_processed
            if self._batches_processed > 0 else 0.0
        )

        return {
            "batches_processed": self._batches_processed,
            "total_requests": self._total_requests,
            "average_batch_size": avg_batch_size,
            "pending_requests": len(self._pending),
            "queue_size": self._queue.qsize(),
        }

    @property
    def is_running(self) -> bool:
        """Check if batcher is running."""
        return self._running
