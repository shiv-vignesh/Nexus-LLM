"""
Throughput benchmarking for inference engines.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from nexus.core.interfaces.inference_engine import (
    IInferenceEngine,
    GenerationRequest,
)
from nexus.evaluation.metrics.latency import LatencyTracker, LatencyMetrics


@dataclass
class BenchmarkConfig:
    """Configuration for throughput benchmark."""

    num_requests: int = 100
    prompt: str = "Hello, how are you today?"
    max_tokens: int = 128
    temperature: float = 0.7
    concurrency: int = 10


@dataclass
class BenchmarkResult:
    """Results from a throughput benchmark."""

    latency: LatencyMetrics
    config: BenchmarkConfig
    errors: int = 0
    success_rate: float = 1.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Throughput Benchmark Results
============================
Requests: {self.latency.total_requests}
Errors: {self.errors}
Success Rate: {self.success_rate:.1%}

Latency:
  Mean: {self.latency.mean_ms:.2f}ms
  P50:  {self.latency.p50_ms:.2f}ms
  P90:  {self.latency.p90_ms:.2f}ms
  P95:  {self.latency.p95_ms:.2f}ms
  P99:  {self.latency.p99_ms:.2f}ms

Throughput:
  Requests/sec: {self.latency.requests_per_second:.2f}
  Tokens/sec:   {self.latency.tokens_per_second:.2f}
"""


class ThroughputBenchmark:
    """Benchmark inference engine throughput.

    Runs concurrent requests to measure:
    - Latency distribution (p50, p90, p95, p99)
    - Throughput (requests/sec, tokens/sec)
    - Error rate

    Example:
        >>> benchmark = ThroughputBenchmark(engine)
        >>> result = await benchmark.run(BenchmarkConfig(num_requests=100))
        >>> print(result.summary())
    """

    def __init__(self, engine: IInferenceEngine):
        """Initialize benchmark.

        Args:
            engine: Inference engine to benchmark
        """
        self.engine = engine

    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run the throughput benchmark.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with all metrics
        """
        tracker = LatencyTracker()
        tracker.start()

        errors = 0
        semaphore = asyncio.Semaphore(config.concurrency)

        async def make_request(request_id: int) -> bool:
            """Make a single request with concurrency control."""
            nonlocal errors

            async with semaphore:
                request = GenerationRequest(
                    request_id=f"bench-{request_id}",
                    prompt=config.prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                start = time.perf_counter()
                try:
                    response = await self.engine.generate(request)
                    latency_ms = (time.perf_counter() - start) * 1000

                    tracker.record_latency(latency_ms)
                    tracker.record_tokens(response.completion_tokens)
                    return True

                except Exception as e:
                    errors += 1
                    return False

        # Run concurrent requests
        tasks = [make_request(i) for i in range(config.num_requests)]
        await asyncio.gather(*tasks)

        # Compute metrics
        latency_metrics = tracker.compute_metrics()
        success_rate = (config.num_requests - errors) / config.num_requests

        return BenchmarkResult(
            latency=latency_metrics,
            config=config,
            errors=errors,
            success_rate=success_rate,
        )

    async def run_warmup(self, num_requests: int = 5) -> None:
        """Run warmup requests before benchmarking.

        Args:
            num_requests: Number of warmup requests
        """
        for i in range(num_requests):
            request = GenerationRequest(
                request_id=f"warmup-{i}",
                prompt="Hello",
                max_tokens=10,
            )
            try:
                await self.engine.generate(request)
            except Exception:
                pass
