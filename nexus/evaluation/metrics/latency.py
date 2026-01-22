"""
Latency and throughput metrics for inference performance.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyMetrics:
    """Latency statistics."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Counts
    total_requests: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
            "requests_per_second": self.requests_per_second,
            "tokens_per_second": self.tokens_per_second,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time_seconds": self.total_time_seconds,
        }


class LatencyTracker:
    """Track and compute latency statistics.

    Example:
        >>> tracker = LatencyTracker()
        >>> with tracker.track():
        ...     # Do inference
        ...     pass
        >>> tracker.record_tokens(128)
        >>> metrics = tracker.compute_metrics()
    """

    def __init__(self):
        """Initialize tracker."""
        self._latencies: list[float] = []
        self._tokens: list[int] = []
        self._start_time: float | None = None
        self._current_start: float | None = None

    def start(self) -> None:
        """Start tracking session."""
        self._latencies = []
        self._tokens = []
        self._start_time = time.perf_counter()

    def track(self):
        """Context manager for tracking a single request.

        Returns:
            Context manager that records latency
        """
        return _LatencyContext(self)

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._latencies.append(latency_ms)

    def record_tokens(self, tokens: int) -> None:
        """Record tokens generated.

        Args:
            tokens: Number of tokens
        """
        self._tokens.append(tokens)

    def compute_metrics(self) -> LatencyMetrics:
        """Compute latency statistics.

        Returns:
            LatencyMetrics with all statistics
        """
        if not self._latencies:
            return LatencyMetrics()

        import numpy as np

        latencies = np.array(self._latencies)
        total_tokens = sum(self._tokens) if self._tokens else 0

        # Calculate time range
        total_time = time.perf_counter() - self._start_time if self._start_time else sum(latencies) / 1000

        return LatencyMetrics(
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            mean_ms=float(np.mean(latencies)),
            median_ms=float(np.median(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p90_ms=float(np.percentile(latencies, 90)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            std_ms=float(np.std(latencies)),
            requests_per_second=len(self._latencies) / total_time if total_time > 0 else 0,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            total_requests=len(self._latencies),
            total_tokens=total_tokens,
            total_time_seconds=total_time,
        )

    def reset(self) -> None:
        """Reset all measurements."""
        self._latencies = []
        self._tokens = []
        self._start_time = None


class _LatencyContext:
    """Context manager for timing operations."""

    def __init__(self, tracker: LatencyTracker):
        self.tracker = tracker
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record_latency(latency_ms)


def compute_throughput(
    total_tokens: int,
    total_time_seconds: float,
) -> dict[str, float]:
    """Compute throughput metrics.

    Args:
        total_tokens: Total tokens generated
        total_time_seconds: Total time in seconds

    Returns:
        Dictionary with throughput metrics
    """
    if total_time_seconds <= 0:
        return {"tokens_per_second": 0.0}

    return {
        "tokens_per_second": total_tokens / total_time_seconds,
    }
