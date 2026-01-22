#!/usr/bin/env python3
"""
CLI script for benchmarking inference performance.

Usage:
    python scripts/benchmark_inference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --requests 100
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nexus.core.config import InferenceConfig, InferenceBackend
from nexus.inference.engines.vllm_engine import VLLMInferenceEngine
from nexus.inference.engines.transformers_engine import TransformersInferenceEngine
from nexus.evaluation.benchmarks.throughput import ThroughputBenchmark, BenchmarkConfig


console = Console()
app = typer.Typer(help="Benchmark inference performance")


@app.command()
def run(
    model: str = typer.Option(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--model",
        "-m",
        help="Model name or path",
    ),
    backend: str = typer.Option(
        "transformers",
        "--backend",
        "-b",
        help="Inference backend (vllm or transformers)",
    ),
    requests: int = typer.Option(
        100,
        "--requests",
        "-n",
        help="Number of requests to run",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-c",
        help="Number of concurrent requests",
    ),
    max_tokens: int = typer.Option(
        128,
        "--max-tokens",
        help="Maximum tokens per request",
    ),
    prompt: str = typer.Option(
        "Explain the concept of machine learning in simple terms.",
        "--prompt",
        "-p",
        help="Prompt to use for benchmarking",
    ),
    warmup: int = typer.Option(
        5,
        "--warmup",
        help="Number of warmup requests",
    ),
):
    """Run inference throughput benchmark."""
    console.print("[bold blue]Nexus-LLM Inference Benchmark[/bold blue]")
    console.print()

    # Create config
    inference_config = InferenceConfig(
        model_name_or_path=model,
        backend=InferenceBackend(backend),
        max_tokens=max_tokens,
    )

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Backend: {backend}")
    console.print(f"  Requests: {requests}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Max tokens: {max_tokens}")
    console.print()

    # Run benchmark
    asyncio.run(_run_benchmark(
        inference_config,
        requests,
        concurrency,
        max_tokens,
        prompt,
        warmup,
    ))


async def _run_benchmark(
    config: InferenceConfig,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    prompt: str,
    warmup: int,
) -> None:
    """Run the async benchmark."""
    # Initialize engine
    console.print("[bold]Initializing inference engine...[/bold]")

    if config.backend == InferenceBackend.VLLM:
        engine = VLLMInferenceEngine(config)
    else:
        engine = TransformersInferenceEngine(config)

    await engine.initialize()
    console.print(f"  Engine ready: {engine.backend_name}")
    console.print()

    # Create benchmark
    benchmark = ThroughputBenchmark(engine)

    # Warmup
    if warmup > 0:
        console.print(f"[bold]Running {warmup} warmup requests...[/bold]")
        await benchmark.run_warmup(warmup)
        console.print()

    # Run benchmark
    console.print(f"[bold]Running {num_requests} benchmark requests...[/bold]")

    benchmark_config = BenchmarkConfig(
        num_requests=num_requests,
        prompt=prompt,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    result = await benchmark.run(benchmark_config)

    # Display results
    console.print()
    console.print("[bold green]Benchmark Complete![/bold green]")
    console.print()

    # Create results table
    table = Table(title="Latency Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Mean Latency", f"{result.latency.mean_ms:.2f} ms")
    table.add_row("P50 Latency", f"{result.latency.p50_ms:.2f} ms")
    table.add_row("P90 Latency", f"{result.latency.p90_ms:.2f} ms")
    table.add_row("P95 Latency", f"{result.latency.p95_ms:.2f} ms")
    table.add_row("P99 Latency", f"{result.latency.p99_ms:.2f} ms")
    table.add_row("Min Latency", f"{result.latency.min_ms:.2f} ms")
    table.add_row("Max Latency", f"{result.latency.max_ms:.2f} ms")

    console.print(table)
    console.print()

    # Throughput table
    throughput_table = Table(title="Throughput Statistics")
    throughput_table.add_column("Metric", style="cyan")
    throughput_table.add_column("Value", style="green")

    throughput_table.add_row("Requests/sec", f"{result.latency.requests_per_second:.2f}")
    throughput_table.add_row("Tokens/sec", f"{result.latency.tokens_per_second:.2f}")
    throughput_table.add_row("Total Requests", str(result.latency.total_requests))
    throughput_table.add_row("Total Tokens", str(result.latency.total_tokens))
    throughput_table.add_row("Errors", str(result.errors))
    throughput_table.add_row("Success Rate", f"{result.success_rate:.1%}")

    console.print(throughput_table)

    # Cleanup
    await engine.shutdown()


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
