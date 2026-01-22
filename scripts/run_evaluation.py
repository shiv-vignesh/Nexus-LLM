#!/usr/bin/env python3
"""
CLI script for evaluating models.

Usage:
    python scripts/run_evaluation.py --model-path ./output/final_model --data-path data/eval.jsonl
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nexus.core.config import TrainingConfig
from nexus.training.reward_model.trainer import RewardModelTrainer
from nexus.training.data.dataset import RLHFDataset, DatasetFormat
from nexus.evaluation.metrics.reward import compute_reward_metrics


console = Console()
app = typer.Typer(help="Evaluate trained models")


@app.command()
def reward_model(
    model_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    data_path: Path = typer.Argument(..., help="Path to evaluation data"),
    batch_size: int = typer.Option(8, "--batch-size", "-b"),
    single_turn: bool = typer.Option(True, "--single-turn/--multi-turn"),
):
    """Evaluate a reward model on preference data."""
    console.print("[bold blue]Nexus-LLM Model Evaluation[/bold blue]")
    console.print()

    console.print(f"[bold]Loading model from {model_path}[/bold]")

    # Create trainer with default config
    config = TrainingConfig()
    trainer = RewardModelTrainer(config)
    trainer.load_checkpoint(model_path)

    console.print(f"[bold]Loading evaluation data from {data_path}[/bold]")

    eval_dataset = RLHFDataset(
        data_path,
        format_type=DatasetFormat.ANTHROPIC,
        single_turn_only=single_turn,
    )
    console.print(f"  Samples: {len(eval_dataset)}")
    console.print()

    # Run evaluation
    console.print("[bold]Running evaluation...[/bold]")
    metrics = trainer.evaluate(eval_dataset)

    # Display results
    console.print()
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for name, value in metrics.items():
        table.add_row(name, f"{value:.4f}")

    console.print(table)


@app.command()
def generation(
    model_path: Path = typer.Argument(..., help="Path to model"),
    references_path: Path = typer.Argument(..., help="Path to reference texts"),
    hypotheses_path: Path = typer.Argument(..., help="Path to generated texts"),
):
    """Evaluate generation quality with BLEU and ROUGE."""
    from nexus.evaluation.metrics.generation import compute_batch_metrics

    console.print("[bold blue]Generation Quality Evaluation[/bold blue]")
    console.print()

    # Load texts
    with open(references_path) as f:
        references = [line.strip() for line in f if line.strip()]

    with open(hypotheses_path) as f:
        hypotheses = [line.strip() for line in f if line.strip()]

    console.print(f"References: {len(references)}")
    console.print(f"Hypotheses: {len(hypotheses)}")
    console.print()

    # Compute metrics
    console.print("[bold]Computing metrics...[/bold]")
    metrics = compute_batch_metrics(references, hypotheses)

    # Display results
    console.print()
    table = Table(title="Generation Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for name, value in metrics.to_dict().items():
        table.add_row(name, f"{value:.4f}")

    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
