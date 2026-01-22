#!/usr/bin/env python3
"""
CLI script for training reward models.

Usage:
    python scripts/train_reward_model.py --config configs/training/reward_model.yaml
    python scripts/train_reward_model.py --base-model facebook/opt-1.3b --data-path data/train.jsonl
"""

import argparse
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from nexus.core.config import NexusConfig, TrainingConfig
from nexus.training.reward_model.trainer import RewardModelTrainer
from nexus.training.data.dataset import RLHFDataset, DatasetFormat
from nexus.registry import ModelRegistry, ModelType


console = Console()
app = typer.Typer(help="Train RLHF reward models")


@app.command()
def train(
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training configuration YAML",
    ),
    base_model: str = typer.Option(
        "facebook/opt-1.3b",
        "--base-model",
        "-m",
        help="Base model name or path",
    ),
    data_path: Path = typer.Option(
        ...,
        "--data-path",
        "-d",
        help="Path to training data (JSONL)",
    ),
    eval_data_path: Path = typer.Option(
        None,
        "--eval-data-path",
        help="Path to evaluation data",
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    use_lora: bool = typer.Option(True, "--lora/--no-lora", help="Use LoRA"),
    lora_rank: int = typer.Option(64, "--lora-rank", help="LoRA rank"),
    register: bool = typer.Option(
        False,
        "--register",
        help="Register trained model in registry",
    ),
    model_id: str = typer.Option(
        None,
        "--model-id",
        help="Model ID for registration",
    ),
):
    """Train a reward model on RLHF preference data."""
    console.print("[bold blue]Nexus-LLM Reward Model Training[/bold blue]")
    console.print()

    # Load or create config
    if config_path and config_path.exists():
        console.print(f"Loading config from {config_path}")
        nexus_config = NexusConfig.from_yaml(config_path)
        training_config = nexus_config.training
    else:
        training_config = TrainingConfig(
            base_model=base_model,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_lora=use_lora,
        )
        training_config.lora.rank = lora_rank

    # Print configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Base model: {training_config.base_model}")
    console.print(f"  LoRA: {training_config.use_lora} (rank={training_config.lora.rank})")
    console.print(f"  Epochs: {training_config.num_epochs}")
    console.print(f"  Batch size: {training_config.batch_size}")
    console.print(f"  Learning rate: {training_config.learning_rate}")
    console.print()

    # Load datasets
    console.print(f"[bold]Loading data from {data_path}[/bold]")

    train_dataset = RLHFDataset(
        data_path,
        format_type=DatasetFormat.ANTHROPIC,
        single_turn_only=True,
    )
    console.print(f"  Training samples: {len(train_dataset)}")

    eval_dataset = None
    if eval_data_path and eval_data_path.exists():
        eval_dataset = RLHFDataset(
            eval_data_path,
            format_type=DatasetFormat.ANTHROPIC,
            single_turn_only=True,
        )
        console.print(f"  Evaluation samples: {len(eval_dataset)}")

    console.print()

    # Create trainer and train
    console.print("[bold]Starting training...[/bold]")

    trainer = RewardModelTrainer(training_config)
    result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
    )

    # Print results
    console.print()
    console.print("[bold green]Training Complete![/bold green]")
    console.print(f"  Final loss: {result.final_loss:.4f}")
    console.print(f"  Best loss: {result.best_loss:.4f}")
    console.print(f"  Total steps: {result.total_steps}")
    console.print(f"  Training time: {result.training_time_seconds:.1f}s")
    console.print(f"  Model saved to: {result.model_path}")

    # Register if requested
    if register:
        model_id = model_id or f"reward-model-{base_model.split('/')[-1]}"
        console.print()
        console.print(f"[bold]Registering model as '{model_id}'[/bold]")

        registry = ModelRegistry()
        version = registry.register_model(
            model_id=model_id,
            model_type=ModelType.REWARD_MODEL,
            artifact_path=result.model_path,
            base_model=base_model,
            training_config=training_config.model_dump(),
            training_metrics=result.metrics,
            description=f"Reward model fine-tuned from {base_model}",
        )

        console.print(f"  Registered version: {version.version}")


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., help="Path to model checkpoint"),
    data_path: Path = typer.Argument(..., help="Path to evaluation data"),
):
    """Evaluate a trained reward model."""
    console.print("[bold blue]Evaluating Reward Model[/bold blue]")

    # Load config from checkpoint
    training_config = TrainingConfig()

    # Load dataset
    eval_dataset = RLHFDataset(
        data_path,
        format_type=DatasetFormat.ANTHROPIC,
        single_turn_only=True,
    )
    console.print(f"Evaluation samples: {len(eval_dataset)}")

    # Create trainer and evaluate
    trainer = RewardModelTrainer(training_config)
    trainer.load_checkpoint(checkpoint_path)

    metrics = trainer.evaluate(eval_dataset)

    console.print()
    console.print("[bold]Evaluation Results:[/bold]")
    for name, value in metrics.items():
        console.print(f"  {name}: {value:.4f}")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
