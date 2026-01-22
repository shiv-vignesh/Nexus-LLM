"""Trainer interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_path: Path
    final_loss: float
    best_loss: float
    total_steps: int
    epochs_completed: int
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_path": str(self.model_path),
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "total_steps": self.total_steps,
            "epochs_completed": self.epochs_completed,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
        }


class ITrainer(ABC):
    """Interface for model trainers.

    Implementations should handle:
    - Model initialization and configuration
    - Training loop execution
    - Checkpoint management
    - Metric logging
    """

    @abstractmethod
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str | Path | None = None,
    ) -> TrainingResult:
        """Execute training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Directory to save checkpoints and final model

        Returns:
            TrainingResult with training metrics and model path
        """
        ...

    @abstractmethod
    def evaluate(self, eval_dataset: Dataset) -> dict[str, float]:
        """Evaluate model on dataset.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Dictionary of metric names to values
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> None:
        """Save current model state.

        Args:
            path: Path to save checkpoint
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state from checkpoint.

        Args:
            path: Path to checkpoint
        """
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if trainer is initialized and ready."""
        ...
