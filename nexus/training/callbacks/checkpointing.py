"""
Training callbacks for checkpointing, early stopping, and logging.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CallbackContext:
    """Context passed to callbacks."""
    step: int
    epoch: int
    loss: float
    metrics: dict[str, float]
    model: Any
    optimizer: Any
    output_dir: Path


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_step_end(self, context: CallbackContext) -> bool:
        """Called at the end of each training step.

        Returns:
            True to continue training, False to stop
        """
        ...

    def on_epoch_end(self, context: CallbackContext) -> bool:
        """Called at the end of each epoch."""
        return True

    def on_train_end(self, context: CallbackContext) -> None:
        """Called when training finishes."""
        pass


class CheckpointCallback(TrainingCallback):
    """Save checkpoints at regular intervals."""

    def __init__(
        self,
        save_steps: int = 500,
        save_best_only: bool = False,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        """Initialize checkpoint callback.

        Args:
            save_steps: Save every N steps
            save_best_only: Only save when metric improves
            metric_name: Metric to monitor
            mode: "min" or "max" for metric comparison
        """
        self.save_steps = save_steps
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_step_end(self, context: CallbackContext) -> bool:
        if context.step % self.save_steps != 0:
            return True

        current_value = context.metrics.get(self.metric_name, context.loss)
        should_save = not self.save_best_only

        if self.save_best_only:
            if self.mode == "min" and current_value < self.best_value:
                self.best_value = current_value
                should_save = True
            elif self.mode == "max" and current_value > self.best_value:
                self.best_value = current_value
                should_save = True

        if should_save:
            checkpoint_dir = context.output_dir / f"checkpoint-{context.step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Actual saving handled by trainer
            print(f"Checkpoint saved at step {context.step}")

        return True


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when metric stops improving."""

    def __init__(
        self,
        patience: int = 5,
        metric_name: str = "eval_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of checks without improvement before stopping
            metric_name: Metric to monitor
            mode: "min" or "max"
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta

        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_step_end(self, context: CallbackContext) -> bool:
        if self.metric_name not in context.metrics:
            return True

        current_value = context.metrics[self.metric_name]

        improved = False
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} checks without improvement")
            return False

        return True


class MetricsLoggerCallback(TrainingCallback):
    """Log metrics to file."""

    def __init__(
        self,
        log_file: str | Path = "metrics.jsonl",
        log_steps: int = 10,
    ):
        """Initialize metrics logger.

        Args:
            log_file: Path to log file
            log_steps: Log every N steps
        """
        self.log_file = Path(log_file)
        self.log_steps = log_steps
        self._file_handle = None

    def on_step_end(self, context: CallbackContext) -> bool:
        if context.step % self.log_steps != 0:
            return True

        log_entry = {
            "step": context.step,
            "epoch": context.epoch,
            "loss": context.loss,
            **context.metrics,
        }

        # Append to log file
        log_path = context.output_dir / self.log_file
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return True

    def on_train_end(self, context: CallbackContext) -> None:
        print(f"Training metrics saved to {context.output_dir / self.log_file}")
