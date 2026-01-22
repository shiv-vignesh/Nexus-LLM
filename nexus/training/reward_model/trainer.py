"""
Reward model trainer implementation.

Provides a complete training loop for RLHF reward models with:
- LoRA fine-tuning support
- Pairwise preference learning
- Comprehensive logging and checkpointing
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, PeftModel

from nexus.core.config import TrainingConfig
from nexus.core.interfaces.trainer import ITrainer, TrainingResult
from nexus.core.exceptions import TrainingError, CheckpointError
from nexus.training.lora.config import create_lora_config, LoRAConfigFactory
from nexus.training.reward_model.value_head import ValueHeadWrapper
from nexus.training.reward_model.loss import RewardModelLoss, LossOutput
from nexus.training.data.collator import RLHFDataCollator, CollatorConfig


@dataclass
class TrainingState:
    """Mutable training state."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    best_accuracy: float = 0.0
    training_loss: float = 0.0
    logs: list[dict] = field(default_factory=list)


class RewardModelTrainer(ITrainer):
    """Trainer for RLHF reward models.

    Implements the ITrainer interface for reward model training with:
    - Automatic LoRA configuration
    - Pairwise preference loss
    - Gradient accumulation
    - Mixed precision training
    - Comprehensive logging
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            model: Optional pretrained model (loaded if not provided)
            tokenizer: Optional tokenizer (loaded if not provided)
        """
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._wrapped_model: ValueHeadWrapper | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._scaler: torch.amp.GradScaler | None = None
        self._state = TrainingState()
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize model, tokenizer, and training components."""
        if self._initialized:
            return

        # Load tokenizer
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base model
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32,
                trust_remote_code=True,
            )

        # Apply LoRA if configured
        if self.config.use_lora:
            model_type = self._infer_model_type()
            lora_config = LoRAConfigFactory.from_settings(
                self.config.lora,
                model_type=model_type,
            )
            self._model = get_peft_model(self._model, lora_config)
            self._log_trainable_params()

        # Wrap with value head
        self._wrapped_model = ValueHeadWrapper(self._model)

        # Move to device
        device = self._resolve_device()
        self._wrapped_model.to(device)

        # Initialize mixed precision
        if self.config.mixed_precision != "no" and device.type == "cuda":
            self._scaler = torch.amp.GradScaler()

        self._initialized = True

    def _infer_model_type(self) -> str:
        """Infer model type from model name."""
        name = self.config.base_model.lower()
        for model_type in ["llama", "mistral", "opt", "gpt2", "falcon", "phi", "qwen", "gemma"]:
            if model_type in name:
                return model_type
        return "opt"  # Default fallback

    def _resolve_device(self) -> torch.device:
        """Resolve the training device."""
        if self.config.device.value == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device.value)

    def _log_trainable_params(self) -> None:
        """Log trainable parameter statistics."""
        if isinstance(self._model, PeftModel):
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self._model.parameters())
            ratio = trainable / total * 100
            print(f"Trainable parameters: {trainable:,} / {total:,} ({ratio:.2f}%)")

    def _setup_optimizer(self, num_training_steps: int) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self._optimizer = AdamW(
            self._wrapped_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler with warmup
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        warmup_scheduler = LinearLR(
            self._optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        main_scheduler = CosineAnnealingLR(
            self._optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str | Path | None = None,
    ) -> TrainingResult:
        """Execute reward model training.

        Args:
            train_dataset: Training dataset with PreferencePair items
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for checkpoints and final model

        Returns:
            TrainingResult with metrics
        """
        self._initialize()
        start_time = time.time()

        output_dir = Path(output_dir) if output_dir else Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup data loader
        collator = RLHFDataCollator(
            self._tokenizer,
            CollatorConfig(
                max_context_length=self.config.max_context_length,
                max_response_length=self.config.max_response_length,
            ),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                collate_fn=collator,
            )

        # Calculate training steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.num_epochs

        # Setup optimizer
        self._setup_optimizer(total_steps)

        # Loss function
        loss_fn = RewardModelLoss(
            preference_weight=self.config.preference_loss_weight,
            lm_weight=self.config.lm_loss_weight,
        )

        # Training loop
        self._wrapped_model.train()
        device = self._wrapped_model.device

        for epoch in range(self.config.num_epochs):
            self._state.epoch = epoch
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(progress):
                loss_output = self._training_step(batch, loss_fn, device)

                epoch_loss += loss_output.total_loss.item()
                epoch_accuracy += loss_output.accuracy.item()
                num_batches += 1

                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self._state.global_step += 1

                    # Logging
                    if self._state.global_step % self.config.logging_steps == 0:
                        self._log_step(loss_output, progress)

                    # Evaluation
                    if eval_loader and self._state.global_step % self.config.eval_steps == 0:
                        eval_metrics = self._evaluate(eval_loader, loss_fn, device)
                        self._state.logs.append({
                            "step": self._state.global_step,
                            "eval": eval_metrics,
                        })

                    # Save checkpoint
                    if self._state.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(output_dir / f"checkpoint-{self._state.global_step}")

            # End of epoch
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches

            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

            if avg_loss < self._state.best_loss:
                self._state.best_loss = avg_loss
                self.save_checkpoint(output_dir / "best_model")

        # Save final model
        self.save_checkpoint(output_dir / "final_model")

        training_time = time.time() - start_time

        return TrainingResult(
            model_path=output_dir / "final_model",
            final_loss=avg_loss,
            best_loss=self._state.best_loss,
            total_steps=self._state.global_step,
            epochs_completed=self.config.num_epochs,
            metrics={
                "final_accuracy": avg_accuracy,
                "best_accuracy": self._state.best_accuracy,
            },
            training_time_seconds=training_time,
        )

    def _training_step(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn: RewardModelLoss,
        device: torch.device,
    ) -> LossOutput:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Determine autocast dtype
        autocast_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        use_autocast = self._scaler is not None

        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            # Forward pass for chosen
            chosen_outputs = self._wrapped_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                labels=batch["chosen_labels"],
                return_logits=self.config.lm_loss_weight > 0,
            )

            # Forward pass for rejected
            rejected_outputs = self._wrapped_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                labels=batch["rejected_labels"],
                return_logits=self.config.lm_loss_weight > 0,
            )

            # Compute loss
            loss_output = loss_fn(
                chosen_rewards=chosen_outputs["rewards"],
                rejected_rewards=rejected_outputs["rewards"],
                chosen_logits=chosen_outputs.get("logits"),
                chosen_labels=batch["chosen_labels"],
                rejected_logits=rejected_outputs.get("logits"),
                rejected_labels=batch["rejected_labels"],
            )

        # Backward pass
        scaled_loss = loss_output.total_loss / self.config.gradient_accumulation_steps

        if self._scaler:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss_output

    def _optimizer_step(self) -> None:
        """Execute optimizer step with gradient clipping."""
        if self._scaler:
            self._scaler.unscale_(self._optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self._wrapped_model.parameters(),
            self.config.max_grad_norm,
        )

        if self._scaler:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        self._scheduler.step()
        self._optimizer.zero_grad()

    def _log_step(self, loss_output: LossOutput, progress: tqdm) -> None:
        """Log training metrics."""
        lr = self._optimizer.param_groups[0]["lr"]
        progress.set_postfix({
            "loss": f"{loss_output.total_loss.item():.4f}",
            "acc": f"{loss_output.accuracy.item():.4f}",
            "lr": f"{lr:.2e}",
        })

        self._state.logs.append({
            "step": self._state.global_step,
            "loss": loss_output.total_loss.item(),
            "pref_loss": loss_output.preference_loss.item(),
            "accuracy": loss_output.accuracy.item(),
            "chosen_reward": loss_output.chosen_reward.item(),
            "rejected_reward": loss_output.rejected_reward.item(),
            "lr": lr,
        })

    def _evaluate(
        self,
        eval_loader: DataLoader,
        loss_fn: RewardModelLoss,
        device: torch.device,
    ) -> dict[str, float]:
        """Run evaluation."""
        self._wrapped_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                chosen_outputs = self._wrapped_model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                )
                rejected_outputs = self._wrapped_model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                )

                loss_output = loss_fn(
                    chosen_rewards=chosen_outputs["rewards"],
                    rejected_rewards=rejected_outputs["rewards"],
                )

                total_loss += loss_output.total_loss.item()
                total_accuracy += loss_output.accuracy.item()
                num_batches += 1

        self._wrapped_model.train()

        return {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
        }

    def evaluate(self, eval_dataset: Dataset) -> dict[str, float]:
        """Evaluate model on dataset."""
        self._initialize()

        collator = RLHFDataCollator(
            self._tokenizer,
            CollatorConfig(
                max_context_length=self.config.max_context_length,
                max_response_length=self.config.max_response_length,
            ),
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size * 2,
            collate_fn=collator,
        )

        loss_fn = RewardModelLoss()
        return self._evaluate(eval_loader, loss_fn, self._wrapped_model.device)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Save PEFT adapter if using LoRA
            if isinstance(self._model, PeftModel):
                self._model.save_pretrained(path / "adapter")

            # Save value head
            torch.save(
                self._wrapped_model.value_head.state_dict(),
                path / "value_head.pt",
            )

            # Save tokenizer
            self._tokenizer.save_pretrained(path / "tokenizer")

            # Save training state
            torch.save({
                "global_step": self._state.global_step,
                "epoch": self._state.epoch,
                "best_loss": self._state.best_loss,
                "config": self.config.model_dump(),
            }, path / "training_state.pt")

            print(f"Checkpoint saved to {path}")

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        path = Path(path)

        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}")

        try:
            self._initialize()

            # Load PEFT adapter
            if (path / "adapter").exists() and isinstance(self._model, PeftModel):
                self._model.load_adapter(path / "adapter", "default")

            # Load value head
            if (path / "value_head.pt").exists():
                self._wrapped_model.value_head.load_state_dict(
                    torch.load(path / "value_head.pt", map_location=self._wrapped_model.device)
                )

            # Load training state
            if (path / "training_state.pt").exists():
                state = torch.load(path / "training_state.pt")
                self._state.global_step = state.get("global_step", 0)
                self._state.epoch = state.get("epoch", 0)
                self._state.best_loss = state.get("best_loss", float("inf"))

            print(f"Checkpoint loaded from {path}")

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if trainer is initialized."""
        return self._initialized

    @property
    def model(self) -> ValueHeadWrapper | None:
        """Get the wrapped model."""
        return self._wrapped_model

    @property
    def tokenizer(self) -> PreTrainedTokenizer | None:
        """Get the tokenizer."""
        return self._tokenizer
