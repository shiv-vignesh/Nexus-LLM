"""
Value head wrapper for reward model training.

Wraps a causal language model with a value head for reward prediction.
"""

import torch
import torch.nn as nn
from typing import Any

from transformers import PreTrainedModel


class ValueHead(nn.Module):
    """A value head that outputs a scalar reward.

    Takes the hidden states from a transformer and produces
    a single scalar value (reward) per sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        """Initialize the value head.

        Args:
            hidden_size: Size of input hidden states
            dropout: Dropout probability
            layer_norm: Whether to apply layer normalization
        """
        super().__init__()

        layers = []

        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))

        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        ])

        self.head = nn.Sequential(*layers)

        # Initialize with small weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute reward from hidden states.

        Args:
            hidden_states: Last hidden states [batch, seq, hidden]

        Returns:
            Rewards [batch, seq, 1]
        """
        return self.head(hidden_states)


class ValueHeadWrapper(nn.Module):
    """Wrapper that adds a value head to a pretrained model.

    This wrapper:
    1. Runs the base model to get hidden states
    2. Applies a value head to compute rewards
    3. Extracts the reward at the last non-padded position

    Attributes:
        base_model: The pretrained transformer model
        value_head: The value head module
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_size: int | None = None,
        dropout: float = 0.1,
    ):
        """Initialize the wrapper.

        Args:
            base_model: Pretrained model to wrap
            hidden_size: Hidden size (inferred from model if not provided)
            dropout: Dropout for value head
        """
        super().__init__()
        self.base_model = base_model

        # Infer hidden size from model config
        if hidden_size is None:
            config = base_model.config
            hidden_size = getattr(
                config,
                "hidden_size",
                getattr(config, "n_embd", getattr(config, "d_model", 768))
            )

        self.value_head = ValueHead(hidden_size, dropout=dropout)
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_logits: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through model and value head.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq]
            labels: Optional labels for LM loss [batch, seq]
            return_logits: Whether to return LM logits
            **kwargs: Additional arguments for base model

        Returns:
            Dictionary with:
                - rewards: Scalar rewards [batch]
                - values: Per-token values [batch, seq]
                - logits: LM logits (if return_logits=True)
                - lm_loss: LM loss (if labels provided)
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]

        # Compute values for all positions
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq]

        # Extract reward at last non-padded position
        if attention_mask is not None:
            # Find last non-padded position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            rewards = values[batch_indices, seq_lengths]
        else:
            # No padding, use last position
            rewards = values[:, -1]

        result = {
            "rewards": rewards,
            "values": values,
        }

        # Include logits if requested
        if return_logits and hasattr(outputs, "logits"):
            result["logits"] = outputs.logits

        # Compute LM loss if labels provided
        if labels is not None and hasattr(outputs, "logits"):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            logits = outputs.logits

            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            result["lm_loss"] = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return result

    def get_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get rewards for inputs (inference mode).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Rewards tensor [batch]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs["rewards"]

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on base model."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.parameters()).dtype
