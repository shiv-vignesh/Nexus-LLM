"""
Data collator for RLHF reward model training.

Handles tokenization, padding, and masking for preference pair batches.
The collator implements the dialogue-aware masking strategy where context
tokens are masked (IGNORE_INDEX) to focus the loss on response tokens.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from nexus.training.data.dataset import PreferencePair
from nexus.training.data.preprocessing import DialogueParser

# Index used to ignore tokens in loss computation
IGNORE_INDEX = -100


@dataclass
class CollatorConfig:
    """Configuration for the RLHF data collator."""
    max_context_length: int = 256
    max_response_length: int = 256
    padding_side: str = "right"
    mask_context: bool = True
    truncation: bool = True


class RLHFDataCollator:
    """Data collator for RLHF reward model training.

    Tokenizes preference pairs and creates batched tensors with proper
    attention masks and labels. Implements dialogue-aware masking where
    context tokens receive IGNORE_INDEX labels.

    Key design decisions:
    1. Context/response split happens at collation time for efficiency
    2. Left-padding for inference, right-padding for training
    3. Labels are masked for context tokens to focus loss on responses
    4. Returns both chosen and rejected in a single batch

    Attributes:
        tokenizer: HuggingFace tokenizer
        config: Collator configuration
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        config: CollatorConfig | None = None,
    ):
        """Initialize the collator.

        Args:
            tokenizer: Tokenizer to use for encoding
            config: Collator configuration
        """
        self.tokenizer = tokenizer
        self.config = config or CollatorConfig()

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store original padding side for restoration
        self._original_padding_side = tokenizer.padding_side

    def __call__(self, batch: list[PreferencePair]) -> dict[str, torch.Tensor]:
        """Collate a batch of preference pairs.

        Args:
            batch: List of PreferencePair objects

        Returns:
            Dictionary with:
                - chosen_input_ids: [batch_size, seq_len]
                - chosen_attention_mask: [batch_size, seq_len]
                - chosen_labels: [batch_size, seq_len]
                - rejected_input_ids: [batch_size, seq_len]
                - rejected_attention_mask: [batch_size, seq_len]
                - rejected_labels: [batch_size, seq_len]
        """
        # Set padding side for this batch
        self.tokenizer.padding_side = self.config.padding_side

        try:
            # Process chosen and rejected separately
            chosen_batch = self._process_batch([p.chosen for p in batch])
            rejected_batch = self._process_batch([p.rejected for p in batch])

            return {
                "chosen_input_ids": chosen_batch["input_ids"],
                "chosen_attention_mask": chosen_batch["attention_mask"],
                "chosen_labels": chosen_batch["labels"],
                "rejected_input_ids": rejected_batch["input_ids"],
                "rejected_attention_mask": rejected_batch["attention_mask"],
                "rejected_labels": rejected_batch["labels"],
            }
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = self._original_padding_side

    def _process_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Process a batch of dialogue texts.

        Args:
            texts: List of dialogue texts

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for text in texts:
            result = self._process_single(text)
            all_input_ids.append(result["input_ids"])
            all_attention_masks.append(result["attention_mask"])
            all_labels.append(result["labels"])

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for input_ids, attn_mask, labels in zip(all_input_ids, all_attention_masks, all_labels):
            pad_len = max_len - len(input_ids)

            if self.config.padding_side == "right":
                padded_input_ids.append(input_ids + [pad_id] * pad_len)
                padded_attention_masks.append(attn_mask + [0] * pad_len)
                padded_labels.append(labels + [IGNORE_INDEX] * pad_len)
            else:  # left padding
                padded_input_ids.append([pad_id] * pad_len + input_ids)
                padded_attention_masks.append([0] * pad_len + attn_mask)
                padded_labels.append([IGNORE_INDEX] * pad_len + labels)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def _process_single(self, text: str) -> dict[str, list[int]]:
        """Process a single dialogue text.

        Parses the dialogue, tokenizes context and response separately,
        and creates labels with context tokens masked.

        Args:
            text: Full dialogue text

        Returns:
            Dictionary with input_ids, attention_mask, labels as lists
        """
        # Parse dialogue to get context and response
        parsed = DialogueParser.parse_and_split(text, "anthropic")

        # Tokenize context and response separately
        context_tokens = self.tokenizer.encode(
            parsed.context,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_context_length,
        )

        response_tokens = self.tokenizer.encode(
            parsed.response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_response_length,
        )

        # Combine tokens
        input_ids = context_tokens + response_tokens
        attention_mask = [1] * len(input_ids)

        # Create labels with context masked
        if self.config.mask_context:
            labels = [IGNORE_INDEX] * len(context_tokens) + response_tokens
        else:
            labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InferenceCollator:
    """Simplified collator for inference (no labels, left padding)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Collate texts for inference.

        Args:
            texts: List of prompts

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Use left padding for inference
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return encoded
        finally:
            self.tokenizer.padding_side = original_side
