"""
Loss functions for reward model training.

Implements the pairwise preference loss and combined training objectives
for RLHF reward model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple


class LossOutput(NamedTuple):
    """Output from loss computation."""
    total_loss: torch.Tensor
    preference_loss: torch.Tensor
    lm_loss: torch.Tensor | None
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    accuracy: torch.Tensor


class PairwisePreferenceLoss(nn.Module):
    """Pairwise preference loss for reward model training.

    Implements the Bradley-Terry model loss:
        L = -log(sigmoid(r_chosen - r_rejected))

    This loss maximizes the probability that the chosen response
    receives a higher reward than the rejected response.
    """

    def __init__(self, margin: float = 0.0):
        """Initialize the loss.

        Args:
            margin: Optional margin to add to the preference difference.
                   A positive margin makes the model more conservative,
                   requiring larger differences to be confident.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the pairwise preference loss.

        Args:
            chosen_rewards: Rewards for chosen responses [batch_size]
            rejected_rewards: Rewards for rejected responses [batch_size]

        Returns:
            Tuple of (loss, accuracy)
        """
        # Compute reward difference
        reward_diff = chosen_rewards - rejected_rewards

        # Apply margin if specified
        if self.margin > 0:
            reward_diff = reward_diff - self.margin

        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(reward_diff).mean()

        # Compute accuracy (how often chosen > rejected)
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, accuracy


class RewardModelLoss(nn.Module):
    """Combined loss for reward model training.

    Combines:
    1. Pairwise preference loss (primary objective)
    2. Language modeling loss (optional regularization)

    The LM loss acts as a regularizer to prevent the model from
    forgetting its language modeling capabilities during RM training.
    """

    def __init__(
        self,
        preference_weight: float = 1.0,
        lm_weight: float = 0.1,
        margin: float = 0.0,
        ignore_index: int = -100,
    ):
        """Initialize the combined loss.

        Args:
            preference_weight: Weight for preference loss
            lm_weight: Weight for language modeling loss
            margin: Margin for preference loss
            ignore_index: Index to ignore in LM loss
        """
        super().__init__()
        self.preference_weight = preference_weight
        self.lm_weight = lm_weight
        self.ignore_index = ignore_index

        self.preference_loss = PairwisePreferenceLoss(margin=margin)
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        chosen_logits: torch.Tensor | None = None,
        chosen_labels: torch.Tensor | None = None,
        rejected_logits: torch.Tensor | None = None,
        rejected_labels: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute the combined loss.

        Args:
            chosen_rewards: Rewards for chosen [batch_size]
            rejected_rewards: Rewards for rejected [batch_size]
            chosen_logits: Optional LM logits for chosen [batch, seq, vocab]
            chosen_labels: Optional labels for chosen [batch, seq]
            rejected_logits: Optional LM logits for rejected
            rejected_labels: Optional labels for rejected

        Returns:
            LossOutput with all loss components
        """
        # Compute preference loss
        pref_loss, accuracy = self.preference_loss(chosen_rewards, rejected_rewards)

        # Compute LM loss if logits provided
        lm_loss = None
        if self.lm_weight > 0 and chosen_logits is not None and chosen_labels is not None:
            # Reshape for cross entropy: [batch * seq, vocab] vs [batch * seq]
            vocab_size = chosen_logits.size(-1)

            # Chosen LM loss
            chosen_lm = self.lm_loss(
                chosen_logits[:, :-1, :].contiguous().view(-1, vocab_size),
                chosen_labels[:, 1:].contiguous().view(-1),
            )

            # Rejected LM loss (if provided)
            if rejected_logits is not None and rejected_labels is not None:
                rejected_lm = self.lm_loss(
                    rejected_logits[:, :-1, :].contiguous().view(-1, vocab_size),
                    rejected_labels[:, 1:].contiguous().view(-1),
                )
                lm_loss = (chosen_lm + rejected_lm) / 2
            else:
                lm_loss = chosen_lm

        # Compute total loss
        total_loss = self.preference_weight * pref_loss
        if lm_loss is not None:
            total_loss = total_loss + self.lm_weight * lm_loss

        return LossOutput(
            total_loss=total_loss,
            preference_loss=pref_loss,
            lm_loss=lm_loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            accuracy=accuracy,
        )


class MarginRankingRewardLoss(nn.Module):
    """Alternative margin-based ranking loss.

    Uses PyTorch's MarginRankingLoss:
        L = max(0, -y * (x1 - x2) + margin)

    Where y=1 to rank chosen > rejected.
    """

    def __init__(self, margin: float = 1.0):
        """Initialize margin ranking loss.

        Args:
            margin: Margin for ranking loss
        """
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute margin ranking loss.

        Args:
            chosen_rewards: Rewards for chosen [batch_size]
            rejected_rewards: Rewards for rejected [batch_size]

        Returns:
            Tuple of (loss, accuracy)
        """
        # Target: chosen should be ranked higher (y=1)
        target = torch.ones_like(chosen_rewards)

        loss = self.loss_fn(chosen_rewards, rejected_rewards, target)

        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, accuracy
