"""
Reward model evaluation metrics.
"""

from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class RewardMetrics:
    """Metrics for reward model evaluation."""

    accuracy: float = 0.0
    auc_roc: float = 0.0
    avg_chosen_reward: float = 0.0
    avg_rejected_reward: float = 0.0
    reward_margin: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "avg_chosen_reward": self.avg_chosen_reward,
            "avg_rejected_reward": self.avg_rejected_reward,
            "reward_margin": self.reward_margin,
        }


def compute_reward_metrics(
    chosen_rewards: torch.Tensor | list[float],
    rejected_rewards: torch.Tensor | list[float],
) -> RewardMetrics:
    """Compute reward model evaluation metrics.

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses

    Returns:
        RewardMetrics with all computed metrics
    """
    # Convert to numpy
    if isinstance(chosen_rewards, torch.Tensor):
        chosen_rewards = chosen_rewards.detach().cpu().numpy()
    if isinstance(rejected_rewards, torch.Tensor):
        rejected_rewards = rejected_rewards.detach().cpu().numpy()

    # Ensure same length
    n = min(len(chosen_rewards), len(rejected_rewards))
    chosen_rewards = chosen_rewards[:n]
    rejected_rewards = rejected_rewards[:n]

    # Accuracy: how often chosen > rejected
    predictions = (chosen_rewards > rejected_rewards).astype(int)
    labels = [1] * n  # All should be 1 (chosen > rejected)
    accuracy = accuracy_score(labels, predictions)

    # AUC-ROC
    # Combine rewards and create binary labels
    all_rewards = list(chosen_rewards) + list(rejected_rewards)
    all_labels = [1] * n + [0] * n  # 1 for chosen, 0 for rejected

    try:
        auc_roc = roc_auc_score(all_labels, all_rewards)
    except ValueError:
        auc_roc = 0.5  # Default if can't compute

    # Average rewards
    avg_chosen = float(chosen_rewards.mean())
    avg_rejected = float(rejected_rewards.mean())
    margin = avg_chosen - avg_rejected

    return RewardMetrics(
        accuracy=accuracy,
        auc_roc=auc_roc,
        avg_chosen_reward=avg_chosen,
        avg_rejected_reward=avg_rejected,
        reward_margin=margin,
    )


def compute_pairwise_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """Compute pairwise accuracy.

    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses

    Returns:
        Accuracy (proportion of pairs where chosen > rejected)
    """
    correct = (chosen_rewards > rejected_rewards).float()
    return correct.mean().item()
