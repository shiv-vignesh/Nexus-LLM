"""
Generation quality metrics: BLEU, ROUGE, perplexity.
"""

from dataclasses import dataclass, field
from typing import Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# Download NLTK data if not present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@dataclass
class GenerationMetrics:
    """Container for generation quality metrics."""

    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    rouge_1_f: float = 0.0
    rouge_1_p: float = 0.0
    rouge_1_r: float = 0.0
    rouge_2_f: float = 0.0
    rouge_l_f: float = 0.0
    rouge_l_p: float = 0.0
    rouge_l_r: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_3": self.bleu_3,
            "bleu_4": self.bleu_4,
            "rouge_1_f": self.rouge_1_f,
            "rouge_1_p": self.rouge_1_p,
            "rouge_1_r": self.rouge_1_r,
            "rouge_2_f": self.rouge_2_f,
            "rouge_l_f": self.rouge_l_f,
            "rouge_l_p": self.rouge_l_p,
            "rouge_l_r": self.rouge_l_r,
        }


def compute_bleu(
    reference: str,
    hypothesis: str,
    smoothing: bool = True,
) -> dict[str, float]:
    """Compute BLEU scores for a reference-hypothesis pair.

    Args:
        reference: Reference text
        hypothesis: Generated text
        smoothing: Use smoothing function

    Returns:
        Dictionary with BLEU-1 through BLEU-4 scores
    """
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    if not hyp_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Smoothing function to handle zero counts
    smooth_fn = SmoothingFunction().method1 if smoothing else None

    # Compute BLEU scores
    scores = {}

    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        try:
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smooth_fn,
            )
            scores[f"bleu_{n}"] = score
        except Exception:
            scores[f"bleu_{n}"] = 0.0

    return scores


def compute_rouge(
    reference: str,
    hypothesis: str,
    rouge_types: list[str] | None = None,
) -> dict[str, float]:
    """Compute ROUGE scores for a reference-hypothesis pair.

    Args:
        reference: Reference text
        hypothesis: Generated text
        rouge_types: ROUGE types to compute (default: rouge1, rouge2, rougeL)

    Returns:
        Dictionary with ROUGE scores
    """
    rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    result = {}
    for rouge_type, score in scores.items():
        prefix = rouge_type.replace("rouge", "rouge_").lower()
        result[f"{prefix}_f"] = score.fmeasure
        result[f"{prefix}_p"] = score.precision
        result[f"{prefix}_r"] = score.recall

    return result


def compute_generation_metrics(
    reference: str,
    hypothesis: str,
) -> GenerationMetrics:
    """Compute all generation metrics.

    Args:
        reference: Reference text
        hypothesis: Generated text

    Returns:
        GenerationMetrics with all scores
    """
    bleu_scores = compute_bleu(reference, hypothesis)
    rouge_scores = compute_rouge(reference, hypothesis)

    return GenerationMetrics(
        bleu_1=bleu_scores["bleu_1"],
        bleu_2=bleu_scores["bleu_2"],
        bleu_3=bleu_scores["bleu_3"],
        bleu_4=bleu_scores["bleu_4"],
        rouge_1_f=rouge_scores.get("rouge_1_f", 0.0),
        rouge_1_p=rouge_scores.get("rouge_1_p", 0.0),
        rouge_1_r=rouge_scores.get("rouge_1_r", 0.0),
        rouge_2_f=rouge_scores.get("rouge_2_f", 0.0),
        rouge_l_f=rouge_scores.get("rouge_l_f", 0.0),
        rouge_l_p=rouge_scores.get("rouge_l_p", 0.0),
        rouge_l_r=rouge_scores.get("rouge_l_r", 0.0),
    )


def compute_batch_metrics(
    references: list[str],
    hypotheses: list[str],
) -> GenerationMetrics:
    """Compute average metrics over a batch.

    Args:
        references: List of reference texts
        hypotheses: List of generated texts

    Returns:
        GenerationMetrics with averaged scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")

    if not references:
        return GenerationMetrics()

    # Compute metrics for each pair
    all_metrics = [
        compute_generation_metrics(ref, hyp)
        for ref, hyp in zip(references, hypotheses)
    ]

    # Average
    n = len(all_metrics)

    return GenerationMetrics(
        bleu_1=sum(m.bleu_1 for m in all_metrics) / n,
        bleu_2=sum(m.bleu_2 for m in all_metrics) / n,
        bleu_3=sum(m.bleu_3 for m in all_metrics) / n,
        bleu_4=sum(m.bleu_4 for m in all_metrics) / n,
        rouge_1_f=sum(m.rouge_1_f for m in all_metrics) / n,
        rouge_1_p=sum(m.rouge_1_p for m in all_metrics) / n,
        rouge_1_r=sum(m.rouge_1_r for m in all_metrics) / n,
        rouge_2_f=sum(m.rouge_2_f for m in all_metrics) / n,
        rouge_l_f=sum(m.rouge_l_f for m in all_metrics) / n,
        rouge_l_p=sum(m.rouge_l_p for m in all_metrics) / n,
        rouge_l_r=sum(m.rouge_l_r for m in all_metrics) / n,
    )
