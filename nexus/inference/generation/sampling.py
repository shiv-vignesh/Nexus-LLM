"""
Sampling configuration for text generation.

Provides a unified configuration interface that works with both
vLLM and HuggingFace Transformers backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SamplingStrategy(str, Enum):
    """Common sampling strategies."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TOP_K_TOP_P = "top_k_top_p"


@dataclass
class SamplingConfig:
    """Unified sampling configuration.

    Works with both vLLM SamplingParams and HuggingFace GenerationConfig.

    Attributes:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        presence_penalty: Penalty for token presence
        frequency_penalty: Penalty for token frequency
        repetition_penalty: Penalty for repeating tokens
        stop_sequences: Sequences that stop generation
        n: Number of completions to generate
        best_of: Number of candidates for best_of sampling
        logprobs: Number of log probabilities to return
        seed: Random seed for reproducibility
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    n: int = 1
    best_of: int | None = None
    logprobs: int | None = None
    seed: int | None = None

    @classmethod
    def greedy(cls, max_tokens: int = 256) -> "SamplingConfig":
        """Create greedy decoding configuration."""
        return cls(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )

    @classmethod
    def creative(cls, max_tokens: int = 256) -> "SamplingConfig":
        """Create creative sampling configuration."""
        return cls(
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=50,
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )

    @classmethod
    def balanced(cls, max_tokens: int = 256) -> "SamplingConfig":
        """Create balanced sampling configuration."""
        return cls(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )

    def to_vllm_params(self) -> "SamplingParams":
        """Convert to vLLM SamplingParams.

        Returns:
            vLLM SamplingParams object

        Raises:
            ImportError: If vLLM is not installed
        """
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM is required for to_vllm_params()")

        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else -1,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop_sequences if self.stop_sequences else None,
            n=self.n,
            best_of=self.best_of,
            logprobs=self.logprobs,
            seed=self.seed,
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Convert to HuggingFace generation config dict.

        Returns:
            Dictionary compatible with model.generate()
        """
        config = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature if self.temperature > 0 else None,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
            "num_return_sequences": self.n,
        }

        # Remove None values
        return {k: v for k, v in config.items() if v is not None}

    def with_overrides(self, **kwargs) -> "SamplingConfig":
        """Create a new config with overrides.

        Args:
            **kwargs: Fields to override

        Returns:
            New SamplingConfig with overrides applied
        """
        current = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "stop_sequences": self.stop_sequences.copy(),
            "n": self.n,
            "best_of": self.best_of,
            "logprobs": self.logprobs,
            "seed": self.seed,
        }
        current.update(kwargs)
        return SamplingConfig(**current)

    @property
    def strategy(self) -> SamplingStrategy:
        """Infer the sampling strategy from parameters."""
        if self.temperature == 0:
            return SamplingStrategy.GREEDY
        if self.top_k > 0 and self.top_p < 1.0:
            return SamplingStrategy.TOP_K_TOP_P
        if self.top_k > 0:
            return SamplingStrategy.TOP_K
        if self.top_p < 1.0:
            return SamplingStrategy.TOP_P
        return SamplingStrategy.SAMPLING
