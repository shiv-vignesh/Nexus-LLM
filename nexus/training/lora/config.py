"""
LoRA configuration factory for different model architectures.

Provides sensible defaults and target module mappings for common model families.
"""

from dataclasses import dataclass
from typing import ClassVar

from peft import LoraConfig, TaskType

from nexus.core.config import LoRAConfig as LoRASettings


@dataclass
class ModelLoRAMapping:
    """Mapping of target modules for different model architectures."""

    # Common attention projections across architectures
    ATTENTION_MODULES: ClassVar[dict[str, list[str]]] = {
        "opt": ["q_proj", "v_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "gpt2": ["c_attn", "c_proj"],
        "gpt_neox": ["query_key_value", "dense"],
        "falcon": ["query_key_value", "dense"],
        "phi": ["q_proj", "k_proj", "v_proj", "dense"],
        "qwen": ["c_attn", "c_proj"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

    # MLP modules for more aggressive fine-tuning
    MLP_MODULES: ClassVar[dict[str, list[str]]] = {
        "opt": ["fc1", "fc2"],
        "llama": ["gate_proj", "up_proj", "down_proj"],
        "mistral": ["gate_proj", "up_proj", "down_proj"],
        "gpt2": ["c_fc", "c_proj"],
        "gpt_neox": ["dense_h_to_4h", "dense_4h_to_h"],
        "falcon": ["dense_h_to_4h", "dense_4h_to_h"],
        "phi": ["fc1", "fc2"],
        "qwen": ["w1", "w2", "c_proj"],
        "gemma": ["gate_proj", "up_proj", "down_proj"],
    }

    @classmethod
    def get_target_modules(
        cls,
        model_type: str,
        include_mlp: bool = False
    ) -> list[str]:
        """Get target modules for a model architecture.

        Args:
            model_type: Model architecture name (lowercase)
            include_mlp: Whether to include MLP layers

        Returns:
            List of target module names
        """
        model_type = model_type.lower()

        # Try exact match first
        modules = cls.ATTENTION_MODULES.get(model_type, [])

        if include_mlp:
            modules = modules + cls.MLP_MODULES.get(model_type, [])

        # Fallback to common patterns
        if not modules:
            modules = ["q_proj", "v_proj"]

        return modules


class LoRAConfigFactory:
    """Factory for creating LoRA configurations with model-aware defaults."""

    # Recommended configurations for different scenarios
    PRESETS: ClassVar[dict[str, dict]] = {
        "efficient": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "include_mlp": False,
        },
        "balanced": {
            "rank": 64,
            "alpha": 128,
            "dropout": 0.05,
            "include_mlp": False,
        },
        "expressive": {
            "rank": 128,
            "alpha": 256,
            "dropout": 0.1,
            "include_mlp": True,
        },
        "reward_model": {
            "rank": 64,
            "alpha": 128,
            "dropout": 0.05,
            "include_mlp": False,
        },
    }

    @classmethod
    def from_preset(
        cls,
        preset: str,
        model_type: str,
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> LoraConfig:
        """Create LoRA config from a preset.

        Args:
            preset: Preset name (efficient, balanced, expressive, reward_model)
            model_type: Model architecture name
            task_type: PEFT task type

        Returns:
            Configured LoraConfig
        """
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")

        settings = cls.PRESETS[preset]

        target_modules = ModelLoRAMapping.get_target_modules(
            model_type,
            include_mlp=settings.get("include_mlp", False)
        )

        return LoraConfig(
            r=settings["rank"],
            lora_alpha=settings["alpha"],
            lora_dropout=settings["dropout"],
            target_modules=target_modules,
            bias="none",
            task_type=task_type,
        )

    @classmethod
    def from_settings(
        cls,
        settings: LoRASettings,
        model_type: str | None = None,
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> LoraConfig:
        """Create LoRA config from Nexus settings.

        Args:
            settings: LoRAConfig from Nexus configuration
            model_type: Optional model type to infer target modules
            task_type: PEFT task type

        Returns:
            Configured LoraConfig
        """
        target_modules = settings.target_modules

        # If target modules not specified, try to infer from model type
        if not target_modules and model_type:
            target_modules = ModelLoRAMapping.get_target_modules(model_type)

        return LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            target_modules=target_modules,
            bias=settings.bias,
            task_type=task_type,
        )


def create_lora_config(
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    model_type: str | None = None,
    include_mlp: bool = False,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Create a LoRA configuration.

    Convenience function for creating LoRA configs with sensible defaults.

    Args:
        rank: LoRA rank (r parameter)
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate for LoRA layers
        target_modules: Explicit list of target modules
        model_type: Model architecture to infer target modules
        include_mlp: Whether to include MLP layers in targets
        task_type: PEFT task type

    Returns:
        Configured LoraConfig

    Example:
        >>> config = create_lora_config(rank=32, model_type="llama")
        >>> print(config.target_modules)
        ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    """
    if target_modules is None:
        if model_type:
            target_modules = ModelLoRAMapping.get_target_modules(model_type, include_mlp)
        else:
            target_modules = ["q_proj", "v_proj"]  # Safe default

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )
