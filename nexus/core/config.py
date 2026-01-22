"""
Configuration management for Nexus-LLM using Pydantic Settings.

Supports loading from environment variables, .env files, and YAML configs.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    """Supported device types for training and inference."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class InferenceBackend(str, Enum):
    """Supported inference backends."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class LoRAConfig(BaseSettings):
    """LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_LORA_")

    rank: int = Field(default=64, ge=1, le=1024, description="LoRA rank (r)")
    alpha: int = Field(default=128, ge=1, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate")
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj"],
        description="Target modules for LoRA adaptation"
    )
    bias: str = Field(default="none", description="Bias training strategy")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for PEFT")


class TrainingConfig(BaseSettings):
    """Configuration for model training."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_TRAINING_")

    # Model settings
    base_model: str = Field(
        default="facebook/opt-1.3b",
        description="Base model name or path"
    )
    use_lora: bool = Field(default=True, description="Whether to use LoRA fine-tuning")
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)

    # Sequence lengths
    max_context_length: int = Field(default=256, ge=1)
    max_response_length: int = Field(default=256, ge=1)

    # Reward model specific
    preference_loss_weight: float = Field(default=1.0, ge=0.0)
    lm_loss_weight: float = Field(default=0.1, ge=0.0)

    # Checkpointing
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)

    # Device and precision
    device: DeviceType = Field(default=DeviceType.AUTO)
    mixed_precision: str = Field(default="fp16", pattern="^(no|fp16|bf16)$")

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str | DeviceType) -> DeviceType:
        if isinstance(v, str):
            return DeviceType(v.lower())
        return v


class InferenceConfig(BaseSettings):
    """Configuration for model inference."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_INFERENCE_")

    # Backend selection
    backend: InferenceBackend = Field(
        default=InferenceBackend.VLLM,
        description="Inference backend to use"
    )

    # Model settings
    model_name_or_path: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Model to load for inference"
    )

    # vLLM specific settings
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, gt=0, le=1.0)
    max_model_len: int = Field(default=2048, ge=1)
    enable_prefix_caching: bool = Field(default=True)
    enforce_eager: bool = Field(default=False)

    # Generation defaults
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    # Batching
    max_batch_size: int = Field(default=32, ge=1)
    batch_timeout_ms: int = Field(default=50, ge=1)


class RedisConfig(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: str | None = Field(default=None)

    # Stream settings
    stream_name: str = Field(default="nexus:inference:requests")
    consumer_group: str = Field(default="nexus-workers")
    result_ttl_seconds: int = Field(default=3600)

    @property
    def url(self) -> str:
        """Construct Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class ServingConfig(BaseSettings):
    """Configuration for the FastAPI serving layer."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_SERVING_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    # API settings
    api_prefix: str = Field(default="/v1")
    enable_docs: bool = Field(default=True)
    cors_origins: list[str] = Field(default=["*"])

    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    # Request handling
    request_timeout_seconds: int = Field(default=300, ge=1)
    max_concurrent_requests: int = Field(default=1000, ge=1)

    # Redis and inference configs
    redis: RedisConfig = Field(default_factory=RedisConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class RegistryConfig(BaseSettings):
    """Configuration for model registry."""

    model_config = SettingsConfigDict(env_prefix="NEXUS_REGISTRY_")

    storage_path: Path = Field(
        default=Path("./model_registry"),
        description="Local path for model storage"
    )
    metadata_backend: str = Field(
        default="local",
        description="Metadata storage backend (local, redis)"
    )


class NexusConfig(BaseSettings):
    """Root configuration for the Nexus-LLM platform."""

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    # Global settings
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    seed: int = Field(default=42)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NexusConfig":
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
