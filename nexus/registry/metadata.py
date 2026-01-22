"""
Model artifact and metadata schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(str, Enum):
    """Types of model artifacts."""
    FULL_MODEL = "full_model"
    LORA_ADAPTER = "lora_adapter"
    VALUE_HEAD = "value_head"
    TOKENIZER = "tokenizer"
    CONFIG = "config"


@dataclass
class ModelArtifact:
    """Represents a model artifact file or directory."""

    artifact_type: ArtifactType
    path: Path
    size_bytes: int = 0
    checksum: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_type": self.artifact_type.value,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        """Deserialize from dictionary."""
        return cls(
            artifact_type=ArtifactType(data["artifact_type"]),
            path=Path(data["path"]),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrainingRun:
    """Information about a training run that produced a model."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    config: dict = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)

    # Hardware info
    device: str = ""
    gpu_memory_gb: float = 0.0

    # Data info
    train_samples: int = 0
    eval_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "device": self.device,
            "gpu_memory_gb": self.gpu_memory_gb,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRun":
        """Deserialize from dictionary."""
        return cls(
            run_id=data["run_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            hyperparameters=data.get("hyperparameters", {}),
            device=data.get("device", ""),
            gpu_memory_gb=data.get("gpu_memory_gb", 0.0),
            train_samples=data.get("train_samples", 0),
            eval_samples=data.get("eval_samples", 0),
        )


@dataclass
class ModelCard:
    """Model card with documentation and usage info."""

    model_id: str
    model_name: str
    description: str = ""
    use_cases: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    training_data: str = ""
    evaluation_results: dict = field(default_factory=dict)
    example_usage: str = ""
    license: str = "MIT"
    authors: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown model card."""
        sections = [
            f"# {self.model_name}",
            "",
            f"**Model ID:** {self.model_id}",
            "",
            "## Description",
            self.description or "No description provided.",
            "",
        ]

        if self.use_cases:
            sections.extend([
                "## Use Cases",
                *[f"- {use_case}" for use_case in self.use_cases],
                "",
            ])

        if self.limitations:
            sections.extend([
                "## Limitations",
                *[f"- {limitation}" for limitation in self.limitations],
                "",
            ])

        if self.training_data:
            sections.extend([
                "## Training Data",
                self.training_data,
                "",
            ])

        if self.evaluation_results:
            sections.extend([
                "## Evaluation Results",
                "| Metric | Value |",
                "|--------|-------|",
                *[f"| {k} | {v} |" for k, v in self.evaluation_results.items()],
                "",
            ])

        if self.example_usage:
            sections.extend([
                "## Example Usage",
                "```python",
                self.example_usage,
                "```",
                "",
            ])

        sections.extend([
            "## License",
            self.license,
            "",
        ])

        if self.authors:
            sections.extend([
                "## Authors",
                *[f"- {author}" for author in self.authors],
            ])

        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "description": self.description,
            "use_cases": self.use_cases,
            "limitations": self.limitations,
            "training_data": self.training_data,
            "evaluation_results": self.evaluation_results,
            "example_usage": self.example_usage,
            "license": self.license,
            "authors": self.authors,
        }
