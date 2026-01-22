"""Model registry interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    """Types of models in the registry."""

    BASE = "base"
    FINE_TUNED = "fine_tuned"
    REWARD_MODEL = "reward_model"
    LORA_ADAPTER = "lora_adapter"


@dataclass
class ModelVersion:
    """A specific version of a model."""

    version: str
    created_at: datetime
    stage: ModelStage = ModelStage.DEVELOPMENT
    artifact_path: Path | None = None

    # Training lineage
    base_model: str | None = None
    training_config: dict = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)

    # Optional metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "stage": self.stage.value,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "base_model": self.base_model,
            "training_config": self.training_config,
            "training_metrics": self.training_metrics,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            stage=ModelStage(data.get("stage", "development")),
            artifact_path=Path(data["artifact_path"]) if data.get("artifact_path") else None,
            base_model=data.get("base_model"),
            training_config=data.get("training_config", {}),
            training_metrics=data.get("training_metrics", {}),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    model_id: str
    model_type: ModelType
    created_at: datetime
    updated_at: datetime

    # All versions
    versions: dict[str, ModelVersion] = field(default_factory=dict)

    # Current active version per stage
    production_version: str | None = None
    staging_version: str | None = None

    # Model info
    description: str = ""
    tags: list[str] = field(default_factory=list)
    owner: str = ""

    @property
    def latest_version(self) -> str | None:
        """Get the latest version by creation time."""
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: v.created_at).version

    def get_version(self, version: str | None = None) -> ModelVersion | None:
        """Get a specific version or the latest."""
        if version is None:
            version = self.latest_version
        return self.versions.get(version) if version else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "production_version": self.production_version,
            "staging_version": self.staging_version,
            "description": self.description,
            "tags": self.tags,
            "owner": self.owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Deserialize from dictionary."""
        return cls(
            model_id=data["model_id"],
            model_type=ModelType(data["model_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            versions={k: ModelVersion.from_dict(v) for k, v in data.get("versions", {}).items()},
            production_version=data.get("production_version"),
            staging_version=data.get("staging_version"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            owner=data.get("owner", ""),
        )


class IModelRegistry(ABC):
    """Interface for model registry.

    Provides model versioning, artifact storage, and lifecycle management.
    """

    @abstractmethod
    def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        artifact_path: Path,
        version: str | None = None,
        base_model: str | None = None,
        training_config: dict | None = None,
        training_metrics: dict[str, float] | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelVersion:
        """Register a new model or version.

        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (base, fine_tuned, etc.)
            artifact_path: Path to model artifacts
            version: Optional version string (auto-generated if not provided)
            base_model: Base model this was derived from
            training_config: Training configuration used
            training_metrics: Metrics from training
            description: Human-readable description
            tags: Searchable tags

        Returns:
            Created ModelVersion
        """
        ...

    @abstractmethod
    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata or None if not found
        """
        ...

    @abstractmethod
    def get_model_version(
        self, model_id: str, version: str | None = None
    ) -> ModelVersion | None:
        """Get a specific model version.

        Args:
            model_id: Model identifier
            version: Version string (None for latest)

        Returns:
            ModelVersion or None if not found
        """
        ...

    @abstractmethod
    def list_models(
        self,
        model_type: ModelType | None = None,
        tags: list[str] | None = None,
        stage: ModelStage | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filters.

        Args:
            model_type: Filter by model type
            tags: Filter by tags (AND)
            stage: Filter by having a version in this stage

        Returns:
            List of matching ModelMetadata
        """
        ...

    @abstractmethod
    def promote_version(
        self, model_id: str, version: str, stage: ModelStage
    ) -> ModelVersion:
        """Promote a model version to a stage.

        Args:
            model_id: Model identifier
            version: Version to promote
            stage: Target stage

        Returns:
            Updated ModelVersion
        """
        ...

    @abstractmethod
    def load_model_artifacts(
        self, model_id: str, version: str | None = None
    ) -> Path:
        """Get path to model artifacts.

        Args:
            model_id: Model identifier
            version: Version (None for latest)

        Returns:
            Path to model artifacts
        """
        ...

    @abstractmethod
    def delete_version(self, model_id: str, version: str) -> bool:
        """Delete a model version.

        Args:
            model_id: Model identifier
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        ...
