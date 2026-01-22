"""
Model registry implementation.

Provides model versioning, artifact storage, and lifecycle management.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.core.config import RegistryConfig
from nexus.core.interfaces.model_registry import (
    IModelRegistry,
    ModelMetadata,
    ModelVersion,
    ModelStage,
    ModelType,
)
from nexus.core.exceptions import (
    ModelNotFoundError,
    ModelExistsError,
    StorageError,
)
from nexus.registry.storage.local import LocalStorage
from nexus.registry.storage.base import StorageBackend


class ModelRegistry(IModelRegistry):
    """Model registry with versioning and artifact management.

    Provides:
    - Model version control
    - Artifact storage and retrieval
    - Stage-based lifecycle management (dev, staging, production)
    - Metadata and lineage tracking

    Storage structure:
        {base_path}/
            models/
                {model_id}/
                    metadata.json
                    {version}/
                        artifacts/
                        version.json
    """

    METADATA_FILE = "metadata.json"
    VERSION_FILE = "version.json"
    ARTIFACTS_DIR = "artifacts"

    def __init__(
        self,
        config: RegistryConfig | None = None,
        storage: StorageBackend | None = None,
    ):
        """Initialize the model registry.

        Args:
            config: Registry configuration
            storage: Optional storage backend (uses LocalStorage if not provided)
        """
        self.config = config or RegistryConfig()

        if storage:
            self.storage = storage
        else:
            self.storage = LocalStorage(self.config.storage_path)

        self._models_path = Path("models")

        # Ensure models directory exists
        models_dir = self.config.storage_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_id: str) -> Path:
        """Get path to model directory."""
        return self._models_path / model_id

    def _get_version_path(self, model_id: str, version: str) -> Path:
        """Get path to version directory."""
        return self._models_path / model_id / version

    def _generate_version(self) -> str:
        """Generate a new version string."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"v_{timestamp}_{short_uuid}"

    def _load_metadata(self, model_id: str) -> ModelMetadata | None:
        """Load model metadata from storage."""
        metadata_key = str(self._get_model_path(model_id) / self.METADATA_FILE)

        if not self.storage.exists(metadata_key):
            return None

        try:
            # Read metadata file
            metadata_path = self.config.storage_path / metadata_key
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception:
            return None

    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to storage."""
        metadata_path = (
            self.config.storage_path /
            self._get_model_path(metadata.model_id) /
            self.METADATA_FILE
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_version(self, model_id: str, version: str) -> ModelVersion | None:
        """Load version info from storage."""
        version_path = (
            self.config.storage_path /
            self._get_version_path(model_id, version) /
            self.VERSION_FILE
        )

        if not version_path.exists():
            return None

        try:
            with open(version_path, "r") as f:
                data = json.load(f)
            return ModelVersion.from_dict(data)
        except Exception:
            return None

    def _save_version(self, model_id: str, version: ModelVersion) -> None:
        """Save version info to storage."""
        version_path = (
            self.config.storage_path /
            self._get_version_path(model_id, version.version) /
            self.VERSION_FILE
        )
        version_path.parent.mkdir(parents=True, exist_ok=True)

        with open(version_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

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
        """Register a new model or version."""
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise StorageError(f"Artifact path does not exist: {artifact_path}")

        # Generate version if not provided
        version = version or self._generate_version()

        # Load or create model metadata
        metadata = self._load_metadata(model_id)
        now = datetime.utcnow()

        if metadata is None:
            # New model
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                created_at=now,
                updated_at=now,
                description=description,
                tags=tags or [],
            )
        else:
            # Check if version already exists
            if version in metadata.versions:
                raise ModelExistsError(model_id, version)
            metadata.updated_at = now

        # Copy artifacts to storage
        storage_key = str(
            self._get_version_path(model_id, version) / self.ARTIFACTS_DIR
        )
        stored_path = self.storage.save(artifact_path, storage_key)

        # Create version record
        model_version = ModelVersion(
            version=version,
            created_at=now,
            stage=ModelStage.DEVELOPMENT,
            artifact_path=Path(stored_path),
            base_model=base_model,
            training_config=training_config or {},
            training_metrics=training_metrics or {},
            description=description,
            tags=tags or [],
        )

        # Update metadata
        metadata.versions[version] = model_version

        # Save metadata and version
        self._save_metadata(metadata)
        self._save_version(model_id, model_version)

        print(f"Registered model '{model_id}' version '{version}'")
        return model_version

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata."""
        return self._load_metadata(model_id)

    def get_model_version(
        self, model_id: str, version: str | None = None
    ) -> ModelVersion | None:
        """Get a specific model version."""
        metadata = self._load_metadata(model_id)
        if metadata is None:
            return None

        if version is None:
            version = metadata.latest_version

        return metadata.versions.get(version) if version else None

    def list_models(
        self,
        model_type: ModelType | None = None,
        tags: list[str] | None = None,
        stage: ModelStage | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filters."""
        models_dir = self.config.storage_path / "models"

        if not models_dir.exists():
            return []

        results = []

        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata = self._load_metadata(model_dir.name)
            if metadata is None:
                continue

            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue

            if tags:
                if not all(tag in metadata.tags for tag in tags):
                    continue

            if stage:
                has_stage = any(
                    v.stage == stage for v in metadata.versions.values()
                )
                if not has_stage:
                    continue

            results.append(metadata)

        return results

    def promote_version(
        self, model_id: str, version: str, stage: ModelStage
    ) -> ModelVersion:
        """Promote a model version to a stage."""
        metadata = self._load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(model_id)

        if version not in metadata.versions:
            raise ModelNotFoundError(model_id, version)

        # Update version stage
        model_version = metadata.versions[version]
        model_version.stage = stage

        # Update metadata pointers
        if stage == ModelStage.PRODUCTION:
            metadata.production_version = version
        elif stage == ModelStage.STAGING:
            metadata.staging_version = version

        metadata.updated_at = datetime.utcnow()

        # Save updates
        self._save_metadata(metadata)
        self._save_version(model_id, model_version)

        print(f"Promoted '{model_id}' version '{version}' to {stage.value}")
        return model_version

    def load_model_artifacts(
        self, model_id: str, version: str | None = None
    ) -> Path:
        """Get path to model artifacts."""
        model_version = self.get_model_version(model_id, version)
        if model_version is None:
            raise ModelNotFoundError(model_id, version)

        if model_version.artifact_path is None:
            raise StorageError(f"No artifacts stored for {model_id}:{version}")

        return model_version.artifact_path

    def delete_version(self, model_id: str, version: str) -> bool:
        """Delete a model version."""
        metadata = self._load_metadata(model_id)
        if metadata is None:
            return False

        if version not in metadata.versions:
            return False

        # Remove from storage
        storage_key = str(self._get_version_path(model_id, version))
        self.storage.delete(storage_key)

        # Update metadata
        del metadata.versions[version]

        if metadata.production_version == version:
            metadata.production_version = None
        if metadata.staging_version == version:
            metadata.staging_version = None

        metadata.updated_at = datetime.utcnow()
        self._save_metadata(metadata)

        print(f"Deleted '{model_id}' version '{version}'")
        return True

    def get_production_model(self, model_id: str) -> ModelVersion | None:
        """Get the production version of a model."""
        metadata = self._load_metadata(model_id)
        if metadata is None or metadata.production_version is None:
            return None
        return metadata.versions.get(metadata.production_version)

    def export_model_card(self, model_id: str, output_path: Path) -> None:
        """Export a markdown model card."""
        from nexus.registry.metadata import ModelCard

        metadata = self._load_metadata(model_id)
        if metadata is None:
            raise ModelNotFoundError(model_id)

        # Build model card
        latest = metadata.get_version()
        card = ModelCard(
            model_id=model_id,
            model_name=model_id.replace("-", " ").title(),
            description=metadata.description,
            training_data=latest.training_config.get("dataset", "") if latest else "",
            evaluation_results=latest.training_metrics if latest else {},
        )

        # Write markdown
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card.to_markdown())

        print(f"Model card exported to {output_path}")
