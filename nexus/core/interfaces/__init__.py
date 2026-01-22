"""Interface definitions (protocols) for dependency injection and testability."""

from nexus.core.interfaces.trainer import ITrainer, TrainingResult
from nexus.core.interfaces.inference_engine import (
    IInferenceEngine,
    GenerationRequest,
    GenerationResponse,
)
from nexus.core.interfaces.model_registry import (
    IModelRegistry,
    ModelMetadata,
    ModelVersion,
)

__all__ = [
    "ITrainer",
    "TrainingResult",
    "IInferenceEngine",
    "GenerationRequest",
    "GenerationResponse",
    "IModelRegistry",
    "ModelMetadata",
    "ModelVersion",
]
