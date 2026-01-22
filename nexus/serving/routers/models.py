"""
Models API router.

Provides endpoints for model information and management.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from nexus.serving.schemas.common import ModelInfo
from nexus.serving.dependencies import get_inference_engine, get_registry
from nexus.core.interfaces.inference_engine import IInferenceEngine
from nexus.registry import ModelRegistry

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models(
    engine: IInferenceEngine = Depends(get_inference_engine),
    registry: ModelRegistry | None = Depends(get_registry),
):
    """List available models."""
    models = []

    # Add currently loaded model
    if engine.is_ready:
        info = await engine.get_model_info()
        models.append(ModelInfo(
            model_id=info.get("model_name", "default"),
            model_name=info.get("model_name", "default"),
            backend=engine.backend_name,
            max_context_length=info.get("max_model_len", 2048),
            capabilities=["text-generation", "chat"],
            loaded_at=datetime.utcnow(),
        ))

    # Add registered models if registry is available
    if registry:
        for metadata in registry.list_models():
            if metadata.model_id not in [m.model_id for m in models]:
                models.append(ModelInfo(
                    model_id=metadata.model_id,
                    model_name=metadata.model_id,
                    backend="registered",
                    max_context_length=2048,
                    capabilities=["text-generation"],
                ))

    return models


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    engine: IInferenceEngine = Depends(get_inference_engine),
    registry: ModelRegistry | None = Depends(get_registry),
):
    """Get information about a specific model."""
    # Check loaded model
    if engine.is_ready:
        info = await engine.get_model_info()
        if info.get("model_name") == model_id or model_id == "default":
            return ModelInfo(
                model_id=model_id,
                model_name=info.get("model_name", model_id),
                backend=engine.backend_name,
                max_context_length=info.get("max_model_len", 2048),
                capabilities=["text-generation", "chat"],
                loaded_at=datetime.utcnow(),
            )

    # Check registry
    if registry:
        metadata = registry.get_model(model_id)
        if metadata:
            return ModelInfo(
                model_id=metadata.model_id,
                model_name=metadata.model_id,
                backend="registered",
                max_context_length=2048,
                capabilities=["text-generation"],
            )

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
