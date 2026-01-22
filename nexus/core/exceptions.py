"""
Custom exception hierarchy for Nexus-LLM.

Provides granular exception types for different failure modes across
training, inference, registry, and serving components.
"""

from typing import Any


class NexusError(Exception):
    """Base exception for all Nexus-LLM errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# Training Errors
class TrainingError(NexusError):
    """Base exception for training-related errors."""
    pass


class DatasetError(TrainingError):
    """Error loading or processing training data."""
    pass


class CheckpointError(TrainingError):
    """Error saving or loading model checkpoints."""
    pass


class ConfigurationError(TrainingError):
    """Invalid training configuration."""
    pass


# Registry Errors
class RegistryError(NexusError):
    """Base exception for model registry errors."""
    pass


class ModelNotFoundError(RegistryError):
    """Requested model not found in registry."""

    def __init__(self, model_id: str, version: str | None = None):
        details = {"model_id": model_id}
        if version:
            details["version"] = version
            message = f"Model '{model_id}' version '{version}' not found in registry"
        else:
            message = f"Model '{model_id}' not found in registry"
        super().__init__(message, details)
        self.model_id = model_id
        self.version = version


class ModelExistsError(RegistryError):
    """Model already exists in registry."""

    def __init__(self, model_id: str, version: str):
        super().__init__(
            f"Model '{model_id}' version '{version}' already exists",
            {"model_id": model_id, "version": version}
        )


class StorageError(RegistryError):
    """Error accessing model storage."""
    pass


# Inference Errors
class InferenceError(NexusError):
    """Base exception for inference-related errors."""
    pass


class ModelLoadError(InferenceError):
    """Error loading model for inference."""

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model from '{model_path}': {reason}",
            {"model_path": model_path, "reason": reason}
        )


class GenerationError(InferenceError):
    """Error during text generation."""
    pass


class BatchingError(InferenceError):
    """Error in request batching."""
    pass


class EngineNotReadyError(InferenceError):
    """Inference engine not initialized or ready."""

    def __init__(self, engine_type: str):
        super().__init__(
            f"Inference engine '{engine_type}' is not ready",
            {"engine_type": engine_type}
        )


# Serving Errors
class ServingError(NexusError):
    """Base exception for serving-related errors."""
    pass


class RequestError(ServingError):
    """Invalid or malformed request."""
    pass


class RateLimitError(ServingError):
    """Rate limit exceeded."""

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            {"limit": limit, "window_seconds": window_seconds}
        )


class TimeoutError(ServingError):
    """Request timeout."""

    def __init__(self, timeout_seconds: int, request_id: str | None = None):
        details = {"timeout_seconds": timeout_seconds}
        if request_id:
            details["request_id"] = request_id
        super().__init__(f"Request timed out after {timeout_seconds}s", details)


class QueueFullError(ServingError):
    """Request queue is full."""

    def __init__(self, max_size: int):
        super().__init__(
            f"Request queue is full (max: {max_size})",
            {"max_size": max_size}
        )
