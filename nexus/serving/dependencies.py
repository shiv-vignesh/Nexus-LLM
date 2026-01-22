"""
FastAPI dependency injection for shared resources.
"""

from typing import AsyncIterator

from nexus.core.config import NexusConfig, InferenceConfig, ServingConfig
from nexus.core.interfaces.inference_engine import IInferenceEngine
from nexus.inference.engines.vllm_engine import VLLMInferenceEngine
from nexus.inference.engines.transformers_engine import TransformersInferenceEngine
from nexus.inference.batching.batcher import DynamicBatcher, BatcherConfig
from nexus.infrastructure.redis.client import RedisClient
from nexus.registry import ModelRegistry


class AppState:
    """Application state container for shared resources."""

    def __init__(self):
        self.config: NexusConfig | None = None
        self.inference_engine: IInferenceEngine | None = None
        self.batcher: DynamicBatcher | None = None
        self.redis_client: RedisClient | None = None
        self.registry: ModelRegistry | None = None
        self._initialized = False

    async def initialize(self, config: NexusConfig) -> None:
        """Initialize all resources.

        Args:
            config: Application configuration
        """
        if self._initialized:
            return

        self.config = config

        # Initialize inference engine
        inference_config = config.serving.inference

        if inference_config.backend.value == "vllm":
            self.inference_engine = VLLMInferenceEngine(inference_config)
        else:
            self.inference_engine = TransformersInferenceEngine(inference_config)

        await self.inference_engine.initialize()

        # Initialize batcher
        batcher_config = BatcherConfig(
            max_batch_size=inference_config.max_batch_size,
            max_wait_ms=inference_config.batch_timeout_ms,
        )
        self.batcher = DynamicBatcher(
            batcher_config,
            self.inference_engine.generate_batch,
        )
        await self.batcher.start()

        # Initialize Redis
        try:
            self.redis_client = RedisClient(config.serving.redis)
            await self.redis_client.connect()
        except Exception as e:
            print(f"Redis connection failed (optional): {e}")
            self.redis_client = None

        # Initialize registry
        try:
            self.registry = ModelRegistry(config.registry)
        except Exception as e:
            print(f"Registry initialization failed (optional): {e}")
            self.registry = None

        self._initialized = True
        print("Application state initialized")

    async def shutdown(self) -> None:
        """Cleanup all resources."""
        if self.batcher:
            await self.batcher.stop()

        if self.inference_engine:
            await self.inference_engine.shutdown()

        if self.redis_client:
            await self.redis_client.disconnect()

        self._initialized = False
        print("Application state shutdown complete")

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# Global app state instance
_app_state = AppState()


def get_app_state() -> AppState:
    """Get the global application state."""
    return _app_state


def get_inference_engine() -> IInferenceEngine:
    """Dependency to get the inference engine."""
    state = get_app_state()
    if state.inference_engine is None:
        raise RuntimeError("Inference engine not initialized")
    return state.inference_engine


def get_batcher() -> DynamicBatcher | None:
    """Dependency to get the batcher (optional)."""
    return get_app_state().batcher


def get_redis_client() -> RedisClient | None:
    """Dependency to get the Redis client (optional)."""
    return get_app_state().redis_client


def get_registry() -> ModelRegistry | None:
    """Dependency to get the model registry (optional)."""
    return get_app_state().registry


def get_config() -> NexusConfig:
    """Dependency to get the configuration."""
    state = get_app_state()
    if state.config is None:
        raise RuntimeError("Configuration not loaded")
    return state.config
