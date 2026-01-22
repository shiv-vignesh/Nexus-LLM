"""Storage backends for model artifacts."""

from nexus.registry.storage.local import LocalStorage
from nexus.registry.storage.base import StorageBackend

__all__ = ["LocalStorage", "StorageBackend"]
