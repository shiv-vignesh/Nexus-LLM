"""
Base storage backend interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageBackend(ABC):
    """Abstract base class for model storage backends."""

    @abstractmethod
    def save(self, source_path: Path, destination_key: str) -> str:
        """Save artifacts from source path to storage.

        Args:
            source_path: Local path to artifacts
            destination_key: Key/path in storage

        Returns:
            URI or path to stored artifacts
        """
        ...

    @abstractmethod
    def load(self, key: str, destination_path: Path) -> Path:
        """Load artifacts from storage to local path.

        Args:
            key: Key/path in storage
            destination_path: Local path to save to

        Returns:
            Path to loaded artifacts
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Key/path in storage

        Returns:
            True if exists
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete artifacts from storage.

        Args:
            key: Key/path in storage

        Returns:
            True if deleted
        """
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys in storage.

        Args:
            prefix: Optional prefix filter

        Returns:
            List of keys
        """
        ...

    @abstractmethod
    def get_uri(self, key: str) -> str:
        """Get URI for a storage key.

        Args:
            key: Key/path in storage

        Returns:
            URI string
        """
        ...
