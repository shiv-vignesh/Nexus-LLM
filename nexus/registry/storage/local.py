"""
Local filesystem storage backend.
"""

import hashlib
import shutil
from pathlib import Path

from nexus.registry.storage.base import StorageBackend
from nexus.core.exceptions import StorageError


class LocalStorage(StorageBackend):
    """Local filesystem storage backend.

    Stores model artifacts in a local directory structure:
        {base_path}/{model_id}/{version}/artifacts/
    """

    def __init__(self, base_path: str | Path):
        """Initialize local storage.

        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, source_path: Path, destination_key: str) -> str:
        """Save artifacts to local storage.

        Copies the entire directory structure from source to storage.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise StorageError(f"Source path does not exist: {source_path}")

        dest_path = self.base_path / destination_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if source_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)

            return str(dest_path)

        except Exception as e:
            raise StorageError(f"Failed to save to storage: {e}")

    def load(self, key: str, destination_path: Path) -> Path:
        """Load artifacts from local storage."""
        source_path = self.base_path / key
        destination_path = Path(destination_path)

        if not source_path.exists():
            raise StorageError(f"Key does not exist in storage: {key}")

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.is_dir():
                if destination_path.exists():
                    shutil.rmtree(destination_path)
                shutil.copytree(source_path, destination_path)
            else:
                shutil.copy2(source_path, destination_path)

            return destination_path

        except Exception as e:
            raise StorageError(f"Failed to load from storage: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in storage."""
        return (self.base_path / key).exists()

    def delete(self, key: str) -> bool:
        """Delete artifacts from storage."""
        path = self.base_path / key

        if not path.exists():
            return False

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True

        except Exception as e:
            raise StorageError(f"Failed to delete from storage: {e}")

    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys in storage."""
        search_path = self.base_path / prefix if prefix else self.base_path

        if not search_path.exists():
            return []

        keys = []
        for path in search_path.rglob("*"):
            if path.is_file() or (path.is_dir() and not any(path.iterdir())):
                rel_path = path.relative_to(self.base_path)
                keys.append(str(rel_path))

        return sorted(keys)

    def get_uri(self, key: str) -> str:
        """Get file URI for a key."""
        path = self.base_path / key
        return f"file://{path.absolute()}"

    def get_size(self, key: str) -> int:
        """Get size of stored artifacts in bytes."""
        path = self.base_path / key

        if not path.exists():
            return 0

        if path.is_file():
            return path.stat().st_size

        # Sum size of all files in directory
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size

        return total

    def compute_checksum(self, key: str) -> str:
        """Compute MD5 checksum of stored artifacts."""
        path = self.base_path / key

        if not path.exists():
            raise StorageError(f"Key does not exist: {key}")

        md5 = hashlib.md5()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5.update(chunk)
        else:
            # Hash all files in sorted order for reproducibility
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    # Include relative path in hash
                    rel_path = file_path.relative_to(path)
                    md5.update(str(rel_path).encode())

                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            md5.update(chunk)

        return md5.hexdigest()
