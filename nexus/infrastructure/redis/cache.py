"""
Redis-based caching layer for inference results and model outputs.
"""

import json
import hashlib
import time
from typing import Any

from nexus.infrastructure.redis.client import RedisClient


class RedisCache:
    """Redis-based cache with TTL and statistics.

    Provides caching for:
    - Inference results (for idempotent requests)
    - Tokenized prompts
    - Model outputs

    Example:
        >>> cache = RedisCache(client, prefix="inference")
        >>> await cache.set("request_123", response_data)
        >>> cached = await cache.get("request_123")
    """

    def __init__(
        self,
        redis_client: RedisClient,
        prefix: str = "cache",
        default_ttl: int = 3600,
    ):
        """Initialize cache.

        Args:
            redis_client: Redis client
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
        """
        self._client = redis_client
        self.prefix = prefix
        self.default_ttl = default_ttl

        # Local stats (not persisted)
        self._hits = 0
        self._misses = 0

    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        full_key = self._make_key(key)
        value = await self._client.get(full_key)

        if value is not None:
            self._hits += 1
        else:
            self._misses += 1

        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if not specified)

        Returns:
            True if set successfully
        """
        full_key = self._make_key(key)
        return await self._client.set(
            full_key,
            value,
            ttl_seconds=ttl or self.default_ttl,
        )

    async def delete(self, key: str) -> bool:
        """Delete cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        full_key = self._make_key(key)
        return await self._client.delete(full_key)

    async def exists(self, key: str) -> bool:
        """Check if key is cached.

        Args:
            key: Cache key

        Returns:
            True if cached
        """
        full_key = self._make_key(key)
        return await self._client.exists(full_key)

    async def get_or_set(
        self,
        key: str,
        factory: Any,
        ttl: int | None = None,
    ) -> Any:
        """Get cached value or compute and cache it.

        Args:
            key: Cache key
            factory: Callable or coroutine to compute value
            ttl: TTL in seconds

        Returns:
            Cached or computed value
        """
        value = await self.get(key)

        if value is not None:
            return value

        # Compute value
        if callable(factory):
            import asyncio
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
        else:
            value = factory

        await self.set(key, value, ttl)
        return value

    def compute_key(self, *args, **kwargs) -> str:
        """Compute a cache key from arguments.

        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            MD5 hash key
        """
        data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "prefix": self.prefix,
            "default_ttl": self.default_ttl,
        }

    def reset_stats(self) -> None:
        """Reset local statistics."""
        self._hits = 0
        self._misses = 0
