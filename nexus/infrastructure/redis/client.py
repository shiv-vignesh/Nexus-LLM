"""
Redis client wrapper with connection management.
"""

import asyncio
import json
from typing import Any

import redis.asyncio as redis
from redis.asyncio import Redis

from nexus.core.config import RedisConfig


class RedisClient:
    """Async Redis client with connection pooling.

    Provides a clean interface for common Redis operations
    with proper connection management and error handling.

    Example:
        >>> config = RedisConfig()
        >>> client = RedisClient(config)
        >>> await client.connect()
        >>> await client.set("key", {"data": "value"})
        >>> result = await client.get("key")
    """

    def __init__(self, config: RedisConfig):
        """Initialize Redis client.

        Args:
            config: Redis configuration
        """
        self.config = config
        self._client: Redis | None = None
        self._pool: redis.ConnectionPool | None = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is not None:
            return

        self._pool = redis.ConnectionPool.from_url(
            self.config.url,
            decode_responses=True,
            max_connections=20,
        )

        self._client = Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()
        print(f"Connected to Redis at {self.config.host}:{self.config.port}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

    async def ping(self) -> bool:
        """Check if Redis is responsive."""
        if self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False

    # Key-Value Operations

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Set a key-value pair.

        Args:
            key: Redis key
            value: Value (will be JSON serialized if not string)
            ttl_seconds: Optional TTL in seconds

        Returns:
            True if set successfully
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        if not isinstance(value, str):
            value = json.dumps(value)

        return await self._client.set(key, value, ex=ttl_seconds)

    async def get(self, key: str) -> Any | None:
        """Get value for key.

        Args:
            key: Redis key

        Returns:
            Value (JSON decoded if applicable) or None
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        value = await self._client.get(key)

        if value is None:
            return None

        # Try to decode JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: Redis key

        Returns:
            True if key was deleted
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        return await self._client.delete(key) > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Redis key

        Returns:
            True if key exists
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        return await self._client.exists(key) > 0

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set TTL on existing key.

        Args:
            key: Redis key
            ttl_seconds: TTL in seconds

        Returns:
            True if TTL was set
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        return await self._client.expire(key, ttl_seconds)

    # Hash Operations

    async def hset(self, name: str, mapping: dict) -> int:
        """Set multiple hash fields.

        Args:
            name: Hash name
            mapping: Field-value mapping

        Returns:
            Number of fields set
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        # Serialize values
        serialized = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in mapping.items()
        }

        return await self._client.hset(name, mapping=serialized)

    async def hget(self, name: str, key: str) -> Any | None:
        """Get hash field value.

        Args:
            name: Hash name
            key: Field name

        Returns:
            Field value or None
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        value = await self._client.hget(name, key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def hgetall(self, name: str) -> dict:
        """Get all hash fields.

        Args:
            name: Hash name

        Returns:
            Dictionary of field-value pairs
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        data = await self._client.hgetall(name)

        # Deserialize values
        result = {}
        for k, v in data.items():
            try:
                result[k] = json.loads(v)
            except json.JSONDecodeError:
                result[k] = v

        return result

    # Pipeline for batch operations

    def pipeline(self):
        """Create a pipeline for batch operations."""
        if self._client is None:
            raise RuntimeError("Redis client not connected")

        return self._client.pipeline()

    @property
    def client(self) -> Redis | None:
        """Get the underlying Redis client."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._client is not None
