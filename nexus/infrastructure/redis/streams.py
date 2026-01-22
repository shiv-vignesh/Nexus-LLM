"""
Redis Streams for distributed task queuing.

Provides a reliable message queue for distributing inference
requests across multiple workers.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from redis.asyncio import Redis

from nexus.core.config import RedisConfig


@dataclass
class StreamMessage:
    """A message from a Redis stream."""

    message_id: str
    data: dict[str, Any]
    stream: str


class RedisStreamQueue:
    """Redis Streams-based task queue.

    Uses Redis Streams with consumer groups for:
    - Reliable message delivery
    - Horizontal scaling across workers
    - Message acknowledgment
    - Dead letter queue support

    Example:
        >>> queue = RedisStreamQueue(client, config)
        >>> await queue.initialize()
        >>>
        >>> # Producer
        >>> message_id = await queue.enqueue({"request_id": "123", "prompt": "..."})
        >>>
        >>> # Consumer
        >>> async for message in queue.consume():
        ...     # Process message
        ...     await queue.acknowledge(message)
    """

    def __init__(
        self,
        redis_client: Redis,
        config: RedisConfig,
        consumer_name: str | None = None,
    ):
        """Initialize stream queue.

        Args:
            redis_client: Redis client
            config: Redis configuration
            consumer_name: Unique name for this consumer
        """
        self._client = redis_client
        self.config = config
        self.consumer_name = consumer_name or f"consumer-{time.time_ns()}"

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the stream and consumer group."""
        if self._initialized:
            return

        try:
            # Create consumer group (MKSTREAM creates stream if not exists)
            await self._client.xgroup_create(
                name=self.config.stream_name,
                groupname=self.config.consumer_group,
                id="0",
                mkstream=True,
            )
        except Exception as e:
            # Group may already exist
            if "BUSYGROUP" not in str(e):
                raise

        self._initialized = True
        print(f"Stream queue initialized: {self.config.stream_name}")

    async def enqueue(
        self,
        data: dict[str, Any],
        max_len: int | None = None,
    ) -> str:
        """Add a message to the queue.

        Args:
            data: Message data (will be JSON serialized)
            max_len: Optional max stream length (for trimming)

        Returns:
            Message ID
        """
        # Serialize data
        serialized = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in data.items()
        }

        message_id = await self._client.xadd(
            name=self.config.stream_name,
            fields=serialized,
            maxlen=max_len,
            approximate=True,
        )

        return message_id

    async def consume(
        self,
        batch_size: int = 10,
        block_ms: int = 1000,
    ) -> AsyncIterator[StreamMessage]:
        """Consume messages from the queue.

        Args:
            batch_size: Max messages per batch
            block_ms: Block time in milliseconds

        Yields:
            StreamMessage objects
        """
        while True:
            try:
                # Read new messages
                result = await self._client.xreadgroup(
                    groupname=self.config.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.config.stream_name: ">"},
                    count=batch_size,
                    block=block_ms,
                )

                if not result:
                    continue

                for stream_name, messages in result:
                    for message_id, data in messages:
                        # Deserialize data
                        deserialized = {}
                        for k, v in data.items():
                            try:
                                deserialized[k] = json.loads(v)
                            except json.JSONDecodeError:
                                deserialized[k] = v

                        yield StreamMessage(
                            message_id=message_id,
                            data=deserialized,
                            stream=stream_name,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Stream consume error: {e}")
                await asyncio.sleep(1)

    async def acknowledge(self, message: StreamMessage) -> bool:
        """Acknowledge a processed message.

        Args:
            message: Message to acknowledge

        Returns:
            True if acknowledged
        """
        count = await self._client.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message.message_id,
        )
        return count > 0

    async def reject(self, message: StreamMessage) -> None:
        """Reject a message (will be redelivered).

        The message stays in the pending list and can be
        claimed by another consumer.

        Args:
            message: Message to reject
        """
        # Simply not acknowledging keeps it pending
        pass

    async def get_pending_count(self) -> int:
        """Get count of pending messages."""
        info = await self._client.xpending(
            name=self.config.stream_name,
            groupname=self.config.consumer_group,
        )
        return info.get("pending", 0) if info else 0

    async def get_stream_length(self) -> int:
        """Get total messages in stream."""
        return await self._client.xlen(self.config.stream_name)

    async def claim_stale_messages(
        self,
        min_idle_ms: int = 60000,
        count: int = 10,
    ) -> list[StreamMessage]:
        """Claim messages that have been pending too long.

        Useful for recovering from crashed workers.

        Args:
            min_idle_ms: Minimum idle time to claim
            count: Max messages to claim

        Returns:
            List of claimed messages
        """
        try:
            messages = await self._client.xautoclaim(
                name=self.config.stream_name,
                groupname=self.config.consumer_group,
                consumername=self.consumer_name,
                min_idle_time=min_idle_ms,
                count=count,
            )

            claimed = []
            if messages and len(messages) > 1:
                for message_id, data in messages[1]:
                    if data:  # Skip deleted messages
                        deserialized = {}
                        for k, v in data.items():
                            try:
                                deserialized[k] = json.loads(v)
                            except json.JSONDecodeError:
                                deserialized[k] = v

                        claimed.append(StreamMessage(
                            message_id=message_id,
                            data=deserialized,
                            stream=self.config.stream_name,
                        ))

            return claimed

        except Exception:
            return []

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message from the stream.

        Args:
            message_id: Message ID to delete

        Returns:
            True if deleted
        """
        count = await self._client.xdel(
            self.config.stream_name,
            message_id,
        )
        return count > 0
