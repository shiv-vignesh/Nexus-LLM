"""
Prefix caching for efficient repeated prompt handling.

Caches tokenized prompts and their prefixes to avoid redundant
tokenization and enable vLLM's automatic prefix caching.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix cache."""

    max_entries: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    min_prefix_length: int = 10  # Minimum tokens to cache


@dataclass
class CacheEntry:
    """A cached prefix entry."""

    prefix_hash: str
    tokens: list[int]
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        self.last_accessed = self.created_at


class PrefixCache:
    """LRU cache for prompt prefixes.

    Caches tokenized prefixes to:
    1. Avoid redundant tokenization
    2. Enable vLLM's automatic prefix caching
    3. Track common prompt patterns

    Example:
        >>> cache = PrefixCache(config)
        >>> tokens = cache.get_or_tokenize(prompt, tokenizer)
        >>> # Subsequent calls with same prefix are fast
    """

    def __init__(self, config: PrefixCacheConfig | None = None):
        """Initialize prefix cache.

        Args:
            config: Cache configuration
        """
        self.config = config or PrefixCacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _compute_hash(self, text: str) -> str:
        """Compute hash for cache key."""
        return hashlib.md5(text.encode()).hexdigest()

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.created_at > self.config.ttl_seconds
        ]
        for key in expired:
            del self._cache[key]

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)

    def get(self, text: str) -> list[int] | None:
        """Get cached tokens for text.

        Args:
            text: Text to look up

        Returns:
            Cached tokens or None if not found
        """
        self._evict_expired()

        key = self._compute_hash(text)

        if key in self._cache:
            entry = self._cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.tokens

        self._misses += 1
        return None

    def put(self, text: str, tokens: list[int]) -> None:
        """Cache tokens for text.

        Args:
            text: Original text
            tokens: Tokenized representation
        """
        if len(tokens) < self.config.min_prefix_length:
            return  # Don't cache very short prefixes

        self._evict_lru()

        key = self._compute_hash(text)
        now = time.time()

        self._cache[key] = CacheEntry(
            prefix_hash=key,
            tokens=tokens,
            created_at=now,
        )

    def get_or_tokenize(
        self,
        text: str,
        tokenizer: Any,
    ) -> list[int]:
        """Get cached tokens or tokenize and cache.

        Args:
            text: Text to tokenize
            tokenizer: Tokenizer to use if not cached

        Returns:
            Token IDs
        """
        cached = self.get(text)
        if cached is not None:
            return cached

        # Tokenize
        tokens = tokenizer.encode(text)

        # Cache if long enough
        self.put(text, tokens)

        return tokens

    def find_common_prefix(
        self,
        texts: list[str],
        tokenizer: Any,
    ) -> tuple[str, list[int]]:
        """Find the common prefix among multiple texts.

        Useful for RAG scenarios where context is shared.

        Args:
            texts: List of texts to find common prefix
            tokenizer: Tokenizer for encoding

        Returns:
            Tuple of (common_prefix_text, common_prefix_tokens)
        """
        if not texts:
            return "", []

        # Find common string prefix
        common_prefix = texts[0]
        for text in texts[1:]:
            while not text.startswith(common_prefix):
                common_prefix = common_prefix[:-1]
                if not common_prefix:
                    return "", []

        # Tokenize the common prefix
        tokens = self.get_or_tokenize(common_prefix, tokenizer)

        return common_prefix, tokens

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "entries": len(self._cache),
            "max_entries": self.config.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
