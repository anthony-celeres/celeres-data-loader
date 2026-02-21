"""
CDL Prefetcher — Async block prefetching with LRU cache.

Loads upcoming blocks in background threads while the main thread
processes the current block, hiding I/O latency behind computation.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Block Cache
# ---------------------------------------------------------------------------

class BlockCache:
    """Thread-safe LRU block cache.

    Parameters
    ----------
    max_blocks : int
        Maximum number of blocks to keep in cache.
    """

    __slots__ = ("max_blocks", "_cache", "_order", "_lock")

    def __init__(self, max_blocks: int = 2) -> None:
        self.max_blocks = max_blocks
        self._cache: Dict[int, Any] = {}
        self._order: list[int] = []
        self._lock = threading.Lock()

    def get(self, block_id: int) -> Optional[Any]:
        """Return cached block or ``None``."""
        with self._lock:
            if block_id in self._cache:
                if block_id in self._order:
                    self._order.remove(block_id)
                self._order.append(block_id)
                return self._cache[block_id]
            return None

    def put(self, block_id: int, data: Any) -> None:
        """Insert a block, evicting the oldest if at capacity."""
        with self._lock:
            if block_id in self._cache:
                return
            while len(self._cache) >= self.max_blocks and self._order:
                self._cache.pop(self._order.pop(0), None)
            self._cache[block_id] = data
            self._order.append(block_id)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def __contains__(self, block_id: int) -> bool:
        with self._lock:
            return block_id in self._cache


# ---------------------------------------------------------------------------
# Async Prefetcher
# ---------------------------------------------------------------------------

class AsyncPrefetcher:
    """Background block prefetcher backed by a bounded thread pool.

    Parameters
    ----------
    load_fn : callable
        ``load_fn(block_id) -> block_data``
    prefetch_depth : int
        How many blocks ahead to prefetch.
    cache_size : int
        Maximum blocks to hold in the LRU cache.
    num_workers : int
        Number of background loading threads.
    """

    def __init__(
        self,
        load_fn: Callable[[int], Any],
        prefetch_depth: int = 2,
        cache_size: int = 4,
        num_workers: int = 2,
    ) -> None:
        self.load_fn = load_fn
        self.prefetch_depth = prefetch_depth
        self.num_workers = num_workers
        self.cache = BlockCache(max_blocks=cache_size)

        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: Dict[int, Any] = {}
        self._pending_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._pending_lock:
            self._pending.clear()
        self.cache.clear()

    def schedule(self, block_id: int) -> None:
        """Submit *block_id* for background loading (no-op if cached)."""
        if not self._running or self._executor is None:
            return
        if block_id in self.cache:
            return
        with self._pending_lock:
            if block_id in self._pending:
                return
            future = self._executor.submit(self._load_block, block_id)
            self._pending[block_id] = future

    def get(self, block_id: int, timeout: float = 30.0) -> Any:
        """Return block data, waiting for a pending prefetch if needed."""
        cached = self.cache.get(block_id)
        if cached is not None:
            return cached

        with self._pending_lock:
            future = self._pending.get(block_id)

        if future is not None:
            try:
                return future.result(timeout=timeout)
            except Exception:
                pass  # fall through to synchronous load

        data = self.load_fn(block_id)
        self.cache.put(block_id, data)
        return data

    def prefetch_ahead(self, current_pos: int, block_order: list[int]) -> None:
        """Schedule the next *prefetch_depth* blocks from *current_pos*."""
        for i in range(1, self.prefetch_depth + 1):
            nxt = current_pos + i
            if nxt < len(block_order):
                self.schedule(block_order[nxt])

    # -- internal -----------------------------------------------------------

    def _load_block(self, block_id: int) -> Any:
        try:
            data = self.load_fn(block_id)
            self.cache.put(block_id, data)
            return data
        finally:
            with self._pending_lock:
                self._pending.pop(block_id, None)


# ---------------------------------------------------------------------------
# Synchronous fallback
# ---------------------------------------------------------------------------

class SyncPrefetcher:
    """Same interface as :class:`AsyncPrefetcher` but loads synchronously.

    Useful for debugging and single-threaded environments.
    """

    def __init__(self, load_fn: Callable[[int], Any], cache_size: int = 2, **_kw: Any) -> None:
        self.load_fn = load_fn
        self.cache = BlockCache(max_blocks=cache_size)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.cache.clear()

    def schedule(self, block_id: int) -> None:
        pass

    def get(self, block_id: int, timeout: float = 30.0) -> Any:
        cached = self.cache.get(block_id)
        if cached is not None:
            return cached
        data = self.load_fn(block_id)
        self.cache.put(block_id, data)
        return data

    def prefetch_ahead(self, current_pos: int, block_order: list[int]) -> None:
        pass
