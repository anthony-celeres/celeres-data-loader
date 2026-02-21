"""
CDL Loader — I/O-efficient data loading with O(1)-memory shuffling.

Combines CDL's Feistel-based shuffle with block-aware sequential disk
reads and async prefetching.  This is the primary user-facing class for
full I/O-optimised training pipelines.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from cdl.shuffle import CDLShuffle, SplitMix64
from cdl.datasets import BlockDataset
from cdl.prefetch import AsyncPrefetcher, SyncPrefetcher


class CDLLoader:
    """O(1)-memory, I/O-efficient data loader.

    Parameters
    ----------
    dataset : BlockDataset
        A block-aware dataset adapter.
    batch_size : int
        Samples per batch.
    seed : int
        Base random seed.
    prefetch : bool
        Enable background block prefetching.
    prefetch_depth : int
        Blocks to prefetch ahead.
    num_workers : int
        Background loading threads.
    drop_last : bool
        Drop the last incomplete batch.
    collate_fn : callable or None
        Custom batch collation; defaults to NumPy stacking.

    Examples
    --------
    >>> from cdl import CDLLoader, HDF5BlockDataset
    >>> dataset = HDF5BlockDataset("train.h5")
    >>> loader = CDLLoader(dataset, batch_size=64, seed=42)
    >>> for epoch in range(100):
    ...     loader.set_epoch(epoch)   # O(1)
    ...     for images, labels in loader:
    ...         train(images, labels)
    """

    def __init__(
        self,
        dataset: BlockDataset,
        batch_size: int = 32,
        seed: int = 42,
        prefetch: bool = True,
        prefetch_depth: int = 2,
        num_workers: int = 2,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.collate_fn: Callable = collate_fn or self._default_collate

        self.shuffler = CDLShuffle(
            dataset_size=len(dataset),
            block_size=dataset.block_size,
            seed=seed,
        )

        if prefetch:
            self.prefetcher = AsyncPrefetcher(
                load_fn=dataset.load_block,
                prefetch_depth=prefetch_depth,
                cache_size=prefetch_depth + 2,
                num_workers=num_workers,
            )
        else:
            self.prefetcher = SyncPrefetcher(
                load_fn=dataset.load_block,
                cache_size=2,
            )

        self._epoch = 0
        self._block_order: Optional[List[int]] = None

    # -- epoch management ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Derive a new shuffle order for *epoch*.  O(1) operation."""
        self._epoch = epoch
        self.shuffler = CDLShuffle(
            dataset_size=len(self.dataset),
            block_size=self.dataset.block_size,
            seed=SplitMix64.derive_key(self.seed, epoch),
        )
        self._block_order = None

    # -- iteration ----------------------------------------------------------

    @property
    def block_order(self) -> List[int]:
        if self._block_order is None:
            self._block_order = self.shuffler.get_block_order()
        return self._block_order

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        self.prefetcher.start()
        try:
            yield from self._iterate_batches()
        finally:
            self.prefetcher.stop()

    def _iterate_batches(self) -> Iterator[Any]:
        batch: list[Any] = []
        order = self.block_order

        for pos, block_id in enumerate(order):
            self.prefetcher.prefetch_ahead(pos, order)
            block_data = self.prefetcher.get(block_id)
            intra_order = self.shuffler.get_intra_block_order(block_id)

            for offset in intra_order:
                batch.append(self.dataset.get_item(block_data, offset))
                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    # -- collation ----------------------------------------------------------

    @staticmethod
    def _default_collate(batch: List[Tuple]) -> Tuple:
        if not batch:
            return (np.array([]), None)
        items = [b[0] for b in batch]
        labels = [b[1] for b in batch if b[1] is not None]
        if isinstance(items[0], np.ndarray):
            items = np.stack(items)
        if labels:
            labels = np.stack(labels) if isinstance(labels[0], np.ndarray) else np.array(labels)
        else:
            labels = None
        return (items, labels)

    # -- checkpointing ------------------------------------------------------

    def get_state(self) -> dict:
        """Return minimal checkpoint dict (~24 bytes of essential state)."""
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "dataset_size": len(self.dataset),
            "block_size": self.dataset.block_size,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, dataset: BlockDataset, **kwargs: Any) -> CDLLoader:
        """Restore a loader from a checkpoint dict."""
        loader = cls(dataset=dataset, seed=checkpoint["seed"], **kwargs)
        loader.set_epoch(checkpoint["epoch"])
        return loader

    # -- diagnostics --------------------------------------------------------

    def get_io_stats(self) -> dict:
        """Return I/O efficiency statistics."""
        n = len(self.dataset)
        nb = self.shuffler.num_blocks
        bs = self.dataset.block_size
        return {
            "dataset_size": n,
            "num_blocks": nb,
            "block_size": bs,
            "expected_locality_pct": (bs - 1) / bs * 100 if bs > 0 else 0,
            "seeks_per_epoch": nb,
            "seeks_random_shuffle": n,
            "seek_reduction": n / nb if nb > 0 else 1,
        }


# ---------------------------------------------------------------------------
# Optional PyTorch tensor collation
# ---------------------------------------------------------------------------

try:
    import torch

    def torch_collate(batch: List[Tuple]) -> Tuple:
        """Collate into ``torch.Tensor`` s."""
        if not batch:
            return (torch.tensor([]), None)
        items = [b[0] for b in batch]
        labels = [b[1] for b in batch if b[1] is not None]
        if isinstance(items[0], np.ndarray):
            items = torch.from_numpy(np.stack(items))
        elif isinstance(items[0], torch.Tensor):
            items = torch.stack(items)
        if labels:
            if isinstance(labels[0], (int, float)):
                labels = torch.tensor(labels)
            elif isinstance(labels[0], np.ndarray):
                labels = torch.from_numpy(np.stack(labels))
        else:
            labels = None
        return (items, labels)

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    torch_collate = None  # type: ignore[assignment]
