"""
CDL Sampler — PyTorch-compatible drop-in for RandomSampler.

Provides O(1)-memory index generation with block-sequential I/O order.
Works with ``torch.utils.data.DataLoader`` exactly like the built-in
samplers — just set ``sampler=`` instead of ``shuffle=True``.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sized

from cdl.shuffle import CDLShuffle, SplitMix64


class CDLSampler:
    """O(1)-memory sampler backed by CDL's Feistel permutation.

    Parameters
    ----------
    data_source : Sized, optional
        Dataset (used for ``len()``).  Provide either this or *dataset_size*.
    dataset_size : int, optional
        Explicit dataset size (alternative to *data_source*).
    block_size : int
        Storage block size — controls I/O locality.
    seed : int
        Base random seed.
    rounds : int
        Feistel rounds (default 6).

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> sampler = CDLSampler(dataset, block_size=1024, seed=42)
    >>> loader = DataLoader(dataset, sampler=sampler, batch_size=64)
    >>> for epoch in range(100):
    ...     sampler.set_epoch(epoch)   # O(1)!
    ...     for batch in loader:
    ...         train(batch)
    """

    def __init__(
        self,
        data_source: Optional[Sized] = None,
        dataset_size: Optional[int] = None,
        block_size: int = 1024,
        seed: int = 42,
        rounds: int = 6,
    ) -> None:
        if data_source is not None:
            self._dataset_size = len(data_source)
        elif dataset_size is not None:
            self._dataset_size = dataset_size
        else:
            raise ValueError("Must provide either data_source or dataset_size")

        self.block_size = block_size
        self.seed = seed
        self.rounds = rounds
        self._epoch = 0

        self._shuffler = CDLShuffle(
            dataset_size=self._dataset_size,
            block_size=block_size,
            seed=seed,
            rounds=rounds,
        )

    # -- core ---------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Derive a new shuffle order for *epoch*.  O(1) operation."""
        self._epoch = epoch
        epoch_seed = SplitMix64.derive_key(self.seed, epoch)
        self._shuffler = CDLShuffle(
            dataset_size=self._dataset_size,
            block_size=self.block_size,
            seed=epoch_seed,
            rounds=self.rounds,
        )

    def __iter__(self) -> Iterator[int]:
        for _, _, global_idx in self._shuffler:
            yield global_idx

    def __len__(self) -> int:
        return self._dataset_size

    # -- convenience --------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def num_blocks(self) -> int:
        return self._shuffler.num_blocks

    def get_shuffled_indices(self) -> list[int]:
        """Return all shuffled indices as a list."""
        return list(self)

    def get_block_order(self) -> list[int]:
        """Return block IDs in shuffled loading order."""
        return self._shuffler.get_block_order()

    # -- checkpointing ------------------------------------------------------

    def get_state(self) -> dict:
        """Return minimal checkpoint dict (~24 bytes of essential state)."""
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "dataset_size": self._dataset_size,
            "block_size": self.block_size,
            "rounds": self.rounds,
        }

    @classmethod
    def from_state(cls, state: dict) -> CDLSampler:
        """Restore a sampler from a checkpoint dict."""
        sampler = cls(
            dataset_size=state["dataset_size"],
            block_size=state["block_size"],
            seed=state["seed"],
            rounds=state.get("rounds", 6),
        )
        sampler.set_epoch(state["epoch"])
        return sampler


# ---------------------------------------------------------------------------
# PyTorch integration (optional — only when torch is installed)
# ---------------------------------------------------------------------------

try:
    from torch.utils.data import Sampler

    class TorchCDLSampler(Sampler, CDLSampler):  # type: ignore[type-arg]
        """CDLSampler that also inherits ``torch.utils.data.Sampler``.

        Provides full PyTorch type-checking compatibility.
        """

        def __init__(
            self,
            data_source: Sized,
            block_size: int = 1024,
            seed: int = 42,
            rounds: int = 6,
        ) -> None:
            CDLSampler.__init__(
                self,
                data_source=data_source,
                block_size=block_size,
                seed=seed,
                rounds=rounds,
            )

    TORCH_SAMPLER_AVAILABLE = True

except ImportError:
    TorchCDLSampler = CDLSampler  # type: ignore[misc,assignment]
    TORCH_SAMPLER_AVAILABLE = False
