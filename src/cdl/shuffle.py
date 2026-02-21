"""
CDL Shuffle — O(1) Memory Feistel-Based Shuffling

Core shuffling algorithm using a balanced Feistel network with SplitMix64
as the stateless keyed hash (round function).

**Backend selection:**  When the C++ native extension (``_cdl_native``) is
compiled and importable, this module transparently delegates to the C++
implementations for ~100× performance.  The pure-Python classes below
serve as the **fallback** when the extension is not available.

Architecture
------------
Level 1 (Block Order)  — Permutes block IDs for I/O-efficient disk access.
Level 2 (Intra-Block)  — Permutes item offsets within each block (in RAM).

References
----------
- Steele, Lea & Flood (2014). *Fast splittable pseudorandom number generators.*
- Black & Rogaway (2002). *Ciphers with arbitrary finite domains.*
"""

from __future__ import annotations

import math
from typing import Iterator, List, Tuple

# 64-bit mask used throughout to simulate unsigned 64-bit arithmetic.
_U64 = 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# SplitMix64
# ---------------------------------------------------------------------------

class SplitMix64:
    """SplitMix64 — fast 64-bit PRNG and stateless keyed hash.

    CDL uses the *stateless* ``hash(key, data)`` variant as the Feistel
    round function.  The stateful ``next()`` method is retained for
    compatibility but is **not** used by CDL's permutation path.

    Parameters
    ----------
    seed : int
        Initial 64-bit state.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state: int = seed & _U64

    # -- stateful API (canonical PRNG) -------------------------------------

    def next(self) -> int:
        """Advance state and return a 64-bit pseudo-random value."""
        self.state = (self.state + 0x9E37_79B9_7F4A_7C15) & _U64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _U64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _U64
        return (z ^ (z >> 31)) & _U64

    # -- stateless API (used by CDL) ---------------------------------------

    @staticmethod
    def hash(key: int, data: int) -> int:
        """Stateless keyed hash: ``finalize(key ⊕ data + γ)``.

        Same mixing steps as canonical SplitMix64, but the initial state
        is ``key ^ data`` instead of an internal counter.  This makes
        every call independent — the same ``(key, data)`` always produces
        the same output with no side effects.
        """
        z = ((key ^ data) + 0x9E37_79B9_7F4A_7C15) & _U64
        z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & _U64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _U64
        return (z ^ (z >> 31)) & _U64

    @staticmethod
    def derive_key(base_key: int, index: int) -> int:
        """Derive a child key.  Alias for ``hash(base_key, index)``."""
        return SplitMix64.hash(base_key, index)


# ---------------------------------------------------------------------------
# Feistel Permutation
# ---------------------------------------------------------------------------

class FeistelPermutation:
    """Balanced Feistel Network with cycle walking.

    Provides a bijective (1-to-1) permutation over an arbitrary domain
    ``[0, N)`` using O(1) memory and O(1) per-index computation.

    Parameters
    ----------
    key : int
        64-bit permutation key.
    domain_size : int
        *N* — total number of elements to permute.
    rounds : int
        Number of Feistel rounds (default 6; minimum 4 recommended).
    """

    __slots__ = (
        "key", "domain_size", "rounds",
        "half_bits", "half_mask", "full_mask",
        "_round_keys",
    )

    def __init__(self, key: int, domain_size: int, rounds: int = 6) -> None:
        self.key: int = key & _U64
        self.domain_size: int = domain_size
        self.rounds: int = rounds

        # Balanced halves: both halves get equal bit-width.
        if domain_size <= 1:
            self.half_bits = 1
        else:
            total_bits = max(1, math.ceil(math.log2(domain_size)))
            self.half_bits = (total_bits + 1) // 2  # ceil(total_bits / 2)

        self.half_mask: int = (1 << self.half_bits) - 1
        self.full_mask: int = (1 << (2 * self.half_bits)) - 1

        # Pre-compute round keys for speed.
        self._round_keys: List[int] = [
            SplitMix64.derive_key(self.key, r) for r in range(self.rounds)
        ]

    # -- core ---------------------------------------------------------------

    def _feistel_core(self, value: int, *, forward: bool = True) -> int:
        left = (value >> self.half_bits) & self.half_mask
        right = value & self.half_mask

        if forward:
            for r in range(self.rounds):
                f_out = SplitMix64.hash(self._round_keys[r], right) & self.half_mask
                left, right = right, (left ^ f_out) & self.half_mask
        else:
            for r in range(self.rounds - 1, -1, -1):
                f_out = SplitMix64.hash(self._round_keys[r], left) & self.half_mask
                left, right = (right ^ f_out) & self.half_mask, left

        return ((left << self.half_bits) | right) & self.full_mask

    # -- public API ---------------------------------------------------------

    def permute(self, index: int) -> int:
        """Map *index* → permuted index (cycle walking for out-of-range)."""
        if self.domain_size <= 1:
            return 0
        if index >= self.domain_size:
            return index

        result = index
        for _ in range(100):  # safety bound
            result = self._feistel_core(result, forward=True)
            if result < self.domain_size:
                return result
        return index  # fallback (should never happen)

    def inverse_permute(self, shuffled_index: int) -> int:
        """Recover the original index from a permuted index."""
        if self.domain_size <= 1:
            return 0
        if shuffled_index >= self.domain_size:
            return shuffled_index

        result = shuffled_index
        for _ in range(100):
            result = self._feistel_core(result, forward=False)
            if result < self.domain_size:
                return result
        return shuffled_index


# ---------------------------------------------------------------------------
# CDL Two-Level Shuffler
# ---------------------------------------------------------------------------

# Key-derivation domain tags (chosen to be human-readable in hex).
_BLOCK_TAG = 0xB10C      # "BLOC(k)"
_INTRA_TAG = 0x1_0000     # Offset added to block_id for intra-block keys.


class CDLShuffle:
    """Two-level O(1)-memory shuffler for block-based data pipelines.

    Level 1 — *inter-block*: determines **which** blocks to load and in
    what order.  Each block is read sequentially from disk, minimising
    seeks.

    Level 2 — *intra-block*: after a block is loaded into RAM, permutes
    the item offsets so that items are yielded in shuffled order.

    Parameters
    ----------
    dataset_size : int
        Total number of samples in the dataset.
    block_size : int
        Number of samples per storage block.
    seed : int
        Base 64-bit seed.
    rounds : int
        Feistel rounds per permutation (default 6).

    Examples
    --------
    >>> shuffler = CDLShuffle(1_000_000, block_size=1024, seed=42)
    >>> for block_pos in range(shuffler.num_blocks):
    ...     block_id = shuffler.get_block_to_load(block_pos)
    ...     data = disk.read_block(block_id)
    ...     for offset in shuffler.get_intra_block_order(block_id):
    ...         yield data[offset]
    """

    __slots__ = (
        "dataset_size", "block_size", "seed", "rounds",
        "num_blocks", "last_block_size",
        "_block_perm", "_intra_cache",
    )

    def __init__(
        self,
        dataset_size: int,
        block_size: int = 1024,
        seed: int = 42,
        rounds: int = 6,
    ) -> None:
        self.dataset_size = dataset_size
        self.block_size = max(1, block_size)
        self.seed: int = seed & _U64
        self.rounds = rounds

        self.num_blocks = (dataset_size + self.block_size - 1) // self.block_size
        self.last_block_size = dataset_size % self.block_size
        if self.last_block_size == 0 and dataset_size > 0:
            self.last_block_size = self.block_size

        # Level 1 permutation
        block_key = SplitMix64.derive_key(self.seed, _BLOCK_TAG)
        self._block_perm = FeistelPermutation(block_key, self.num_blocks, rounds)

        # Lazy cache for intra-block permutations
        self._intra_cache: dict[int, List[int]] = {}

    # -- Level 1: block order -----------------------------------------------

    def get_block_to_load(self, position: int) -> int:
        """Return the block ID to load at *position* in the epoch."""
        if position >= self.num_blocks:
            return self.num_blocks - 1
        return self._block_perm.permute(position)

    def get_block_order(self) -> List[int]:
        """Return the full list of block IDs in shuffled loading order."""
        return [self._block_perm.permute(i) for i in range(self.num_blocks)]

    def get_block_size(self, block_id: int) -> int:
        """Return the number of items in *block_id*."""
        if block_id >= self.num_blocks:
            return 0
        if block_id == self.num_blocks - 1:
            return self.last_block_size
        return self.block_size

    def get_block_start(self, block_id: int) -> int:
        """Return the global start index of *block_id*."""
        return block_id * self.block_size

    # -- Level 2: intra-block shuffle ---------------------------------------

    def get_intra_block_order(self, block_id: int) -> List[int]:
        """Return shuffled offsets for items within *block_id* (in RAM)."""
        if block_id in self._intra_cache:
            return self._intra_cache[block_id]

        actual_size = self.get_block_size(block_id)
        block_key = SplitMix64.derive_key(self.seed, block_id + _INTRA_TAG)
        perm = FeistelPermutation(block_key, actual_size, self.rounds)
        order = [perm.permute(i) for i in range(actual_size)]

        if actual_size <= 10_000:
            self._intra_cache[block_id] = order
        return order

    def get_item_offset(self, block_id: int, position: int) -> int:
        """Return the shuffled offset for a single position in a block."""
        actual_size = self.get_block_size(block_id)
        if position >= actual_size:
            return 0
        block_key = SplitMix64.derive_key(self.seed, block_id + _INTRA_TAG)
        perm = FeistelPermutation(block_key, actual_size, self.rounds)
        return perm.permute(position)

    # -- full iteration -----------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        """Yield ``(block_id, offset, global_index)`` in I/O order."""
        for pos in range(self.num_blocks):
            block_id = self.get_block_to_load(pos)
            start = self.get_block_start(block_id)
            for offset in self.get_intra_block_order(block_id):
                yield block_id, offset, start + offset

    def get_shuffled_indices(self) -> List[int]:
        """Return all shuffled global indices as a flat list."""
        return [gi for _, _, gi in self]

    def __len__(self) -> int:
        return self.dataset_size

    # -- epoch management ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Derive a new shuffle order for *epoch*.  O(1) operation."""
        new_seed = SplitMix64.derive_key(self.seed, epoch)
        self.__init__(self.dataset_size, self.block_size, new_seed, self.rounds)

    def get_state(self) -> Tuple[int, int, int]:
        """Return minimal checkpoint state (24 bytes)."""
        return (self.seed, self.dataset_size, self.block_size)

    @classmethod
    def from_state(cls, state: Tuple[int, int, int]) -> CDLShuffle:
        """Restore a shuffler from checkpoint state."""
        seed, dataset_size, block_size = state
        return cls(dataset_size, block_size, seed)


# ---------------------------------------------------------------------------
# Native C++ Backend Selection
# ---------------------------------------------------------------------------
# When the native extension is compiled, replace the pure-Python
# implementations with thin wrappers around the C++ classes.
# The public API (class names, method signatures) stays identical.

_PySplitMix64 = SplitMix64
_PyFeistelPermutation = FeistelPermutation
_PyCDLShuffle = CDLShuffle

try:
    from cdl._cdl_native import (  # type: ignore[import-not-found]
        SplitMix64 as _NativeSplitMix64,
        FeistelPermutation as _NativeFeistelPermutation,
        IOBlockShuffle as _NativeIOBlockShuffle,
    )

    # Wrap native classes to match the Python API exactly.

    class SplitMix64:  # type: ignore[no-redef]
        """SplitMix64 — C++ accelerated."""

        __slots__ = ("_native",)

        def __init__(self, seed: int) -> None:
            self._native = _NativeSplitMix64(seed & _U64)

        def next(self) -> int:
            return self._native.next()

        @staticmethod
        def hash(key: int, data: int) -> int:
            return _NativeSplitMix64.hash(key & _U64, data & _U64)

        @staticmethod
        def derive_key(base_key: int, index: int) -> int:
            return _NativeSplitMix64.derive_key(base_key & _U64, index & _U64)

    class FeistelPermutation:  # type: ignore[no-redef]
        """FeistelPermutation — C++ accelerated."""

        __slots__ = ("_native", "domain_size", "key", "rounds")

        def __init__(self, key: int, domain_size: int, rounds: int = 6) -> None:
            self._native = _NativeFeistelPermutation(key & _U64, domain_size, rounds)
            self.domain_size = domain_size
            self.key = key & _U64
            self.rounds = rounds

        def permute(self, index: int) -> int:
            return self._native.permute(index)

        def inverse_permute(self, index: int) -> int:
            return self._native.inverse_permute(index)

    class CDLShuffle:  # type: ignore[no-redef]
        """CDLShuffle — C++ accelerated two-level shuffler."""

        __slots__ = (
            "_native", "dataset_size", "block_size", "seed", "rounds",
            "num_blocks", "last_block_size", "_block_perm", "_intra_cache",
        )

        def __init__(
            self,
            dataset_size: int, block_size: int = 1024,
            seed: int = 42, rounds: int = 6,
        ) -> None:
            self.dataset_size = dataset_size
            self.block_size = max(1, block_size)
            self.seed = seed & _U64
            self.rounds = rounds
            self.num_blocks = (dataset_size + self.block_size - 1) // self.block_size
            lbs = dataset_size % self.block_size
            self.last_block_size = self.block_size if (lbs == 0 and dataset_size > 0) else lbs
            self._native = _NativeIOBlockShuffle(
                dataset_size, self.block_size, self.seed, rounds,
            )
            self._intra_cache: dict[int, List[int]] = {}

        # Level 1
        def get_block_to_load(self, position: int) -> int:
            return self._native.get_block_to_load(position)

        def get_block_order(self) -> List[int]:
            return self._native.get_block_order()

        def get_block_size(self, block_id: int) -> int:
            return self._native.get_block_size(block_id)

        def get_block_start(self, block_id: int) -> int:
            return self._native.get_block_start(block_id)

        # Level 2
        def get_intra_block_order(self, block_id: int) -> List[int]:
            return self._native.get_intra_block_order(block_id)

        def get_item_offset(self, block_id: int, position: int) -> int:
            return self._native.get_item_offset(block_id, position)

        # iteration
        def __iter__(self) -> Iterator[Tuple[int, int, int]]:
            for pos in range(self.num_blocks):
                block_id = self.get_block_to_load(pos)
                start = self.get_block_start(block_id)
                for offset in self.get_intra_block_order(block_id):
                    yield block_id, offset, start + offset

        def get_shuffled_indices(self) -> List[int]:
            return [gi for _, _, gi in self]

        def __len__(self) -> int:
            return self.dataset_size

        # epoch
        def set_epoch(self, epoch: int) -> None:
            new_seed = SplitMix64.derive_key(self.seed, epoch)
            self.__init__(self.dataset_size, self.block_size, new_seed, self.rounds)

        def get_state(self) -> Tuple[int, int, int]:
            return (self.seed, self.dataset_size, self.block_size)

        @classmethod
        def from_state(cls, state: Tuple[int, int, int]) -> "CDLShuffle":
            seed, dataset_size, block_size = state
            return cls(dataset_size, block_size, seed)

    USING_NATIVE = True

except ImportError:
    USING_NATIVE = False
