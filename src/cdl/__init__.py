"""
CDL — Celeres Data Loader

Constant-memory shuffling and I/O-efficient data loading for deep learning.

CDL replaces traditional O(N) Fisher-Yates shuffling with an O(1) Feistel
permutation, enabling block-sequential disk access patterns while preserving
statistical shuffle quality. The result: 24 bytes of shuffle state instead
of gigabytes, 1000× fewer disk seeks, and instant epoch transitions.

Quick Start
-----------
As a PyTorch sampler (drop-in replacement for RandomSampler):

    >>> from cdl import CDLSampler
    >>> sampler = CDLSampler(dataset, block_size=1024, seed=42)
    >>> loader = DataLoader(dataset, sampler=sampler, batch_size=64)
    >>> for epoch in range(100):
    ...     sampler.set_epoch(epoch)   # O(1) — just derives a new key
    ...     for batch in loader:
    ...         train(batch)

Full I/O-optimised loader (with async prefetch):

    >>> from cdl import CDLLoader, HDF5BlockDataset
    >>> dataset = HDF5BlockDataset("train.h5")
    >>> loader = CDLLoader(dataset, batch_size=64, seed=42)
    >>> for epoch in range(100):
    ...     loader.set_epoch(epoch)
    ...     for batch in loader:
    ...         train(batch)
"""

__version__ = "0.1.0"

# Core algorithm ----------------------------------------------------------
from cdl.shuffle import SplitMix64, FeistelPermutation, CDLShuffle

# PyTorch integration -----------------------------------------------------
from cdl.sampler import CDLSampler

# I/O pipeline ------------------------------------------------------------
from cdl.loader import CDLLoader
from cdl.datasets import (
    BlockDataset,
    HDF5BlockDataset,
    DirectoryBlockDataset,
    NumpyBlockDataset,
    infer_block_size,
)
from cdl.prefetch import AsyncPrefetcher, BlockCache

__all__ = [
    # Core
    "SplitMix64",
    "FeistelPermutation",
    "CDLShuffle",
    # PyTorch
    "CDLSampler",
    # I/O
    "CDLLoader",
    "BlockDataset",
    "HDF5BlockDataset",
    "DirectoryBlockDataset",
    "NumpyBlockDataset",
    "infer_block_size",
    "AsyncPrefetcher",
    "BlockCache",
]
