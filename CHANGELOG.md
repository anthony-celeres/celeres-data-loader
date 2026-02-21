# Changelog

All notable changes to CDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-02-21

### Added

- Core Feistel permutation with cycle walking (`FeistelPermutation`).
- SplitMix64 stateless keyed hash (`SplitMix64`).
- Two-level CDL shuffler (`CDLShuffle`).
- PyTorch-compatible sampler (`CDLSampler`).
- Full I/O-optimised loader with async prefetching (`CDLLoader`).
- Block-aware dataset adapters: `HDF5BlockDataset`, `DirectoryBlockDataset`, `NumpyBlockDataset`.
- Format-aware block-size inference from HDF5 chunks, directory shards, and Parquet row groups.
- Async block prefetcher with LRU cache (`AsyncPrefetcher`).
- 24-byte checkpointing and O(1) epoch transitions.
