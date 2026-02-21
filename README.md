# CDL — Celeres Data Loader

**Constant-memory shuffling and I/O-efficient data loading for deep learning.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-cdl-orange.svg)]()

---

## The Problem

Training large models requires shuffling billions of samples every epoch.
Traditional shuffling creates two scaling bottlenecks:

| | Fisher-Yates Shuffle | CDL |
|---|---|---|
| **Shuffle memory** | O(N) — 8 GB for 1B samples | **O(1) — 24 bytes** |
| **Disk seeks per epoch** | N random seeks | **N / block_size** sequential reads |
| **Epoch initialisation** | O(N) permutation | **O(1) key derivation** |
| **Checkpoint size** | ~8 GB shuffle state | **24 bytes** |
| **Distributed sync** | Broadcast O(N) array | **Zero sync** (same seed) |

## How It Works

CDL replaces the O(N) permutation array with a **two-level Feistel network**
that computes shuffled indices on the fly:

```
Level 1 — Block Order               Level 2 — Intra-Block
┌──────────────────────┐            ┌──────────────────────┐
│ Block 0 ──────►  B3  │            │ Offset 0 ──────► O4  │
│ Block 1 ──────►  B0  │            │ Offset 1 ──────► O2  │
│ Block 2 ──────►  B4  │            │ Offset 2 ──────► O0  │
│  ...                 │            │   ...                │
│ Block N ──────►  B1  │            │ Offset B ──────► O3  │
└──────────────────────┘            └──────────────────────┘
  Shuffles WHICH blocks               Shuffles items WITHIN
  to read from disk                    each block (in RAM)
```

The disk reads stay **sequential within each block** (minimising seeks),
while items are still **fully shuffled** across the epoch.

---

## Quick Start

### Install

```bash
pip install cdl
```

### As a PyTorch Sampler (drop-in replacement)

```python
from torch.utils.data import DataLoader
from cdl import CDLSampler

dataset = torchvision.datasets.ImageNet(root="./data", split="train")

sampler = CDLSampler(dataset, block_size=1024, seed=42)
loader = DataLoader(dataset, sampler=sampler, batch_size=128)

for epoch in range(100):
    sampler.set_epoch(epoch)   # O(1) — instant!
    for batch in loader:
        train(batch)
```

### Full I/O-Optimised Loader

```python
from cdl import CDLLoader, HDF5BlockDataset

dataset = HDF5BlockDataset("train.h5")          # block size auto-inferred
loader = CDLLoader(dataset, batch_size=64, seed=42)

for epoch in range(100):
    loader.set_epoch(epoch)                      # 24-byte state change
    for images, labels in loader:
        train(images, labels)

    state = loader.get_state()                   # checkpoint: 24 bytes!
    save_checkpoint(model, state)
```

---

## Benchmarks

### Memory Usage

| Dataset Size | Fisher-Yates | CDL | Savings |
|---|---|---|---|
| 1 Million | 8 MB | 24 B | 99.9997% |
| 100 Million | 800 MB | 24 B | 99.9999% |
| 1 Billion | 8 GB | 24 B | 99.9999% |

### I/O Efficiency

| Metric | Random Shuffle | CDL (block=1024) |
|---|---|---|
| **I/O Locality** | 0.03–0.43% | 98–99% |
| **Disk Seeks / Epoch** | N | N / 1024 |
| **HDD Speedup** | 1× | 25–30× |

### Shuffle Quality

| Metric | CDL | vs Random |
|---|---|---|
| **Batch Entropy** | 96.85% | 0.03% gap |
| **Training Accuracy** (CIFAR-10) | 69.79% | +0.50% vs 69.29% |

---

## API Reference

### `CDLSampler`

```python
from cdl import CDLSampler

sampler = CDLSampler(
    data_source=dataset,     # or dataset_size=N
    block_size=1024,
    seed=42,
    rounds=6,
)

sampler.set_epoch(epoch)     # O(1)
state = sampler.get_state()  # 24-byte checkpoint
```

### `CDLLoader`

```python
from cdl import CDLLoader, HDF5BlockDataset

loader = CDLLoader(
    dataset=HDF5BlockDataset("data.h5"),
    batch_size=64,
    seed=42,
    prefetch=True,           # async block prefetching
    prefetch_depth=2,
)
```

### Block Datasets

| Adapter | Auto-infers block size from |
|---|---|
| `HDF5BlockDataset` | HDF5 chunk metadata |
| `DirectoryBlockDataset` | Subdirectory (shard) file counts |
| `NumpyBlockDataset` | User-specified (in-memory) |

---

## Architecture

```
src/cdl/
├── __init__.py       # Public API surface
├── shuffle.py        # Core: SplitMix64, FeistelPermutation, CDLShuffle
├── sampler.py        # PyTorch Sampler integration
├── loader.py         # Full I/O-optimised loader
├── datasets.py       # Block-aware dataset adapters
├── prefetch.py       # Async block prefetcher + LRU cache
└── ext/              # Optional C++ native extension
    └── __init__.py
```

### Algorithm

```
Input index X
├── Split into (Left, Right) halves
├── For r = 0 to 5:
│   ├── F = SplitMix64.hash(RoundKey[r], Right) & half_mask
│   ├── new_LEFT  = Right
│   └── new_RIGHT = (Left ⊕ F) & half_mask
├── Combine: result = (Left << half_bits) | Right
└── Cycle walk: if result >= N, re-apply
```

---

## When CDL Helps Most

| Scenario | Impact |
|---|---|
| GPU + HDD storage | ⭐⭐⭐⭐⭐ Massive speedup |
| GPU + Network (S3, NFS) | ⭐⭐⭐⭐⭐ Massive speedup |
| Memory-constrained (>100M samples) | ⭐⭐⭐⭐⭐ Essential |
| Distributed training | ⭐⭐⭐⭐ Stateless checkpointing |
| GPU + NVMe SSD | ⭐⭐ Moderate benefit |

---

## Citation

```bibtex
@thesis{cdl2026,
  title   = {CDL: A Block-Based Constant-Memory Shuffling System
             for I/O-Efficient Deep Learning Data Pipelines},
  author  = {Your Name},
  year    = {2026},
  school  = {Your University},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## References

1. Steele, G. L., Lea, D., & Flood, C. H. (2014). *Fast splittable pseudorandom number generators.* OOPSLA 2014.
2. Black, J., & Rogaway, P. (2002). *Ciphers with arbitrary finite domains.* CT-RSA 2002.
3. Morris, B., Rogaway, P., & Stegers, T. (2009). *How to encipher messages on a small domain.* CRYPTO 2009.
