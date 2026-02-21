"""
CDL Quick Start Example

Demonstrates CDL's three main usage patterns:
1. CDLShuffle  — direct index permutation (lowest level)
2. CDLSampler  — PyTorch DataLoader drop-in
3. CDLLoader   — full I/O-optimised pipeline
"""

import numpy as np

from cdl import CDLShuffle, CDLSampler, CDLLoader, NumpyBlockDataset


def demo_shuffle():
    """Low-level: generate shuffled indices directly."""
    print("=" * 60)
    print("1. CDLShuffle — O(1) memory index permutation")
    print("=" * 60)

    dataset_size = 1_000_000
    block_size = 1024
    seed = 42

    shuffler = CDLShuffle(dataset_size, block_size, seed)

    print(f"  Dataset size   : {dataset_size:,}")
    print(f"  Block size     : {block_size}")
    print(f"  Num blocks     : {shuffler.num_blocks:,}")
    print(f"  Shuffle state  : 24 bytes  (vs {dataset_size * 8 / 1e6:.0f} MB for Fisher-Yates)")
    print()

    # Show block loading order (first 10)
    order = shuffler.get_block_order()[:10]
    print(f"  Block order (first 10): {order}")

    # Show intra-block shuffle (first block)
    intra = shuffler.get_intra_block_order(order[0])[:10]
    print(f"  Intra-block offsets   : {intra}")
    print()


def demo_sampler():
    """Mid-level: PyTorch sampler replacement."""
    print("=" * 60)
    print("2. CDLSampler — PyTorch drop-in replacement")
    print("=" * 60)

    sampler = CDLSampler(dataset_size=10_000, block_size=256, seed=42)

    indices = list(sampler)
    print(f"  Total indices : {len(indices):,}")
    print(f"  All unique    : {len(set(indices)) == len(indices)}")
    print(f"  First 10      : {indices[:10]}")

    # Epoch transition
    sampler.set_epoch(1)
    epoch1_indices = list(sampler)
    print(f"  Epoch 1 first : {epoch1_indices[:10]}")
    print(f"  Different     : {indices[:10] != epoch1_indices[:10]}")

    # Checkpointing
    state = sampler.get_state()
    print(f"  Checkpoint    : {state}")
    print()


def demo_loader():
    """Full pipeline: loader with prefetch and batching."""
    print("=" * 60)
    print("3. CDLLoader — full I/O-optimised pipeline")
    print("=" * 60)

    # Create a synthetic dataset
    data = np.random.randn(500, 3, 32, 32).astype(np.float32)
    labels = np.random.randint(0, 10, size=500)
    dataset = NumpyBlockDataset(data, labels, block_size=64)

    loader = CDLLoader(dataset, batch_size=32, seed=42, prefetch=False)

    print(f"  Dataset size  : {len(dataset):,}")
    print(f"  Block size    : {dataset.block_size}")
    print(f"  Batches/epoch : {len(loader)}")

    # Iterate one epoch
    total_samples = 0
    for batch_data, batch_labels in loader:
        total_samples += len(batch_data)

    print(f"  Samples seen  : {total_samples}")

    # I/O stats
    stats = loader.get_io_stats()
    print(f"  Seek reduction: {stats['seek_reduction']:.0f}×")
    print(f"  I/O locality  : {stats['expected_locality_pct']:.1f}%")
    print()


if __name__ == "__main__":
    demo_shuffle()
    demo_sampler()
    demo_loader()
    print("[OK] All demos completed successfully.")
