"""Tests for CDLLoader with NumpyBlockDataset."""

import numpy as np
import pytest

from cdl.loader import CDLLoader
from cdl.datasets import NumpyBlockDataset


@pytest.fixture
def numpy_dataset():
    data = np.arange(200).reshape(200, 1).astype(np.float32)
    labels = np.arange(200)
    return NumpyBlockDataset(data, labels, block_size=32)


class TestCDLLoader:
    def test_iterates_all_samples(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42, prefetch=False)
        all_items = []
        for batch_data, batch_labels in loader:
            all_items.extend(batch_data.flatten().tolist())
        assert sorted(all_items) == list(range(200))

    def test_batch_size(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42, prefetch=False)
        batches = list(loader)
        # All but last should be full size
        for data, labels in batches[:-1]:
            assert len(data) == 16

    def test_drop_last(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42, drop_last=True, prefetch=False)
        batches = list(loader)
        for data, labels in batches:
            assert len(data) == 16

    def test_len(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42)
        # 200 / 16 = 12.5 → ceil = 13
        assert len(loader) == 13

    def test_set_epoch_changes_order(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=200, seed=42, prefetch=False)
        batch0 = list(loader)[0][0].flatten().tolist()
        loader.set_epoch(1)
        batch1 = list(loader)[0][0].flatten().tolist()
        assert batch0 != batch1

    def test_get_state_roundtrip(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42, prefetch=False)
        loader.set_epoch(5)
        state = loader.get_state()
        assert state["epoch"] == 5
        assert state["seed"] == 42

    def test_io_stats(self, numpy_dataset):
        loader = CDLLoader(numpy_dataset, batch_size=16, seed=42)
        stats = loader.get_io_stats()
        assert stats["dataset_size"] == 200
        assert stats["num_blocks"] == 7  # ceil(200/32)
        assert stats["seek_reduction"] > 1
