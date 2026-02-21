"""Tests for CDLSampler."""

import pytest

from cdl.sampler import CDLSampler


class TestCDLSampler:
    def test_len(self):
        sampler = CDLSampler(dataset_size=1000, block_size=100, seed=42)
        assert len(sampler) == 1000

    def test_iter_covers_all_indices(self):
        n = 200
        sampler = CDLSampler(dataset_size=n, block_size=32, seed=42)
        indices = list(sampler)
        assert sorted(indices) == list(range(n))

    def test_set_epoch_changes_order(self):
        sampler = CDLSampler(dataset_size=100, block_size=16, seed=42)
        order0 = list(sampler)
        sampler.set_epoch(1)
        order1 = list(sampler)
        assert order0 != order1

    def test_get_state_roundtrip(self):
        s1 = CDLSampler(dataset_size=500, block_size=64, seed=42)
        s1.set_epoch(3)
        state = s1.get_state()
        s2 = CDLSampler.from_state(state)
        assert list(s1) == list(s2)

    def test_data_source_interface(self):
        """Accepts a sized object (like a PyTorch dataset)."""
        class FakeDataset:
            def __len__(self):
                return 100
        sampler = CDLSampler(data_source=FakeDataset(), block_size=16, seed=42)
        assert len(sampler) == 100
        assert sorted(list(sampler)) == list(range(100))

    def test_requires_either_source_or_size(self):
        with pytest.raises(ValueError):
            CDLSampler()
