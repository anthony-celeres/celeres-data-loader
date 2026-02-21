"""Tests for the core shuffle module (SplitMix64, FeistelPermutation, CDLShuffle)."""

import pytest

from cdl.shuffle import SplitMix64, FeistelPermutation, CDLShuffle


# ---------------------------------------------------------------------------
# SplitMix64
# ---------------------------------------------------------------------------

class TestSplitMix64:
    def test_hash_deterministic(self):
        """Same (key, data) always produces the same output."""
        assert SplitMix64.hash(42, 7) == SplitMix64.hash(42, 7)

    def test_hash_varies_with_key(self):
        assert SplitMix64.hash(1, 0) != SplitMix64.hash(2, 0)

    def test_hash_varies_with_data(self):
        assert SplitMix64.hash(0, 1) != SplitMix64.hash(0, 2)

    def test_derive_key_is_hash(self):
        assert SplitMix64.derive_key(42, 7) == SplitMix64.hash(42, 7)

    def test_output_is_64_bit(self):
        result = SplitMix64.hash(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF)
        assert 0 <= result < 2**64

    def test_stateful_next_advances_state(self):
        rng = SplitMix64(42)
        a = rng.next()
        b = rng.next()
        assert a != b


# ---------------------------------------------------------------------------
# FeistelPermutation
# ---------------------------------------------------------------------------

class TestFeistelPermutation:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 100, 1000, 1023, 1024, 1025])
    def test_bijective(self, n):
        """Every input maps to a unique output within [0, N)."""
        perm = FeistelPermutation(key=42, domain_size=n)
        outputs = {perm.permute(i) for i in range(n)}
        assert outputs == set(range(n))

    @pytest.mark.parametrize("n", [2, 10, 100, 1024])
    def test_inverse(self, n):
        """inverse_permute(permute(i)) == i for all i."""
        perm = FeistelPermutation(key=42, domain_size=n)
        for i in range(n):
            assert perm.inverse_permute(perm.permute(i)) == i

    def test_different_keys_give_different_permutations(self):
        n = 100
        p1 = [FeistelPermutation(key=1, domain_size=n).permute(i) for i in range(n)]
        p2 = [FeistelPermutation(key=2, domain_size=n).permute(i) for i in range(n)]
        assert p1 != p2

    def test_domain_size_one(self):
        perm = FeistelPermutation(key=42, domain_size=1)
        assert perm.permute(0) == 0

    def test_domain_size_zero(self):
        perm = FeistelPermutation(key=42, domain_size=0)
        assert perm.permute(0) == 0


# ---------------------------------------------------------------------------
# CDLShuffle
# ---------------------------------------------------------------------------

class TestCDLShuffle:
    def test_produces_all_indices(self):
        """Every global index appears exactly once."""
        n = 100
        shuffler = CDLShuffle(dataset_size=n, block_size=10, seed=42)
        indices = shuffler.get_shuffled_indices()
        assert sorted(indices) == list(range(n))

    def test_different_seeds_different_order(self):
        n = 100
        s1 = CDLShuffle(n, block_size=10, seed=1).get_shuffled_indices()
        s2 = CDLShuffle(n, block_size=10, seed=2).get_shuffled_indices()
        assert s1 != s2

    def test_set_epoch_changes_order(self):
        shuffler = CDLShuffle(100, block_size=10, seed=42)
        order0 = shuffler.get_shuffled_indices()
        shuffler.set_epoch(1)
        order1 = shuffler.get_shuffled_indices()
        assert order0 != order1

    def test_block_order_covers_all_blocks(self):
        shuffler = CDLShuffle(100, block_size=10, seed=42)
        assert sorted(shuffler.get_block_order()) == list(range(10))

    def test_intra_block_covers_all_offsets(self):
        shuffler = CDLShuffle(100, block_size=10, seed=42)
        order = shuffler.get_intra_block_order(0)
        assert sorted(order) == list(range(10))

    def test_get_state_roundtrip(self):
        s1 = CDLShuffle(1000, block_size=32, seed=42)
        state = s1.get_state()
        s2 = CDLShuffle.from_state(state)
        assert s1.get_shuffled_indices() == s2.get_shuffled_indices()

    def test_last_block_handles_remainder(self):
        shuffler = CDLShuffle(dataset_size=105, block_size=10, seed=42)
        assert shuffler.num_blocks == 11
        assert shuffler.last_block_size == 5
        # All indices still covered
        assert sorted(shuffler.get_shuffled_indices()) == list(range(105))

    def test_len(self):
        shuffler = CDLShuffle(1000, block_size=100, seed=42)
        assert len(shuffler) == 1000

    def test_iter_yields_tuples(self):
        shuffler = CDLShuffle(20, block_size=5, seed=42)
        for block_id, offset, global_idx in shuffler:
            assert 0 <= block_id < shuffler.num_blocks
            assert 0 <= offset < shuffler.block_size
            assert 0 <= global_idx < 20
