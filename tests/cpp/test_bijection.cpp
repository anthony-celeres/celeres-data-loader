/**
 * CDL C++ Tests — Bijection and Shuffle Verification
 *
 * Tests that FeistelPermutation produces valid bijections and
 * IOBlockShuffle covers all indices.
 */

#include <cdl/cdl.hpp>
#include <cassert>
#include <vector>
#include <algorithm>
#include <set>
#include <cstdio>

void test_feistel_bijection(size_t n, uint64_t seed = 42) {
    cdl::FeistelPermutation perm(seed, n);
    std::set<size_t> seen;

    for (size_t i = 0; i < n; ++i) {
        size_t p = perm.permute(i);
        assert(p < n && "Permuted value out of range");
        assert(seen.find(p) == seen.end() && "Duplicate in permutation");
        seen.insert(p);

        // Test inverse
        size_t inv = perm.inverse_permute(p);
        assert(inv == i && "Inverse permutation failed");
    }
    assert(seen.size() == n && "Not all values produced");
}

void test_io_block_shuffle(size_t dataset_size, size_t block_size, uint64_t seed = 42) {
    cdl::IOBlockShuffle shuffler(dataset_size, block_size, seed);

    auto order = shuffler.get_full_iteration_order();
    assert(order.size() == dataset_size && "Full iteration missing items");

    std::set<size_t> seen(order.begin(), order.end());
    assert(seen.size() == dataset_size && "Duplicate indices in iteration");

    for (size_t idx : order) {
        assert(idx < dataset_size && "Index out of range");
    }
}

int main() {
    printf("Testing FeistelPermutation bijection...\n");

    // Test various domain sizes
    size_t sizes[] = {1, 2, 3, 7, 10, 100, 1000, 1023, 1024, 1025, 4096};
    for (size_t n : sizes) {
        test_feistel_bijection(n);
        printf("  n=%zu: PASS\n", n);
    }

    printf("\nTesting IOBlockShuffle coverage...\n");

    // Test various dataset/block combinations
    struct { size_t ds; size_t bs; } combos[] = {
        {100, 10}, {100, 7}, {1000, 128}, {1024, 1024}, {1025, 512}
    };
    for (auto [ds, bs] : combos) {
        test_io_block_shuffle(ds, bs);
        printf("  dataset=%zu, block=%zu: PASS\n", ds, bs);
    }

    printf("\nAll C++ tests passed!\n");
    return 0;
}
