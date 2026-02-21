/**
 * CDL — Celeres Data Loader
 *
 * I/O-Aware Block Shuffle
 *
 * Implements the core two-level shuffle that makes CDL's data loading
 * I/O-efficient while maintaining full permutation quality.
 *
 * Level 1 (Inter-block): Shuffles which blocks to load from disk.
 *   → Determines the ORDER blocks are read (sequential I/O within blocks).
 *
 * Level 2 (Intra-block): Shuffles items within a loaded block.
 *   → Applied AFTER the block is in memory (no I/O penalty).
 */

#ifndef CDL_IO_BLOCK_SHUFFLE_HPP
#define CDL_IO_BLOCK_SHUFFLE_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include "feistel.hpp"

namespace cdl {

/**
 * IOBlockShuffle — Two-level I/O-aware shuffler
 */
class IOBlockShuffle {
private:
    size_t dataset_size_;
    size_t block_size_;
    size_t num_blocks_;
    uint64_t seed_;
    size_t rounds_;
    FeistelPermutation block_order_;   // Level 1: inter-block shuffle

public:
    IOBlockShuffle(size_t dataset_size, size_t block_size, uint64_t seed, size_t rounds = 6)
        : dataset_size_(dataset_size)
        , block_size_(block_size)
        , num_blocks_((dataset_size + block_size - 1) / block_size)
        , seed_(seed)
        , rounds_(rounds)
        , block_order_(SplitMix64::derive_key(seed, 0), num_blocks_, rounds)
    {}

    // ========================================================================
    // Level 1 — Block loading order
    // ========================================================================

    /** Which block to load at position `pos` in the epoch */
    size_t get_block_to_load(size_t position) const {
        return block_order_.permute(position);
    }

    /** Full shuffled block order */
    std::vector<size_t> get_block_order() const {
        std::vector<size_t> order(num_blocks_);
        for (size_t i = 0; i < num_blocks_; ++i)
            order[i] = block_order_.permute(i);
        return order;
    }

    /** Actual size of block (last block may be smaller) */
    size_t get_block_size(size_t block_id) const {
        size_t start = block_id * block_size_;
        return std::min(block_size_, dataset_size_ - std::min(start, dataset_size_));
    }

    /** Starting global index of a block */
    size_t get_block_start(size_t block_id) const {
        return block_id * block_size_;
    }

    // ========================================================================
    // Level 2 — Intra-block shuffle (applied in memory)
    // ========================================================================

    /** Shuffled offsets for items within a block */
    std::vector<size_t> get_intra_block_order(size_t block_id) const {
        size_t actual = get_block_size(block_id);
        if (actual == 0) return {};

        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);

        std::vector<size_t> order(actual);
        for (size_t i = 0; i < actual; ++i)
            order[i] = item_perm.permute(i);
        return order;
    }

    /** Single item offset within a block */
    size_t get_item_offset(size_t block_id, size_t position) const {
        size_t actual = get_block_size(block_id);
        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);
        return item_perm.permute(position);
    }

    /** Shuffle a data array in-place using the intra-block permutation */
    template<typename T>
    void shuffle_block_inplace(size_t block_id, T* data, size_t count) const {
        auto order = get_intra_block_order(block_id);
        size_t n = std::min(count, order.size());

        std::vector<T> temp(data, data + n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = temp[order[i]];
        }
    }

    // ========================================================================
    // Legacy — Global permutation interface
    // ========================================================================

    /** Map global index through two-level permutation */
    size_t permute(size_t index) const {
        size_t block_pos = index / block_size_;
        size_t offset_pos = index % block_size_;
        size_t block_id = block_order_.permute(block_pos);

        size_t actual = get_block_size(block_id);
        size_t clamped = std::min(offset_pos, actual > 0 ? actual - 1 : 0u);

        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);
        size_t offset = item_perm.permute(clamped);

        return block_id * block_size_ + offset;
    }

    /** Inverse of global permutation */
    size_t inverse_permute(size_t index) const {
        size_t block_id = index / block_size_;
        size_t offset = index % block_size_;

        size_t actual = get_block_size(block_id);
        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);
        size_t orig_offset = item_perm.inverse_permute(offset);

        size_t block_pos = block_order_.inverse_permute(block_id);
        return block_pos * block_size_ + orig_offset;
    }

    // ========================================================================
    // Full iteration order (for testing/debugging)
    // ========================================================================

    /** Get complete iteration order: for each block in shuffled order, yield shuffled items */
    std::vector<size_t> get_full_iteration_order() const {
        std::vector<size_t> result;
        result.reserve(dataset_size_);

        for (size_t pos = 0; pos < num_blocks_; ++pos) {
            size_t block_id = block_order_.permute(pos);
            size_t start = block_id * block_size_;
            auto intra = get_intra_block_order(block_id);

            for (size_t offset : intra) {
                size_t global = start + offset;
                if (global < dataset_size_)
                    result.push_back(global);
            }
        }
        return result;
    }

    /** Expected locality factor */
    double expected_locality_factor() const {
        return static_cast<double>(block_size_) / dataset_size_;
    }

    // Accessors
    size_t dataset_size() const { return dataset_size_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }
    uint64_t seed() const { return seed_; }
};

} // namespace cdl

#endif // CDL_IO_BLOCK_SHUFFLE_HPP
