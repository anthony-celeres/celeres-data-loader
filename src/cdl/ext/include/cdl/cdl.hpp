/**
 * CDL — Celeres Data Loader
 *
 * Main header that includes all CDL components and provides
 * convenience classes.
 *
 * Usage:
 *   #include <cdl/cdl.hpp>
 */

#ifndef CDL_CDL_HPP
#define CDL_CDL_HPP

// Version
#define CDL_VERSION_MAJOR 0
#define CDL_VERSION_MINOR 1
#define CDL_VERSION_PATCH 0
#define CDL_VERSION_STRING "0.1.0"

// Core components
#include "splitmix64.hpp"
#include "feistel.hpp"
#include "io_block_shuffle.hpp"

namespace cdl {

// ============================================================================
// SimpleShuffle — Single-level Feistel shuffle (for comparison)
// ============================================================================

/**
 * Simple single-level shuffle using a Feistel permutation.
 * Does NOT optimize for I/O — included for benchmarking comparison.
 */
class SimpleShuffle {
private:
    FeistelPermutation perm_;
    uint64_t seed_;

public:
    SimpleShuffle(size_t dataset_size, uint64_t seed, size_t rounds = 6)
        : perm_(seed, dataset_size, rounds), seed_(seed) {}

    size_t permute(size_t index) const { return perm_.permute(index); }
    size_t inverse_permute(size_t index) const { return perm_.inverse_permute(index); }
    size_t dataset_size() const { return perm_.size(); }
    uint64_t seed() const { return seed_; }
};

// ============================================================================
// Sampler — PyTorch-compatible O(1) memory sampler
// ============================================================================

/**
 * CDL Sampler for epoch-based training.
 *
 * Provides O(1) memory index generation with block-sequential I/O order.
 * Drop-in replacement for torch.utils.data.RandomSampler.
 */
class Sampler {
private:
    uint64_t base_seed_;
    size_t dataset_size_;
    size_t block_size_;
    size_t rounds_;
    size_t epoch_;
    IOBlockShuffle shuffler_;

public:
    Sampler(size_t dataset_size, size_t block_size = 1024,
            uint64_t seed = 42, size_t rounds = 6)
        : base_seed_(seed)
        , dataset_size_(dataset_size)
        , block_size_(block_size)
        , rounds_(rounds)
        , epoch_(0)
        , shuffler_(dataset_size, block_size,
                   SplitMix64::derive_key(seed, 0), rounds)
    {}

    void set_epoch(size_t epoch) {
        epoch_ = epoch;
        uint64_t epoch_seed = SplitMix64::derive_key(base_seed_, epoch);
        shuffler_ = IOBlockShuffle(dataset_size_, block_size_, epoch_seed, rounds_);
    }

    size_t operator[](size_t i) const {
        return shuffler_.permute(i);
    }

    // Accessors
    size_t size() const { return dataset_size_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return shuffler_.num_blocks(); }
    size_t epoch() const { return epoch_; }
    uint64_t seed() const { return base_seed_; }

    size_t get_block_to_load(size_t position) const {
        return shuffler_.get_block_to_load(position);
    }

    std::vector<size_t> get_intra_block_order(size_t block_id) const {
        return shuffler_.get_intra_block_order(block_id);
    }
};

} // namespace cdl

#endif // CDL_CDL_HPP
