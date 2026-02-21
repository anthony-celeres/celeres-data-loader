/**
 * CDL Python Bindings (pybind11)
 *
 * Exposes the C++ core as `_cdl_native` for use by the Python package.
 * The Python layer in `cdl.shuffle` auto-detects and prefers these
 * native implementations when available.
 *
 * Build:
 *   pip install pybind11
 *   cd CDL && mkdir build && cd build
 *   cmake .. -DCDL_BUILD_PYTHON=ON
 *   cmake --build .
 */

// MinGW workaround: include these before pybind11
#include <mutex>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cdl/cdl.hpp>
#include <vector>

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Generate a batch of shuffled indices as numpy array
 */
py::array_t<size_t> get_shuffled_indices(
    uint64_t seed,
    size_t dataset_size,
    size_t block_size,
    size_t start_idx,
    size_t count
) {
    cdl::IOBlockShuffle shuffler(dataset_size, block_size, seed);

    py::array_t<size_t> result(count);
    auto buf = result.mutable_unchecked<1>();

    for (size_t i = 0; i < count; ++i) {
        buf(i) = shuffler.permute(start_idx + i);
    }

    return result;
}

/**
 * Generate all shuffled indices for a dataset
 */
py::array_t<size_t> get_all_shuffled_indices(
    uint64_t seed,
    size_t dataset_size,
    size_t block_size
) {
    return get_shuffled_indices(seed, dataset_size, block_size, 0, dataset_size);
}

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(_cdl_native, m) {
    m.doc() = R"doc(
CDL — Celeres Data Loader (Native C++ Extension)

High-performance C++ core for O(1)-memory shuffling.
This module is auto-detected by `cdl.shuffle` when available.

Do not import directly — use `from cdl import CDLShuffle` instead.
)doc";

    // Version info
    m.attr("__version__") = CDL_VERSION_STRING;

    // ========================================================================
    // Utility Functions
    // ========================================================================
    m.def("get_shuffled_indices", &get_shuffled_indices,
        py::arg("seed"),
        py::arg("dataset_size"),
        py::arg("block_size"),
        py::arg("start_idx"),
        py::arg("count"),
        "Generate a batch of shuffled indices as numpy array");

    m.def("get_all_shuffled_indices", &get_all_shuffled_indices,
        py::arg("seed"),
        py::arg("dataset_size"),
        py::arg("block_size"),
        "Generate all shuffled indices for a dataset");

    // ========================================================================
    // SplitMix64 — Core PRNG
    // ========================================================================
    py::class_<cdl::SplitMix64>(m, "SplitMix64",
        "Fast PRNG based on SplitMix64 algorithm")
        .def(py::init<uint64_t>(), py::arg("seed"))
        .def("next", &cdl::SplitMix64::next, "Generate next random number")
        .def_static("hash", &cdl::SplitMix64::hash,
            py::arg("key"), py::arg("data"),
            "Stateless hash function")
        .def_static("derive_key", &cdl::SplitMix64::derive_key,
            py::arg("base_key"), py::arg("data"),
            "Derive a new key from base key and data");

    // ========================================================================
    // FeistelPermutation — Core permutation
    // ========================================================================
    py::class_<cdl::FeistelPermutation>(m, "FeistelPermutation",
        R"doc(
Feistel Network Permutation.

Provides bijective (1-to-1) mapping over arbitrary domain sizes.
O(1) memory, O(1) per-index computation.

Args:
    seed: Permutation key
    domain_size: Size of the permutation domain
    rounds: Number of Feistel rounds (default 6)
)doc")
        .def(py::init<uint64_t, size_t, size_t>(),
            py::arg("seed"),
            py::arg("domain_size"),
            py::arg("rounds") = 6)
        .def("permute", &cdl::FeistelPermutation::permute, py::arg("index"))
        .def("inverse_permute", &cdl::FeistelPermutation::inverse_permute, py::arg("index"))
        .def("size", &cdl::FeistelPermutation::size)
        .def("key", &cdl::FeistelPermutation::key)
        .def("__len__", &cdl::FeistelPermutation::size)
        .def("__getitem__", &cdl::FeistelPermutation::permute);

    // ========================================================================
    // IOBlockShuffle — Two-level shuffler
    // ========================================================================
    py::class_<cdl::IOBlockShuffle>(m, "IOBlockShuffle",
        R"doc(
I/O-Aware Block Shuffler.

Level 1: Shuffles which BLOCKS to load from disk (sequential I/O).
Level 2: Shuffles items AFTER block is loaded into RAM.

Args:
    dataset_size: Total number of samples
    block_size: Samples per block
    seed: Epoch seed
    rounds: Feistel rounds (default 6)
)doc")
        .def(py::init<size_t, size_t, uint64_t, size_t>(),
            py::arg("dataset_size"),
            py::arg("block_size"),
            py::arg("seed"),
            py::arg("rounds") = 6)
        // Level 1
        .def("get_block_to_load", &cdl::IOBlockShuffle::get_block_to_load, py::arg("position"))
        .def("get_block_order", &cdl::IOBlockShuffle::get_block_order)
        .def("get_block_size", &cdl::IOBlockShuffle::get_block_size, py::arg("block_id"))
        .def("get_block_start", &cdl::IOBlockShuffle::get_block_start, py::arg("block_id"))
        // Level 2
        .def("get_intra_block_order", &cdl::IOBlockShuffle::get_intra_block_order, py::arg("block_id"))
        .def("get_item_offset", &cdl::IOBlockShuffle::get_item_offset,
            py::arg("block_id"), py::arg("position"))
        // Legacy
        .def("permute", &cdl::IOBlockShuffle::permute, py::arg("index"))
        .def("inverse_permute", &cdl::IOBlockShuffle::inverse_permute, py::arg("index"))
        .def("expected_locality_factor", &cdl::IOBlockShuffle::expected_locality_factor)
        // Accessors
        .def("dataset_size", &cdl::IOBlockShuffle::dataset_size)
        .def("block_size", &cdl::IOBlockShuffle::block_size)
        .def("num_blocks", &cdl::IOBlockShuffle::num_blocks)
        .def("seed", &cdl::IOBlockShuffle::seed)
        .def("__len__", &cdl::IOBlockShuffle::dataset_size)
        .def("__getitem__", &cdl::IOBlockShuffle::permute);

    // Alias
    m.attr("BlockShuffle") = m.attr("IOBlockShuffle");

    // ========================================================================
    // SimpleShuffle
    // ========================================================================
    py::class_<cdl::SimpleShuffle>(m, "SimpleShuffle",
        "Simple single-level Feistel shuffle (no block awareness)")
        .def(py::init<size_t, uint64_t, size_t>(),
            py::arg("dataset_size"),
            py::arg("seed"),
            py::arg("rounds") = 6)
        .def("permute", &cdl::SimpleShuffle::permute, py::arg("index"))
        .def("inverse_permute", &cdl::SimpleShuffle::inverse_permute, py::arg("index"))
        .def("dataset_size", &cdl::SimpleShuffle::dataset_size)
        .def("seed", &cdl::SimpleShuffle::seed)
        .def("__len__", &cdl::SimpleShuffle::dataset_size)
        .def("__getitem__", &cdl::SimpleShuffle::permute);

    // ========================================================================
    // Sampler
    // ========================================================================
    py::class_<cdl::Sampler>(m, "Sampler",
        R"doc(
PyTorch-compatible CDL Sampler.

Drop-in replacement for torch.utils.data.RandomSampler with O(1) memory.

Args:
    dataset_size: Total number of samples
    block_size: Samples per block (default 1024)
    seed: Base seed for reproducibility (default 42)
    rounds: Feistel rounds (default 6)
)doc")
        .def(py::init<size_t, size_t, uint64_t, size_t>(),
            py::arg("dataset_size"),
            py::arg("block_size") = 1024,
            py::arg("seed") = 42,
            py::arg("rounds") = 6)
        .def("set_epoch", &cdl::Sampler::set_epoch, py::arg("epoch"))
        .def("__len__", &cdl::Sampler::size)
        .def("__getitem__", &cdl::Sampler::operator[])
        .def("size", &cdl::Sampler::size)
        .def("block_size", &cdl::Sampler::block_size)
        .def("num_blocks", &cdl::Sampler::num_blocks)
        .def("epoch", &cdl::Sampler::epoch)
        .def("seed", &cdl::Sampler::seed)
        .def("get_block_to_load", &cdl::Sampler::get_block_to_load, py::arg("position"))
        .def("get_intra_block_order", &cdl::Sampler::get_intra_block_order, py::arg("block_id"));
}
