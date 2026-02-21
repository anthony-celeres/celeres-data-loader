"""
CDL native extension — optional C++ acceleration.

When the C++ extension is compiled, this module exposes the native
:class:`FeistelPermutation`, :class:`SplitMix64`, and :class:`IOBlockShuffle`
implementations.  The pure-Python fallbacks in :mod:`cdl.shuffle` are used
automatically when the extension is absent.

Build the extension:
    cd CDL && mkdir build && cd build
    cmake .. -DCDL_BUILD_PYTHON=ON
    cmake --build . --config Release
"""

from __future__ import annotations

try:
    from cdl._cdl_native import (  # type: ignore[import-not-found]
        SplitMix64 as NativeSplitMix64,
        FeistelPermutation as NativeFeistelPermutation,
        IOBlockShuffle as NativeIOBlockShuffle,
        Sampler as NativeSampler,
        get_shuffled_indices,
        get_all_shuffled_indices,
    )

    NATIVE_AVAILABLE = True

except ImportError:
    NativeSplitMix64 = None  # type: ignore[misc,assignment]
    NativeFeistelPermutation = None  # type: ignore[misc,assignment]
    NativeIOBlockShuffle = None  # type: ignore[misc,assignment]
    NativeSampler = None  # type: ignore[misc,assignment]
    get_shuffled_indices = None  # type: ignore[misc,assignment]
    get_all_shuffled_indices = None  # type: ignore[misc,assignment]
    NATIVE_AVAILABLE = False

__all__ = [
    "NativeSplitMix64",
    "NativeFeistelPermutation",
    "NativeIOBlockShuffle",
    "NativeSampler",
    "get_shuffled_indices",
    "get_all_shuffled_indices",
    "NATIVE_AVAILABLE",
]
