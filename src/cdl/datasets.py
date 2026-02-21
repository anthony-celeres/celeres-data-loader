"""
CDL Datasets — Block-aware dataset adapters.

Each adapter exposes a uniform ``load_block`` / ``get_item`` interface
so that CDLLoader can work with any storage format while preserving
block-sequential I/O.

Supported formats
-----------------
- **HDF5** — block size auto-inferred from chunk metadata.
- **Directory** — block size auto-inferred from shard structure.
- **NumPy** — in-memory; useful for testing.
- **Custom** — subclass :class:`BlockDataset`.
"""

from __future__ import annotations

import math
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

# Fallback when block size cannot be inferred from metadata.
DEFAULT_BLOCK_SIZE: int = 256


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BlockDataset(ABC):
    """Abstract base class for block-aware datasets.

    Subclasses must implement ``__len__``, ``num_blocks``, ``block_size``,
    ``load_block``, and ``get_item``.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples."""

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Number of blocks."""

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Items per block (except possibly the last)."""

    @abstractmethod
    def load_block(self, block_id: int) -> Any:
        """Load an entire block from storage."""

    @abstractmethod
    def get_item(self, block_data: Any, offset: int) -> Any:
        """Extract a single item from loaded block data."""

    def get_block_size(self, block_id: int) -> int:
        """Actual size of *block_id* (handles partial last block)."""
        if block_id == self.num_blocks - 1:
            remainder = len(self) % self.block_size
            return remainder if remainder > 0 else self.block_size
        return self.block_size


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

class HDF5BlockDataset(BlockDataset):
    """HDF5 block dataset — block size auto-inferred from chunk metadata.

    Parameters
    ----------
    path : str
        Path to the ``.h5`` file.
    data_key : str
        HDF5 dataset key for the data (default ``"data"``).
    label_key : str or None
        HDF5 dataset key for labels (default ``"labels"``; ``None`` to skip).
    block_size : int or None
        Override block size.  ``None`` → infer from HDF5 chunk metadata.
    transform : callable or None
        Optional per-item transform.
    """

    def __init__(
        self,
        path: str,
        data_key: str = "data",
        label_key: Optional[str] = "labels",
        block_size: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required: pip install h5py")

        self.path = path
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform

        with h5py.File(path, "r") as f:
            self._dataset_size: int = f[data_key].shape[0]
            self._data_shape = f[data_key].shape[1:]
            self._data_dtype = f[data_key].dtype
            self._has_labels: bool = label_key is not None and label_key in f

            if block_size is not None:
                self._block_size = block_size
            elif f[data_key].chunks is not None:
                self._block_size = f[data_key].chunks[0]
            else:
                self._block_size = DEFAULT_BLOCK_SIZE
                warnings.warn(
                    f"HDF5 dataset '{data_key}' is not chunked. "
                    f"Using default block_size={DEFAULT_BLOCK_SIZE}. "
                    f"For optimal I/O, re-create the file with chunking "
                    f"or specify block_size explicitly.",
                    UserWarning,
                    stacklevel=2,
                )

        self._num_blocks = math.ceil(self._dataset_size / self._block_size)
        self._file: Any = None  # lazily opened

    def _ensure_open(self) -> None:
        if self._file is None:
            import h5py
            self._file = h5py.File(self.path, "r")

    def __len__(self) -> int:
        return self._dataset_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def load_block(self, block_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._ensure_open()
        start = block_id * self._block_size
        end = min(start + self._block_size, self._dataset_size)
        data = self._file[self.data_key][start:end]
        labels = self._file[self.label_key][start:end] if self._has_labels else None
        return (data, labels)

    def get_item(
        self, block_data: Tuple[np.ndarray, Optional[np.ndarray]], offset: int,
    ) -> Tuple[Any, Optional[Any]]:
        data, labels = block_data
        item = data[offset]
        if self.transform is not None:
            item = self.transform(item)
        label = labels[offset] if labels is not None else None
        return (item, label)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------

class DirectoryBlockDataset(BlockDataset):
    """Image-folder dataset — block size auto-inferred from shard structure.

    Parameters
    ----------
    root : str
        Root directory containing image files (or subdirectory shards).
    extensions : list[str]
        File extensions to include.
    block_size : int or None
        Override block size.  ``None`` → infer from subdirectory counts.
    transform : callable or None
        Optional per-item image transform.
    label_fn : callable or None
        ``label_fn(filepath) -> int`` for extracting labels.
    """

    def __init__(
        self,
        root: str,
        extensions: Optional[List[str]] = None,
        block_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        label_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.root = root
        self.extensions = [e.lower() for e in extensions]
        self.transform = transform
        self.label_fn = label_fn

        # Collect all files
        self.files: list[str] = []
        self.labels: list[int] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    filepath = os.path.join(dirpath, fname)
                    self.files.append(filepath)
                    if label_fn is not None:
                        self.labels.append(label_fn(filepath))

        self._dataset_size = len(self.files)

        if block_size is not None:
            self._block_size = block_size
        else:
            inferred = self._infer_block_size_from_shards()
            if inferred is not None:
                self._block_size = inferred
            else:
                self._block_size = DEFAULT_BLOCK_SIZE
                warnings.warn(
                    f"Could not infer block size from '{root}'. "
                    f"Using default block_size={DEFAULT_BLOCK_SIZE}. "
                    f"Organise files into subdirectories or specify block_size.",
                    UserWarning,
                    stacklevel=2,
                )

        self._num_blocks = math.ceil(self._dataset_size / self._block_size)

    def _infer_block_size_from_shards(self) -> Optional[int]:
        try:
            entries = sorted(os.listdir(self.root))
            subdirs = [e for e in entries if os.path.isdir(os.path.join(self.root, e))]
            if not subdirs:
                return None
            first_shard = os.path.join(self.root, subdirs[0])
            count = sum(
                1 for f in os.listdir(first_shard)
                if any(f.lower().endswith(ext) for ext in self.extensions)
            )
            return count if count > 0 else None
        except OSError:
            return None

    def __len__(self) -> int:
        return self._dataset_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def load_block(self, block_id: int) -> List[Tuple[str, Optional[int]]]:
        start = block_id * self._block_size
        end = min(start + self._block_size, self._dataset_size)
        block_files = self.files[start:end]
        block_labels = self.labels[start:end] if self.labels else [None] * len(block_files)
        return list(zip(block_files, block_labels))

    def get_item(
        self, block_data: List[Tuple[str, Optional[int]]], offset: int,
    ) -> Tuple[Any, Optional[int]]:
        filepath, label = block_data[offset]
        try:
            from PIL import Image
            image = Image.open(filepath).convert("RGB")
        except ImportError:
            if filepath.endswith(".npy"):
                image = np.load(filepath)
            else:
                raise ImportError(
                    f"Pillow is required to load '{filepath}': pip install Pillow"
                )
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)


# ---------------------------------------------------------------------------
# NumPy (in-memory)
# ---------------------------------------------------------------------------

class NumpyBlockDataset(BlockDataset):
    """In-memory NumPy block dataset — for testing and small datasets."""

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        block_size: int = 1024,
    ) -> None:
        if len(data) == 0:
            raise ValueError("dataset must not be empty")
        self.data = data
        self.labels = labels
        self._block_size = block_size
        self._dataset_size = len(data)
        self._num_blocks = (self._dataset_size + block_size - 1) // block_size

    def __len__(self) -> int:
        return self._dataset_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    def load_block(self, block_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        start = block_id * self._block_size
        end = min(start + self._block_size, self._dataset_size)
        data = self.data[start:end]
        labels = self.labels[start:end] if self.labels is not None else None
        return (data, labels)

    def get_item(
        self, block_data: Tuple[np.ndarray, Optional[np.ndarray]], offset: int,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        data, labels = block_data
        return (data[offset], labels[offset] if labels is not None else None)


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def infer_block_size(source: str, data_key: str = "data") -> int:
    """Infer optimal block size from a data source path.

    Supports HDF5 (``.h5``), Parquet, and sharded directories.

    Raises
    ------
    ValueError
        If block size cannot be inferred.
    """
    src = source.lower()

    if src.endswith((".h5", ".hdf5", ".he5")):
        try:
            import h5py
            with h5py.File(source, "r") as f:
                if data_key in f and f[data_key].chunks is not None:
                    return f[data_key].chunks[0]
        except (ImportError, OSError, KeyError):
            pass

    elif src.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            meta = pq.read_metadata(source)
            if meta.num_row_groups > 0:
                return meta.row_group(0).num_rows
        except (ImportError, OSError):
            pass

    elif os.path.isdir(source):
        try:
            entries = sorted(os.listdir(source))
            subdirs = [e for e in entries if os.path.isdir(os.path.join(source, e))]
            if subdirs:
                count = len(os.listdir(os.path.join(source, subdirs[0])))
                if count > 0:
                    return count
        except OSError:
            pass

    raise ValueError(
        f"Could not infer block size from '{source}'. "
        f"Please specify block_size explicitly."
    )
