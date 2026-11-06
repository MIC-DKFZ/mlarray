from collections.abc import Mapping
from dataclasses import replace as _replace
import numpy as np
import blosc2
from typing import Any, Dict, Optional, Sequence, Union, List, Tuple
from pathlib import Path
import os
from mlarray.metadata import (
    Axis,
    Meta,
    MetadataError,
    SPEC_VERSION,
    Space,
    SourceEntry,
    Values,
    space_from_geometry,
)
from mlarray import blosc2_layout_strategies as _layout
from mlarray import fingerprint as _fingerprint
from mlarray import metadata_io
from mlarray import storage as _storage
from mlarray.constructors import MLArrayConstructors
from mlarray.storage import MLARRAY_SUFFIX, MLARRAY_DEFAULT_PATCH_SIZE  # re-exported

MLARRAY_DEFAULT_READ_AHEAD = 4

_read_ahead_configured = False


def set_read_ahead(nthreads: int = MLARRAY_DEFAULT_READ_AHEAD) -> int:
    """Set blosc2's read-ahead for this process (``blosc2.set_prefetch``).

    With read-ahead, reading a slice of an on-disk array fetches the chunks it
    needs with ``nthreads`` background threads in parallel, instead of one
    small request after another. For data that is not in the page cache this
    makes patch reads several times faster, most on network storage (in our
    benchmark with 12 nnU-Net-style loader processes on network storage, 3D
    patches took 64-82% less time). For data already in the page cache it can
    cost a little (up to about 17% per read with a single reader).

    MLArray turns it on with ``MLARRAY_DEFAULT_READ_AHEAD`` (4) threads the
    first time a file is opened, unless this function was called before or
    the ``BLOSC_PREFETCH`` environment variable is set. The setting applies to
    the whole process, including other blosc2 arrays it reads; each reader
    thread gets its own read-ahead threads. Call ``set_read_ahead(0)`` to turn
    it off. Without read-ahead support in the installed blosc2 this does
    nothing.

    Args:
        nthreads (int): Read-ahead threads per slice being read; 0 turns
            read-ahead off.

    Returns:
        int: The previous number of read-ahead threads (0 if unsupported).
    """
    global _read_ahead_configured
    _read_ahead_configured = True
    if not hasattr(blosc2, "set_prefetch"):
        return 0
    return int(blosc2.set_prefetch(int(nthreads)))


def _apply_default_read_ahead() -> None:
    """Turn on read-ahead once per process unless the user configured it."""
    global _read_ahead_configured
    if _read_ahead_configured:
        return
    _read_ahead_configured = True
    if "BLOSC_PREFETCH" in os.environ or not hasattr(blosc2, "set_prefetch"):
        return
    blosc2.set_prefetch(MLARRAY_DEFAULT_READ_AHEAD)


def _writer() -> str:
    from mlarray import __version__
    return f"mlarray {__version__}"


def _metadata_parts(axes=None, spaces=None, values=None, profiles=None, user=None, source=None) -> Dict[str, Any]:
    """The metadata keyword arguments that were given."""
    parts = dict(axes=axes, spaces=spaces, values=values, profiles=profiles, user=user, source=source)
    return {key: value for key, value in parts.items() if value is not None}


def _compose_space(spaces: Sequence[Space], axes: Sequence[Axis], space, spacing, origin, direction) -> List[Space]:
    """Spaces with ``space`` (re)built from spacing, origin and direction, placed first.

    When a space of that name exists and maps the same array axes, the values
    not given are taken from it, so e.g. only the spacing can be changed.
    """
    array_axes = [axis.name for axis in axes if axis.type == "space"]
    if not array_axes:
        raise MetadataError("Convenience geometry needs array axes of type space; declare axes=[...].")
    name = space if isinstance(space, str) else (space.name if isinstance(space, Space) else dict(space).get("name"))
    existing = next((s for s in spaces if s.name == name), None)
    if existing is not None and list(existing.from_array.array_axes) == array_axes:
        if isinstance(space, str):
            space = existing
        spacing = existing.spacing if spacing is None else spacing
        origin = existing.origin if origin is None else origin
        direction = existing.direction if direction is None else direction
    new = space_from_geometry(space, array_axes, spacing=spacing, origin=origin, direction=direction)
    return [new] + [s for s in spaces if s.name != new.name]


class MLArray(MLArrayConstructors):
    def __init__(
            self,
            array: Optional[Union[np.ndarray, str, Path]] = None,
            *,
            axes: Optional[Sequence[Union[Axis, str, Mapping]]] = None,
            spaces: Optional[Sequence[Union[Space, Mapping]]] = None,
            values: Optional[Union[Values, Mapping]] = None,
            profiles: Optional[Mapping] = None,
            user: Optional[Mapping] = None,
            source: Optional[Sequence[Union[SourceEntry, Mapping]]] = None,
            spacing: Optional[Sequence[float]] = None,
            origin: Optional[Sequence[float]] = None,
            direction: Optional[Any] = None,
            space: Optional[Union[str, Mapping, Space]] = None,
            meta: Optional[Union[Meta, Mapping]] = None,
            copy: Optional['MLArray'] = None,
            copy_source: bool = False,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ) -> None:
        """Create an MLArray from an array, or load a ".mla" file.

        The MLArray file format (".mla") is a Blosc2-compressed array with
        explicit, domain-neutral metadata (spec ``docs/design/spec.md``).

        Args:
            array (Optional[Union[np.ndarray, str, Path]]): A NumPy array, or the
                path of a ".mla" file to load. None creates an instance without
                array data.
            axes: One entry per array axis, in array order: names (``"z"``),
                :class:`~mlarray.metadata.Axis` objects or dicts with ``name``,
                ``type`` (space, time, channel, component, other) and optional
                labels or sampling. Defaults to ``axis_0``, ``axis_1``, ... of
                type space.
            spaces: Coordinate systems with their from-array transform
                (:class:`~mlarray.metadata.Space` or dicts). The first space
                is the default one.
            values: What the numbers mean (:class:`~mlarray.metadata.Values`
                or a dict with ``kind``, ``unit``, ``labels`` ...).
            profiles: Domain metadata by profile name, e.g. ``{"ml": {...}}``.
            user: Arbitrary user metadata; never interpreted.
            source: Original metadata of converted files (read-only afterwards).
            spacing, origin, direction: Convenience geometry, composed into the
                from-array matrix of ``space`` (see
                :func:`mlarray.metadata.space_from_geometry`). They follow the
                array's space axes in array order: ``spacing[c]`` and
                ``direction[:, c]`` belong to array axis ``c``; ``origin`` is the
                centre of index (0, ..., 0) in space coordinates.
            space: The space the convenience geometry lives in: a preset
                (``"patient"``: DICOM patient frame, LPS, millimeter) or a
                space description without ``from_array``. Required with
                ``spacing``, ``origin`` or ``direction``.
            meta: Complete metadata (:class:`~mlarray.metadata.Meta` or its
                dict form) to start from; it is copied.
            copy: Another MLArray to take metadata from: axes, spaces, values,
                profiles and user, but not its history, its ``ml`` statistics
                and boxes, nor its ``source`` (unless ``copy_source=True``).
            copy_source: Also copy the ``source`` of ``copy``.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters used when creating in-memory compressed array data.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used for chunk access.
            compressed (bool): Keep the data Blosc2-compressed in memory
                (default) or as a NumPy array.
        """
        self.filepath = None
        self.support_metadata = None
        self.mode = None
        self.mmap_mode = None
        self._meta = None
        self._store = None
        self._backend = None
        self._has_array = False
        parts = _metadata_parts(axes, spaces, values, profiles, user, source)
        geometry = dict(spacing=spacing, origin=origin, direction=direction, space=space)
        if isinstance(array, (str, Path)):
            given = sorted(parts) + sorted(k for k, v in geometry.items() if v is not None)
            if meta is not None or copy is not None or patch_size != "default" or chunk_size is not None \
                    or block_size is not None or cparams is not None or dparams is not None:
                given.append("meta/copy/layout/cparams/dparams")
            if given:
                raise RuntimeError(f"{', '.join(given)} cannot be set when array is a filepath.")
            self._load(array, compressed=compressed)
            return
        meta = self._build_meta(meta, copy, copy_source, like=False, parts=parts, geometry=geometry,
                                shape=None if array is None else np.shape(array))
        if array is not None:
            self._asarray(
                array,
                meta=meta,
                patch_size=patch_size,
                chunk_size=chunk_size,
                block_size=block_size,
                cparams=cparams,
                dparams=dparams,
                compressed=compressed,
            )
        else:
            self._meta = meta
            self.support_metadata = True
            if compressed:
                self._store = blosc2.empty((0,))
                self._backend = "blosc2"
            else:
                self._store = np.empty((0,))
                self._backend = "numpy"

    @classmethod
    def open(
            cls,
            filepath: Union[str, Path],
            mode: str = 'r',
            mmap_mode: str = 'r',
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Open an existing MLArray file with memory mapping.

        The first open in a process turns on blosc2's read-ahead with 4
        threads, which makes reads of data not in the page cache several
        times faster; see :func:`mlarray.set_read_ahead` to change or turn it
        off.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be either 'r' (default) or 'a'. Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'r' (default), 'r+', 'c' or None. Leave at default if unsure.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for opening.
        """
        class_instance = cls()
        class_instance._open(filepath, mode, mmap_mode, dparams)
        return class_instance

    @classmethod
    def create(            
            cls,
            filepath: Union[str, Path],
            shape: Union[List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Meta, Mapping]] = None,
            mode: str = 'w',
            mmap_mode: str = 'w+',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Create a new MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            shape (Union[List, Tuple, np.ndarray]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (np.dtype): Numpy dtype for a newly created array.
            meta (Optional[Union[Meta, Mapping]]): Metadata of the created
                ``MLArray`` (copied), e.g. ``{"axes": [...], "spaces": [...]}``.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be 'w' (default). Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'w+' (default) or None. Leave at default if unsure.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of spatial
                dimensions. Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 5, 'nthreads': 4}`` with blosc2's default filter
                (SHUFFLE); the data is not known yet, so no filter is chosen
                from it.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for creation.
        """
        class_instance = cls()
        class_instance._create(filepath, shape, dtype, meta, mode, mmap_mode, patch_size, chunk_size, block_size, cparams, dparams)
        return class_instance

    @classmethod
    def load(
            cls,
            filepath: Union[str, Path], 
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Load a MLArray file as a whole without memory mapping.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        class_instance = cls()
        class_instance._load(filepath, dparams, compressed=compressed)
        return class_instance

    def compress(
            self,
            patch_size: Optional[Union[int, List, Tuple]] = "default",
            chunk_size: Optional[Union[int, List, Tuple]] = None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
        ) -> None:
        """Compress this MLArray in-place using the Blosc2 backend.

        The instance is detached from any previously opened file state and
        becomes an in-memory object. Persist changes by calling ``save()``.

        Args:
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Use ``"default"`` to use the default
                patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2 compression
                parameters.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()

        converted = MLArray.asarray(
            self.to_numpy(),
            meta=self.meta,
            patch_size=patch_size,
            chunk_size=chunk_size,
            block_size=block_size,
            cparams=cparams,
            dparams=dparams,
            compressed=True,
        )
        self._replace_with_converted(converted)

    def decompress(self) -> None:
        """Decompress this MLArray in-place to a NumPy-backed array.

        The instance is detached from any previously opened file state and
        becomes an in-memory object. Persist changes by calling ``save()``.
        ``meta.storage`` keeps the layout record (patch, chunk and block
        size, compression), as ``load(compressed=False)`` does, so a later
        ``save()`` or ``compress()`` reuses it.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()

        meta = self.meta.copy()
        self._replace_with_converted(MLArray.asarray(self.to_numpy(), meta=meta, patch_size=None, compressed=False))
        self._meta._storage = meta.storage

    def save(
            self,
            filepath: Union[str, Path],
        ):
        """Save the array to a MLArray file.

        The file is written to a temporary file next to the target and then
        atomically moved into place, so an existing target is only replaced
        once the new file is complete. The source file of an opened MLArray
        is never modified. If an opened MLArray is saved to its own file, it
        is reopened afterwards with the same mode and mmap_mode.

        Args:
            filepath (Union[str, Path]): Path to save the file. Must end with
                ".mla".

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        _storage.check_suffix(filepath)

        self.support_metadata = True

        is_file_backed = self.mode is not None or self.mmap_mode is not None
        saves_onto_source = (
            is_file_backed
            and self.filepath is not None
            and Path(filepath).is_file()
            and os.path.samefile(self.filepath, filepath)
        )

        store, meta = self._blosc2_store_for_save()

        _storage.save_atomically(
            store, filepath, lambda saved: metadata_io.write(saved, self._serialize_metadata(meta), unchanged_in=store)
        )

        if saves_onto_source:
            # The open store still refers to the replaced file; rebind it to the new one.
            reopen_mode, reopen_mmap_mode = _storage.reopen_modes(self.mode, self.mmap_mode)
            self._open(filepath, reopen_mode, reopen_mmap_mode, self._store.schunk.dparams)

    def close(self):
        """Flush metadata and close the underlying store.

        After closing, the MLArray instance no longer has an attached array.
        """
        self._write_metadata()
        self._store = None
        self._backend = None
        self._has_array = False
        self.filepath = None
        self.support_metadata = None
        self.mode = None  
        self.mmap_mode = None
        self._meta = None

    def flush(self):
        """Write pending metadata changes to the opened file.

        Array data written through a writable memory map is persisted by
        Blosc2 directly; metadata changes are only written on ``flush()`` or
        ``close()``. Does nothing for read-only or in-memory arrays.
        """
        self._write_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getstate__(self):
        """Pickle support, e.g. for spawn/forkserver DataLoader workers.

        - Opened files are pickled by reference (filepath, mode, mmap_mode,
          dparams) and reopened on unpickling. Files created with ``create``
          are reopened with ``mode='a'`` / ``mmap_mode='r+'`` so they are not
          truncated. Metadata in memory, including changes not yet flushed,
          is carried over; array changes made through a copy-on-write map
          (``mmap_mode='c'``) are not.
        - In-memory Blosc2 arrays are pickled as a compressed frame.
        - NumPy-backed arrays pickle the NumPy array.
        """
        state = self.__dict__.copy()
        store = state.pop("_store")
        if store is None or self._backend == "numpy":
            state["_store_state"] = ("value", store)
        elif self.mode is not None or self.mmap_mode is not None:
            mode, mmap_mode = _storage.reopen_modes(self.mode, self.mmap_mode)
            meta = self._meta
            reread = {name: not getattr(meta, name).loaded and getattr(meta, name).origin is store
                      for name in ("_source", "_user")}
            if any(reread.values()):  # read undecoded parts from the file again instead of pickling them
                meta = meta.copy()
                for name, again in reread.items():
                    if again:
                        setattr(meta, name, None)
                state["_meta"] = meta
            state["_store_state"] = ("file", {
                "mode": mode,
                "mmap_mode": mmap_mode,
                "dparams": store.schunk.dparams,
                "reread": reread,
            })
        else:
            state["_store_state"] = ("cframe", _storage.in_memory_state(store))
        return state

    def __setstate__(self, state):
        kind, payload = state.pop("_store_state")
        self.__dict__.update(state)
        if kind == "value":
            self._store = payload
        elif kind == "cframe":
            self._store = _storage.restore_in_memory(payload)
        else:
            meta = self._meta
            self._store = None
            self._open(self.filepath, payload["mode"], payload["mmap_mode"], payload["dparams"])
            for name, again in payload["reread"].items():
                if again:
                    setattr(meta, name, getattr(self._meta, name))
            meta._set_shape(self._store.shape)
            self._meta = meta

    def _has_array_data(self) -> bool:
        """True when array data is loaded (not a metadata-only instance)."""
        return self._store is not None and self._has_array

    def _require_array(self) -> None:
        """Raise ``TypeError`` unless array data is loaded."""
        if not self._has_array_data():
            raise TypeError("MLArray has no array data loaded.")

    def to_numpy(self):
        """Return the underlying data as a NumPy array.

        Returns:
            np.ndarray: A NumPy view or copy of the stored array data.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()
        return self._store[...]

    def __array__(self, dtype=None, copy=None):
        """NumPy array interface for implicit conversion.

        Args:
            dtype (Optional[np.dtype]): Optional dtype to cast to.

        Returns:
            np.ndarray: The underlying data as a NumPy array.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()
        if self._backend == "numpy":
            return np.array(self._store, dtype=dtype, copy=copy)
        if copy is False:
            raise ValueError("A Blosc2-backed MLArray cannot be converted to a NumPy array without copying.")
        arr = self._store[...]
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, key):
        """Return a slice or element from the underlying array.

        Args:
            key (Any): Any valid NumPy/Blosc2 indexing key (slices, ints, tuples,
                boolean arrays).

        Returns:
            Any: The indexed value or subarray.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()
        return self._store[key]

    def __setitem__(self, key, value):
        """Assign to a slice or element in the underlying array.

        Args:
            key (Any): Any valid NumPy/Blosc2 indexing key.
            value (Any): Value(s) to assign. Must be broadcastable to the
                selected region.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()
        self._store[key] = value

    def __iter__(self):
        """Iterate over the first axis of the underlying array.

        Returns:
            Iterator: Iterator over the array's first dimension.

        Raises:
            TypeError: If no array data is loaded.
        """
        self._require_array()
        return iter(self._store)

    def __len__(self):
        """Return the length of the first array dimension.

        Returns:
            int: Size of axis 0, or 0 if no array is loaded.
        """
        if not self._has_array_data():
            return 0
        return len(self._store)

    @property
    def meta(self) -> Optional[Meta]:
        """All metadata (:class:`~mlarray.metadata.Meta`)."""
        return self._meta

    # -- core metadata ---------------------------------------------------------

    @property
    def axes(self) -> Tuple[Axis, ...]:
        """The array axes, in array order. Assign a new list to replace them (validated)."""
        return self._meta.axes

    @axes.setter
    def axes(self, value):
        self._meta.axes = value

    @property
    def spaces(self) -> Tuple[Space, ...]:
        """Coordinate systems the array is placed in; the first is the default."""
        return self._meta.spaces

    @spaces.setter
    def spaces(self, value):
        self._meta.spaces = value

    @property
    def values(self) -> Values:
        """What the stored numbers mean."""
        return self._meta.values

    @values.setter
    def values(self, value):
        self._meta.values = value

    @property
    def profiles(self) -> Dict[str, Dict[str, Any]]:
        """Domain metadata by profile name (plain dicts, validated when written)."""
        return self._meta.profiles

    @profiles.setter
    def profiles(self, value):
        self._meta.update(profiles=value)

    @property
    def user(self) -> Dict[str, Any]:
        """Arbitrary user metadata; loaded from the file on first access."""
        return self._meta.user

    @user.setter
    def user(self, value):
        self._meta.user = value

    @property
    def source(self) -> Tuple[SourceEntry, ...]:
        """Original metadata of the converted file(s); read-only, loaded on first access."""
        return self._meta.source

    @property
    def history(self) -> Tuple[Mapping, ...]:
        """Conversions and operations applied to this image (append with ``meta.add_history``)."""
        return self._meta.history

    # -- geometry (derived from the spaces, never stored) ----------------------

    def space(self, name: Optional[str] = None) -> Optional[Space]:
        """A space by name; without a name the default (first) space, or None if there is none."""
        spaces = self._meta.spaces
        if name is None:
            return spaces[0] if spaces else None
        for candidate in spaces:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No space named {name!r}; spaces: {[s.name for s in spaces]}.")

    def _space_or_raise(self, name: Optional[str]) -> Space:
        space = self.space(name)
        if space is None:
            raise MetadataError("The array has no spaces: it lives in index space only.")
        return space

    def matrix(self, space: Optional[str] = None) -> np.ndarray:
        """Homogeneous from-array matrix of a space (default: the first one).

        Rows are the space axes (+1), columns the space's ``from_array.array_axes``
        (+ translation). An integer index is the centre of its voxel.
        """
        return self._space_or_raise(space).matrix()

    @property
    def spacing(self) -> Optional[Tuple[float, ...]]:
        """Voxel spacing along each mapped array axis of the default space (array order), or None."""
        space = self.space()
        return None if space is None else space.spacing

    @property
    def origin(self) -> Optional[Tuple[float, ...]]:
        """Default-space coordinates of the centre of index (0, ..., 0), or None."""
        space = self.space()
        return None if space is None else space.origin

    @property
    def direction(self) -> Optional[np.ndarray]:
        """Unit vector of each mapped array axis in the default space (rows: space axes), or None."""
        space = self.space()
        return None if space is None else space.direction

    def letter_codes(self, space: Optional[str] = None) -> Tuple[str, ...]:
        """Display letters: the direction each mapped array axis increases toward, e.g. ("S", "P", "L")."""
        return self._space_or_raise(space).letter_codes()

    def set_geometry(
            self,
            spacing: Optional[Sequence[float]] = None,
            origin: Optional[Sequence[float]] = None,
            direction: Optional[Any] = None,
            space: Optional[Union[str, Mapping, Space]] = None,
        ) -> None:
        """Set a space from spacing, origin and direction (see the constructor).

        ``space`` defaults to the current default space. If the space exists,
        values not given are kept, e.g. ``set_geometry(spacing=[1, 1, 2])``
        changes only the spacing. The space becomes the first (default) one.
        """
        if space is None:
            space = self.space()
            if space is None:
                raise MetadataError("Give space= (e.g. 'patient'): the array has no space yet.")
            space = space.name
        self._meta.spaces = _compose_space(self._meta.spaces, self._meta.axes, space, spacing, origin, direction)

    def fingerprint(self) -> Dict[str, Any]:
        """Fingerprint of the array and its geometry (spec 6.3), as written into ``source`` entries."""
        self._require_array()
        return _fingerprint.compute(self, self._meta)

    # -- array properties ------------------------------------------------------

    @property
    def shape(self):
        """Returns the shape of the array.

        Returns:
            tuple: Shape of the underlying array, or None if no array is loaded.
        """
        if not self._has_array_data():
            return None
        return tuple(self._store.shape)

    @property
    def dtype(self):
        """Returns the dtype of the array.

        Returns:
            np.dtype: Dtype of the underlying array, or None if no array is
                loaded.
        """
        if not self._has_array_data():
            return None
        return self._store.dtype

    @property
    def ndim(self) -> Optional[int]:
        """Number of array dimensions, or None if no array is loaded."""
        shape = self.shape
        return None if shape is None else len(shape)

    @classmethod
    def comp_blosc2_params(
            cls,
            image_size: Tuple[int, ...],
            patch_size: Tuple[int, ...],
            spatial_axis_mask: Optional[list[bool]] = None,
            bytes_per_pixel: int = 4,
            max_block_nbytes: int = _layout.DEFAULT_MAX_BLOCK_NBYTES,
            max_chunk_nbytes: int = _layout.DEFAULT_MAX_CHUNK_NBYTES,
        ):
        """
        Compute recommended Blosc2 chunk and block sizes from a patch-size hint.

        Uses nnU-Net's layout heuristic, generalized to any number of spatial
        and non-spatial axes (:func:`mlarray.blosc2_layout_strategies.comp_blosc2_params`):

            1. Non-spatial axes (channels, time, ...) get chunk and block
               size 1, so reading one channel touches only its own data.
            2. Blocks start at the next power of two of the patch size per
               spatial axis and are halved until they fit ``max_block_nbytes``
               (128 KiB).
            3. Chunks grow by whole blocks on the least covered spatial axis
               until they reach ``max_chunk_nbytes`` (6 MiB), the image extent,
               or 1.5x the patch size per axis.

        For 2 and 3 spatial axes the layout is identical to nnU-Net's.

        Args:
            image_size (Tuple[int, ...]): Full array shape.
            patch_size (Tuple[int, ...]): Patch size over spatial axes only.
                A 1 keeps that axis at chunk size 1 (e.g. ``(1, 256, 256)``
                for 2D slices of a 3D image).
            spatial_axis_mask (Optional[list[bool]]): Mask indicating for every
                array axis whether it is spatial. If omitted, all axes are
                treated as spatial.
            bytes_per_pixel (int): Number of bytes per element. Defaults to 4
                for float32.
            max_block_nbytes (int): Largest block in bytes.
            max_chunk_nbytes (int): Largest chunk in bytes.

        Returns:
            Tuple[List[int], List[int]]: Recommended chunk size and block size.
        """
        return _layout.comp_blosc2_params(
            image_size=tuple(int(v) for v in image_size),
            patch_size=tuple(int(v) for v in patch_size),
            spatial_axis_mask=spatial_axis_mask,
            bytes_per_pixel=bytes_per_pixel,
            max_block_nbytes=max_block_nbytes,
            max_chunk_nbytes=max_chunk_nbytes,
        )

    def _open(
            self,
            filepath: Union[str, Path],
            mode: str = 'r',
            mmap_mode: str = 'r',
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Internal open method. Open an existing MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be either 'r' (default) or 'a'. Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'r' (default), 'r+', 'c' or None. Leave at default if unsure.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for opening.
        """
        _storage.check_suffix(filepath)
        self.filepath = str(filepath)
        
        if not Path(filepath).is_file():
            raise RuntimeError(f"No MLArray file exists under {filepath}.")
        if mode not in ('r', 'a'):
            raise RuntimeError("mode must be one of the following: 'r', 'a'")
        if mmap_mode not in ('r', 'r+', 'c', None):
            raise RuntimeError("mmap_mode must be one of the following: 'r', 'r+', 'c', None")
        
        self.support_metadata = True

        dparams = _storage.resolve_dparams(dparams)
        
        _apply_default_read_ahead()
        self._store = blosc2.open(urlpath=str(filepath), dparams=dparams, mode=mode, mmap_mode=mmap_mode)
        self._backend = "blosc2"
        self._read_meta()
        self.mode = mode
        self.mmap_mode = mmap_mode

    def _create(            
            self,
            filepath: Union[str, Path],
            shape: Union[List, Tuple, np.ndarray],
            dtype: np.dtype,
            meta: Optional[Union[Meta, Mapping]] = None,
            mode: str = 'w',
            mmap_mode: str = 'w+',
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MLArray image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            cparams: Optional[Union[Dict, blosc2.CParams]] = None,
            dparams: Optional[Union[Dict, blosc2.DParams]] = None
        ):
        """Internal create method. Create a new MLArray file with memory mapping.

        Args:
            filepath (Union[str, Path]): Target file path. Must end with
                ".mla".
            shape (Union[List, Tuple, np.ndarray]): Shape of the array
                to create. If provided, a new file is created. Length must match
                the full array dimensionality (including non-spatial axes if present).
            dtype (np.dtype): Numpy dtype for a newly created array.
            meta (Optional[Union[Meta, Mapping]]): Metadata of the created
                ``MLArray`` (copied), e.g. ``{"axes": [...], "spaces": [...]}``.
            mode (str): Controls storage open/creation permissions when using standard Blosc2 I/O (read-only, read/write, overwrite). 
                Does not affect lazy loading or decompression; data is accessed and decompressed on demand by Blosc2.
                Must be 'w' (default). Leave at default if unsure.
            mmap_mode (str): Controls access via OS-level memory mapping of the compressed data, including read/write permissions. 
                Changes how bytes are fetched from disk (paging rather than explicit reads), while chunks are still decompressed on demand by Blosc2.
                Overrides `mode` if set. Must be either 'w+' (default) or None. Leave at default if unsure.
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of spatial
                dimensions. Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or tuple/list with length equal to the array
                dimensions. Cannot be combined with an explicit ``patch_size``.
            cparams (Optional[Union[Dict, blosc2.CParams]]): Blosc2
                compression parameters used when
                creating/writing array data (for example codec, compression
                level, and filters). Controls how data is compressed when
                stored. If None, defaults to ``{'codec': blosc2.Codec.LZ4HC,
                'clevel': 5, 'nthreads': 4}`` with blosc2's default filter
                (SHUFFLE); the data is not known yet, so no filter is chosen
                from it.
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when
                reading/accessing compressed chunks (for example number of
                decompression threads). Controls runtime decompression behavior.
                If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is invalid or if mode/mmap_mode is invalid for creation.
        """
        _storage.check_suffix(filepath)
        self.filepath = str(filepath)

        if mode != 'w':
            raise RuntimeError("mode must be 'w'.")
        if mmap_mode not in ('w+', None):
            raise RuntimeError("mmap_mode must be one of the following: 'w+', None")
    
        shape = _storage.normalize_shape(shape)
        meta = self._coerce_meta(meta, shape)
        storage, cparams = _storage.comp_layout(
            meta.storage, patch_size, chunk_size, block_size, shape, np.dtype(dtype).itemsize,
            MLArray._space_axis_mask(meta), cparams,
        )
        self.support_metadata = True
        self._store = blosc2.empty(shape=shape, dtype=np.dtype(dtype), urlpath=str(filepath), chunks=storage.chunk_size, blocks=storage.block_size, cparams=cparams, dparams=_storage.resolve_dparams(dparams), mode=mode, mmap_mode=mmap_mode)
        self._backend = "blosc2"
        self._has_array = True
        self.mode = mode
        self.mmap_mode = mmap_mode
        self._meta = meta
        self._record_storage(storage)
        self._write_metadata()

    def _load(
            self,
            filepath: Union[str, Path], 
            dparams: Optional[Union[Dict, blosc2.DParams]] = None,
            compressed: bool = True,
        ):
        """Internal MLArray load method. Load a MLArray file without memory mapping.

        Args:
            filepath (Union[str, Path]): Path to the MLArray file to be loaded.
                The filepath needs to have the extension ".mla".
            dparams (Optional[Union[Dict, blosc2.DParams]]): Blosc2
                decompression parameters used when loading/accessing compressed
                chunks. If None, defaults to ``{'nthreads': 1}``.

        Raises:
            RuntimeError: If the file extension is not ".mla".
        """
        _storage.check_suffix(filepath)
        self.filepath = str(filepath)
        self.support_metadata = True
        self._store = _storage.load_into_memory(filepath, _storage.resolve_dparams(dparams))
        self._backend = "blosc2"
        self.mode = None
        self.mmap_mode = None
        self._read_meta()
        if not compressed:
            self._store = np.asarray(self._store[...])
            self._backend = "numpy"

    # -- metadata internals ----------------------------------------------------

    @staticmethod
    def _space_axis_mask(meta: Meta) -> List[bool]:
        """For every array axis whether it is a space axis (drives the layout heuristic)."""
        return [axis.type == "space" for axis in meta.axes]

    @staticmethod
    def _coerce_meta(meta: Optional[Union[Meta, Mapping]], shape: Optional[Sequence[int]] = None) -> Meta:
        """A private copy of ``meta`` (Meta or its dict form), bound to ``shape``."""
        if meta is None:
            meta = Meta()
        elif isinstance(meta, Meta):
            meta = meta.copy()
        elif isinstance(meta, Mapping):
            meta = Meta.from_dict({"version": SPEC_VERSION, **meta})
        else:
            raise TypeError(f"meta must be a Meta or a mapping, got {type(meta).__name__}.")
        if shape is not None:
            meta._set_shape(shape)
        return meta

    @staticmethod
    def _derived_meta(other: Meta, like: bool, copy_source: bool) -> Meta:
        """Metadata for an image derived from another one (spec 6.1 rule 4).

        Geometry, values, profiles and user metadata are copied; the history
        and the data-dependent ``ml`` statistics and boxes are not, and
        ``source`` only with ``copy_source``. ``*_like`` constructors also drop
        the label semantics.
        """
        meta = other.copy(source=copy_source)
        meta._history = []
        meta.writer = None
        ml = meta.profiles.get("ml")
        if ml is not None:
            for key in ("statistics", "boxes"):
                ml.pop(key, None)
        if like:
            meta._values = meta.values.without_label_semantics()
        meta._shape = None
        return meta

    @classmethod
    def _build_meta(cls, meta, copy, copy_source, like, parts, geometry, shape) -> Meta:
        """Metadata of a new array from ``meta``/``copy`` plus the given parts and convenience geometry."""
        if meta is not None and copy is not None:
            raise ValueError("Give either meta or copy, not both.")
        if copy is not None:
            if not isinstance(copy, MLArray) or copy.meta is None:
                raise TypeError("copy must be an MLArray with metadata.")
            result = cls._derived_meta(copy.meta, like=like, copy_source=copy_source)
        else:
            if copy_source:
                raise ValueError("copy_source needs copy.")
            result = cls._coerce_meta(meta)
        source = parts.pop("source", None)
        if source is not None:
            result._source = Meta(source=source)._source
        user = parts.pop("user", None)
        if user is not None:
            result.user = user
        has_geometry = any(geometry[key] is not None for key in ("spacing", "origin", "direction"))
        if geometry["space"] is not None and not has_geometry:
            raise ValueError("space= needs spacing, origin or direction.")
        if has_geometry:
            if geometry["space"] is None:
                raise MetadataError(
                    "spacing, origin and direction place the array in a space; say which with space= "
                    "(e.g. space='patient' for the DICOM patient frame) or pass spaces=[...]."
                )
            if "spaces" in parts:
                raise ValueError("Give either spaces or spacing/origin/direction, not both.")
        if shape is not None:
            result._set_shape(tuple(int(v) for v in shape), axes=parts.pop("axes", None))
        if parts:
            result.update(**parts)
        if has_geometry:
            result.spaces = _compose_space(result.spaces, result.axes, geometry["space"],
                                           geometry["spacing"], geometry["origin"], geometry["direction"])
        return result

    def _record_storage(self, storage) -> None:
        """Record the store's layout: the computed storage with chunk/block size as blosc2 chose them."""
        if self._backend == "blosc2" and self._has_array:
            storage = _replace(storage, chunk_size=tuple(self._store.chunks), block_size=tuple(self._store.blocks))
        self._meta._storage = storage

    def _read_meta(self):
        """Read the metadata of the opened or loaded store."""
        meta = metadata_io.read(self._store, shape=None)
        meta._set_shape(tuple(self._store.shape))
        self._meta = meta
        self._has_array = True

    def _write_metadata(self):
        """Write MLArray metadata to the underlying store if it is writable."""
        is_writable = False
        if self.support_metadata:
            if self.mode in ('a', 'w') and self.mmap_mode is None:
                is_writable = True
            elif self.mmap_mode in ('r+', 'w+'):
                is_writable = True

        if not is_writable:
            return

        if self._backend != "blosc2":
            return

        metadata_io.write(self._store, self._serialize_metadata())

    def _serialize_metadata(self, meta: Optional[Meta] = None) -> Meta:
        """Validate the metadata (default: ``self.meta``) and stamp the writer before it is written."""
        meta = self.meta if meta is None else meta
        meta.validate()
        meta.writer = _writer()
        return meta

    def _blosc2_store_for_save(self) -> Tuple[Any, Meta]:
        """Return the Blosc2 store and metadata to write, without changing this instance.

        Blosc2-backed arrays are saved as they are. NumPy-backed arrays are
        compressed into a temporary store whose layout follows the recorded
        ``meta.storage`` (patch size first, then chunk/block size, then the
        default patch size); the instance stays NumPy-backed.
        """
        if not self._has_array_data():
            # The file format always holds an array: save an empty one with one axis per declared axis.
            meta = self.meta.copy()
            shape = (0,) * max(1, len(meta.axes))
            meta._set_shape(shape)
            return blosc2.empty(shape), meta

        if self._backend == "blosc2":
            return self._store, self.meta

        array = np.ascontiguousarray(self._store)
        meta = self.meta.copy()
        recorded = meta.storage
        cparams = None if recorded is None or not recorded.compression else recorded.to_cparams()
        storage, cparams = _storage.comp_layout(
            recorded,
            patch_size="default",
            chunk_size=None,
            block_size=None,
            shape=array.shape,
            dtype_itemsize=np.dtype(array.dtype).itemsize,
            spatial_axis_mask=MLArray._space_axis_mask(meta),
            cparams=cparams,
            array=array,
        )
        store = blosc2.asarray(
            array,
            chunks=storage.chunk_size,
            blocks=storage.block_size,
            cparams=cparams,
        )
        meta._storage = _replace(storage, chunk_size=tuple(store.chunks), block_size=tuple(store.blocks))
        return store, meta

    def _replace_with_converted(self, other: "MLArray") -> None:
        """Replace in-memory state from another MLArray and detach file state."""
        old_store = self._store
        self._store = other._store
        self._backend = other._backend
        self._has_array = other._has_array
        self._meta = other._meta
        self.support_metadata = other.support_metadata
        self.filepath = None
        self.mode = None
        self.mmap_mode = None
        del old_store
