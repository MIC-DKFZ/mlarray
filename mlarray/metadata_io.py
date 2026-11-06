"""The only place where MLArray metadata is serialized and deserialized (spec section 3).

Metadata lives in three blosc2 vlmeta keys, each a msgpack map that blosc2
compresses separately:

- ``mlarray``: core, profiles, history and storage; decoded on open;
- ``mlarray.source``: ``{"source": [...]}``; decoded on first access;
- ``mlarray.user``: ``{"user": {...}}``; decoded on first access.

Keys are read as raw bytes and decoded with plain msgpack, never through
blosc2's vlmeta decoder, which can rebuild Python objects. msgpack extension
types are refused. Nothing is ever pickled.
"""

from typing import Any, Callable, Optional, Sequence

import msgpack

from mlarray.metadata import Meta, _Lazy

CORE_KEY = "mlarray"
SOURCE_KEY = "mlarray.source"
USER_KEY = "mlarray.user"
KEYS = (CORE_KEY, SOURCE_KEY, USER_KEY)


class MetadataFormatError(RuntimeError):
    """A file's metadata is missing or not MLArray v1 msgpack metadata."""


def _refuse_extension(code: int, data: bytes) -> Any:
    raise MetadataFormatError(f"MLArray metadata contains a msgpack extension type ({code}); only plain msgpack is allowed.")


def unpack(data: bytes, key: str, str_keys: bool = True) -> Any:
    """Decode raw msgpack bytes of a vlmeta key."""
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=str_keys, ext_hook=_refuse_extension)
    except MetadataFormatError:
        raise
    except Exception as error:  # msgpack raises several unrelated types for malformed data
        raise MetadataFormatError(f"vlmeta key {key!r} is not valid msgpack: {error}") from error


def pack(value: Any) -> bytes:
    return msgpack.packb(value, use_bin_type=True)


def _vlmeta(store):
    return store.schunk.vlmeta


def _wrapped(data: bytes, key: str, field: str) -> Any:
    value = unpack(data, key, str_keys=False)
    if not isinstance(value, dict) or field not in value:
        raise MetadataFormatError(f"vlmeta key {key!r} must be a map with a {field!r} entry.")
    return value[field]


def _decode_source(data: bytes) -> Any:
    return _wrapped(data, SOURCE_KEY, "source")


def _decode_user(data: bytes) -> Any:
    value = _wrapped(data, USER_KEY, "user")
    if not isinstance(value, dict):
        raise MetadataFormatError(f"vlmeta key {USER_KEY!r}: 'user' must be a map.")
    return value


def _lazy(store, key: str, decode: Callable[[bytes], Any]) -> _Lazy:
    vlmeta = _vlmeta(store)
    return _Lazy(raw=lambda: vlmeta.get_vlmeta(key), decode=decode, origin=store)


def read(store, shape: Optional[Sequence[int]]) -> Meta:
    """Read the metadata of a blosc2 store; ``source`` and ``user`` stay undecoded."""
    vlmeta = _vlmeta(store)
    keys = set(vlmeta.keys())
    if CORE_KEY not in keys:
        raise MetadataFormatError("Not an MLArray file: the 'mlarray' metadata key is missing.")
    core = unpack(vlmeta.get_vlmeta(CORE_KEY), CORE_KEY)
    if not isinstance(core, dict):
        raise MetadataFormatError(
            "The 'mlarray' metadata key is not a msgpack map. Files written by MLArray before v1 are not "
            "supported; convert them again from their original files."
        )
    source = _lazy(store, SOURCE_KEY, _decode_source) if SOURCE_KEY in keys else None
    user = _lazy(store, USER_KEY, _decode_user) if USER_KEY in keys else None
    return Meta.from_dict(core, shape=shape, source=source, user=user)


def write(store, meta: Meta, unchanged_in: Any = None) -> None:
    """Write ``meta`` into the vlmeta of ``store``.

    ``source`` and ``user`` that were never decoded are skipped when their
    bytes already are in ``store`` (``unchanged_in`` is the store they were
    copied from, e.g. by ``blosc2.NDArray.save``); otherwise their raw bytes
    are copied without decoding. Empty parts remove their key.
    """
    vlmeta = _vlmeta(store)
    vlmeta.set_vlmeta(CORE_KEY, pack(meta.to_dict()), typesize=1)
    keys = set(vlmeta.keys())
    for key, part, field in ((SOURCE_KEY, meta._source, "source"), (USER_KEY, meta._user, "user")):
        if not part.loaded:
            if part.origin is store or (unchanged_in is not None and part.origin is unchanged_in):
                continue
            vlmeta.set_vlmeta(key, part.raw(), typesize=1)
            continue
        if field == "source":
            value = [entry.to_dict() for entry in meta.source]
        else:
            value = meta.user
        if not value:
            if key in keys:
                del vlmeta[key]
            continue
        vlmeta.set_vlmeta(key, pack({field: value}), typesize=1)


def copy_raw(source_store, target_store) -> None:
    """Copy all vlmeta keys as raw bytes, without decoding them."""
    source, target = _vlmeta(source_store), _vlmeta(target_store)
    for key in source.keys():
        target.set_vlmeta(key, source.get_vlmeta(key), typesize=1)
