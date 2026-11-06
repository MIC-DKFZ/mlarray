"""A standardized blosc2 array format with explicit metadata for scientific and medical images."""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlarray.mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE, MLARRAY_DEFAULT_READ_AHEAD, set_read_ahead
    from mlarray.metadata import (
        Axis,
        FromArray,
        Meta,
        MetadataError,
        SourceEntry,
        Space,
        SpaceAxis,
        Storage,
        Values,
        space_from_geometry,
    )
    from mlarray.cli import cli_print_header, cli_convert_to_mlarray

__all__ = [
    "__version__",
    "MLArray",
    "MLARRAY_DEFAULT_PATCH_SIZE",
    "MLARRAY_DEFAULT_READ_AHEAD",
    "set_read_ahead",
    "Axis",
    "FromArray",
    "Meta",
    "MetadataError",
    "SourceEntry",
    "Space",
    "SpaceAxis",
    "Storage",
    "Values",
    "space_from_geometry",
    "cli_print_header",
    "cli_convert_to_mlarray",
]

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - during editable installs pre-build
    __version__ = "0.0.0"


_LAZY_ATTRS = {
    "MLArray": ("mlarray.mlarray", "MLArray"),
    "MLARRAY_DEFAULT_PATCH_SIZE": ("mlarray.mlarray", "MLARRAY_DEFAULT_PATCH_SIZE"),
    "MLARRAY_DEFAULT_READ_AHEAD": ("mlarray.mlarray", "MLARRAY_DEFAULT_READ_AHEAD"),
    "set_read_ahead": ("mlarray.mlarray", "set_read_ahead"),
    "Axis": ("mlarray.metadata", "Axis"),
    "FromArray": ("mlarray.metadata", "FromArray"),
    "Meta": ("mlarray.metadata", "Meta"),
    "MetadataError": ("mlarray.metadata", "MetadataError"),
    "SourceEntry": ("mlarray.metadata", "SourceEntry"),
    "Space": ("mlarray.metadata", "Space"),
    "SpaceAxis": ("mlarray.metadata", "SpaceAxis"),
    "Storage": ("mlarray.metadata", "Storage"),
    "Values": ("mlarray.metadata", "Values"),
    "space_from_geometry": ("mlarray.metadata", "space_from_geometry"),
    "cli_print_header": ("mlarray.cli", "cli_print_header"),
    "cli_convert_to_mlarray": ("mlarray.cli", "cli_convert_to_mlarray"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
