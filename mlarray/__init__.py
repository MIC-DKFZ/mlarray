"""A standardized blosc2 image reader and writer for medical images."""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlarray.mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE
    from mlarray.meta import (
        Meta,
        MetaBbox,
        MetaBlosc2,
        MetaExtra,
        MetaHasArray,
        MetaOriginal,
        MetaImageFormat,
        MetaIsSeg,
        MetaSpatial,
        MetaStatistics,
        MetaVersion,
    )
    from mlarray.utils import is_serializable
    from mlarray.cli import cli_print_header, cli_convert_to_mlarray

__all__ = [
    "__version__",
    "MLArray",
    "MLARRAY_DEFAULT_PATCH_SIZE",
    "Meta",
    "MetaBbox",
    "MetaBlosc2",
    "MetaExtra",
    "MetaHasArray",
    "MetaOriginal",
    "MetaImageFormat",
    "MetaIsSeg",
    "MetaSpatial",
    "MetaStatistics",
    "MetaVersion",
    "is_serializable",
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
    "Meta": ("mlarray.meta", "Meta"),
    "MetaBbox": ("mlarray.meta", "MetaBbox"),
    "MetaBlosc2": ("mlarray.meta", "MetaBlosc2"),
    "MetaExtra": ("mlarray.meta", "MetaExtra"),
    "MetaHasArray": ("mlarray.meta", "MetaHasArray"),
    "MetaOriginal": ("mlarray.meta", "MetaOriginal"),
    "MetaImageFormat": ("mlarray.meta", "MetaImageFormat"),
    "MetaIsSeg": ("mlarray.meta", "MetaIsSeg"),
    "MetaSpatial": ("mlarray.meta", "MetaSpatial"),
    "MetaStatistics": ("mlarray.meta", "MetaStatistics"),
    "MetaVersion": ("mlarray.meta", "MetaVersion"),
    "is_serializable": ("mlarray.utils", "is_serializable"),
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
