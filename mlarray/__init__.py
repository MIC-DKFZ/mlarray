"""A standardized blosc2 image reader and writer for medical images.."""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlarray.mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE
    from mlarray.meta import Meta, MetaBlosc2, MetaSpatial
    from mlarray.utils import is_serializable
    from mlarray.cli import cli_print_header, cli_convert_to_mlarray

__all__ = [
    "__version__",
    "MLArray",
    "MLARRAY_DEFAULT_PATCH_SIZE",
    "Meta",
    "MetaBlosc2",
    "MetaSpatial",
    "is_serializable",
    "cli_print_header",
    "cli_convert_to_mlarray",
]

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - during editable installs pre-build
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name in {"MLArray", "MLARRAY_DEFAULT_PATCH_SIZE"}:
        from mlarray.mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE

        return MLArray if name == "MLArray" else MLARRAY_DEFAULT_PATCH_SIZE
    if name in {"Meta", "MetaBlosc2", "MetaSpatial"}:
        from mlarray.meta import Meta, MetaBlosc2, MetaSpatial

        return {"Meta": Meta, "MetaBlosc2": MetaBlosc2, "MetaSpatial": MetaSpatial}[name]
    if name == "is_serializable":
        from mlarray.utils import is_serializable

        return is_serializable
    if name in {"cli_print_header", "cli_convert_to_mlarray"}:
        from mlarray.cli import cli_print_header, cli_convert_to_mlarray

        return {
            "cli_print_header": cli_print_header,
            "cli_convert_to_mlarray": cli_convert_to_mlarray,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
