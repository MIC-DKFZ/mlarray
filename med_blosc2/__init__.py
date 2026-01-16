"""A standardized blosc2 image reader and writer for medical images.."""

from importlib import metadata as _metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from med_blosc2.med_blosc2 import MedBlosc2, MED_BLOSC2_DEFAULT_PATCH_SIZE
    from med_blosc2.meta import Meta, MetaBlosc2, MetaSpatial
    from med_blosc2.utils import is_serializable
    from med_blosc2.cli import cli_print_header, cli_convert_to_medblosc2

__all__ = [
    "__version__",
    "MedBlosc2",
    "MED_BLOSC2_DEFAULT_PATCH_SIZE",
    "Meta",
    "MetaBlosc2",
    "MetaSpatial",
    "is_serializable",
    "cli_print_header",
    "cli_convert_to_medblosc2",
]

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - during editable installs pre-build
    __version__ = "0.0.0"


def __getattr__(name: str):
    if name in {"MedBlosc2", "MED_BLOSC2_DEFAULT_PATCH_SIZE"}:
        from med_blosc2.med_blosc2 import MedBlosc2, MED_BLOSC2_DEFAULT_PATCH_SIZE

        return MedBlosc2 if name == "MedBlosc2" else MED_BLOSC2_DEFAULT_PATCH_SIZE
    if name in {"Meta", "MetaBlosc2", "MetaSpatial"}:
        from med_blosc2.meta import Meta, MetaBlosc2, MetaSpatial

        return {"Meta": Meta, "MetaBlosc2": MetaBlosc2, "MetaSpatial": MetaSpatial}[name]
    if name == "is_serializable":
        from med_blosc2.utils import is_serializable

        return is_serializable
    if name in {"cli_print_header", "cli_convert_to_medblosc2"}:
        from med_blosc2.cli import cli_print_header, cli_convert_to_medblosc2

        return {
            "cli_print_header": cli_print_header,
            "cli_convert_to_medblosc2": cli_convert_to_medblosc2,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
