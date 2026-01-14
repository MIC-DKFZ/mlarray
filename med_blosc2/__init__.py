"""A standardized blosc2 image reader and writer for medical images.."""

from importlib import metadata as _metadata
from med_blosc2.med_blosc2 import MedBlosc2, MED_BLOSC2_DEFAULT_PATCH_SIZE

__all__ = ["__version__"]

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover - during editable installs pre-build
    __version__ = "0.0.0"
