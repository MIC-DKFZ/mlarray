import argparse
import json
from collections.abc import Mapping
from typing import Optional, Union
from pathlib import Path
import numpy as np
from mlarray import MLArray
from mlarray.meta import _meta_internal_write

try:
    from medvol import MedVol
except ImportError:
    MedVol = None


def _require_medvol() -> None:
    if MedVol is None:
        raise RuntimeError(
            "medvol is required for mlarray_convert; install with 'pip install mlarray[all]'."
        )


def _is_mlarray_path(filepath: Union[str, Path]) -> bool:
    return str(filepath).endswith(".mla")


def _is_nifti_path(filepath: Union[str, Path]) -> bool:
    filepath_str = str(filepath)
    return filepath_str.endswith(".nii.gz") or filepath_str.endswith(".nii")


def _is_nrrd_path(filepath: Union[str, Path]) -> bool:
    return str(filepath).endswith(".nrrd")


def _is_medvol_path(filepath: Union[str, Path]) -> bool:
    return _is_nifti_path(filepath) or _is_nrrd_path(filepath)


def _source_format_for_path(filepath: Union[str, Path]) -> Optional[str]:
    if _is_nifti_path(filepath):
        return "nifti"
    if _is_nrrd_path(filepath):
        return "nrrd"
    return None


def _medvol_backend_for_path(filepath: Union[str, Path]) -> Optional[str]:
    """Return the explicit backend to use when creating a MedVol for saving.

    NIfTI: None → MedVol auto-selects nibabel (canonical RAS+ geometry,
    fresh header from affine — dict source-metadata is intentionally not
    passed through since nibabel expects a Nifti1Header object).
    NRRD:  "pynrrd" → MedVol backend flag must match so pynrrd.save() picks
    up the source-metadata dict as the base header.
    """
    if _is_nrrd_path(filepath):
        return "pynrrd"
    return None



def _jsonable_header_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonable_header_value(value.item())
        return [_jsonable_header_value(v) for v in value.tolist()]
    if isinstance(value, bytes):
        # Must precede np.generic: np.bytes_ is a subclass of both bytes and
        # np.generic, and .item() on it returns Python bytes (not str).
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable_header_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_header_value(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        return _jsonable_header_value(converted)
    return value


def _header_to_source_meta(header) -> dict:
    if header is None:
        return {}

    raw = None
    if isinstance(header, Mapping):
        raw = dict(header)
    elif hasattr(header, "items"):
        raw = {str(k): v for k, v in header.items()}
    elif hasattr(header, "keys"):
        raw = {str(k): header[k] for k in header.keys()}
    else:
        raise TypeError(
            f"Unsupported MedVol header type for MLArray conversion: {type(header).__name__}"
        )

    converted = _jsonable_header_value(raw)
    if not isinstance(converted, dict):
        raise TypeError("Converted MedVol header must be a mapping.")
    return converted


def _coord_system_alias(coord_system: Optional[str]) -> Optional[str]:
    if coord_system in {"RAS", "RAS+"}:
        return "RAS+"
    if coord_system in {"LPS", "LPS+"}:
        return "LPS+"
    return coord_system


def print_header(filepath: Union[str, Path]) -> None:
    """Print the MLArray metadata header for a file.

    Args:
        filepath: Path to a ".mla" file.
    """
    meta = MLArray(filepath).meta
    if meta is None:
        print("null")
        return
    print(json.dumps(meta.to_plain(include_none=True), indent=2, sort_keys=True))


def convert_to_mlarray(load_filepath: Union[str, Path], save_filepath: Union[str, Path]):
    _require_medvol()
    image_meta_format = _source_format_for_path(load_filepath)
    # Let MedVol auto-detect the backend (nibabel for NIfTI, pynrrd for NRRD).
    # The default canonicalize=True reorients the array+affine to RAS+.
    image_medvol = MedVol(load_filepath)
    image_mlarray = MLArray(
        image_medvol.array,
        affine=image_medvol.affine,
        meta=_header_to_source_meta(image_medvol.header),
    )
    coord_system = _coord_system_alias(image_medvol.coordinate_system)
    if coord_system is not None:
        image_mlarray.meta.spatial.coord_system = coord_system
    with _meta_internal_write():
        image_mlarray.meta._image_meta_format = image_meta_format
    image_mlarray.save(save_filepath)


def convert_from_mlarray(load_filepath: Union[str, Path], save_filepath: Union[str, Path]):
    _require_medvol()
    image_mlarray = MLArray(load_filepath)
    backend = _medvol_backend_for_path(save_filepath)

    # MedVol always canonicalises to RAS+ on load, so data stored in an
    # MLArray is in RAS+ orientation.  The only exception is legacy LPS+
    # files created before that default existed.  Anything else (None,
    # "unknown", custom strings) is therefore safely treated as RAS+.
    stored_cs = image_mlarray.meta.spatial.coord_system
    coordinate_system = "LPS+" if stored_cs in {"LPS", "LPS+"} else "RAS+"

    header = dict(image_mlarray.meta.source.to_plain())
    if _is_nrrd_path(save_filepath):
        # Strip NRRD coordinate-system keys; pynrrd will re-set "space" from
        # the coordinate_system we pass.  "NRRD_space" is a SimpleITK
        # metadata artifact that must not appear in clean NRRD output.
        header.pop("space", None)
        header.pop("NRRD_space", None)

    MedVol(
        image_mlarray.to_numpy(),
        affine=image_mlarray.affine,
        header=header,
        backend=backend,
        coordinate_system=coordinate_system,
        canonicalize=False,
    ).save(save_filepath, backend=backend)


def convert_image(load_filepath: Union[str, Path], save_filepath: Union[str, Path]):
    if _is_medvol_path(load_filepath) and _is_mlarray_path(save_filepath):
        convert_to_mlarray(load_filepath, save_filepath)
        return

    if _is_mlarray_path(load_filepath) and _is_medvol_path(save_filepath):
        convert_from_mlarray(load_filepath, save_filepath)
        return

    raise RuntimeError(
        "Supported conversions are NIfTI/NRRD -> MLArray and MLArray -> NIfTI/NRRD."
    )


def cli_print_header() -> None:
    parser = argparse.ArgumentParser(
        prog="mlarray_header",
        description="Print the MLArray metadata header for a file.",
    )
    parser.add_argument("filepath", help="Path to a .mla file.")
    args = parser.parse_args()
    print_header(args.filepath)


def cli_convert_to_mlarray() -> None:
    parser = argparse.ArgumentParser(
        prog="mlarray_convert",
        description="Convert between MLArray and NIfTI/NRRD files.",
    )
    parser.add_argument(
        "load_filepath",
        help="Input path: MLArray (.mla), NIfTI (.nii.gz, .nii), or NRRD (.nrrd).",
    )
    parser.add_argument(
        "save_filepath",
        help="Output path: MLArray (.mla), NIfTI (.nii.gz, .nii), or NRRD (.nrrd).",
    )
    args = parser.parse_args()
    convert_image(args.load_filepath, args.save_filepath)
