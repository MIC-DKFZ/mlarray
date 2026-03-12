import argparse
import json
from typing import Optional, Union
from pathlib import Path
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


def _spatial_components_for_medvol(image_mlarray: MLArray):
    if image_mlarray.meta.spatial.affine is not None:
        return image_mlarray.scale, image_mlarray.translation, image_mlarray.rotation
    return image_mlarray.spacing, image_mlarray.origin, image_mlarray.direction


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
    image_medvol = MedVol(load_filepath)
    image_mlarray = MLArray(
        image_medvol.array,
        spacing=image_medvol.spacing,
        origin=image_medvol.origin,
        direction=image_medvol.direction,
        meta=image_medvol.header,
    )
    with _meta_internal_write():
        image_mlarray.meta._image_meta_format = image_meta_format
    image_mlarray.save(save_filepath)


def convert_from_mlarray(load_filepath: Union[str, Path], save_filepath: Union[str, Path]):
    _require_medvol()
    image_mlarray = MLArray(load_filepath)
    spacing, origin, direction = _spatial_components_for_medvol(image_mlarray)
    image_medvol = MedVol(
        image_mlarray.to_numpy(),
        spacing=spacing,
        origin=origin,
        direction=direction,
        header=image_mlarray.meta.source.to_plain(),
    )
    image_medvol.save(save_filepath)


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
