import argparse
import json
from typing import Union
from pathlib import Path
from mlarray import MLArray

try:
    from medvol import MedVol
except ImportError:
    MedVol = None


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
    if MedVol is None:
        raise RuntimeError("medvol is required for mlarray_convert; install with 'pip install mlarray[all]'.")
    image_meta_format = None
    if str(load_filepath).endswith(f".nii.gz") or str(load_filepath).endswith(f".nii"):
        image_meta_format = "nifti"
    elif str(load_filepath).endswith(f".nrrd"):
        image_meta_format = "nrrd"
    image_medvol = MedVol(load_filepath)
    image_mlarray = MLArray(image_medvol.array, spacing=image_medvol.spacing, origin=image_medvol.origin, direction=image_medvol.direction, meta=image_medvol.header)
    image_mlarray.meta._image_meta_format = image_meta_format
    image_mlarray.save(save_filepath)


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
        description="Convert a NiFTi or NRRD file to MLArray and copy all metadata.",
    )
    parser.add_argument("load_filepath", help="Path to the NiFTi (.nii.gz, .nii) or NRRD (.nrrd) file to load.")
    parser.add_argument("save_filepath", help="Path to the MLArray (.mla) file to save.")
    args = parser.parse_args()
    convert_to_mlarray(args.load_filepath, args.save_filepath)
