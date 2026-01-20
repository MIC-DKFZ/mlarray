import argparse
import json
from typing import Union
from pathlib import Path
from med_blosc2 import MedBlosc2

try:
    from medvol import MedVol
except ImportError:
    MedVol = None


def print_header(filepath: Union[str, Path]) -> None:
    """Print the MedBlosc2 metadata header for a file.

    Args:
        filepath: Path to a ".mb2nd" or ".b2nd" file.
    """
    meta = MedBlosc2(filepath).meta
    if meta is None:
        print("null")
        return
    print(json.dumps(meta.to_dict(include_none=True), indent=2, sort_keys=True))


def convert_to_medblosc2(load_filepath: Union[str, Path], save_filepath: Union[str, Path]):
    if MedVol is None:
        raise RuntimeError("medvol is required for medblosc2_convert; install with 'pip install med-blosc2[all]'.")
    image_medvol = MedVol(load_filepath)
    image_medblosc2 = MedBlosc2(image_medvol.array, spacing=image_medvol.spacing, origin=image_medvol.origin, direction=image_medvol.direction, meta=image_medvol.header)
    image_medblosc2.save(save_filepath)


def cli_print_header() -> None:
    parser = argparse.ArgumentParser(
        prog="medblosc2_header",
        description="Print the MedBlosc2 metadata header for a file.",
    )
    parser.add_argument("filepath", help="Path to a .mb2nd or .b2nd file.")
    args = parser.parse_args()
    print_header(args.filepath)


def cli_convert_to_medblosc2() -> None:
    parser = argparse.ArgumentParser(
        prog="medblosc2_convert",
        description="Convert a NiFTi or NRRD file to MedBlosc2 and copy all metadata.",
    )
    parser.add_argument("load_filepath", help="Path to the NiFTi (.nii.gz, .nii) or NRRD (.nrrd) file to load.")
    parser.add_argument("save_filepath", help="Path to the MedBlosc2 (.mb2nd) file to save.")
    args = parser.parse_args()
    convert_to_medblosc2(args.load_filepath, args.save_filepath)
