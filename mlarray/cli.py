import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

from mlarray import MLArray


def _json_safe(value: Any) -> Any:
    """Make metadata printable as strict JSON.

    Non-finite floats become the strings "NaN", "Infinity" and "-Infinity"
    (bare NaN is not JSON); binary values are shown as ``<binary, N bytes>``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary, {len(bytes(value))} bytes>"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def header(filepath: Union[str, Path], source: bool = False, user: bool = False) -> dict:
    """The metadata of an ``.mla`` file as plain data; ``source`` and ``user`` only on request."""
    image = MLArray.open(filepath)
    out = image.meta.to_dict()
    if source:
        out["source"] = [entry.to_dict() for entry in image.source]
    if user:
        out["user"] = image.user
    return _json_safe(out)


def print_header(filepath: Union[str, Path], source: bool = False, user: bool = False) -> None:
    """Print the metadata of an ``.mla`` file as JSON."""
    print(json.dumps(header(filepath, source=source, user=user), indent=2, allow_nan=False))


def _medvol():
    """MedVol 2, which does all format conversion (MLArray itself reads and writes only .mla)."""
    try:
        import medvol
    except ImportError as error:
        if error.name != "medvol":   # MedVol is installed but one of its dependencies is missing
            raise
        raise RuntimeError("mlarray_convert needs MedVol 2: pip install 'mlarray[all]'") from error
    major = str(getattr(medvol, "__version__", "0")).split(".")[0]
    if not major.isdigit() or int(major) < 2:
        raise RuntimeError(f"mlarray_convert needs MedVol 2 or newer; found {medvol.__version__}. "
                           "Upgrade with: pip install -U 'mlarray[all]'")
    return medvol


def convert_image(load_filepath: Union[str, Path], save_filepath: Union[str, Path], **options):
    """Convert between any two formats MedVol supports (``.mla``, NIfTI, NRRD, MetaImage, MGH, DICOM).

    ``options`` are MedVol policy options (e.g. ``nifti_form="sform"``) and ``source_format`` /
    ``target_format``. Returns MedVol's report.
    """
    return _medvol().convert(load_filepath, save_filepath, **options)


def cli_print_header() -> None:
    parser = argparse.ArgumentParser(
        prog="mlarray_header",
        description="Print the metadata of an MLArray file as JSON.",
    )
    parser.add_argument("filepath", help="Path to a .mla file.")
    parser.add_argument("--source", action="store_true", help="Also print the original metadata of converted files.")
    parser.add_argument("--user", action="store_true", help="Also print the user metadata.")
    args = parser.parse_args()
    print_header(args.filepath, source=args.source, user=args.user)


def cli_convert_to_mlarray() -> None:
    parser = argparse.ArgumentParser(
        prog="mlarray_convert",
        description="Convert between MLArray (.mla) and NIfTI, NRRD, MetaImage, MGH/MGZ and DICOM with MedVol. "
                    "Nothing is reoriented, rescaled or dropped silently; the report lists every choice.",
    )
    parser.add_argument("load_filepath", help="Input file or DICOM directory.")
    parser.add_argument("save_filepath", help="Output file (or directory for DICOM).")
    parser.add_argument("--source-format", help="Input format if it cannot be detected (e.g. dicom).")
    parser.add_argument("--target-format", help="Output format if the extension does not say (e.g. dicom).")
    try:
        from medvol.policy import OPTIONS
    except ImportError:
        OPTIONS = {}
    for name, choices in OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", choices=choices, help=f"MedVol policy (default {choices[0]}).")
    args = parser.parse_args()
    options = {name: getattr(args, name) for name in OPTIONS if getattr(args, name) is not None}
    if args.source_format:
        options["source_format"] = args.source_format
    if args.target_format:
        options["target_format"] = args.target_format
    try:
        print(convert_image(args.load_filepath, args.save_filepath, **options))
    except (OSError, ValueError, RuntimeError, NotImplementedError) as error:
        parser.exit(1, f"mlarray_convert: error: {error}\n")
