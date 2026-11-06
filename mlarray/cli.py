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


def convert_image(load_filepath: Union[str, Path], save_filepath: Union[str, Path]) -> None:
    raise RuntimeError(
        "mlarray_convert is not available on this development branch: conversion is being rebuilt "
        "on MedVol 2, which writes MLArray v1 metadata."
    )


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
        description="Convert between MLArray and NIfTI/NRRD files.",
    )
    parser.add_argument("load_filepath", help="Input path.")
    parser.add_argument("save_filepath", help="Output path.")
    args = parser.parse_args()
    convert_image(args.load_filepath, args.save_filepath)
