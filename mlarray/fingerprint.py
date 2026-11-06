"""Fingerprint of an array and its geometry (MLArray spec section 6.3).

MedVol writes a fingerprint into each ``source`` entry at conversion time and
compares it on export to decide whether data-dependent source fields (e.g.
NIfTI ``cal_min``, DICOM ``LargestPixelValue``) still describe the array.

- ``shape`` and ``dtype`` (NumPy dtype name);
- ``geometry``: ``"xxh3_64:<hex>"`` of the axes and spaces, with every float
  rounded to 10 significant digits, so differences far below the accuracy
  standard (spec 9: 1e-9 relative) do not change it;
- ``data``: ``"xxh3_128:<hex>"`` of the array values in C order.
"""

from typing import Any, Dict, Mapping, Optional

import msgpack
import numpy as np
import xxhash

#: Bytes hashed per step for arrays that are not in memory as one block.
_STEP_NBYTES = 64 * 1024**2


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, Mapping):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def geometry_hash(meta) -> str:
    """Hash of the core geometry: the axes and spaces of ``meta``."""
    geometry = {
        "axes": [axis.to_dict() for axis in meta.axes],
        "spaces": [space.to_dict() for space in meta.spaces],
    }
    return "xxh3_64:" + xxhash.xxh3_64_hexdigest(msgpack.packb(_rounded(geometry)))


def data_hash(array) -> str:
    """Hash of the array values in C order (NumPy arrays, Blosc2 arrays or MLArrays)."""
    hasher = xxhash.xxh3_128()
    shape = tuple(array.shape)
    if isinstance(array, np.ndarray) or not shape:
        hasher.update(np.ascontiguousarray(np.asarray(array)).tobytes())
    else:
        row_nbytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(array.dtype).itemsize)
        step = max(1, _STEP_NBYTES // row_nbytes)
        for start in range(0, shape[0], step):
            hasher.update(np.ascontiguousarray(array[start:start + step]).tobytes())
    return "xxh3_128:" + hasher.hexdigest()


def compute(array, meta) -> Dict[str, Any]:
    """Fingerprint of ``array`` (anything with ``shape``, ``dtype`` and slicing) and its ``meta``."""
    return {
        "shape": [int(v) for v in array.shape],
        "dtype": np.dtype(array.dtype).name,
        "geometry": geometry_hash(meta),
        "data": data_hash(array),
    }


def matches(fingerprint: Optional[Mapping], array, meta) -> bool:
    """True when ``fingerprint`` still describes ``array`` and its geometry.

    Shape, dtype and geometry are compared first, so the data is only hashed
    when they agree.
    """
    if not fingerprint:
        return False
    if list(fingerprint.get("shape", [])) != [int(v) for v in array.shape]:
        return False
    if fingerprint.get("dtype") != np.dtype(array.dtype).name:
        return False
    if fingerprint.get("geometry") != geometry_hash(meta):
        return False
    return fingerprint.get("data") == data_hash(array)
