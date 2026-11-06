"""MLArray v1 metadata model (spec ``docs/design/spec.md`` sections 4-8).

Core parts are immutable value objects (:class:`Axis`, :class:`Space`,
:class:`Values`, :class:`Storage`, :class:`SourceEntry`). They are replaced as a
whole through :class:`Meta`, which validates every change against the array
shape before applying it, so a failed change leaves the previous state intact.

``profiles`` and ``user`` are plain, mutable dicts; profiles are validated when
the metadata is written. ``source`` is read-only.

Unknown keys are kept at every level and written back unchanged.
"""

from __future__ import annotations

import copy as _copy
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlarray.units import unit_code as _curated_unit_code

SPEC_VERSION = "1.0"

AXIS_TYPES = ("space", "time", "channel", "component", "other")
VALUE_KINDS = ("measurement", "label", "instance", "probability", "vector", "other")
LABEL_LAYOUTS = ("index", "channel")
LABEL_KINDS = ("label", "instance", "probability")

# increases_toward vocabulary: word -> (opposite, display letter)
DIRECTION_WORDS: Dict[str, Tuple[str, str]] = {
    "left": ("right", "L"), "right": ("left", "R"),
    "anterior": ("posterior", "A"), "posterior": ("anterior", "P"),
    "superior": ("inferior", "S"), "inferior": ("superior", "I"),
    "east": ("west", "E"), "west": ("east", "W"),
    "north": ("south", "N"), "south": ("north", "S"),
    "up": ("down", "U"), "down": ("up", "D"),
}

_INT64_MIN, _UINT64_MAX = -(2**63), 2**64 - 1


class MetadataError(ValueError):
    """Metadata violates the MLArray specification."""


# ----------------------------------------------------------------------------
# Plain msgpack-native values
# ----------------------------------------------------------------------------

def plain(value: Any, path: str = "value", str_keys: bool = True) -> Any:
    """Return a deep, plain copy of ``value`` made of msgpack-native types.

    NumPy scalars and arrays are converted to Python values and lists, tuples
    to lists. Read-only views (:class:`FrozenMap`) are copied into dicts.

    Raises:
        TypeError: For a value msgpack cannot store, naming its path.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if not _INT64_MIN <= value <= _UINT64_MAX:
            raise TypeError(f"{path}: integer {value} does not fit in 64 bits.")
        return value
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        return plain(value.tolist(), path, str_keys)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if str_keys and not isinstance(key, str):
                raise TypeError(f"{path}: keys must be strings, got {type(key).__name__} {key!r}.")
            key = plain(key, f"{path} key", str_keys)
            out[key] = plain(item, f"{path}[{key!r}]", str_keys)
        return out
    if isinstance(value, (list, tuple)):
        return [plain(item, f"{path}[{i}]", str_keys) for i, item in enumerate(value)]
    raise TypeError(f"{path}: values of type {type(value).__name__} cannot be stored in MLArray metadata.")


class FrozenMap(Mapping):
    """An immutable mapping (read-only metadata). Picklable and copyable."""

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Mapping] = None):
        object.__setattr__(self, "_data", dict(data or {}))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __setattr__(self, name, value):
        raise AttributeError("Read-only metadata cannot be changed.")

    def __eq__(self, other):
        if isinstance(other, Mapping):
            return self._data == dict(other.items())
        return NotImplemented

    __hash__ = None

    def __repr__(self):
        return f"FrozenMap({self._data!r})"

    def __reduce__(self):
        return (FrozenMap, (self._data,))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def frozen(value: Any) -> Any:
    """Deep read-only copy of a plain value: dicts become FrozenMap, lists tuples."""
    if isinstance(value, Mapping):
        return FrozenMap({key: frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    return value


def thawed(value: Any) -> Any:
    """Deep plain copy of a (possibly frozen) value: mappings become dicts, tuples lists."""
    if isinstance(value, Mapping):
        return {key: thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thawed(item) for item in value]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _number(value: Any, path: str, finite: bool = True) -> float:
    if not _is_number(value):
        raise MetadataError(f"{path} must be a number, got {value!r}.")
    value = float(value)
    if finite and not math.isfinite(value):
        raise MetadataError(f"{path} must be finite, got {value!r}.")
    return value


def _number_or_int(value: Any, path: str, finite: bool = True) -> Union[int, float]:
    """A number that stays an int when given as one (keeps exact integer values)."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return _number(value, path, finite)


def _int(value: Any, path: str, minimum: Optional[int] = None) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise MetadataError(f"{path} must be an integer, got {value!r}.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise MetadataError(f"{path} must be >= {minimum}, got {value}.")
    return value


def _string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise MetadataError(f"{path} must be a non-empty string, got {value!r}.")
    return value


def _optional(value: Any, convert: Callable[[Any, str], Any], path: str) -> Any:
    return None if value is None else convert(value, path)


def _sequence(value: Any, path: str) -> list:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise MetadataError(f"{path} must be a list, got {value!r}.")
    return list(value)


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise MetadataError(f"{path} must be a map, got {value!r}.")
    return value


def _split(data: Mapping, known: Iterable[str], path: str) -> Tuple[Dict[str, Any], Mapping]:
    """Split a mapping into known fields and frozen unknown fields (kept verbatim)."""
    data = _mapping(data, path)
    known = tuple(known)
    unknown = {key: value for key, value in data.items() if key not in known}
    return {key: data[key] for key in known if key in data}, frozen(plain(unknown, path))


def _with_unknown(out: Dict[str, Any], unknown: Mapping) -> Dict[str, Any]:
    out.update(thawed(unknown))
    return out


def _unique(names: Sequence[str], path: str) -> None:
    seen = set()
    for name in names:
        if name in seen:
            raise MetadataError(f"{path}: name {name!r} is used more than once.")
        seen.add(name)


# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------

def _unit_fields(data: Mapping, path: str) -> Tuple[Optional[str], Optional[str]]:
    unit = _optional(data.get("unit"), _string, f"{path}.unit")
    code = _optional(data.get("unit_code"), _string, f"{path}.unit_code")
    if code is None:
        code = _curated_unit_code(unit)
    return unit, code


def _put_unit(out: Dict[str, Any], unit: Optional[str], code: Optional[str]) -> None:
    if unit is not None:
        out["unit"] = unit
    if code is not None:
        out["unit_code"] = code


# ----------------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """One array axis (spec 4.2). ``name`` is a name only, never a direction."""

    name: str
    type: str = "space"
    labels: Optional[Tuple[str, ...]] = None
    start: Optional[float] = None
    step: Optional[float] = None
    coordinates: Optional[Tuple[float, ...]] = None
    unit: Optional[str] = None
    unit_code: Optional[str] = None
    unknown: Mapping = field(default_factory=FrozenMap, compare=True, repr=False)

    _KEYS = ("name", "type", "labels", "start", "step", "coordinates", "unit", "unit_code")

    @classmethod
    def coerce(cls, value: Union["Axis", str, Mapping], path: str = "axes") -> "Axis":
        if isinstance(value, Axis):
            return value
        if isinstance(value, str):
            return cls.from_dict({"name": value, "type": "space"}, path)
        return cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str = "axes") -> "Axis":
        known, unknown = _split(data, cls._KEYS, path)
        if "name" not in known:
            raise MetadataError(f"{path}.name is required.")
        axis_type = known.get("type", "space")
        if axis_type not in AXIS_TYPES:
            raise MetadataError(f"{path}.type must be one of {AXIS_TYPES}, got {axis_type!r}.")
        labels = known.get("labels")
        if labels is not None:
            labels = tuple(_string(v, f"{path}.labels[{i}]", allow_empty=True) for i, v in enumerate(_sequence(labels, f"{path}.labels")))
        coordinates = known.get("coordinates")
        if coordinates is not None:
            coordinates = tuple(_number(v, f"{path}.coordinates[{i}]") for i, v in enumerate(_sequence(coordinates, f"{path}.coordinates")))
        start = _optional(known.get("start"), _number, f"{path}.start")
        step = _optional(known.get("step"), _number, f"{path}.step")
        if coordinates is not None and (start is not None or step is not None):
            raise MetadataError(f"{path}: give either coordinates or start/step, not both.")
        unit, code = _unit_fields(known, path)
        if axis_type == "space" and (start is not None or step is not None or coordinates is not None or unit is not None):
            raise MetadataError(f"{path}: space axes carry no sampling or unit; their geometry belongs in spaces.")
        return cls(
            name=_string(known["name"], f"{path}.name"),
            type=axis_type,
            labels=labels,
            start=start,
            step=step,
            coordinates=coordinates,
            unit=unit,
            unit_code=code,
            unknown=unknown,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type}
        if self.labels is not None:
            out["labels"] = list(self.labels)
        if self.start is not None:
            out["start"] = self.start
        if self.step is not None:
            out["step"] = self.step
        if self.coordinates is not None:
            out["coordinates"] = list(self.coordinates)
        _put_unit(out, self.unit, self.unit_code)
        return _with_unknown(out, self.unknown)

    def _check_length(self, length: int, path: str) -> None:
        for name in ("labels", "coordinates"):
            values = getattr(self, name)
            if values is not None and len(values) != length:
                raise MetadataError(f"{path}.{name} has {len(values)} entries, but the axis has length {length}.")


def default_axes(ndim: int) -> Tuple[Axis, ...]:
    """Axes for an array without declared axes: ``axis_0``, ``axis_1``, ... of type space."""
    return tuple(Axis(name=f"axis_{i}", type="space") for i in range(ndim))


# ----------------------------------------------------------------------------
# Spaces
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceAxis:
    """One axis of a space: its name, unit and the direction its coordinate grows."""

    name: str
    unit: Optional[str] = None
    unit_code: Optional[str] = None
    increases_toward: Optional[str] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("name", "unit", "unit_code", "increases_toward")

    @classmethod
    def coerce(cls, value: Union["SpaceAxis", str, Mapping], path: str) -> "SpaceAxis":
        if isinstance(value, SpaceAxis):
            return value
        if isinstance(value, str):
            value = {"name": value}
        return cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str) -> "SpaceAxis":
        known, unknown = _split(data, cls._KEYS, path)
        if "name" not in known:
            raise MetadataError(f"{path}.name is required.")
        unit, code = _unit_fields(known, path)
        if unit is None:
            raise MetadataError(f"{path}.unit is required.")
        return cls(
            name=_string(known["name"], f"{path}.name"),
            unit=unit,
            unit_code=code,
            increases_toward=_optional(known.get("increases_toward"), _string, f"{path}.increases_toward"),
            unknown=unknown,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        _put_unit(out, self.unit, self.unit_code)
        if self.increases_toward is not None:
            out["increases_toward"] = self.increases_toward
        return _with_unknown(out, self.unknown)


@dataclass(frozen=True)
class FromArray:
    """How array indices map into a space (spec 4.3).

    Either ``matrix`` (homogeneous, rows: space axes + 1, columns: ``array_axes``
    + translation) or the axis-aligned shorthand ``scale`` + ``translation``.
    An integer index is the centre of its voxel.
    """

    array_axes: Tuple[str, ...]
    matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    scale: Optional[Tuple[float, ...]] = None
    translation: Optional[Tuple[float, ...]] = None
    profile_mapping: Optional[Mapping] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("array_axes", "matrix", "scale", "translation", "profile_mapping")

    @classmethod
    def coerce(cls, value: Union["FromArray", Mapping], path: str) -> "FromArray":
        return value if isinstance(value, FromArray) else cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str) -> "FromArray":
        known, unknown = _split(data, cls._KEYS, path)
        if "array_axes" not in known:
            raise MetadataError(f"{path}.array_axes is required.")
        array_axes = tuple(_string(v, f"{path}.array_axes[{i}]") for i, v in enumerate(_sequence(known["array_axes"], f"{path}.array_axes")))
        if not array_axes:
            raise MetadataError(f"{path}.array_axes must not be empty.")
        _unique(array_axes, f"{path}.array_axes")
        matrix = known.get("matrix")
        scale = known.get("scale")
        translation = known.get("translation")
        if (matrix is None) == (scale is None):
            raise MetadataError(f"{path}: give exactly one of matrix or scale (+ translation).")
        if matrix is None and translation is None:
            translation = [0.0] * len(scale)
        if matrix is not None and translation is not None:
            raise MetadataError(f"{path}: translation belongs to the scale shorthand; the matrix has its own last column.")
        n = len(array_axes)
        if matrix is not None:
            rows = _sequence(matrix, f"{path}.matrix")
            matrix = tuple(
                tuple(_number(v, f"{path}.matrix[{r}][{c}]") for c, v in enumerate(_sequence(row, f"{path}.matrix[{r}]")))
                for r, row in enumerate(rows)
            )
            if len(matrix) < 2 or any(len(row) != n + 1 for row in matrix):
                raise MetadataError(f"{path}.matrix must have {n + 1} columns (array_axes + translation) and at least 2 rows.")
            if matrix[-1] != tuple([0.0] * n + [1.0]):
                raise MetadataError(f"{path}.matrix last row must be [0, ..., 0, 1], got {list(matrix[-1])}.")
            linear = np.array([row[:-1] for row in matrix[:-1]])
            for c in range(n):
                if not np.any(linear[:, c]):
                    raise MetadataError(f"{path}.matrix column {c} ({array_axes[c]}) is zero.")
        else:
            scale = tuple(_number(v, f"{path}.scale[{i}]") for i, v in enumerate(_sequence(scale, f"{path}.scale")))
            translation = tuple(_number(v, f"{path}.translation[{i}]") for i, v in enumerate(_sequence(translation, f"{path}.translation")))
            if len(scale) != n or len(translation) != n:
                raise MetadataError(f"{path}: scale and translation need one entry per array axis ({n}).")
            if any(s == 0 for s in scale):
                raise MetadataError(f"{path}.scale entries must not be zero, got {list(scale)}.")
        profile_mapping = known.get("profile_mapping")
        if profile_mapping is not None:
            profile_mapping = frozen(plain(_mapping(profile_mapping, f"{path}.profile_mapping"), f"{path}.profile_mapping"))
        return cls(array_axes, matrix, scale, translation, profile_mapping, unknown)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"array_axes": list(self.array_axes)}
        if self.matrix is not None:
            out["matrix"] = [list(row) for row in self.matrix]
        else:
            out["scale"] = list(self.scale)
            out["translation"] = list(self.translation)
        if self.profile_mapping is not None:
            out["profile_mapping"] = thawed(self.profile_mapping)
        return _with_unknown(out, self.unknown)

    @property
    def space_ndim(self) -> int:
        """Number of space axes this mapping produces."""
        return len(self.matrix) - 1 if self.matrix is not None else len(self.scale)

    def to_matrix(self) -> np.ndarray:
        """The homogeneous from-array matrix as float64 (the shorthand composed)."""
        if self.matrix is not None:
            return np.array(self.matrix, dtype=np.float64)
        n = len(self.scale)
        matrix = np.eye(n + 1, dtype=np.float64)
        matrix[:n, :n] = np.diag(self.scale)
        matrix[:n, n] = self.translation
        return matrix


@dataclass(frozen=True)
class Space:
    """A named coordinate system and how the array sits in it (spec 4.3)."""

    name: str
    axes: Tuple[SpaceAxis, ...]
    from_array: FromArray
    reference: Optional[Mapping] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("name", "reference", "axes", "from_array")

    @classmethod
    def coerce(cls, value: Union["Space", Mapping], path: str) -> "Space":
        return value if isinstance(value, Space) else cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str = "spaces") -> "Space":
        known, unknown = _split(data, cls._KEYS, path)
        for key in ("name", "axes", "from_array"):
            if key not in known:
                raise MetadataError(f"{path}.{key} is required.")
        axes = tuple(SpaceAxis.coerce(v, f"{path}.axes[{i}]") for i, v in enumerate(_sequence(known["axes"], f"{path}.axes")))
        if not axes:
            raise MetadataError(f"{path}.axes must not be empty.")
        _unique([a.name for a in axes], f"{path}.axes")
        from_array = FromArray.coerce(known["from_array"], f"{path}.from_array")
        if from_array.space_ndim != len(axes):
            raise MetadataError(
                f"{path}.from_array maps into {from_array.space_ndim} space axes, but the space has {len(axes)}."
            )
        reference = known.get("reference")
        if reference is not None:
            reference = plain(_mapping(reference, f"{path}.reference"), f"{path}.reference")
            for key in ("scheme", "id"):
                _string(reference.get(key), f"{path}.reference.{key}")
            reference = frozen(reference)
        return cls(_string(known["name"], f"{path}.name"), axes, from_array, reference, unknown)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.reference is not None:
            out["reference"] = thawed(self.reference)
        out["axes"] = [axis.to_dict() for axis in self.axes]
        out["from_array"] = self.from_array.to_dict()
        return _with_unknown(out, self.unknown)

    # -- derived geometry (never stored, spec 4.3 rule 6) ---------------------

    def matrix(self) -> np.ndarray:
        """Homogeneous from-array matrix (float64 copy). Columns follow ``from_array.array_axes``."""
        return self.from_array.to_matrix()

    @property
    def spacing(self) -> Tuple[float, ...]:
        """Distance between neighbouring voxel centres along each mapped array axis."""
        linear = self.matrix()[:-1, :-1]
        return tuple(float(v) for v in np.linalg.norm(linear, axis=0))

    @property
    def origin(self) -> Tuple[float, ...]:
        """Space coordinates of the centre of index (0, ..., 0)."""
        return tuple(float(v) for v in self.matrix()[:-1, -1])

    @property
    def direction(self) -> np.ndarray:
        """Unit vector of each mapped array axis in the space (rows: space axes, columns: array axes).

        For sheared grids the columns are not orthogonal; they are still unit vectors.
        """
        linear = self.matrix()[:-1, :-1]
        return linear / np.linalg.norm(linear, axis=0)

    def letter_codes(self) -> Tuple[str, ...]:
        """Display letters: the direction each mapped array axis increases toward.

        For each array axis, the space axis its column points along most, with
        the sign applied: e.g. ``("S", "P", "L")`` for a (z, y, x) array in the
        DICOM patient frame. ``"?"`` when the space axis has no known
        ``increases_toward``. Letters are derived for display and never stored.
        """
        letters = []
        for column in self.direction.T:
            row = int(np.argmax(np.abs(column)))
            word = self.axes[row].increases_toward
            if word not in DIRECTION_WORDS:
                letters.append("?")
                continue
            if column[row] < 0:
                word = DIRECTION_WORDS[word][0]
            letters.append(DIRECTION_WORDS[word][1])
        return tuple(letters)

    def same_frame(self, other: "Space") -> bool:
        """True when both spaces name the same physical frame (equal reference scheme and id)."""
        if self.reference is None or other.reference is None:
            return False
        return (self.reference["scheme"], self.reference["id"]) == (other.reference["scheme"], other.reference["id"])


#: Space presets for the convenience geometry arguments.
SPACE_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    # DICOM patient frame (LPS), recommended world frame for medical images (spec 5.1).
    "patient": [
        {"name": "x", "unit": "millimeter", "increases_toward": "left"},
        {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
        {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
    ],
}


def space_from_geometry(
        space: Union[str, Mapping, "Space"],
        array_axes: Sequence[str],
        spacing: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        direction: Optional[Any] = None,
) -> Space:
    """Build a space from spacing, origin and direction.

    All three follow the array axes ``array_axes`` (array order):

    - ``spacing[c]``: distance between neighbouring voxel centres along array axis ``c``;
    - ``direction[:, c]``: unit vector of array axis ``c`` in the space (rows: space axes);
    - ``origin``: space coordinates of the centre of index ``(0, ..., 0)``.

    The matrix is ``[[direction * spacing, origin], [0, ..., 0, 1]]``.

    Args:
        space: A preset name (``"patient"``: DICOM patient frame, LPS, millimeter),
            or a space description with ``name``, ``axes`` and optional
            ``reference`` (its ``from_array`` is replaced).
    """
    if isinstance(space, str):
        if space not in SPACE_PRESETS:
            raise MetadataError(f"Unknown space preset {space!r}; use one of {sorted(SPACE_PRESETS)} or describe the space.")
        description = {"name": space, "axes": _copy.deepcopy(SPACE_PRESETS[space])}
    elif isinstance(space, Space):
        description = {"name": space.name, "axes": [a.to_dict() for a in space.axes]}
        if space.reference is not None:
            description["reference"] = thawed(space.reference)
    else:
        description = {k: v for k, v in dict(space).items() if k != "from_array"}
    n_array = len(array_axes)
    n_space = len(description.get("axes", []))
    spacing = np.ones(n_array) if spacing is None else np.asarray(spacing, dtype=np.float64)
    origin = np.zeros(n_space) if origin is None else np.asarray(origin, dtype=np.float64)
    if direction is None:
        if n_space != n_array:
            raise MetadataError("direction is required when the space and the array have different numbers of axes.")
        direction = np.eye(n_array)
    direction = np.asarray(direction, dtype=np.float64)
    if spacing.shape != (n_array,):
        raise MetadataError(f"spacing needs one entry per array axis {list(array_axes)}, got {spacing.tolist()}.")
    if origin.shape != (n_space,):
        raise MetadataError(f"origin needs one entry per space axis ({n_space}), got {origin.tolist()}.")
    if direction.shape != (n_space, n_array):
        raise MetadataError(f"direction must have shape ({n_space}, {n_array}) (rows: space axes, columns: array axes), got {direction.shape}.")
    if not (np.all(np.isfinite(spacing)) and np.all(spacing > 0)):
        raise MetadataError(f"spacing must be positive and finite, got {spacing.tolist()}.")
    norms = np.linalg.norm(direction, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise MetadataError(f"direction columns must be unit vectors, got norms {norms.tolist()}.")
    matrix = np.zeros((n_space + 1, n_array + 1))
    matrix[:n_space, :n_array] = direction * spacing
    matrix[:n_space, n_array] = origin
    matrix[n_space, n_array] = 1.0
    description["from_array"] = {"array_axes": list(array_axes), "matrix": matrix.tolist()}
    return Space.from_dict(description, "space")


# ----------------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Values:
    """What the stored numbers mean (spec 4.4). Replaces v0's ``is_seg``."""

    kind: str = "measurement"
    unit: Optional[str] = None
    unit_code: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    fill: Optional[Union[int, float]] = None
    labels: Optional[Mapping] = None
    label_layout: Optional[str] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("kind", "unit", "unit_code", "scale", "offset", "fill", "labels", "label_layout")

    @classmethod
    def coerce(cls, value: Union["Values", Mapping, None], path: str = "values") -> "Values":
        if value is None:
            return cls()
        return value if isinstance(value, Values) else cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str = "values") -> "Values":
        known, unknown = _split(data, cls._KEYS, path)
        kind = known.get("kind", "measurement")
        if kind not in VALUE_KINDS:
            raise MetadataError(f"{path}.kind must be one of {VALUE_KINDS}, got {kind!r}.")
        unit, code = _unit_fields(known, path)
        labels = known.get("labels")
        if labels is not None:
            labels = frozen(_labels(labels, f"{path}.labels"))
        layout = known.get("label_layout")
        if layout is not None and layout not in LABEL_LAYOUTS:
            raise MetadataError(f"{path}.label_layout must be one of {LABEL_LAYOUTS}, got {layout!r}.")
        if (labels is not None or layout is not None) and kind not in LABEL_KINDS:
            raise MetadataError(f"{path}: labels and label_layout need kind label, instance or probability, not {kind!r}.")
        return cls(
            kind=kind,
            unit=unit,
            unit_code=code,
            scale=_optional(known.get("scale"), _number, f"{path}.scale"),
            offset=_optional(known.get("offset"), _number, f"{path}.offset"),
            fill=_optional(known.get("fill"), lambda v, p: _number_or_int(v, p, finite=False), f"{path}.fill"),
            labels=labels,
            label_layout=layout,
            unknown=unknown,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        _put_unit(out, self.unit, self.unit_code)
        for name in ("scale", "offset", "fill"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.labels is not None:
            out["labels"] = thawed(self.labels)
        if self.label_layout is not None:
            out["label_layout"] = self.label_layout
        return _with_unknown(out, self.unknown)

    def without_label_semantics(self) -> "Values":
        """These values with labels, label layout and label kinds removed."""
        kind = "measurement" if self.kind in LABEL_KINDS else self.kind
        return replace(self, kind=kind, labels=None, label_layout=None)


def _labels(value: Any, path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in _mapping(value, path).items():
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            key = str(int(key))
        if not isinstance(key, str) or not key.lstrip("-").isdigit():
            raise MetadataError(f"{path}: keys must be integer values (or their decimal strings), got {key!r}.")
        entry = plain(_mapping(entry, f"{path}[{key!r}]"), f"{path}[{key!r}]")
        if "name" in entry:
            _string(entry["name"], f"{path}[{key!r}].name", allow_empty=True)
        if "color" in entry:
            color = _sequence(entry["color"], f"{path}[{key!r}].color")
            if len(color) not in (3, 4) or any(_int(c, f"{path}[{key!r}].color") > 255 or c < 0 for c in color):
                raise MetadataError(f"{path}[{key!r}].color must be 3 or 4 integers in 0..255, got {color!r}.")
        if "code" in entry:
            code = _mapping(entry["code"], f"{path}[{key!r}].code")
            for part in ("scheme", "value"):
                _string(code.get(part), f"{path}[{key!r}].code.{part}")
        out[key] = entry
    return out


# ----------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Storage:
    """Portable blosc2 layout of the file (spec 8). Never thread counts or other runtime settings."""

    chunk_size: Optional[Tuple[int, ...]] = None
    block_size: Optional[Tuple[int, ...]] = None
    patch_size: Optional[Tuple[int, ...]] = None
    compression: Optional[Mapping] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("chunk_size", "block_size", "patch_size", "compression")
    _COMPRESSION_KEYS = ("codec", "level", "filters")

    @classmethod
    def coerce(cls, value: Union["Storage", Mapping, None], path: str = "storage") -> Optional["Storage"]:
        if value is None:
            return None
        return value if isinstance(value, Storage) else cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str = "storage") -> "Storage":
        known, unknown = _split(data, cls._KEYS, path)
        sizes = {}
        for name in ("chunk_size", "block_size", "patch_size"):
            value = known.get(name)
            if value is not None:
                value = tuple(_int(v, f"{path}.{name}[{i}]", minimum=0) for i, v in enumerate(_sequence(value, f"{path}.{name}")))
            sizes[name] = value
        compression = known.get("compression")
        if compression is not None:
            compression = plain(_mapping(compression, f"{path}.compression"), f"{path}.compression")
            if "codec" in compression:
                _string(compression["codec"], f"{path}.compression.codec")
            if "level" in compression:
                _int(compression["level"], f"{path}.compression.level", minimum=0)
            if "filters" in compression:
                for i, name in enumerate(_sequence(compression["filters"], f"{path}.compression.filters")):
                    _string(name, f"{path}.compression.filters[{i}]")
            compression = frozen(compression)
        return cls(compression=compression, unknown=unknown, **sizes)

    def to_cparams(self, nthreads: Optional[int] = None):
        """The recorded compression as ``blosc2.CParams``.

        ``nthreads`` is a runtime choice and never stored; it defaults to
        MLArray's 4 compression threads.
        """
        from mlarray.storage import cparams_from_compression

        return cparams_from_compression(self.compression, nthreads=nthreads)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in ("chunk_size", "block_size", "patch_size"):
            if getattr(self, name) is not None:
                out[name] = list(getattr(self, name))
        if self.compression is not None:
            out["compression"] = thawed(self.compression)
        return _with_unknown(out, self.unknown)


# ----------------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceEntry:
    """Original metadata of one converted source (spec 6). Read-only."""

    format: str
    header: Mapping = field(default_factory=FrozenMap)
    format_version: Optional[str] = None
    files: Tuple[str, ...] = ()
    reader: Optional[Mapping] = None
    index_map: Optional[Mapping] = None
    fingerprint: Optional[Mapping] = None
    unknown: Mapping = field(default_factory=FrozenMap, repr=False)

    _KEYS = ("format", "format_version", "files", "reader", "index_map", "fingerprint", "header")

    @classmethod
    def coerce(cls, value: Union["SourceEntry", Mapping], path: str) -> "SourceEntry":
        return value if isinstance(value, SourceEntry) else cls.from_dict(value, path)

    @classmethod
    def from_dict(cls, data: Mapping, path: str = "source") -> "SourceEntry":
        known, unknown = _split(data, cls._KEYS, path)
        if "format" not in known:
            raise MetadataError(f"{path}.format is required.")
        files = tuple(_string(v, f"{path}.files[{i}]") for i, v in enumerate(_sequence(known.get("files", []), f"{path}.files")))
        header = known.get("header", {})
        _mapping(header, f"{path}.header")
        reader = known.get("reader")
        if reader is not None:
            reader = frozen(plain(_mapping(reader, f"{path}.reader"), f"{path}.reader"))
        index_map = known.get("index_map")
        if index_map is not None:
            index_map = frozen(_index_map(index_map, f"{path}.index_map"))
        fingerprint = known.get("fingerprint")
        if fingerprint is not None:
            fingerprint = frozen(plain(_mapping(fingerprint, f"{path}.fingerprint"), f"{path}.fingerprint"))
        return cls(
            format=_string(known["format"], f"{path}.format"),
            header=frozen(plain(header, f"{path}.header", str_keys=False)),
            format_version=_optional(known.get("format_version"), _string, f"{path}.format_version"),
            files=files,
            reader=reader,
            index_map=index_map,
            fingerprint=fingerprint,
            unknown=unknown,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"format": self.format}
        if self.format_version is not None:
            out["format_version"] = self.format_version
        if self.files:
            out["files"] = list(self.files)
        for name in ("reader", "index_map", "fingerprint"):
            if getattr(self, name) is not None:
                out[name] = thawed(getattr(self, name))
        out["header"] = thawed(self.header)
        return _with_unknown(out, self.unknown)


def _index_map(value: Any, path: str) -> Dict[str, Any]:
    data = plain(_mapping(value, path), path)
    order = [_int(v, f"{path}.axis_order[{i}]", minimum=0) for i, v in enumerate(_sequence(data.get("axis_order", []), f"{path}.axis_order"))]
    if sorted(order) != list(range(len(order))):
        raise MetadataError(f"{path}.axis_order must be a permutation of 0..n-1, got {order}.")
    n = len(order)
    flipped = _sequence(data.get("flipped", [False] * n), f"{path}.flipped")
    offset = _sequence(data.get("offset", [0] * n), f"{path}.offset")
    if len(flipped) != n or any(not isinstance(v, bool) for v in flipped):
        raise MetadataError(f"{path}.flipped must be {n} booleans, got {flipped!r}.")
    if len(offset) != n:
        raise MetadataError(f"{path}.offset must have {n} entries, got {offset!r}.")
    for i, v in enumerate(offset):
        _int(v, f"{path}.offset[{i}]")
    return data


# ----------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------

KNOWN_PROFILES = ("medical", "ml")


def validate_profiles(profiles: Mapping, axes: Sequence[Axis], spaces: Sequence[Space], shape: Optional[Sequence[int]]) -> Dict[str, Dict[str, Any]]:
    """Validate profiles and return a plain copy. Unknown profiles are kept as they are."""
    out = plain(_mapping(profiles, "profiles"), "profiles")
    for name, profile in out.items():
        path = f"profiles.{name}"
        _mapping(profile, path)
        if "profile_version" in profile:
            _string(profile["profile_version"], f"{path}.profile_version")
        if name == "ml":
            _validate_ml(profile, path, axes, spaces, shape)
        elif name == "medical":
            if "dicom" in profile:
                _mapping(profile["dicom"], f"{path}.dicom")
    return out


def _validate_ml(profile: Mapping, path: str, axes: Sequence[Axis], spaces: Sequence[Space], shape) -> None:
    statistics = profile.get("statistics")
    if statistics is not None:
        _mapping(statistics, f"{path}.statistics")
        for key in ("min", "max", "mean", "median", "standard_deviation"):
            if statistics.get(key) is not None:
                _number(statistics[key], f"{path}.statistics.{key}", finite=False)
        percentiles = statistics.get("percentiles")
        if percentiles is not None:
            for key, value in _mapping(percentiles, f"{path}.statistics.percentiles").items():
                _number(value, f"{path}.statistics.percentiles[{key!r}]", finite=False)
        if statistics.get("min") is not None and statistics.get("max") is not None and statistics["min"] > statistics["max"]:
            raise MetadataError(f"{path}.statistics: min {statistics['min']} > max {statistics['max']}.")
    boxes = profile.get("boxes")
    if boxes is not None:
        _validate_boxes(_mapping(boxes, f"{path}.boxes"), f"{path}.boxes", axes, spaces, shape)
    patch_size = profile.get("patch_size")
    if patch_size is not None:
        for i, v in enumerate(_sequence(patch_size, f"{path}.patch_size")):
            _int(v, f"{path}.patch_size[{i}]", minimum=1)
    if profile.get("split") is not None:
        _string(profile["split"], f"{path}.split")


def _validate_boxes(boxes: Mapping, path: str, axes: Sequence[Axis], spaces: Sequence[Space], shape) -> None:
    frame = _string(boxes.get("frame"), f"{path}.frame")
    names = [_string(v, f"{path}.axes[{i}]") for i, v in enumerate(_sequence(boxes.get("axes"), f"{path}.axes"))]
    _unique(names, f"{path}.axes")
    if frame == "index":
        known = [axis.name for axis in axes]
    else:
        space = next((s for s in spaces if s.name == frame), None)
        if space is None:
            raise MetadataError(f"{path}.frame must be 'index' or a space name, got {frame!r}.")
        known = [axis.name for axis in space.axes]
    for name in names:
        if name not in known:
            raise MetadataError(f"{path}.axes: {name!r} is not an axis of frame {frame!r} ({known}).")
    for i, item in enumerate(_sequence(boxes.get("items", []), f"{path}.items")):
        item_path = f"{path}.items[{i}]"
        _mapping(item, item_path)
        lo = [_number(v, f"{item_path}.min") for v in _sequence(item.get("min"), f"{item_path}.min")]
        hi = [_number(v, f"{item_path}.max") for v in _sequence(item.get("max"), f"{item_path}.max")]
        if len(lo) != len(names) or len(hi) != len(names):
            raise MetadataError(f"{item_path}: min and max need one entry per box axis ({len(names)}).")
        if any(a > b for a, b in zip(lo, hi)):
            raise MetadataError(f"{item_path}: min {lo} exceeds max {hi}.")
        if "score" in item:
            _number(item["score"], f"{item_path}.score")


# ----------------------------------------------------------------------------
# Lazily decoded parts (source, user)
# ----------------------------------------------------------------------------

class _Lazy:
    """A metadata part decoded on first access.

    While unloaded it holds ``raw``, a callable returning the stored msgpack
    bytes, and ``decode``. ``origin`` identifies the store the bytes live in,
    so rewriting that store can skip the part, and saving elsewhere can copy
    the bytes without decoding them.
    """

    __slots__ = ("_value", "_raw", "_decode", "origin")

    def __init__(self, value: Any = None, raw: Optional[Callable[[], bytes]] = None,
                 decode: Optional[Callable[[bytes], Any]] = None, origin: Any = None):
        self._value = value
        self._raw = raw
        self._decode = decode
        self.origin = origin

    @property
    def loaded(self) -> bool:
        return self._raw is None

    def raw(self) -> bytes:
        """The stored bytes of an unloaded part."""
        return self._raw()

    def get(self) -> Any:
        if self._raw is not None:
            self._value = self._decode(self._raw())
            self._raw = None
            self._decode = None
        return self._value

    def set(self, value: Any) -> None:
        self._value = value
        self._raw = None
        self._decode = None
        self.origin = None

    def clone(self) -> "_Lazy":
        """Independent copy; an unloaded part stays unloaded."""
        if self._raw is not None:
            return _Lazy(raw=self._raw, decode=self._decode, origin=self.origin)
        return _Lazy(_copy.deepcopy(self._value))

    def __deepcopy__(self, memo):
        return _Lazy(_copy.deepcopy(self.get(), memo))

    def __reduce__(self):
        return (_Lazy, (self.get(),))


# ----------------------------------------------------------------------------
# Meta
# ----------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Meta:
    """All metadata of an MLArray: core, profiles, history, storage, source and user.

    Core parts are replaced as a whole (``meta.axes = [...]``); each change is
    validated against the array shape and leaves the metadata unchanged when it
    fails. ``profiles`` and ``user`` are plain dicts that can be edited in place;
    profiles are validated whenever the metadata is validated or written.
    ``source`` is read-only.
    """

    _CORE_KEYS = ("version", "writer", "axes", "spaces", "values", "profiles", "history", "storage")

    def __init__(
            self,
            axes: Optional[Sequence[Union[Axis, str, Mapping]]] = None,
            spaces: Optional[Sequence[Union[Space, Mapping]]] = None,
            values: Optional[Union[Values, Mapping]] = None,
            profiles: Optional[Mapping] = None,
            history: Optional[Sequence[Mapping]] = None,
            storage: Optional[Union[Storage, Mapping]] = None,
            source: Optional[Sequence[Union[SourceEntry, Mapping]]] = None,
            user: Optional[Mapping] = None,
            version: str = SPEC_VERSION,
            writer: Optional[str] = None,
            unknown: Optional[Mapping] = None,
            shape: Optional[Sequence[int]] = None,
    ) -> None:
        self._shape = None if shape is None else tuple(int(v) for v in shape)
        self.version = _string(version, "version")
        if not self.version.startswith("1."):
            raise MetadataError(f"Unsupported metadata version {self.version!r}; this library reads 1.x.")
        self.writer = _optional(writer, _string, "writer")
        self._axes: Tuple[Axis, ...] = ()
        self._spaces: Tuple[Space, ...] = ()
        self._values = Values()
        self._storage: Optional[Storage] = None
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []
        self._source = _Lazy(())
        self._user = _Lazy({})
        self.unknown = frozen(plain(unknown or {}, "unknown keys"))
        self.update(
            axes=axes if axes is not None else (default_axes(len(self._shape)) if self._shape is not None else ()),
            spaces=spaces or (),
            values=values,
            profiles=profiles or {},
            storage=storage,
        )
        for entry in history or ():
            self._history.append(_history_entry(entry))
        if source is not None:
            self._source = _Lazy(_source_entries(source))
        if user is not None:
            self.user = user

    # -- core parts ----------------------------------------------------------

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the array the metadata describes (None without an array)."""
        return self._shape

    @property
    def axes(self) -> Tuple[Axis, ...]:
        return self._axes

    @axes.setter
    def axes(self, value):
        self.update(axes=value)

    @property
    def spaces(self) -> Tuple[Space, ...]:
        return self._spaces

    @spaces.setter
    def spaces(self, value):
        self.update(spaces=value)

    @property
    def values(self) -> Values:
        return self._values

    @values.setter
    def values(self, value):
        self.update(values=value)

    @property
    def storage(self) -> Optional[Storage]:
        return self._storage

    @property
    def history(self) -> Tuple[Mapping, ...]:
        """Append-only history of conversions and operations (read-only view)."""
        return tuple(frozen(entry) for entry in self._history)

    def add_history(self, action: str, **fields: Any) -> None:
        """Append a history entry; ``time`` defaults to now (UTC)."""
        fields.setdefault("time", _now())
        self._history.append(_history_entry({"action": action, **fields}))

    def update(self, **parts: Any) -> None:
        """Replace several core parts at once, validated together (all or nothing).

        Accepts ``axes``, ``spaces``, ``values``, ``profiles`` and ``storage``.
        """
        allowed = {"axes", "spaces", "values", "profiles", "storage"}
        unknown = set(parts) - allowed
        if unknown:
            raise TypeError(f"Unknown metadata parts {sorted(unknown)}; allowed: {sorted(allowed)}.")
        axes = self._axes if "axes" not in parts else tuple(
            Axis.coerce(v, f"axes[{i}]") for i, v in enumerate(_sequence(parts["axes"], "axes"))
        )
        spaces = self._spaces if "spaces" not in parts else tuple(
            Space.coerce(v, f"spaces[{i}]") for i, v in enumerate(_sequence(parts["spaces"], "spaces"))
        )
        values = self._values if "values" not in parts else Values.coerce(parts["values"])
        storage = self._storage if "storage" not in parts else Storage.coerce(parts["storage"])
        profiles = self.profiles if "profiles" not in parts else parts["profiles"]
        _validate_core(axes, spaces, values, self._shape)
        profiles = validate_profiles(profiles, axes, spaces, self._shape)
        self._axes, self._spaces, self._values, self._storage, self.profiles = axes, spaces, values, storage, profiles

    def validate(self) -> None:
        """Check the whole metadata against the spec and the array shape."""
        _validate_core(self._axes, self._spaces, self._values, self._shape)
        self.profiles = validate_profiles(self.profiles, self._axes, self._spaces, self._shape)
        if self._user.loaded:  # undecoded user metadata is written back as it was read
            plain(self.user, "user", str_keys=False)

    def _set_shape(self, shape: Optional[Sequence[int]], axes: Optional[Sequence] = None) -> None:
        """Bind the metadata to an array shape (internal), validating it."""
        new_shape = None if shape is None else tuple(int(v) for v in shape)
        old_shape = self._shape
        self._shape = new_shape
        try:
            if axes is not None:
                self.update(axes=axes)
            elif new_shape is not None and not self._axes:
                self.update(axes=default_axes(len(new_shape)))
            else:
                self.validate()
        except BaseException:
            self._shape = old_shape
            raise

    # -- lazy parts ----------------------------------------------------------

    @property
    def source(self) -> Tuple[SourceEntry, ...]:
        """Original metadata of the converted file(s); read-only, decoded on first access."""
        value = self._source.get()
        if not isinstance(value, tuple):
            value = _source_entries(value)
            self._source._value = value
        return value

    @property
    def user(self) -> Dict[str, Any]:
        """Arbitrary user metadata, never interpreted; decoded on first access."""
        return self._user.get()

    @user.setter
    def user(self, value: Mapping) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"user metadata must be a map, got {type(value).__name__}.")
        self._user.set(plain(value, "user", str_keys=False))

    # -- copies and equality --------------------------------------------------

    def copy(self, source: bool = True) -> "Meta":
        """Deep copy that shares no mutable state with this metadata."""
        clone = Meta.__new__(Meta)
        clone.__dict__.update(self.__dict__)
        clone.profiles = _copy.deepcopy(self.profiles)
        clone._history = _copy.deepcopy(self._history)
        clone._user = self._user.clone()
        clone._source = self._source.clone() if source else _Lazy(())
        return clone

    def __deepcopy__(self, memo):
        return self.copy()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Meta):
            return NotImplemented
        return self.to_dict(include_lazy=True) == other.to_dict(include_lazy=True)

    def __repr__(self) -> str:
        return f"Meta(axes={[a.name for a in self._axes]}, spaces={[s.name for s in self._spaces]}, values={self._values.kind!r})"

    # -- serialisation -------------------------------------------------------

    def to_dict(self, include_lazy: bool = False) -> Dict[str, Any]:
        """Plain content of the ``mlarray`` key (plus ``source``/``user`` with ``include_lazy``)."""
        out: Dict[str, Any] = {"version": self.version, "writer": self.writer}
        out["axes"] = [axis.to_dict() for axis in self._axes]
        if self._spaces:
            out["spaces"] = [space.to_dict() for space in self._spaces]
        out["values"] = self._values.to_dict()
        if self.profiles:
            out["profiles"] = _copy.deepcopy(self.profiles)
        if self._history:
            out["history"] = _copy.deepcopy(self._history)
        if self._storage is not None:
            out["storage"] = self._storage.to_dict()
        out.update(thawed(self.unknown))
        if include_lazy:
            out["source"] = [entry.to_dict() for entry in self.source]
            out["user"] = _copy.deepcopy(self.user)
        return out

    @classmethod
    def from_dict(cls, data: Mapping, shape: Optional[Sequence[int]] = None,
                  source: Optional[_Lazy] = None, user: Optional[_Lazy] = None) -> "Meta":
        """Build metadata from the plain content of the ``mlarray`` key.

        ``source`` and ``user`` entries in ``data`` (the format of
        ``to_dict(include_lazy=True)``) are accepted too.
        """
        data = _mapping(data, "mlarray")
        if "version" not in data:
            raise MetadataError("mlarray.version is required.")
        if source is None and "source" in data:
            source = _Lazy(_source_entries(data["source"]))
        if user is None and "user" in data:
            user = _Lazy(plain(_mapping(data["user"], "user"), "user", str_keys=False))
        meta = cls(
            axes=data.get("axes", []),
            spaces=data.get("spaces", []),
            values=data.get("values"),
            profiles=data.get("profiles", {}),
            history=data.get("history", []),
            storage=data.get("storage"),
            version=data["version"],
            writer=data.get("writer"),
            unknown={k: v for k, v in data.items() if k not in cls._CORE_KEYS + ("source", "user")},
            shape=shape,
        )
        if source is not None:
            meta._source = source
        if user is not None:
            meta._user = user
        return meta


def _history_entry(entry: Mapping) -> Dict[str, Any]:
    entry = plain(_mapping(entry, "history entry"), "history entry")
    _string(entry.get("action"), "history entry action")
    return entry


def _source_entries(value: Any) -> Tuple[SourceEntry, ...]:
    return tuple(SourceEntry.coerce(v, f"source[{i}]") for i, v in enumerate(_sequence(value, "source")))


def _validate_core(axes: Sequence[Axis], spaces: Sequence[Space], values: Values, shape: Optional[Sequence[int]]) -> None:
    names = [axis.name for axis in axes]
    _unique(names, "axes")
    if shape is not None:
        if len(axes) != len(shape):
            raise MetadataError(f"axes has {len(axes)} entries, but the array has {len(shape)} dimensions.")
        for i, (axis, length) in enumerate(zip(axes, shape)):
            axis._check_length(length, f"axes[{i}] ({axis.name})")
    _unique([space.name for space in spaces], "spaces")
    by_name = {axis.name: axis for axis in axes}
    for i, space in enumerate(spaces):
        for name in space.from_array.array_axes:
            axis = by_name.get(name)
            if axis is None:
                raise MetadataError(f"spaces[{i}] ({space.name}).from_array.array_axes: {name!r} is not an array axis {names}.")
            if axis.type != "space":
                raise MetadataError(f"spaces[{i}] ({space.name}): array axis {name!r} has type {axis.type!r}; only space axes map into a space.")
    if values.label_layout == "channel" and not any(axis.type == "channel" for axis in axes):
        raise MetadataError("values.label_layout 'channel' needs an array axis of type channel.")


__all__ = [
    "SPEC_VERSION",
    "MetadataError",
    "Axis",
    "SpaceAxis",
    "FromArray",
    "Space",
    "Values",
    "Storage",
    "SourceEntry",
    "Meta",
    "SPACE_PRESETS",
    "space_from_geometry",
    "default_axes",
    "FrozenMap",
    "plain",
    "frozen",
    "thawed",
]
