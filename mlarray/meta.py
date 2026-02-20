from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Mapping, Optional, Type, TypeVar, Union, TypeAlias, Iterable
from enum import Enum

import numpy as np
from mlarray.utils import is_serializable

T = TypeVar("T", bound="BaseMeta")
SK = TypeVar("SK", bound="SingleKeyBaseMeta")


def _is_unset_value(v: Any) -> bool:
    """Return True when a value should be treated as "unset".

    This is used by BaseMeta.copy_from(overwrite=False) to decide whether to
    overwrite a destination field.

    Args:
        v: Value to test.

    Returns:
        True when v is None or an empty container.
    """
    if v is None:
        return True
    if isinstance(v, (dict, list, tuple, set)) and len(v) == 0:
        return True
    return False


@dataclass(slots=True)
class BaseMeta:
    """Base class for metadata containers.

    Subclasses should implement _validate_and_cast to coerce and validate
    fields after initialization or mutation.
    """

    def __post_init__(self) -> None:
        """Validate and normalize fields after dataclass initialization."""
        self._validate_and_cast()

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate and normalize fields in subclasses.

        Args:
            **_: Optional context for validation (ignored here).
        """
        return

    def __repr__(self) -> str:
        """Return a debug representation based on plain values."""
        return repr(self.to_plain())

    def __str__(self) -> str:
        """Return a user-friendly string based on plain values."""
        return str(self.to_plain())

    def to_mapping(self, *, include_none: bool = True) -> dict[str, Any]:
        """Serialize to a mapping, recursively expanding nested BaseMeta.

        Args:
            include_none: Include fields with None values when True.

        Returns:
            A dict of field names to serialized values.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None and not include_none:
                continue
            if isinstance(v, BaseMeta):
                out[f.name] = v.to_mapping(include_none=include_none)
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_mapping(cls: Type[T], d: Mapping[str, Any]) -> T:
        """Construct an instance from a mapping.

        Args:
            d: Input mapping matching dataclass field names.

        Returns:
            A new instance of cls.

        Raises:
            TypeError: If d is not a Mapping.
            KeyError: If unknown keys are present.
        """
        if not isinstance(d, Mapping):
            raise TypeError(
                f"{cls.__name__}.from_mapping expects a mapping, got {type(d).__name__}"
            )

        dd = dict(d)
        known = {f.name for f in fields(cls)}
        unknown = set(dd) - known
        if unknown:
            raise KeyError(
                f"Unknown {cls.__name__} keys in from_mapping: {sorted(unknown)}"
            )

        for f in fields(cls):
            if f.name not in dd:
                continue
            v = dd[f.name]
            if isinstance(v, Mapping):
                anno = f.type
                if isinstance(anno, type) and issubclass(anno, BaseMeta):
                    dd[f.name] = anno.from_mapping(v)

        return cls(**dd)  # type: ignore[arg-type]

    def to_plain(self, *, include_none: bool = False) -> Any:
        """Convert to plain Python objects recursively.

        Args:
            include_none: Include fields with None values when True.

        Returns:
            A dict of field values, with nested BaseMeta expanded. SingleKeyBaseMeta
            overrides this to return its wrapped value.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None and not include_none:
                continue
            if isinstance(v, BaseMeta):
                out[f.name] = v.to_plain(include_none=include_none)
            else:
                out[f.name] = v
        return out

    def is_default(self) -> bool:
        """Return True if this equals a default-constructed instance."""
        default = self.__class__()  # type: ignore[call-arg]

        for f in fields(self):
            a = getattr(self, f.name)
            b = getattr(default, f.name)

            if isinstance(a, BaseMeta) and isinstance(b, BaseMeta):
                if not a.is_default():
                    return False
            else:
                if a != b:
                    return False
        return True

    def reset(self) -> None:
        """Reset all fields to their default or None."""
        for f in fields(self):
            if f.default_factory is not MISSING:  # type: ignore[attr-defined]
                setattr(self, f.name, f.default_factory())  # type: ignore[misc]
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            else:
                setattr(self, f.name, None)

    def copy_from(self: T, other: T, *, overwrite: bool = False) -> None:
        """Copy fields from another instance of the same class.

        Args:
            other: Source instance.
            overwrite: When True, overwrite all fields. When False, only fill
                destination fields that are "unset" (None or empty containers).
                Nested BaseMeta fields are merged recursively unless the entire
                destination sub-meta is default, in which case it is replaced.

        Raises:
            TypeError: If other is not the same class as self.
        """
        if other.__class__ is not self.__class__:
            raise TypeError(f"copy_from expects {self.__class__.__name__}")

        for f in fields(self):
            src = getattr(other, f.name)
            dst = getattr(self, f.name)

            if overwrite:
                setattr(self, f.name, src)
                continue

            if isinstance(dst, BaseMeta) and isinstance(src, BaseMeta):
                if dst.is_default():
                    setattr(self, f.name, src)
                else:
                    dst.copy_from(src, overwrite=False)
                continue

            if _is_unset_value(dst):
                setattr(self, f.name, src)

    @classmethod
    def ensure(cls: Type[T], x: Any) -> T:
        """Coerce x into an instance of cls.

        Args:
            x: None, an instance of cls, or a mapping of fields.

        Returns:
            An instance of cls.

        Raises:
            TypeError: If x is not None, cls, or a mapping.
        """
        if x is None:
            return cls()
        if isinstance(x, cls):
            return x
        if isinstance(x, Mapping):
            return cls.from_mapping(x)
        raise TypeError(f"Expected None, mapping, or {cls.__name__}; got {type(x).__name__}")


@dataclass(slots=True)
class SingleKeyBaseMeta(BaseMeta):
    """BaseMeta subclass that wraps a single field as a raw value."""

    @classmethod
    def _key_name(cls) -> str:
        """Return the single dataclass field name for this meta.

        Raises:
            TypeError: If the subclass does not define exactly one field.
        """
        flds = fields(cls)
        if len(flds) != 1:
            raise TypeError(
                f"{cls.__name__} must define exactly one dataclass field (found {len(flds)})"
            )
        return flds[0].name

    @property
    def value(self) -> Any:
        """Return the wrapped value."""
        return getattr(self, self._key_name())

    @value.setter
    def value(self, v: Any) -> None:
        """Set the wrapped value and re-validate."""
        setattr(self, self._key_name(), v)
        self._validate_and_cast()

    def set(self, v: Any) -> None:
        """Set the wrapped value."""
        self.value = v

    def to_mapping(self, *, include_none: bool = True) -> dict[str, Any]:
        """Serialize to a mapping with the single key.

        Args:
            include_none: Include the key when the value is None.

        Returns:
            A dict with the single field name as the key, or an empty dict.
        """
        k = self._key_name()
        v = self.value
        if v is None and not include_none:
            return {}
        return {k: v}

    @classmethod
    def from_mapping(cls: Type[SK], d: Any) -> SK:
        """Construct from either schema-shaped mapping or raw value.

        Args:
            d: None, mapping, or raw value.

        Returns:
            A new instance of cls.
        """
        if d is None:
            return cls()  # type: ignore[call-arg]

        k = cls._key_name()

        if isinstance(d, Mapping):
            dd = dict(d)
            if set(dd.keys()) == {k}:
                return cls(**{k: dd[k]})  # type: ignore[arg-type]
            return cls(**{k: d})  # type: ignore[arg-type]

        return cls(**{k: d})  # type: ignore[arg-type]

    def to_plain(self, *, include_none: bool = False) -> Any:
        """Return the wrapped value for plain output.

        Args:
            include_none: Return None when the value is None.

        Returns:
            The wrapped value or None.
        """
        v = self.value
        if v is None and not include_none:
            return None
        return v
    
    @classmethod
    def ensure(cls: Type[SK], x: Any) -> SK:
        """Coerce input into an instance of cls.

        Args:
            x: None, instance of cls, mapping, or raw value.

        Returns:
            An instance of cls.
        """
        if x is None:
            return cls()  # type: ignore[call-arg]
        if isinstance(x, cls):
            return x
        return cls.from_mapping(x)

    def __repr__(self) -> str:
        """Return a debug representation of the wrapped value."""
        return repr(self.to_plain())

    def __bool__(self) -> bool:
        """Return truthiness of the wrapped value."""
        return bool(self.value)

    def __len__(self) -> int:
        """Return length of the wrapped value, or 0 if None."""
        v = self.value
        if v is None:
            return 0
        return len(v)  # type: ignore[arg-type]

    def __iter__(self):
        """Iterate over the wrapped value, or empty when None."""
        v = self.value
        if v is None:
            return iter(())
        return iter(v)

    def __contains__(self, item: Any) -> bool:
        """Return membership test on the wrapped value."""
        v = self.value
        if v is None:
            return False
        return item in v

    def __getitem__(self, key: Any) -> Any:
        """Index into the wrapped value."""
        return self.value[key]

    def __setitem__(self, key: Any, val: Any) -> None:
        """Set an item on the wrapped value and re-validate."""
        self.value[key] = val
        self._validate_and_cast()

    def __eq__(self, other: Any) -> bool:
        """Compare by wrapped value."""
        if isinstance(other, SingleKeyBaseMeta):
            return self.value == other.value
        return self.value == other
    

def _cast_to_list(value: Any, label: str):
    """Cast lists/tuples/ndarrays to nested lists.

    Args:
        value: Input list-like value.
        label: Label used in error messages.

    Returns:
        A (possibly nested) Python list.

    Raises:
        TypeError: If the value cannot be cast to a list.
    """
    if isinstance(value, list):
        out = value
    elif isinstance(value, tuple):
        out = list(value)
    elif np is not None and isinstance(value, np.ndarray):
        out = value.tolist()
    else:
        raise TypeError(f"{label} must be a list, tuple, or numpy array")

    for i, item in enumerate(out):
        if isinstance(item, (list, tuple)) or (np is not None and isinstance(item, np.ndarray)):
            out[i] = _cast_to_list(item, label)
    return out


def _to_jsonable(value: Any) -> Any:
    """Recursively convert values to JSON-serializable plain Python objects."""
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    return value


def _cast_to_jsonable_mapping(value: Any, label: str) -> dict[str, Any]:
    """Cast a value to a JSON-serializable mapping.

    Accepts mappings directly or objects exposing ``__dict__`` (for example
    Blosc2 ``CParams`` / ``DParams`` objects).
    """
    if isinstance(value, Mapping):
        out = dict(value)
    elif hasattr(value, "__dict__"):
        out = dict(vars(value))
    else:
        raise TypeError(f"{label} must be a mapping or object with __dict__")

    out = _to_jsonable(out)
    if not isinstance(out, dict):
        raise TypeError(f"{label} could not be converted to a mapping")
    if not is_serializable(out):
        raise TypeError(f"{label} is not JSON-serializable")
    return out


def _validate_int(value: Any, label: str) -> None:
    """Validate that value is an int.

    Args:
        value: Value to validate.
        label: Label used in error messages.

    Raises:
        TypeError: If value is not an int.
    """
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an int")


def _validate_float_int_list(value: Any, label: str, ndims: Optional[int] = None) -> None:
    """Validate a list of floats/ints, optionally with a fixed length.

    Args:
        value: List to validate.
        label: Label used in error messages.
        ndims: Required length when provided.

    Raises:
        TypeError: If value is not a list or contains non-numbers.
        ValueError: If ndims is provided and the length does not match.
    """
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    if ndims is not None and len(value) != ndims:
        raise ValueError(f"{label} must have length {ndims}")
    for v in value:
        if not isinstance(v, (float, int)):
            raise TypeError(f"{label} must contain only floats or ints")


def _validate_float_int_matrix(value: Any, label: str, ndims: Optional[int] = None) -> None:
    """Validate a square list-of-lists matrix of floats/ints.

    Args:
        value: Matrix to validate.
        label: Label used in error messages.
        ndims: Required shape (ndims x ndims) when provided.

    Raises:
        TypeError: If value is not a list-of-lists or contains non-numbers.
        ValueError: If ndims is provided and the shape does not match.
    """
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list of lists")
    if ndims is not None and len(value) != ndims:
        raise ValueError(f"{label} must have shape [{ndims}, {ndims}]")
    for row in value:
        if not isinstance(row, list):
            raise TypeError(f"{label} must be a list of lists")
        if ndims is not None and len(row) != ndims:
            raise ValueError(f"{label} must have shape [{ndims}, {ndims}]")
        for v in row:
            if not isinstance(v, (float, int)):
                raise TypeError(f"{label} must contain only floats or ints")
            

def validate_and_cast_axis_labels(
    value: Any,
    label: str,
    ndims: Optional[int] = None,
) -> tuple[Optional[list[str]], int, int]:
    """
    Validate axis labels/roles, normalize to list[str], and count spatial/non-spatial axes.

    Args:
        value: None or list-like of axis labels/roles (enum members or strings).
        label: Label used in error messages.
        ndims: If provided, enforce list length == ndims.

    Returns:
        (labels, n_spatial, n_non_spatial)

    Raises:
        TypeError: If value is not None and not list-like, or contains invalid items.
        ValueError: If ndims is provided and length mismatch occurs.
    """
    if value is None:
        return None, 0, 0

    # Cast list / tuple / ndarray -> list
    v = _cast_to_list(value, label)

    # Enforce 1D list
    for i, item in enumerate(v):
        if isinstance(item, list):
            raise TypeError(
                f"{label} must be a 1D list (got nested list at index {i})"
            )

    if ndims is not None and len(v) != ndims:
        raise ValueError(f"{label} must have length {ndims}")

    spatial_enum_roles = {
        AxisLabelEnum.spatial,
        AxisLabelEnum.spatial_x,
        AxisLabelEnum.spatial_y,
        AxisLabelEnum.spatial_z,
    }
    spatial_string_roles = {r.value for r in spatial_enum_roles}

    out: list[str] = []
    n_spatial = 0
    n_non_spatial = 0

    for i, x in enumerate(v):
        if isinstance(x, AxisLabelEnum):
            out.append(x.value)
            if x in spatial_enum_roles:
                n_spatial += 1
            else:
                n_non_spatial += 1
            continue

        if isinstance(x, str):
            out.append(x)
            if x in spatial_string_roles:
                n_spatial += 1
            else:
                n_non_spatial += 1
            continue

        raise TypeError(
            f"{label}[{i}] must be a str or AxisLabelEnum "
            f"(got {type(x).__name__})"
        )

    return out, n_spatial, n_non_spatial


def _is_spatial_axis(label: Union[str, AxisLabelEnum]) -> bool:
    """Return True if an axis label/role represents a spatial axis."""
    if isinstance(label, AxisLabelEnum):
        return label in {
            AxisLabelEnum.spatial,
            AxisLabelEnum.spatial_x,
            AxisLabelEnum.spatial_y,
            AxisLabelEnum.spatial_z,
        }

    if isinstance(label, str):
        return label in {
            AxisLabelEnum.spatial.value,
            AxisLabelEnum.spatial_x.value,
            AxisLabelEnum.spatial_y.value,
            AxisLabelEnum.spatial_z.value,
        }

    return False


def _spatial_axis_mask(
    labels: Iterable[Union[str, AxisLabelEnum]],
) -> list[bool]:
    """Return a boolean mask indicating which axes are spatial."""
    if labels is None:
        return None
    return [_is_spatial_axis(label) for label in labels]


@dataclass(slots=True)
class MetaBlosc2(BaseMeta):
    """Metadata for Blosc2 tiling and chunking.

    Attributes:
        chunk_size: List of per-dimension chunk sizes. Length must match ndims.
        block_size: List of per-dimension block sizes. Length must match ndims.
        patch_size: List of per-dimension patch sizes. Length must match spatial ndims.
        cparams: Blosc2 compression parameters as a JSON-serializable dict.
        dparams: Blosc2 decompression parameters as a JSON-serializable dict.
    """
    chunk_size: Optional[list] = None
    block_size: Optional[list] = None
    patch_size: Optional[list] = None
    cparams: Optional[dict[str, Any]] = None
    dparams: Optional[dict[str, Any]] = None

    def _validate_and_cast(self, *, ndims: Optional[int] = None, spatial_ndims: Optional[int] = None, **_: Any) -> None:
        """Validate and normalize tiling sizes.

        Args:
            ndims: Number of dimensions.
            spatial_ndims: Number of spatial dimensions.
            **_: Unused extra context.
        """
        if self.chunk_size is not None:
            self.chunk_size = _cast_to_list(self.chunk_size, "meta.blosc2.chunk_size")
            _validate_float_int_list(self.chunk_size, "meta.blosc2.chunk_size", ndims)

        if self.block_size is not None:
            self.block_size = _cast_to_list(self.block_size, "meta.blosc2.block_size")
            _validate_float_int_list(self.block_size, "meta.blosc2.block_size", ndims)

        if self.patch_size is not None:
            spatial_ndims = ndims if spatial_ndims is None else spatial_ndims
            self.patch_size = _cast_to_list(self.patch_size, "meta.blosc2.patch_size")
            _validate_float_int_list(self.patch_size, "meta.blosc2.patch_size", spatial_ndims)

        if self.cparams is not None:
            self.cparams = _cast_to_jsonable_mapping(self.cparams, "meta.blosc2.cparams")

        if self.dparams is not None:
            self.dparams = _cast_to_jsonable_mapping(self.dparams, "meta.blosc2.dparams")


class AxisLabelEnum(str, Enum):
    """Axis label/role identifiers used for spatial metadata.

    Attributes:
        spatial: Generic spatial axis (used when no axis-specific label applies).
        spatial_x: Spatial axis representing X.
        spatial_y: Spatial axis representing Y.
        spatial_z: Spatial axis representing Z.
        non_spatial: Generic non-spatial axis.
        channel: Channel axis (e.g., color channels or feature maps).
        temporal: Time axis.
        continuous: Continuous-valued axis (non-spatial).
        components: Component axis (e.g., vector components).
    """
    spatial = "spatial"
    spatial_x = "spatial_x"
    spatial_y = "spatial_y"
    spatial_z = "spatial_z"
    non_spatial = "non_spatial"
    channel = "channel"
    temporal = "temporal"
    continuous = "continuous"
    components = "components"


AxisLabel: TypeAlias = Union[str, AxisLabelEnum]


@dataclass(slots=True)
class MetaSpatial(BaseMeta):
    """Spatial metadata describing geometry and layout.

    Attributes:
        spacing: Per-dimension spacing values. Length must match ndims.
        origin: Per-dimension origin values. Length must match ndims.
        direction: Direction cosine matrix of shape [ndims, ndims].
        affine: Homogeneous affine matrix of shape [ndims + 1, ndims + 1].
        shape: Array shape. Length must match (spatial + non-spatial) ndims.
        axis_labels: Per-axis labels or roles. Length must match ndims.
        axis_units: Per-axis units. Length must match ndims.
        _num_spatial_axes: Cached count of spatial axes derived from axis_labels.
        _num_non_spatial_axes: Cached count of non-spatial axes derived from axis_labels.
    """
    AxisLabel = AxisLabelEnum
    spacing: Optional[list[Union[int,float]]] = None
    origin: Optional[list[Union[int,float]]] = None
    direction: Optional[list[list[Union[int,float]]]] = None
    affine: Optional[list[list[Union[int,float]]]] = None
    shape: Optional[list[int]] = None
    axis_labels: Optional[list[Union[str,AxisLabel]]] = None
    axis_units: Optional[list[str]] = None
    _num_spatial_axes: Optional[int] = None
    _num_non_spatial_axes: Optional[int] = None

    def _validate_and_cast(self, *, ndims: Optional[int] = None, spatial_ndims: Optional[int] = None, **_: Any) -> None:
        """Validate and normalize spatial fields.

        Args:
            ndims: Number of dimensions.
            spatial_ndims: Number of spatial dimensions.
            **_: Unused extra context.
        """
        if self.axis_labels is not None:
            self.axis_labels, self._num_spatial_axes, self._num_non_spatial_axes = validate_and_cast_axis_labels(self.axis_labels, "meta.spatial.axis_labels", ndims)
        spatial_ndims = spatial_ndims if self._num_spatial_axes is None else self._num_spatial_axes

        if self.spacing is not None:
            self.spacing = _cast_to_list(self.spacing, "meta.spatial.spacing")
            _validate_float_int_list(self.spacing, "meta.spatial.spacing", spatial_ndims)

        if self.origin is not None:
            self.origin = _cast_to_list(self.origin, "meta.spatial.origin")
            _validate_float_int_list(self.origin, "meta.spatial.origin", spatial_ndims)

        if self.direction is not None:
            self.direction = _cast_to_list(self.direction, "meta.spatial.direction")
            _validate_float_int_matrix(self.direction, "meta.spatial.direction", spatial_ndims)

        if self.affine is not None:
            self.affine = _cast_to_list(self.affine, "meta.spatial.affine")
            if spatial_ndims is not None:
                _validate_float_int_matrix(
                    self.affine,
                    "meta.spatial.affine",
                    spatial_ndims + 1,
                )
            else:
                _validate_float_int_matrix(self.affine, "meta.spatial.affine")
                n_rows = len(self.affine)
                for row in self.affine:
                    if len(row) != n_rows:
                        raise ValueError("meta.spatial.affine must be a square matrix")

        if self.shape is not None:
            self.shape = _cast_to_list(self.shape, "meta.spatial.shape")
            _validate_float_int_list(self.shape, "meta.spatial.shape", ndims)


@dataclass(slots=True)
class MetaStatistics(BaseMeta):
    """Numeric summary statistics for an array.

    Attributes:
        min: Minimum value.
        max: Maximum value.
        mean: Mean value.
        median: Median value.
        std: Standard deviation.
        percentile_min: Minimum percentile value.
        percentile_max: Maximum percentile value.
        percentile_mean: Mean percentile value.
        percentile_median: Median percentile value.
        percentile_std: Standard deviation of percentile values.
        percentile_min_key: Minimum percentile key used to determine percentile_min (for example 0.05).
        percentile_max_key: Maximum percentile key used to determine percentile_max (for example 0.95).
    """
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    percentile_min: Optional[float] = None
    percentile_max: Optional[float] = None
    percentile_mean: Optional[float] = None
    percentile_median: Optional[float] = None
    percentile_std: Optional[float] = None
    percentile_min_key: Optional[float] = None
    percentile_max_key: Optional[float] = None

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate that all stats are numeric when provided."""
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            if v is not None and not isinstance(v, (float, int)):
                raise TypeError(f"meta.stats.{name} must be a float or int")


@dataclass(slots=True)
class MetaBbox(BaseMeta):
    """Bounding box metadata with optional scores and labels.

    Attributes:
        bboxes: List of bounding boxes with shape [n_boxes, ndims, 2], where
            each inner pair is [min, max] for a dimension. Values must be ints
            or floats.
        scores: Optional confidence scores aligned with bboxes (ints or floats).
        labels: Optional labels aligned with bboxes. Each label may be a string,
            int, or float.
    """
    bboxes: Optional[list[list[list[Union[int, float]]]]] = None
    scores: Optional[list[Union[int, float]]] = None
    labels: Optional[list[Union[str, int, float]]] = None

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate bounding box structure and related fields."""
        if self.bboxes is not None:
            self.bboxes = _cast_to_list(self.bboxes, "meta.bbox.bboxes")

            if not isinstance(self.bboxes, list):
                raise TypeError("meta.bbox.bboxes must be a list")

            for b_i, bbox in enumerate(self.bboxes):
                if not isinstance(bbox, list):
                    raise TypeError("meta.bbox.bboxes must be a list of lists")
                for r_i, row in enumerate(bbox):
                    if not isinstance(row, list) or len(row) != 2:
                        raise ValueError("meta.bbox.bboxes rows must have length 2")
                    for v in row:
                        if isinstance(v, bool) or not isinstance(v, (float, int)):
                            raise TypeError("meta.bbox.bboxes must contain ints or floats only")

        if self.scores is not None:
            self.scores = _cast_to_list(self.scores, "meta.bbox.scores")
            _validate_float_int_list(self.scores, "meta.bbox.scores")

        if self.labels is not None:
            self.labels = _cast_to_list(self.labels, "meta.bbox.labels")
            if not isinstance(self.labels, list):
                raise TypeError("meta.bbox.labels must be a list")
            for v in self.labels:
                if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                    raise TypeError("meta.bbox.labels must contain only str, int, or float")

        if self.bboxes is not None:
            n = len(self.bboxes)
            if self.scores is not None and len(self.scores) != n:
                raise ValueError("meta.bbox.scores must have same length as bboxes")
            if self.labels is not None and len(self.labels) != n:
                raise ValueError("meta.bbox.labels must have same length as bboxes")


@dataclass(slots=True)
class MetaSource(SingleKeyBaseMeta):
    """Source metadata from the original image source stored as JSON-serializable dict.

    Attributes:
        data: Arbitrary JSON-serializable metadata.
    """
    data: dict[str, Any] = field(default_factory=dict)

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate that data is a JSON-serializable dict."""
        if not isinstance(self.data, dict):
            raise TypeError(f"meta.image.data must be a dict, got {type(self.data).__name__}")
        if not is_serializable(self.data):
            raise TypeError("meta.image.data is not JSON-serializable")


@dataclass(slots=True)
class MetaExtra(SingleKeyBaseMeta):
    """Generic extra metadata stored as JSON-serializable dict.

    Attributes:
        data: Arbitrary JSON-serializable metadata.
    """
    data: dict[str, Any] = field(default_factory=dict)

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate that data is a JSON-serializable dict."""
        if not isinstance(self.data, dict):
            raise TypeError(f"meta.extra.data must be a dict, got {type(self.data).__name__}")
        if not is_serializable(self.data):
            raise TypeError("meta.extra.data is not JSON-serializable")


@dataclass(slots=True)
class MetaIsSeg(SingleKeyBaseMeta):
    """Flag indicating whether the array is a segmentation mask.

    Attributes:
        is_seg: True/False when known, None when unknown.
    """
    is_seg: Optional[bool] = None

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate is_seg as bool or None."""
        if self.is_seg is not None and not isinstance(self.is_seg, bool):
            raise TypeError("meta.is_seg must be a bool or None")


@dataclass(slots=True)
class MetaHasArray(SingleKeyBaseMeta):
    """Flag indicating whether an array is present.

    Attributes:
        has_array: True when array data is present.
    """
    has_array: bool = False

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate has_array as bool."""
        if not isinstance(self.has_array, bool):
            raise TypeError("meta._has_array must be a bool")


@dataclass(slots=True)
class MetaImageFormat(SingleKeyBaseMeta):
    """String describing the image metadata format.

    Attributes:
        image_meta_format: Format identifier, or None.
    """
    image_meta_format: Optional[str] = None

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate image_meta_format as str or None."""
        if self.image_meta_format is not None and not isinstance(self.image_meta_format, str):
            raise TypeError("meta._image_meta_format must be a str or None")


@dataclass(slots=True)
class MetaVersion(SingleKeyBaseMeta):
    """Version metadata for mlarray.

    Attributes:
        mlarray_version: Version string, or None.
    """
    mlarray_version: Optional[str] = None

    def _validate_and_cast(self, **_: Any) -> None:
        """Validate mlarray_version as str or None."""
        if self.mlarray_version is not None and not isinstance(self.mlarray_version, str):
            raise TypeError("meta._mlarray_version must be a str or None")


@dataclass(slots=True)
class Meta(BaseMeta):
    """Top-level metadata container for mlarray.

    Attributes:
        source: Source metadata from the original image source (JSON-serializable dict).
        extra: Additional metadata (JSON-serializable dict).
        spatial: Spatial metadata (spacing, origin, direction, affine, shape).
        stats: Summary statistics.
        bbox: Bounding boxes.
        is_seg: Segmentation flag.
        blosc2: Blosc2 chunking/tiling metadata.
        _has_array: Payload presence flag.
        _image_meta_format: Image metadata format identifier.
        _mlarray_version: Version string for mlarray.
    """
    source: "MetaSource" = field(default_factory=lambda: MetaSource())
    extra: "MetaExtra" = field(default_factory=lambda: MetaExtra())
    spatial: "MetaSpatial" = field(default_factory=lambda: MetaSpatial())
    stats: "MetaStatistics" = field(default_factory=lambda: MetaStatistics())
    bbox: "MetaBbox" = field(default_factory=lambda: MetaBbox())
    is_seg: "MetaIsSeg" = field(default_factory=lambda: MetaIsSeg())
    blosc2: "MetaBlosc2" = field(default_factory=lambda: MetaBlosc2())
    _has_array: "MetaHasArray" = field(default_factory=lambda: MetaHasArray())
    _image_meta_format: "MetaImageFormat" = field(default_factory=lambda: MetaImageFormat())
    _mlarray_version: "MetaVersion" = field(default_factory=lambda: MetaVersion())

    def _validate_and_cast(self, *, ndims: Optional[int] = None, spatial_ndims: Optional[int] = None, **_: Any) -> None:
        """Coerce child metas and validate with optional context.

        Args:
            ndims: Number of dimensions for context-aware validation.
            spatial_ndims: Number of spatial dimensions for context-aware validation.
            **_: Unused extra context.
        """
        self.source = MetaSource.ensure(self.source)
        self.extra = MetaExtra.ensure(self.extra)
        self.spatial = MetaSpatial.ensure(self.spatial)
        self.stats = MetaStatistics.ensure(self.stats)
        self.bbox = MetaBbox.ensure(self.bbox)
        self.is_seg = MetaIsSeg.ensure(self.is_seg)
        self.blosc2 = MetaBlosc2.ensure(self.blosc2)
        self._has_array = MetaHasArray.ensure(self._has_array)
        self._image_meta_format = MetaImageFormat.ensure(self._image_meta_format)
        self._mlarray_version = MetaVersion.ensure(self._mlarray_version)

        self.spatial._validate_and_cast(ndims=ndims, spatial_ndims=spatial_ndims)
        self.blosc2._validate_and_cast(ndims=ndims, spatial_ndims=spatial_ndims)

    def to_plain(self, *, include_none: bool = False) -> Any:
        """Convert to plain values, suppressing default sub-metas.

        Args:
            include_none: Include None values when True.

        Returns:
            A dict of field values where default child metas are represented
            as None and optionally filtered out.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)

            if isinstance(v, BaseMeta):
                out[f.name] = None if v.is_default() else v.to_plain(include_none=include_none)
            else:
                if v is None and not include_none:
                    continue
                out[f.name] = v

        if not include_none:
            out = {k: val for k, val in out.items() if val is not None}
        return out
    
