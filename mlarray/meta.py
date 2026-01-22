from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union
import numpy as np
from mlarray.utils import is_serializable


@dataclass(slots=True)
class MetaBlosc2:
    """Blosc2 storage metadata for chunking, blocking, and patch hints."""
    chunk_size: Optional[list] = None
    block_size: Optional[list] = None
    patch_size: Optional[list] = None

    def __post_init__(self) -> None:
        """Validate and normalize any provided sizes."""
        self._validate_and_cast()

    def __repr__(self) -> str:
        """Return a repr using the dict form for readability."""
        return repr(self.to_dict())

    def to_dict(self, *, include_none: bool = True) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Args:
            include_none (bool): If False, keys with None values are omitted.

        Returns:
            Dict[str, Any]: Serialized metadata.
        """
        out: Dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "block_size": self.block_size,
            "patch_size": self.patch_size,
        }
        if not include_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out

    def _validate_and_cast(self, ndims: Optional[int] = None, channel_axis: Optional[int] = None) -> None:
        """Validate and cast sizes to list form.

        Args:
            ndims (Optional[int]): Expected number of array dimensions.
            channel_axis (Optional[int]): Channel axis index for patch size
                validation when channels are present.
        """
        if self.chunk_size is not None:
            self.chunk_size = _cast_to_list(self.chunk_size, "meta._blosc2.chunk_size")
            _validate_float_int_list(self.chunk_size, f"meta._blosc2.chunk_size", ndims)
        if self.block_size is not None:
            self.block_size = _cast_to_list(self.block_size, "meta._blosc2.block_size")
            _validate_float_int_list(self.block_size, f"meta._blosc2.block_size", ndims)
        if self.patch_size is not None:
            _ndims = ndims if (ndims is None or channel_axis is None) else ndims-1
            self.patch_size = _cast_to_list(self.patch_size, "meta._blosc2.patch_size")
            _validate_float_int_list(self.patch_size, f"meta._blosc2.patch_size", _ndims)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, strict: bool = True) -> MetaBlosc2:
        """Create a MetaBlosc2 instance from a mapping.

        Args:
            d (Mapping[str, Any]): Source mapping.
            strict (bool): If True, unknown keys raise a KeyError.

        Returns:
            MetaBlosc2: Parsed instance.
        """
        if not isinstance(d, Mapping):
            raise TypeError(f"MetaBlosc2.from_dict expects a mapping, got {type(d).__name__}")
        known = {"chunk_size", "block_size", "patch_size"}
        d = dict(d)
        unknown = set(d.keys()) - known
        if unknown and strict:
            raise KeyError(f"Unknown MetaBlosc2 keys in from_dict: {sorted(unknown)}")
        return cls(
            chunk_size=d.get("chunk_size"),
            block_size=d.get("block_size"),
            patch_size=d.get("patch_size"),
        )


@dataclass(slots=True)
class MetaSpatial:
    """Spatial metadata describing array geometry in physical space."""
    spacing: Optional[List] = None
    origin: Optional[List] = None
    direction: Optional[List[List]] = None
    shape: Optional[List] = None
    channel_axis: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate and normalize spatial fields."""
        self._validate_and_cast()

    def __repr__(self) -> str:
        """Return a repr using the dict form for readability."""
        return repr(self.to_dict())

    def to_dict(self, *, include_none: bool = True) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Args:
            include_none (bool): If False, keys with None values are omitted.

        Returns:
            Dict[str, Any]: Serialized metadata.
        """
        out: Dict[str, Any] = {
            "spacing": self.spacing,
            "origin": self.origin,
            "direction": self.direction,
            "shape": self.shape,
            "channel_axis": self.channel_axis
        }
        if not include_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out

    def _validate_and_cast(self, ndims: Optional[int] = None) -> None:
        """Validate and cast spatial fields to list form.

        Args:
            ndims (Optional[int]): Expected number of spatial dimensions.
        """
        if self.channel_axis is not None:
            _validate_int(self.channel_axis, "meta.spatial.channel_axis")
        if self.spacing is not None:
            self.spacing = _cast_to_list(self.spacing, "meta.spatial.spacing")
            _validate_float_int_list(self.spacing, "meta.spatial.spacing", ndims)
        if self.origin is not None:
            self.origin = _cast_to_list(self.origin, "meta.spatial.origin")
            _validate_float_int_list(self.origin, "meta.spatial.origin", ndims)
        if self.direction is not None:
            self.direction = _cast_to_list(self.direction, "meta.spatial.direction")
            _validate_float_int_matrix(self.direction, "meta.spatial.direction", ndims)
        if self.shape is not None:
            _ndims = ndims if (ndims is None or self.channel_axis is None) else ndims+1
            self.shape = _cast_to_list(self.shape, "meta.spatial.shape")
            _validate_float_int_list(self.shape, "meta.spatial.shape", _ndims)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, strict: bool = True) -> MetaSpatial:
        """Create a MetaSpatial instance from a mapping.

        Args:
            d (Mapping[str, Any]): Source mapping.
            strict (bool): If True, unknown keys raise a KeyError.

        Returns:
            MetaSpatial: Parsed instance.
        """
        if not isinstance(d, Mapping):
            raise TypeError(f"MetaSpatial.from_dict expects a mapping, got {type(d).__name__}")
        known = {"spacing", "origin", "direction", "shape", "channel_axis"}
        d = dict(d)
        unknown = set(d.keys()) - known
        if unknown and strict:
            raise KeyError(f"Unknown MetaSpatial keys in from_dict: {sorted(unknown)}")
        return cls(
            spacing=d.get("spacing"),
            origin=d.get("origin"),
            direction=d.get("direction"),
            shape=d.get("shape"),
            channel_axis=d.get("channel_axis")
        )


@dataclass(slots=True)
class MetaStatistics:
    """Summary statistics for an array or image."""
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

    def __post_init__(self) -> None:
        """Validate that all provided values are numeric."""
        self._validate_and_cast()

    def __repr__(self) -> str:
        """Return a repr using the dict form for readability."""
        return repr(self.to_dict())

    def to_dict(self, *, include_none: bool = True) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Args:
            include_none (bool): If False, keys with None values are omitted.

        Returns:
            Dict[str, Any]: Serialized metadata.
        """
        out: Dict[str, Any] = {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "percentile_min": self.percentile_min,
            "percentile_max": self.percentile_max,
            "percentile_mean": self.percentile_mean,
            "percentile_median": self.percentile_median,
            "percentile_std": self.percentile_std,
        }
        if not include_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out

    def _validate_and_cast(self) -> None:
        """Validate that all statistic fields are float or int."""
        for name in (
            "min",
            "max",
            "mean",
            "median",
            "std",
            "percentile_min",
            "percentile_max",
            "percentile_mean",
            "percentile_median",
            "percentile_std",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (float, int)):
                raise TypeError(f"meta.stats.{name} must be a float or int")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, strict: bool = True) -> MetaStatistics:
        """Create a MetaStatistics instance from a mapping.

        Args:
            d (Mapping[str, Any]): Source mapping.
            strict (bool): If True, unknown keys raise a KeyError.

        Returns:
            MetaStatistics: Parsed instance.
        """
        if not isinstance(d, Mapping):
            raise TypeError(f"MetaStatistics.from_dict expects a mapping, got {type(d).__name__}")
        known = {
            "min",
            "max",
            "mean",
            "median",
            "std",
            "percentile_min",
            "percentile_max",
            "percentile_mean",
            "percentile_median",
            "percentile_std",
        }
        d = dict(d)
        unknown = set(d.keys()) - known
        if unknown and strict:
            raise KeyError(f"Unknown MetaStatistics keys in from_dict: {sorted(unknown)}")
        return cls(
            min=d.get("min"),
            max=d.get("max"),
            mean=d.get("mean"),
            median=d.get("median"),
            std=d.get("std"),
            percentile_min=d.get("percentile_min"),
            percentile_max=d.get("percentile_max"),
            percentile_mean=d.get("percentile_mean"),
            percentile_median=d.get("percentile_median"),
            percentile_std=d.get("percentile_std"),
        )


@dataclass(slots=True)
class MetaBbox:
    """Bounding boxes stored as per-dimension min/max pairs."""
    bboxes: Optional[List[List[List[int]]]] = None

    def __post_init__(self) -> None:
        """Validate and normalize bounding box lists."""
        self._validate_and_cast()

    def __iter__(self):
        """Iterate over bounding boxes."""
        return iter(self.bboxes or [])

    def __getitem__(self, index):
        """Return the bbox at ``index``."""
        if self.bboxes is None:
            raise TypeError("meta.bbox is None")
        return self.bboxes[index]

    def __setitem__(self, index, value):
        """Set the bbox at ``index``."""
        if self.bboxes is None:
            raise TypeError("meta.bbox is None")
        self.bboxes[index] = value

    def __len__(self):
        """Return the number of bounding boxes."""
        return len(self.bboxes or [])

    def __repr__(self) -> str:
        """Return a repr of the bounding box list."""
        return repr(self.bboxes)

    def to_list(self) -> Optional[List[List[List[int]]]]:
        """Return the raw bbox list."""
        return self.bboxes

    def _validate_and_cast(self, ndims: Optional[int] = None) -> None:
        """Validate bbox structure and cast to list form.

        Args:
            ndims (Optional[int]): Expected number of spatial dimensions.
        """
        if self.bboxes is None:
            return
        self.bboxes = _cast_to_list(self.bboxes, "meta.bbox.bboxes")
        if not isinstance(self.bboxes, list):
            raise TypeError("meta.bbox.bboxes must be a list of bboxes")
        for bbox in self.bboxes:
            if not isinstance(bbox, list):
                raise TypeError("meta.bbox.bboxes must be a list of bboxes")
            if ndims is not None and len(bbox) != ndims:
                raise ValueError(f"meta.bbox.bboxes entries must have length {ndims}")
            for row in bbox:
                if not isinstance(row, list):
                    raise TypeError("meta.bbox.bboxes must be a list of lists")
                if len(row) != 2:
                    raise ValueError("meta.bbox.bboxes entries must have length 2 per dimension")
                for item in row:
                    if isinstance(item, bool) or not isinstance(item, int):
                        raise TypeError("meta.bbox.bboxes must contain only ints")

    @classmethod
    def from_list(cls, bboxes: Any) -> MetaBbox:
        """Create a MetaBbox instance from a list-like object.

        Args:
            bboxes (Any): List-like structure shaped as
                [[ [min,max], [min,max], ...], ...].

        Returns:
            MetaBbox: Parsed instance.
        """
        return cls(bboxes=bboxes)


@dataclass(slots=True)
class Meta:
    """Container for MLArray metadata sections."""
    image: Optional[Dict[str, Any]] = None
    spatial: MetaSpatial = field(default_factory=MetaSpatial)
    stats: Optional[Union[dict, MetaStatistics]] = None
    bbox: Optional[MetaBbox] = None
    is_seg: Optional[bool] = None
    _blosc2: MetaBlosc2 = field(default_factory=MetaBlosc2)
    _has_array: Optional[bool] = None
    _image_meta_format: Optional[str] = None
    _mlarray_version: Optional[str] = None

    # controlled escape hatch for future/experimental metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize metadata sections."""
        # Validate anything passed in the constructor
        for name in ("image",):
            val = getattr(self, name)
            if val is not None:
                if not isinstance(val, dict):
                    raise TypeError(f"meta.{name} must be a dict or None, got {type(val).__name__}")
                if not is_serializable(val):
                    raise TypeError(f"meta.{name} is not JSON-serializable")
        if self.stats is not None:
            if isinstance(self.stats, MetaStatistics):
                pass
            elif isinstance(self.stats, Mapping):
                self.stats = MetaStatistics.from_dict(self.stats, strict=False)
            else:
                raise TypeError(f"meta.stats must be a MetaStatistics or mapping, got {type(self.stats).__name__}")
        if self.bbox is not None:
            if isinstance(self.bbox, MetaBbox):
                pass
            elif isinstance(self.bbox, (list, tuple)) or (np is not None and isinstance(self.bbox, np.ndarray)):
                self.bbox = MetaBbox(bboxes=self.bbox)
            else:
                raise TypeError(f"meta.bbox must be a MetaBbox or list-like, got {type(self.bbox).__name__}")

        if self.spatial is None:
            self.spatial = MetaSpatial()
        if not isinstance(self.spatial, MetaSpatial):
            raise TypeError(f"meta.spatial must be a MetaSpatial, got {type(self.spatial).__name__}")

        if self._blosc2 is None:
            self._blosc2 = MetaBlosc2()
        if not isinstance(self._blosc2, MetaBlosc2):
            raise TypeError(f"meta._blosc2 must be a MetaBlosc2, got {type(self._blosc2).__name__}")

        if self.is_seg is not None and not isinstance(self.is_seg, bool):
            raise TypeError("meta.is_seg must be a bool or None")
        if self._has_array is not None and not isinstance(self._has_array, bool):
            raise TypeError("meta._has_array must be a bool or None")
        if self._image_meta_format is not None and not isinstance(self._image_meta_format, str):
            raise TypeError("meta._image_meta_format must be a str or None")
        if self._mlarray_version is not None and not isinstance(self._mlarray_version, str):
            raise TypeError("meta._mlarray_version must be a str or None")

        if not isinstance(self.extra, dict):
            raise TypeError(f"meta.extra must be a dict, got {type(self.extra).__name__}")
        if not is_serializable(self.extra):
            raise TypeError("meta.extra is not JSON-serializable")

    def __repr__(self) -> str:
        """Return a repr using the dict form for readability."""
        return repr(self.to_dict())

    def set(self, key: str, value: Any) -> None:
        """Set a known meta section explicitly.

        Args:
            key (str): Name of the meta section (e.g., "image", "spatial",
                "stats", "bbox", "_blosc2", "is_seg").
            value (Any): Value to set. Must be JSON-serializable for dict
                sections.

        Raises:
            AttributeError: If ``key`` is unknown or disallowed.
            TypeError: If the value has an unexpected type.
        """
        if not hasattr(self, key) and key not in {"_blosc2", "_mlarray_version"}:
            raise AttributeError(f"Unknown meta section: {key!r}")
        if key == "extra":
            raise AttributeError("Use meta.extra[...] to add to extra")
        if key == "spatial":
            if isinstance(value, MetaSpatial):
                setattr(self, key, value)
                return
            if isinstance(value, Mapping):
                setattr(self, key, MetaSpatial.from_dict(value, strict=False))
                return
            raise TypeError("meta.spatial must be a MetaSpatial or mapping")
        if key == "stats":
            if isinstance(value, MetaStatistics):
                self.stats = value
                return
            if isinstance(value, Mapping):
                self.stats = MetaStatistics.from_dict(value, strict=False)
                return
            raise TypeError("meta.stats must be a MetaStatistics or mapping")
        if key == "bbox":
            if isinstance(value, MetaBbox):
                self.bbox = value
                return
            if isinstance(value, (list, tuple)) or (np is not None and isinstance(value, np.ndarray)):
                self.bbox = MetaBbox(bboxes=value)
                return
            raise TypeError("meta.bbox must be a MetaBbox or list-like")
        if key == "_blosc2":
            if isinstance(value, MetaBlosc2):
                self._blosc2 = value
                return
            if isinstance(value, Mapping):
                self._blosc2 = MetaBlosc2.from_dict(value, strict=False)
                return
            raise TypeError("meta._blosc2 must be a MetaBlosc2 or mapping")
        if key == "is_seg":
            if value is not None and not isinstance(value, bool):
                raise TypeError("meta.is_seg must be a bool or None")
            setattr(self, key, value)
            return
        if key == "_has_array":
            if value is not None and not isinstance(value, bool):
                raise TypeError("meta._has_array must be a bool or None")
            setattr(self, key, value)
            return
        if key == "_image_meta_format":
            if value is not None and not isinstance(value, str):
                raise TypeError("meta._image_meta_format must be a str or None")
            setattr(self, key, value)
            return
        if key == "_mlarray_version":
            if value is not None and not isinstance(value, str):
                raise TypeError("meta._mlarray_version must be a str or None")
            self._mlarray_version = value
            return

        value_dict = dict(value)

        if not is_serializable(value_dict):
            raise TypeError(f"meta.{key} is not JSON-serializable")

        setattr(self, key, value_dict)

    def to_dict(self, *, include_none: bool = True) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Args:
            include_none (bool): If False, keys with None (and empty extra) are
                omitted.

        Returns:
            Dict[str, Any]: Serialized metadata.
        """
        out: Dict[str, Any] = {
            "image": self.image,
            "stats": self.stats.to_dict() if self.stats is not None else None,
            "bbox": self.bbox.to_list() if self.bbox is not None else None,
            "is_seg": self.is_seg,
            "spatial": self.spatial.to_dict(),
            "_has_array": self._has_array,
            "_image_meta_format": self._image_meta_format,
            "_blosc2": self._blosc2.to_dict(),
            "_mlarray_version": self._mlarray_version,
            "extra": self.extra,
        }

        if not include_none:
            out = {k: v for k, v in out.items() if v is not None and not (k == "extra" and v == {})}

        return out

    def _validate_and_cast(self, ndims: int) -> None:
        """Validate and normalize metadata with a known dimensionality.

        Args:
            ndims (int): Number of spatial dimensions.
        """
        self.spatial._validate_and_cast(ndims)
        if self.bbox is not None:
            self.bbox._validate_and_cast(ndims)
        self._blosc2._validate_and_cast(ndims)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, strict: bool = True) -> Meta:
        """Construct Meta from a mapping.

        Args:
            d (Mapping[str, Any]): Mapping with keys in {"image", "stats",
                "bbox", "spatial", "_blosc2", "_mlarray_version",
                "_image_meta_format", "_has_array", "is_seg", "extra"}.
            strict (bool): If True, unknown keys raise. If False, unknown keys
                are added to ``extra``.

        Returns:
            Meta: Parsed instance.
        """
        if not isinstance(d, Mapping):
            raise TypeError(f"from_dict expects a mapping, got {type(d).__name__}")

        known = {"image", "stats", "bbox", "spatial", "_has_array", "_image_meta_format", "_blosc2", "_mlarray_version", "is_seg", "extra"}
        d = dict(d)
        unknown = set(d.keys()) - known

        if unknown and strict:
            raise KeyError(f"Unknown meta keys in from_dict: {sorted(unknown)}")

        extra = dict(d.get("extra") or {})
        if unknown and not strict:
            for k in unknown:
                extra[k] = d[k]

        spatial = d.get("spatial")
        if spatial is None:
            spatial = MetaSpatial()
        else:
            spatial = MetaSpatial.from_dict(spatial, strict=strict)

        stats = d.get("stats")
        if stats is None:
            stats = None
        else:
            stats = MetaStatistics.from_dict(stats, strict=strict)

        bbox = d.get("bbox")
        if bbox is None:
            bbox = None
        else:
            bbox = MetaBbox.from_list(bbox)

        _blosc2 = d.get("_blosc2")
        if _blosc2 is None:
            _blosc2 = MetaBlosc2()
        else:
            _blosc2 = MetaBlosc2.from_dict(_blosc2, strict=strict)

        return cls(
            image=d.get("image"),
            stats=stats,
            bbox=bbox,
            is_seg=d.get("is_seg"),
            spatial=spatial,
            _has_array=d.get("_has_array"),
            _image_meta_format=d.get("_image_meta_format"),
            _blosc2=_blosc2,
            _mlarray_version=d.get("_mlarray_version"),            
            extra=extra,
        )

    def copy_from(self, other: Meta) -> None:
        """Copy fields from another Meta where this instance is missing data.

        Args:
            other (Meta): Source Meta instance.
        """
        if self.image is None:
            self.image = other.image
        if self.stats is None:
            self.stats = other.stats
        if self.bbox is None:
            self.bbox = other.bbox
        if self.is_seg is None:
            self.is_seg = other.is_seg
        if self.spatial is None:
            self.spatial = other.spatial
        elif other.spatial is not None:
            if self.spatial.spacing is None:
                self.spatial.spacing = other.spatial.spacing
            if self.spatial.origin is None:
                self.spatial.origin = other.spatial.origin
            if self.spatial.direction is None:
                self.spatial.direction = other.spatial.direction
            if self.spatial.shape is None:
                self.spatial.shape = other.spatial.shape
        if self._has_array is not None:
            self._has_array = other._has_array
        if self._image_meta_format is not None:
            self._image_meta_format = other._image_meta_format
        if self._blosc2 is None:
            self._blosc2 = other._blosc2
        elif other._blosc2 is not None:
            if self._blosc2.chunk_size is None:
                self._blosc2.chunk_size = other._blosc2.chunk_size
            if self._blosc2.block_size is None:
                self._blosc2.block_size = other._blosc2.block_size
            if self._blosc2.patch_size is None:
                self._blosc2.patch_size = other._blosc2.patch_size
        if self._mlarray_version is None:
            self._mlarray_version = other._mlarray_version
        if self.extra == {}:
            self.extra = other.extra


def _cast_to_list(value: Any, label: str):
    """Cast a list/tuple/ndarray to a Python list (recursively).

    Args:
        value (Any): Input value to cast.
        label (str): Label used in error messages.

    Returns:
        list: List representation of the input value.
    """
    if isinstance(value, list):
        out = value
    elif isinstance(value, tuple):
        out = list(value)
    elif np is not None and isinstance(value, np.ndarray):
        out = value.tolist()
    else:
        raise TypeError(f"{label} must be a list, tuple, or numpy array")

    if not isinstance(out, list):
        raise TypeError(f"{label} must be a list, tuple, or numpy array")

    for idx, item in enumerate(out):
        if isinstance(item, (list, tuple)) or (np is not None and isinstance(item, np.ndarray)):
            out[idx] = _cast_to_list(item, label)
    return out


def _validate_int(value: Any, label: str) -> None:
    """Validate that a value is an int.

    Args:
        value (Any): Value to validate.
        label (str): Label used in error messages.
    """
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an int")


def _validate_float_int_list(value: Any, label: str, ndims: Optional[int] = None) -> None:
    """Validate a list of float/int values with optional length.

    Args:
        value (Any): Value to validate.
        label (str): Label used in error messages.
        ndims (Optional[int]): Required length if provided.
    """
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    if ndims is not None and len(value) != ndims:
        raise ValueError(f"{label} must have length {ndims}")
    for item in value:
        if not isinstance(item, (float, int)):
            raise TypeError(f"{label} must contain only floats or ints")


def _validate_float_int_matrix(value: Any, label: str, ndims: Optional[int] = None) -> None:
    """Validate a 2D list of float/int values with optional shape.

    Args:
        value (Any): Value to validate.
        label (str): Label used in error messages.
        ndims (Optional[int]): Required square dimension if provided.
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
        for item in row:
            if not isinstance(item, (float, int)):
                raise TypeError(f"{label} must contain only floats or ints")
