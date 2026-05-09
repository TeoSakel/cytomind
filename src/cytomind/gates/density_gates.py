from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, ClassVar, Mapping, Sequence, TYPE_CHECKING, TypeVar

import anndata as ad
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, label as nd_label
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import chi2, false_discovery_control
from skimage.feature import peak_local_max
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_local,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label
from skimage.segmentation import random_walker, watershed

from cytomind.domain.gates import GateNode

from . import GateRegistry
from .base import Gate

if TYPE_CHECKING:
    from pandas import DataFrame
    from cytomind.domain.constants import BooleanArray, FloatArray, IntArray
else:
    DataFrame = object
    BooleanArray = object
    FloatArray = object
    IntArray = object

__all__ = [
    "DensityGate",
    "DensitySegmentationGate",
    "DensityRegionGate",
    "DensityLandscape",
    "Landscape1D",
    "Landscape2D",
    "GridLandscape1D",
    "GridLandscape2D",
    "RegularGridLandscape1D",
    "RegularGridLandscape2D",
    "LandscapeBuilder",
    "HistogramLandscape1DBuilder",
    "HistogramLandscape2DBuilder",
    "FeatureDetector",
    "ModeDetector",
    "CurvatureDetector",
    "FeatureSelector",
    "NearestModeSelector",
    "HighestDensitySelector",
    "ModeIdSelector",
    "RegionExtractor",
    "ProminencePeakRegionExtractor",
    "RandomWalkerRegionExtractor",
    "WatershedRegionExtractor",
    "LevelSetConnectedComponentExtractor",
    "BoundaryBuilder",
    "IntervalBoundaryBuilder",
    "ContourBoundaryBuilder",
    "GateRegion",
    "IntervalRegion",
    "RasterRegion2D",
    "ModeFeature",
]

FLOAT_DTYPE = np.float32


@dataclass
class ModeFeature:
    id: str
    rank: int
    point: list[float]
    grid_index: list[int]
    peak_density: float
    properties: dict[str, Any] = field(default_factory=dict)
    footprint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": int(self.rank),
            "point": [float(v) for v in self.point],
            "grid_index": [int(v) for v in self.grid_index],
            "peak_density": float(self.peak_density),
            "properties": dict(self.properties),
            "footprint": None if self.footprint is None else dict(self.footprint),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModeFeature":
        return cls(
            id=str(payload["id"]),
            rank=int(payload["rank"]),
            point=[float(v) for v in payload["point"]],
            grid_index=[int(v) for v in payload["grid_index"]],
            peak_density=float(payload["peak_density"]),
            properties=dict(payload.get("properties", {})),
            footprint=(
                dict(payload["footprint"])
                if isinstance(payload.get("footprint"), Mapping)
                else None
            ),
        )

    def footprint_bounds(self) -> "tuple[np.ndarray, np.ndarray] | None":
        """Return the AABB of the footprint as ``(lower, upper)`` float32 arrays.

        Supports ``{"bounds": [[lo...], [hi...]]}`` and
        ``{"lower": [...], "upper": [...]}`` formats.
        Returns ``None`` when no usable bounds are found.
        """
        fp = self.footprint
        if not isinstance(fp, Mapping):
            return None
        raw_bounds = fp.get("bounds")
        if isinstance(raw_bounds, Sequence) and len(raw_bounds) == 2:
            lo = np.asarray(raw_bounds[0], dtype=FLOAT_DTYPE).reshape(-1)
            hi = np.asarray(raw_bounds[1], dtype=FLOAT_DTYPE).reshape(-1)
            if lo.shape == hi.shape:
                return lo, hi
        if "lower" in fp and "upper" in fp:
            lo = np.asarray(fp["lower"], dtype=FLOAT_DTYPE).reshape(-1)
            hi = np.asarray(fp["upper"], dtype=FLOAT_DTYPE).reshape(-1)
            if lo.shape == hi.shape:
                return lo, hi
        return None

    def overlaps_roi_with_footprint(self, roi_bounds: Any) -> "bool | None":
        """Check AABB overlap between this feature's footprint and *roi_bounds*.

        Returns ``None`` when no footprint bounds are available (caller should
        fall back to point containment).
        """
        roi = _normalize_roi_bounds(roi_bounds)
        if roi is None:
            return None
        bounds = self.footprint_bounds()
        if bounds is None:
            return None
        roi_lower, roi_upper = roi
        feat_lower, feat_upper = bounds
        if roi_lower.shape != feat_lower.shape or roi_upper.shape != feat_upper.shape:
            return None
        return bool(np.all(feat_lower <= roi_upper) and np.all(feat_upper >= roi_lower))

    def within_roi(self, roi_bounds: Any) -> bool:
        """Return ``True`` if this feature lies inside *roi_bounds*.

        Checks footprint AABB overlap first; falls back to point containment.
        """
        if roi_bounds is None:
            return True
        overlap = self.overlaps_roi_with_footprint(roi_bounds)
        if overlap is not None:
            return overlap
        normalized = _normalize_roi_bounds(roi_bounds)
        if normalized is None:
            return True
        lower_arr, upper_arr = normalized
        point = np.asarray(self.point, dtype=FLOAT_DTYPE).reshape(-1)
        if lower_arr.shape != point.shape or upper_arr.shape != point.shape:
            return True
        return bool(np.all(point >= lower_arr) and np.all(point <= upper_arr))

    def footprint_mask(self, shape: "tuple[int, int]") -> "np.ndarray":
        """Return a boolean mask of shape *shape* representing this feature's footprint.

        Footprint formats resolved in priority order:

        1. ``footprint["grid_mask"]``    – boolean array already shaped like *shape*.
        2. ``footprint["grid_indices"]`` – array/list of ``[row, col]`` pairs.
        3. ``footprint["grid_index"]``   – single pair, promoted to length-1 list.
        4. ``self.grid_index``           – fallback when no footprint is present.

        Out-of-bounds indices raise ``ValueError``.
        """
        fp = self.footprint

        if isinstance(fp, Mapping):
            # format 1: explicit boolean mask
            raw_mask = fp.get("grid_mask")
            if raw_mask is not None:
                mask = np.asarray(raw_mask, dtype=np.bool_)
                if mask.shape != shape:
                    raise ValueError(
                        f"Feature {self.id!r} grid_mask shape {mask.shape!r} "
                        f"does not match density shape {shape!r}."
                    )
                return mask

            # format 2 (+3): index pairs (single pair promoted to length-1)
            raw_indices = fp.get("grid_indices")
            raw_single = fp.get("grid_index")
            if raw_indices is None and isinstance(raw_single, Sequence) and len(raw_single) == 2:
                raw_indices = [raw_single]

            if isinstance(raw_indices, Sequence) and not isinstance(raw_indices, (str, bytes)):
                idx = np.asarray(raw_indices, dtype=np.intp)
                if idx.ndim == 2 and idx.shape[1] == 2:
                    return _indices_to_mask(idx, shape, self.id)

        # fallback: feature.grid_index
        idx = np.asarray([self.grid_index], dtype=np.intp)
        return _indices_to_mask(idx, shape, self.id)


@dataclass
class ExtractionResult:
    regions: Mapping[str, "GateRegion"]
    label_image: IntArray| None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DensityLandscape(ABC):
    density: FloatArray
    counts: IntArray

    @property
    @abstractmethod
    def n_dims(self) -> int:
        pass

    @abstractmethod
    def to_state(self) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def grid_spec(self) -> dict[str, Any]:
        pass

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "DensityLandscape":
        landscape_type = str(payload.get("type"))
        if landscape_type.startswith("grid1d"):
            return Landscape1D.from_state(payload)
        if landscape_type.startswith("grid2d"):
            return Landscape2D.from_state(payload)
        raise ValueError(f"Unsupported density landscape type {landscape_type!r}.")


@dataclass
class Landscape1D(DensityLandscape):
    density: FloatArray
    counts: IntArray

    @property
    def n_dims(self) -> int:
        return 1

    @property
    @abstractmethod
    def bin_edges(self) -> FloatArray:
        pass

    @property
    def axis(self) -> FloatArray:
        edges = np.asarray(self.bin_edges, dtype=FLOAT_DTYPE)
        return edges[:-1] + np.diff(edges) / 2.0

    @property
    def n_bins(self) -> int:
        return max(self.bin_edges.shape[0] - 1, 0)

    @property
    def is_regular(self) -> bool:
        return False

    @property
    def grid_spec(self) -> dict[str, Any]:
        return {
            "type": "grid1d",
            "bin_edges": np.asarray(self.bin_edges, dtype=FLOAT_DTYPE),
        }

    def left_edge_index(self, value: float) -> int:
        return int(np.searchsorted(self.bin_edges, value, side="left"))

    def right_edge_index(self, value: float) -> int:
        return int(np.searchsorted(self.bin_edges, value, side="right"))

    def bin_index(self, value: float) -> int:
        return self.right_edge_index(value) - 1

    def digitize(self, values: FloatArray) -> tuple[IntArray, BooleanArray]:
        arr = np.asarray(values, dtype=FLOAT_DTYPE).reshape(-1)
        idx = np.searchsorted(self.bin_edges, arr, side="right") - 1
        valid = (idx >= 0) & (idx < self.n_bins)
        return np.asarray(idx, dtype=np.int_), np.asarray(valid, dtype=np.bool_)

    def interval_to_bin_slice(self, min_value: float, max_value: float) -> tuple[int, int]:
        left_bin = self.left_edge_index(min_value)
        right_bin = self.left_edge_index(max_value) - 1
        return left_bin, right_bin

    def interval_bounds(self, left_bin: int, right_bin: int) -> tuple[float, float]:
        left_val, right_val = self.bin_edges[[left_bin, right_bin + 1]]
        return float(left_val), float(right_val)

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "Landscape1D":
        landscape_type = str(payload.get("type"))
        if landscape_type == "grid1d":
            return GridLandscape1D.from_state(payload)
        if landscape_type == "grid1d_regular":
            return RegularGridLandscape1D.from_state(payload)
        raise ValueError(f"Unsupported 1D density landscape type {landscape_type!r}.")


@dataclass
class GridLandscape1D(Landscape1D):
    _bin_edges: FloatArray
    density: FloatArray
    counts: IntArray
    kernel: KernelSpec = field(default_factory=lambda: {"type": "identity", "params": {}})

    @property
    def bin_edges(self) -> FloatArray:
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, value: FloatArray):
        self._bin_edges = np.asarray(value, dtype=FLOAT_DTYPE)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "grid1d",
            "bin_edges": np.asarray(self.bin_edges, dtype=FLOAT_DTYPE),
            "density": np.asarray(self.density, dtype=FLOAT_DTYPE),
            "counts": np.asarray(self.counts, dtype=FLOAT_DTYPE),
            "kernel": _serialize_kernel(self.kernel),
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "GridLandscape1D":
        return cls(
            _bin_edges=np.asarray(payload["bin_edges"], dtype=FLOAT_DTYPE),
            density=np.asarray(payload["density"], dtype=FLOAT_DTYPE),
            counts=np.asarray(payload["counts"], dtype=np.int_),
            kernel=_restore_kernel_from_state(payload),
        )


@dataclass
class Landscape2D(DensityLandscape):
    density: FloatArray
    counts: IntArray

    @property
    def n_dims(self) -> int:
        return 2

    @property
    @abstractmethod
    def x_edges(self) -> FloatArray:
        pass

    @property
    @abstractmethod
    def y_edges(self) -> FloatArray:
        pass

    @property
    def x_centers(self) -> FloatArray:
        edges = np.asarray(self.x_edges, dtype=FLOAT_DTYPE)
        return edges[:-1] + np.diff(edges) / 2.0

    @property
    def y_centers(self) -> FloatArray:
        edges = np.asarray(self.y_edges, dtype=FLOAT_DTYPE)
        return edges[:-1] + np.diff(edges) / 2.0

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.density.shape[0]), int(self.density.shape[1])  # type: ignore[attr-defined]

    @property
    def is_regular(self) -> bool:
        return False

    @property
    def grid_spec(self) -> dict[str, Any]:
        return {
            "type": "grid2d",
            "x_edges": np.asarray(self.x_edges, dtype=FLOAT_DTYPE),
            "y_edges": np.asarray(self.y_edges, dtype=FLOAT_DTYPE),
        }

    def x_bin_index(self, value: float) -> int:
        edges = np.asarray(self.x_edges, dtype=FLOAT_DTYPE)
        return int(np.searchsorted(edges, value, side="right")) - 1

    def y_bin_index(self, value: float) -> int:
        edges = np.asarray(self.y_edges, dtype=FLOAT_DTYPE)
        return int(np.searchsorted(edges, value, side="right")) - 1

    def digitize(self, coords: FloatArray) -> tuple[IntArray, IntArray, BooleanArray]:
        arr = np.asarray(coords, dtype=FLOAT_DTYPE)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("GridLandscape2D expects coordinates with shape (n_events, 2).")
        x_idx = np.searchsorted(np.asarray(self.x_edges, dtype=FLOAT_DTYPE), arr[:, 0], side="right") - 1
        y_idx = np.searchsorted(np.asarray(self.y_edges, dtype=FLOAT_DTYPE), arr[:, 1], side="right") - 1
        valid = (
            (x_idx >= 0)
            & (x_idx < self.shape[1])
            & (y_idx >= 0)
            & (y_idx < self.shape[0])
        )
        return (
            np.asarray(x_idx, dtype=np.int_),
            np.asarray(y_idx, dtype=np.int_),
            np.asarray(valid, dtype=np.bool_),
        )

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "Landscape2D":
        landscape_type = str(payload.get("type"))
        if landscape_type == "grid2d":
            return GridLandscape2D.from_state(payload)
        if landscape_type == "grid2d_regular":
            return RegularGridLandscape2D.from_state(payload)
        raise ValueError(f"Unsupported 2D density landscape type {landscape_type!r}.")


@dataclass
class GridLandscape2D(Landscape2D):
    _x_edges: FloatArray
    _y_edges: FloatArray
    density: FloatArray
    counts: IntArray
    kernel: KernelSpec = field(default_factory=lambda: {"type": "identity", "params": {}})

    @property
    def x_edges(self) -> FloatArray:
        return self._x_edges

    @x_edges.setter
    def x_edges(self, value: FloatArray):
        self._x_edges = np.asarray(value, dtype=FLOAT_DTYPE)

    @property
    def y_edges(self) -> FloatArray:
        return self._y_edges

    @y_edges.setter
    def y_edges(self, value: FloatArray):
        self._y_edges = np.asarray(value, dtype=FLOAT_DTYPE)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "grid2d",
            "x_edges": np.asarray(self.x_edges, dtype=FLOAT_DTYPE),
            "y_edges": np.asarray(self.y_edges, dtype=FLOAT_DTYPE),
            "density": np.asarray(self.density, dtype=FLOAT_DTYPE),
            "counts": np.asarray(self.counts, dtype=FLOAT_DTYPE),
            "kernel": _serialize_kernel(self.kernel),
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "GridLandscape2D":
        return cls(
            _x_edges=np.asarray(payload["x_edges"], dtype=FLOAT_DTYPE),
            _y_edges=np.asarray(payload["y_edges"], dtype=FLOAT_DTYPE),
            density=np.asarray(payload["density"], dtype=FLOAT_DTYPE),
            counts=np.asarray(payload["counts"], dtype=np.int_),
            kernel=_restore_kernel_from_state(payload),
        )


@dataclass
class RegularGridLandscape1D(Landscape1D):
    start: float
    step_size: float
    step_count: int
    density: FloatArray
    counts: IntArray
    kernel: KernelSpec = field(default_factory=lambda: {"type": "identity", "params": {}})

    @property
    def bin_edges(self) -> FloatArray:
        return self.start + self.step_size * np.arange(self.step_count + 1, dtype=FLOAT_DTYPE)

    @property
    def is_regular(self) -> bool:
        return True

    def left_edge_index(self, value: float) -> int:
        if self.n_bins <= 0:
            return 0
        scaled = (float(value) - self.start) / self.step_size
        idx = int(np.ceil(scaled - 1e-9))
        return int(np.clip(idx, 0, self.n_bins))

    def right_edge_index(self, value: float) -> int:
        if self.n_bins <= 0:
            return 0
        scaled = (float(value) - self.start) / self.step_size
        idx = int(np.floor(scaled + 1e-9)) + 1
        return int(np.clip(idx, 0, self.n_bins))

    def bin_index(self, value: float) -> int:
        if self.n_bins <= 0:
            return -1
        scaled = (float(value) - self.start) / self.step_size
        idx = int(np.floor(scaled + 1e-9))
        return idx

    def digitize(self, values: FloatArray) -> tuple[IntArray, BooleanArray]:
        arr = np.asarray(values, dtype=FLOAT_DTYPE).reshape(-1)
        if self.n_bins <= 0:
            return np.asarray([], dtype=np.int_), np.asarray([], dtype=np.bool_)
        idx = np.floor((arr - self.start) / self.step_size + 1e-9).astype(np.int_)
        valid = (idx >= 0) & (idx < self.n_bins)
        return idx, np.asarray(valid, dtype=np.bool_)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "grid1d_regular",
            "start": float(self.start),
            "step_size": float(self.step_size),
            "step_count": int(self.step_count),
            "density": np.asarray(self.density, dtype=FLOAT_DTYPE),
            "counts": np.asarray(self.counts, dtype=np.int_),
            "kernel": _serialize_kernel(self.kernel),
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "RegularGridLandscape1D":
        return cls(
            start=float(payload["start"]),
            step_size=float(payload["step_size"]),
            step_count=int(payload["step_count"]),
            density=np.asarray(payload["density"], dtype=FLOAT_DTYPE),
            counts=np.asarray(payload["counts"], dtype=np.int_),
            kernel=_restore_kernel_from_state(payload),
        )


@dataclass
class RegularGridLandscape2D(Landscape2D):
    start: tuple[float, float]
    step_size: tuple[float, float]
    step_count: tuple[int, int]
    density: FloatArray
    counts: IntArray
    kernel: KernelSpec = field(default_factory=lambda: {"type": "identity", "params": {}})

    def __post_init__(self):
        if self.step_count[0] < 0 or self.step_count[1] < 0:
            raise ValueError("step_count values must be non-negative.")
        if self.step_size[0] <= 0 or self.step_size[1] <= 0:
            raise ValueError("step_size values must be positive.")

    @property
    def x_edges(self) -> FloatArray:
        start_x, _ = self.start
        step_x, _ = self.step_size
        cols, _ = self.step_count
        return start_x + step_x * np.arange(cols + 1, dtype=FLOAT_DTYPE)

    @property
    def y_edges(self) -> FloatArray:
        _, start_y = self.start
        _, step_y = self.step_size
        _, rows = self.step_count
        return start_y + step_y * np.arange(rows + 1, dtype=FLOAT_DTYPE)

    @property
    def shape(self) -> tuple[int, int]:
        return self.step_count

    @property
    def is_regular(self) -> bool:
        return True

    def x_bin_index(self, value: float) -> int:
        start_x, _ = self.start
        step_x, _ = self.step_size
        cols = self.shape[1]
        if cols == 0:
            return -1
        return int(np.floor((float(value) - start_x) / step_x))

    def y_bin_index(self, value: float) -> int:
        _, start_y = self.start
        _, step_y = self.step_size
        rows = self.shape[0]
        if rows == 0:
            return -1
        return int(np.floor((float(value) - start_y) / step_y))

    def digitize(self, coords: FloatArray) -> tuple[IntArray, IntArray, BooleanArray]:
        arr = np.asarray(coords, dtype=FLOAT_DTYPE)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("GridLandscape2D expects coordinates with shape (n_events, 2).")
        start_x, start_y = self.start
        step_x, step_y = self.step_size
        rows, cols = self.shape
        if rows == 0 or cols == 0:
            empty = np.asarray([], dtype=np.int_)
            return empty, empty, np.asarray([], dtype=np.bool_)
        x_idx = np.floor((arr[:, 0] - start_x) / step_x).astype(np.int_)
        y_idx = np.floor((arr[:, 1] - start_y) / step_y).astype(np.int_)
        valid = (
            (x_idx >= 0)
            & (x_idx < cols)
            & (y_idx >= 0)
            & (y_idx < rows)
        )
        return x_idx, y_idx, np.asarray(valid, dtype=np.bool_)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "grid2d_regular",
            "start": [float(self.start[0]), float(self.start[1])],
            "step_size": [float(self.step_size[0]), float(self.step_size[1])],
            "step_count": [int(self.step_count[0]), int(self.step_count[1])],
            "density": np.asarray(self.density, dtype=FLOAT_DTYPE),
            "counts": np.asarray(self.counts, dtype=np.int_),
            "kernel": _serialize_kernel(self.kernel),
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "RegularGridLandscape2D":
        start_x, start_y = payload["start"]
        step_x, step_y = payload["step_size"]
        count_x, count_y = payload["step_count"]
        return cls(
            start=(float(start_x), float(start_y)),
            step_size=(float(step_x), float(step_y)),
            step_count=(int(count_x), int(count_y)),
            density=np.asarray(payload["density"], dtype=FLOAT_DTYPE),
            counts=np.asarray(payload["counts"], dtype=np.int_),
            kernel=_restore_kernel_from_state(payload),
        )


class SerializableDensityComponent(ABC):
    component_type: ClassVar[str]

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        pass


class LandscapeBuilder(SerializableDensityComponent, ABC):
    n_dims: ClassVar[int]

    @abstractmethod
    def build(self, data: FloatArray) -> DensityLandscape:
        pass


class FeatureDetector(SerializableDensityComponent, ABC):
    def __init__(self, mask_spec: MaskSpecLike = None) -> None:
        self.mask_spec = _normalize_mask_spec(mask_spec)

    def _serialized_mask_spec(self) -> dict[str, Any] | None:
        return _serialize_mask_spec(self.mask_spec)

    @abstractmethod
    def detect(self, landscape: DensityLandscape) -> list[ModeFeature]:
        pass


class FeatureSelector(SerializableDensityComponent, ABC):
    @abstractmethod
    def select(
        self,
        features: Sequence[ModeFeature],
        selection_hint: Mapping[str, Any] | None = None,
    ) -> ModeFeature:
        pass


class RegionExtractor(SerializableDensityComponent, ABC):
    def __init__(self, mask_spec: MaskSpecLike = None) -> None:
        self.mask_spec = _normalize_mask_spec(mask_spec)

    def _serialized_mask_spec(self) -> dict[str, Any] | None:
        return _serialize_mask_spec(self.mask_spec)

    def extract(
        self,
        landscape: DensityLandscape,
        *,
        target: ModeFeature,
        features: Sequence[ModeFeature],
        region_prefix: str,
    ) -> ExtractionResult:
        segmentation = self.segment(
            landscape,
            features,
            region_prefix=region_prefix,
        )
        selected_id = f"{region_prefix}_{target.id}"
        try:
            selected_region = segmentation.regions[selected_id]
        except KeyError as exc:
            raise KeyError(
                f"Target feature {target.id!r} was not found in the segmentation output."
            ) from exc
        label_image = None
        if isinstance(selected_region, IntervalRegion):
            grid = np.zeros_like(np.asarray(segmentation.label_image), dtype=np.int_) if segmentation.label_image is not None else None
            if grid is not None:
                left_bin, right_bin = landscape.interval_to_bin_slice(selected_region.min_value, selected_region.max_value) # pyright: ignore[reportAttributeAccessIssue]
                grid[left_bin:right_bin + 1] = 1
                label_image = grid
        elif isinstance(selected_region, RasterRegion2D):
            label_image = np.asarray(selected_region.mask, dtype=np.int_)
        return ExtractionResult(
            regions={selected_id: selected_region},
            label_image=label_image,
            diagnostics={**dict(segmentation.diagnostics), "n_regions": 1},
        )

    @abstractmethod
    def segment(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region_prefix: str,
    ) -> ExtractionResult:
        pass


class BoundaryBuilder(SerializableDensityComponent, ABC):
    @abstractmethod
    def build(self, region: "GateRegion") -> dict[str, Any] | None:
        pass


class GateRegion(ABC):
    region_type: ClassVar[str]

    @classmethod
    def from_spec(
        cls,
        payload: Mapping[str, Any],
        *,
        grid_spec: Mapping[str, Any] | None = None,
    ) -> "GateRegion":
        region_type = str(payload.get("type"))
        if region_type == IntervalRegion.region_type:
            return IntervalRegion.from_spec(payload, grid_spec=grid_spec)
        if region_type == RasterRegion2D.region_type:
            return RasterRegion2D.from_spec(payload, grid_spec=grid_spec)
        raise ValueError(f"Unknown density region type {region_type!r}.")

    @abstractmethod
    def contains(self, data: FloatArray) -> BooleanArray:
        pass

    @abstractmethod
    def score(self, data: FloatArray) -> FloatArray:
        pass

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        pass


@dataclass
class IntervalRegion(GateRegion):
    min_value: float
    max_value: float

    region_type: ClassVar[str] = "interval"

    def contains(self, data: FloatArray) -> BooleanArray:
        values = np.asarray(data, dtype=FLOAT_DTYPE).reshape(-1)
        return np.asarray((values >= self.min_value) & (values < self.max_value), dtype=np.bool_)

    def score(self, data: FloatArray) -> FloatArray:
        values = np.asarray(data, dtype=FLOAT_DTYPE).reshape(-1)
        left = values - self.min_value
        right = self.max_value - values
        inside = np.minimum(left, right)
        outside = -np.maximum(self.min_value - values, values - self.max_value)
        scores = np.where((values >= self.min_value) & (values < self.max_value), inside, outside)
        scale = float(np.max(np.abs(scores))) if scores.size else 0.0
        if scale > 0: scores /= scale
        return scores

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.region_type,
            "bounds": [float(self.min_value), float(self.max_value)],
        }

    @classmethod
    def from_spec(
        cls,
        payload: Mapping[str, Any],
        *,
        grid_spec: Mapping[str, Any] | None = None,
    ) -> "IntervalRegion":
        del grid_spec
        bounds = payload.get("bounds", [None, None])
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError("IntervalRegion spec requires exactly two bounds.")
        return cls(min_value=float(bounds[0]), max_value=float(bounds[1]))


@dataclass
class RasterRegion2D(GateRegion):
    x_edges: FloatArray
    y_edges: FloatArray
    mask: BooleanArray

    region_type: ClassVar[str] = "raster2d"

    @property
    def x_centers(self) -> FloatArray:
        return self.x_edges[:-1] + np.diff(self.x_edges) / 2.0

    @property
    def y_centers(self) -> FloatArray:
        return self.y_edges[:-1] + np.diff(self.y_edges) / 2.0

    def _digitize(self, data: FloatArray) -> tuple[IntArray, IntArray, BooleanArray]:
        coords = np.asarray(data, dtype=FLOAT_DTYPE)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("RasterRegion2D expects input data with shape (n_events, 2).")
        x_idx = np.searchsorted(self.x_edges, coords[:, 0], side="right") - 1
        y_idx = np.searchsorted(self.y_edges, coords[:, 1], side="right") - 1
        valid: BooleanArray = (
            (x_idx >= 0)
            & (x_idx < self.mask.shape[1])
            & (y_idx >= 0)
            & (y_idx < self.mask.shape[0])
        )
        return x_idx, y_idx, valid

    def contains(self, data: FloatArray) -> BooleanArray:
        x_idx, y_idx, valid = self._digitize(data)
        result: BooleanArray = np.zeros(len(x_idx), dtype=np.bool_)
        result[valid] = self.mask[y_idx[valid], x_idx[valid]]
        return result

    def score(self, data: FloatArray) -> FloatArray:
        x_idx, y_idx, valid = self._digitize(data)
        inside_dist = distance_transform_edt(self.mask, return_distances=True)
        outside_dist = distance_transform_edt(~self.mask, return_distances=True)
        signed = inside_dist - outside_dist # pyright: ignore[reportOperatorIssue]
        result: FloatArray = np.full(len(x_idx), -1.0, dtype=FLOAT_DTYPE)
        result[valid] = signed[y_idx[valid], x_idx[valid]]
        scale = float(np.max(np.abs(signed))) if signed.size else 0.0
        if scale > 0: result /= scale
        return result

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.region_type,
            "mask": np.asarray(self.mask, dtype=np.bool_),
        }

    @classmethod
    def from_spec(
        cls,
        payload: Mapping[str, Any],
        *,
        grid_spec: Mapping[str, Any] | None = None,
    ) -> "RasterRegion2D":
        if grid_spec is None:
            raise ValueError("RasterRegion2D spec requires grid_spec.")
        return cls(
            x_edges=np.asarray(grid_spec["x_edges"], dtype=FLOAT_DTYPE),
            y_edges=np.asarray(grid_spec["y_edges"], dtype=FLOAT_DTYPE),
            mask=np.asarray(payload["mask"], dtype=np.bool_),
        )


_LANDSCAPE_BUILDERS: dict[str, type[LandscapeBuilder]] = {}
_FEATURE_DETECTORS: dict[str, type[FeatureDetector]] = {}
_FEATURE_SELECTORS: dict[str, type[FeatureSelector]] = {}
_REGION_EXTRACTORS: dict[str, type[RegionExtractor]] = {}
_BOUNDARY_BUILDERS: dict[str, type[BoundaryBuilder]] = {}
ComponentT = TypeVar("ComponentT")
KernelLike = str | Mapping[str, Any] | np.ndarray | None
KernelSpec = dict[str, Any]
MaskSpec = dict[str, Any]
MaskSpecLike = str | Mapping[str, Any] | np.ndarray | None
StructureLike = str | Sequence[Sequence[int | bool]] | np.ndarray

_GLOBAL_MASK_THRESHOLD_METHODS: dict[str, Any] = {
    "isodata": threshold_isodata,
    "li": threshold_li,
    "mean": threshold_mean,
    "otsu": threshold_otsu,
    "triangle": threshold_triangle,
    "yen": threshold_yen,
}
_LOCAL_MASK_THRESHOLD_METHODS = {"local", "local_gaussian", "local_mean", "local_median"}


def _register_component(
    registry: dict[str, type[ComponentT]],
    component_type: str,
):
    def decorator(component_cls: type[ComponentT]) -> type[ComponentT]:
        registry[component_type] = component_cls
        return component_cls

    return decorator


def _serialize_component(component: SerializableDensityComponent | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if component is None:
        return None
    if isinstance(component, Mapping):
        return {"type": str(component["type"]), "params": dict(component.get("params", {}))}
    return component.to_spec()


def _load_component(
    component: ComponentT | Mapping[str, Any] | None,
    *,
    registry: Mapping[str, type[ComponentT]],
    fallback: ComponentT | None = None,
) -> ComponentT | None:
    if component is None:
        return fallback
    if not isinstance(component, Mapping):
        return component
    component_type = str(component["type"])
    params = dict(component.get("params", {}))
    try:
        component_cls = registry[component_type]
    except KeyError as exc:
        raise ValueError(f"Unknown density component type {component_type!r}.") from exc
    return component_cls(**params)


def _normalize_roi_bounds(roi_bounds: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if roi_bounds is None:
        return None
    if not isinstance(roi_bounds, Sequence) or len(roi_bounds) != 2:
        return None
    lower, upper = roi_bounds
    lower_arr = np.asarray(lower, dtype=FLOAT_DTYPE).reshape(-1)
    upper_arr = np.asarray(upper, dtype=FLOAT_DTYPE).reshape(-1)
    return lower_arr, upper_arr



def _indices_to_mask(
    indices: "np.ndarray",
    shape: tuple[int, int],
    feature_id: str,
) -> np.ndarray:
    """Scatter a (N, 2) int array of ``[row, col]`` pairs into a boolean mask.

    Raises ``ValueError`` if any index is out of bounds for *shape*.
    """
    rows, cols = shape
    r_arr, c_arr = indices[:, 0], indices[:, 1]
    oob = (r_arr < 0) | (r_arr >= rows) | (c_arr < 0) | (c_arr >= cols)
    if oob.any():
        bad = indices[oob].tolist()
        raise ValueError(
            f"Feature {feature_id!r} grid_indices contain out-of-bounds pairs {bad!r} "
            f"for grid shape {shape!r}."
        )
    mask = np.zeros(shape, dtype=np.bool_)
    mask[r_arr, c_arr] = True
    return mask



def _build_marker_image_2d(
    features: Sequence["ModeFeature"],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a 2-D integer markers array for watershed / random_walker.

    Each feature is assigned a unique positive integer label (1-based,
    matching its position in *features*).  Unassigned pixels are 0.

    Algorithm:
    1. Convert every feature footprint to a boolean mask via
       ``ModeFeature.footprint_mask``.
    2. Stack masks and count how many features claim each cell.
    3. Label cells that are claimed by exactly one feature (no conflict)
       directly with that feature's marker id.
    4. Any feature that received zero marker pixels after step 3 (all its
       cells were contested) is force-assigned the first cell of its mask.
    """
    n = len(features)
    masks = np.stack([f.footprint_mask(shape) for f in features])  # (n, rows, cols)
    claim_count = masks.sum(axis=0)                                # (rows, cols) int
    unique = claim_count == 1                                      # uncontested cells

    markers = np.zeros(shape, dtype=np.int_)
    for marker_id, mask in enumerate(masks, start=1):
        markers[mask & unique] = marker_id

    # Force-assign any feature that ended up with no marker pixels.
    for marker_id, (feature, mask) in enumerate(zip(features, masks), start=1):
        if not (markers == marker_id).any():
            r_f, c_f = np.nonzero(mask)
            if r_f.size == 0:
                raise ValueError(
                    f"Feature {feature.id!r} has no valid marker cells in a grid of shape {shape!r}."
                )
            markers[r_f[0], c_f[0]] = marker_id

    return markers



def _points_close(a: FloatArray, b: FloatArray, *, atol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=FLOAT_DTYPE), np.asarray(b, dtype=FLOAT_DTYPE), atol=atol, rtol=0.0))


def _remove_duplicate_points(points: FloatArray, *, atol: float = 1e-9) -> FloatArray:
    coords = np.asarray(points, dtype=FLOAT_DTYPE)
    if coords.shape[0] <= 1:
        return coords
    deltas = np.diff(coords, axis=0)
    keep = np.ones(coords.shape[0], dtype=np.bool_)
    keep[1:] = ~np.all(np.abs(deltas) <= atol, axis=1)
    return coords[keep]


def _simplify_polyline(points: FloatArray, *, closed: bool, atol: float = 1e-9) -> FloatArray:
    coords = _remove_duplicate_points(np.asarray(points, dtype=FLOAT_DTYPE), atol=atol)
    if coords.shape[0] <= 2:
        return coords

    if closed and _points_close(coords[0], coords[-1], atol=atol):
        coords = coords[:-1]
    if coords.shape[0] <= 2:
        return coords

    changed = True
    while changed and coords.shape[0] > (3 if closed else 2):
        changed = False
        keep: list[np.ndarray] = []
        n = coords.shape[0]
        for idx in range(n):
            if not closed and idx in {0, n - 1}:
                keep.append(coords[idx])
                continue
            prev_pt = coords[(idx - 1) % n]
            curr_pt = coords[idx]
            next_pt = coords[(idx + 1) % n]
            v1 = curr_pt - prev_pt
            v2 = next_pt - curr_pt
            area = float(v1[0] * v2[1] - v1[1] * v2[0])
            if np.isclose(area, 0.0, atol=atol) and float(np.dot(v1, v2)) >= 0.0:
                changed = True
                continue
            keep.append(curr_pt)
        coords = np.asarray(keep, dtype=FLOAT_DTYPE)
    return coords


def _frame_arc_length(point: FloatArray, *, xmin: float, xmax: float, ymin: float, ymax: float, atol: float) -> float:
    x = float(point[0])
    y = float(point[1])
    if np.isclose(y, ymin, atol=atol):
        return x - xmin
    if np.isclose(x, xmax, atol=atol):
        return (xmax - xmin) + (y - ymin)
    if np.isclose(y, ymax, atol=atol):
        return (xmax - xmin) + (ymax - ymin) + (xmax - x)
    return 2.0 * (xmax - xmin) + (ymax - ymin) + (ymax - y)


def _frame_point_from_s(
    s: float,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> FloatArray:
    width = xmax - xmin
    height = ymax - ymin
    perimeter = 2.0 * (width + height)
    if perimeter <= 0:
        return np.asarray([xmin, ymin], dtype=FLOAT_DTYPE)
    s = s % perimeter
    if s <= width:
        return np.asarray([xmin + s, ymin], dtype=FLOAT_DTYPE)
    s -= width
    if s <= height:
        return np.asarray([xmax, ymin + s], dtype=FLOAT_DTYPE)
    s -= height
    if s <= width:
        return np.asarray([xmax - s, ymax], dtype=FLOAT_DTYPE)
    s -= width
    return np.asarray([xmin, ymax - s], dtype=FLOAT_DTYPE)


def _snap_point_to_frame(
    point: FloatArray,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_tol: float,
    y_tol: float,
) -> FloatArray | None:
    x = float(point[0])
    y = float(point[1])
    distances = {
        "left": abs(x - xmin),
        "right": abs(x - xmax),
        "bottom": abs(y - ymin),
        "top": abs(y - ymax),
    }
    edge = min(distances, key=distances.get) # pyright: ignore[reportArgumentType, reportCallIssue]
    if edge in {"left", "right"} and distances[edge] > x_tol:
        return None
    if edge in {"bottom", "top"} and distances[edge] > y_tol:
        return None
    if edge == "left":
        return np.asarray([xmin, np.clip(y, ymin, ymax)], dtype=FLOAT_DTYPE)
    if edge == "right":
        return np.asarray([xmax, np.clip(y, ymin, ymax)], dtype=FLOAT_DTYPE)
    if edge == "bottom":
        return np.asarray([np.clip(x, xmin, xmax), ymin], dtype=FLOAT_DTYPE)
    return np.asarray([np.clip(x, xmin, xmax), ymax], dtype=FLOAT_DTYPE)


def _frame_path_between(start: FloatArray, end: FloatArray, *, xmin: float, xmax: float, ymin: float, ymax: float, atol: float) -> FloatArray:
    width = xmax - xmin
    height = ymax - ymin
    perimeter = 2.0 * (width + height)
    if perimeter <= 0:
        return np.vstack([start, end]).astype(FLOAT_DTYPE)
    s_start = _frame_arc_length(start, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, atol=atol)
    s_end = _frame_arc_length(end, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, atol=atol)
    delta_ccw = (s_end - s_start) % perimeter
    delta_cw = (s_start - s_end) % perimeter
    direction = 1.0 if delta_ccw <= delta_cw else -1.0
    delta = delta_ccw if direction > 0 else delta_cw

    breakpoints = np.asarray([0.0, width, width + height, 2.0 * width + height, perimeter], dtype=FLOAT_DTYPE)
    s_points = [s_start]
    if direction > 0:
        for bp in breakpoints:
            if s_start < bp < s_start + delta:
                s_points.append(bp)
        s_points.append(s_start + delta)
    else:
        for bp in breakpoints:
            if s_start - delta < bp < s_start:
                s_points.append(bp)
        s_points.append(s_start - delta)
        s_points = list(reversed(s_points))

    path = [_frame_point_from_s(s, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) for s in s_points]
    if not _points_close(path[0], start, atol=atol):
        path[0] = np.asarray(start, dtype=FLOAT_DTYPE)
    if not _points_close(path[-1], end, atol=atol):
        path[-1] = np.asarray(end, dtype=FLOAT_DTYPE)
    return np.asarray(path, dtype=FLOAT_DTYPE)


def _dflt_bandwidth_px(data: np.ndarray, step_sizes: np.ndarray) -> np.ndarray:
    """Compute default Gaussian sigma in pixel units using the dfltBWrange/featureSignif rule.

    For each axis, estimates a robust scale (min of std-dev and IQR-based scale),
    computes lower and upper rule-of-thumb bandwidths in data units, then takes a
    weighted geometric mean (same as ``featureSignif``'s ``h.init``) and converts
    to pixel units by dividing by the histogram step size.

    Parameters
    ----------
    data
        Array of shape ``(n_events, d)``.
    step_sizes
        Per-axis histogram bin widths in data units, shape ``(d,)``.
    """
    data = np.asarray(data, dtype=np.float64)
    step_sizes = np.asarray(step_sizes, dtype=np.float64)
    n, d = data.shape
    # Upper-bound factor (r=2): (4 / ((d + 2r + 2) * n))^(1 / (d + 2r + 4))
    cmb_fac_upp = (4.0 / ((d + 6) * n)) ** (1.0 / (d + 8))
    # Lower-bound factor (r=0): (4 / ((d + 2) * n))^(1 / (d + 4))
    cmb_fac_low = (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))
    # Robust scale: min(std, IQR / 1.349) per axis
    IQR_NORM = 1.3489795  # norm.ppf(0.75) - norm.ppf(0.25)
    std_devs = np.std(data, axis=0, ddof=1)
    iqr_vals = (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)) / IQR_NORM
    sig_hats = np.minimum(std_devs, iqr_vals)
    h_upp = cmb_fac_upp * sig_hats
    h_low = 0.1 * cmb_fac_low * sig_hats
    # Geometric mean at 1/4 lower + 3/4 upper (featureSignif default)
    h_init = h_low ** 0.25 * h_upp ** 0.75
    return h_init / step_sizes


# TODO: consider organizaing all masking functions under LandscapeMasker class.
def _resolve_scalar_threshold(values: FloatArray, threshold_spec: MaskSpecLike) -> float:
    arr = np.asarray(values, dtype=FLOAT_DTYPE).reshape(-1)
    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        return 0.0

    spec = _normalize_mask_spec(threshold_spec)
    spec_type = str(spec["type"])

    def _required_float(params: Mapping[str, Any], *keys: str, label: str) -> float:
        for key in keys:
            if key in params and params[key] is not None:
                return float(params[key])
        joined = ", ".join(keys)
        raise ValueError(f"{label} requires one of params.{joined}.")

    if spec_type == "none":
        return float("-inf")
    if spec_type == "absolute":
        params = dict(spec.get("params", {}))
        return _required_float(params, "value", "threshold", label="Absolute threshold specs")
    if spec_type == "relative":
        params = dict(spec.get("params", {}))
        relative = _required_float(params, "value", "relative", label="Relative threshold specs")
        if relative < 0.0 or relative > 1.0:
            raise ValueError("Relative threshold must be between 0 and 1.")
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        return min_val + relative * (max_val - min_val)
    if spec_type != "threshold":
        raise ValueError("Scalar threshold specs must use type='threshold', 'absolute', or 'relative'.")

    params = dict(spec.get("params", {}))
    method = str(params.get("method", "quantile"))

    if method == "quantile":
        quantile = float(params.get("quantile", params.get("q", 0.95)))
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("Quantile threshold must be between 0 and 1.")
        return float(np.quantile(finite_values, quantile))
    if method in _GLOBAL_MASK_THRESHOLD_METHODS:
        return float(_GLOBAL_MASK_THRESHOLD_METHODS[method](finite_values))
    if method in _LOCAL_MASK_THRESHOLD_METHODS:
        raise ValueError(f"Local threshold method {method!r} is not valid for scalar threshold resolution.")
    raise ValueError(f"Unsupported scalar threshold method {method!r}.")


def _normalize_mask_spec(mask_spec: MaskSpecLike) -> MaskSpec:
    if mask_spec is None:
        return {"type": "none", "params": {}}
    if isinstance(mask_spec, np.ndarray):
        return {
            "type": "array",
            "params": {"mask": np.asarray(mask_spec, dtype=np.bool_)},
        }
    if isinstance(mask_spec, str):
        normalized = mask_spec.strip().lower()
        if normalized in {"", "none"}:
            return {"type": "none", "params": {}}
        supported = _LOCAL_MASK_THRESHOLD_METHODS | set(_GLOBAL_MASK_THRESHOLD_METHODS) | {"quantile"}
        if normalized not in supported:
            raise ValueError(
                f"Unknown mask method {mask_spec!r}. Supported methods are "
                f"{sorted(_GLOBAL_MASK_THRESHOLD_METHODS)} plus local/local_gaussian/local_mean/local_median/quantile."
            )
        return {
            "type": "threshold",
            "params": {"method": normalized},
        }
    if not isinstance(mask_spec, Mapping):
        raise TypeError("mask_spec must be a string, mapping, ndarray, or None.")

    mask_type = str(mask_spec.get("type", "threshold")).strip().lower()
    params = dict(mask_spec.get("params", {}))
    if mask_type == "none":
        return {"type": "none", "params": {}}
    if mask_type == "array":
        mask = np.asarray(params.get("mask"), dtype=np.bool_)
        return {"type": "array", "params": {"mask": mask}}
    if mask_type in {"absolute", "relative"}:
        if "value" not in params and "threshold" not in params and "relative" not in params:
            raise ValueError(
                f"mask_spec.type={mask_type!r} requires params.value (or params.threshold/params.relative)."
            )
        if "threshold" in params and "value" not in params:
            params["value"] = params["threshold"]
        if "relative" in params and "value" not in params:
            params["value"] = params["relative"]
        return {"type": mask_type, "params": params}

    if mask_type != "threshold":
        raise ValueError("mask_spec.type must be one of 'none', 'array', 'threshold', 'absolute', or 'relative'.")

    method = str(params.get("method", "otsu")).strip().lower()
    supported = _LOCAL_MASK_THRESHOLD_METHODS | set(_GLOBAL_MASK_THRESHOLD_METHODS) | {"quantile"}
    if method not in supported:
        raise ValueError(
            f"Unsupported threshold method {method!r}. Supported methods are "
            f"{sorted(_GLOBAL_MASK_THRESHOLD_METHODS)} plus local/local_gaussian/local_mean/local_median/quantile."
        )
    params["method"] = method
    return {"type": "threshold", "params": params}


def _serialize_mask_spec(mask_spec: MaskSpecLike) -> dict[str, Any] | None:
    normalized = _normalize_mask_spec(mask_spec)
    if normalized["type"] == "none":
        return None
    if normalized["type"] == "array":
        raise ValueError("Array mask specs are runtime-only and cannot be serialized in gate hyperparameters.")
    return {
        "type": str(normalized["type"]),
        "params": dict(normalized.get("params", {})),
    }


def _resolve_density_mask(density: FloatArray, mask_spec: MaskSpecLike = None) -> BooleanArray:
    arr = np.asarray(density, dtype=FLOAT_DTYPE)
    finite_mask = np.asarray(np.isfinite(arr), dtype=np.bool_)
    spec = _normalize_mask_spec(mask_spec)

    if spec["type"] == "none":
        return finite_mask
    if spec["type"] == "array":
        mask = np.asarray(spec["params"]["mask"], dtype=np.bool_)
        if mask.shape != arr.shape:
            raise ValueError(f"Array mask shape {mask.shape!r} does not match density shape {arr.shape!r}.")
        return mask & finite_mask

    params = dict(spec.get("params", {}))
    spec_type = str(spec["type"])
    method = str(params.get("method", "otsu"))
    finite_values = arr[finite_mask]
    if finite_values.size == 0:
        return finite_mask

    def _required_float(*keys: str, label: str) -> float:
        for key in keys:
            if key in params and params[key] is not None:
                return float(params[key])
        joined = ", ".join(keys)
        raise ValueError(f"{label} requires one of params.{joined}.")

    threshold_value: float | np.ndarray
    if spec_type == "absolute":
        threshold_value = _required_float("value", "threshold", label="Absolute mask threshold")
    elif spec_type == "relative":
        relative = _required_float("value", "relative", label="Relative mask threshold")
        if relative < 0.0 or relative > 1.0:
            raise ValueError("Relative mask threshold must be between 0 and 1.")
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
        threshold_value = min_val + relative * (max_val - min_val)
    elif method == "quantile":
        quantile = float(params.get("quantile", params.get("q", 0.95)))
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("Quantile mask threshold must be between 0 and 1.")
        threshold_value = float(np.quantile(finite_values, quantile))
    elif method in _GLOBAL_MASK_THRESHOLD_METHODS:
        threshold_value = float(_GLOBAL_MASK_THRESHOLD_METHODS[method](finite_values))
    elif method in _LOCAL_MASK_THRESHOLD_METHODS:
        if arr.ndim != 2:
            raise ValueError(f"Local threshold method {method!r} only supports 2D landscapes.")
        block_size = int(params.get("block_size", 35))
        if block_size < 3 or block_size % 2 == 0:
            raise ValueError("Local threshold block_size must be an odd integer >= 3.")
        local_kwargs: dict[str, Any] = {
            "block_size": block_size,
            "mode": str(params.get("mode", "reflect")),
        }
        local_method = "gaussian"
        if method == "local_mean":
            local_method = "mean"
        elif method == "local_median":
            local_method = "median"
        local_kwargs["method"] = local_method
        threshold_value = np.asarray(
            threshold_local( # pyright: ignore[reportArgumentType]
                arr,
                **local_kwargs,
            ),
            dtype=FLOAT_DTYPE,
        )
    else:
        raise ValueError(f"Unsupported threshold method {method!r}.")

    resolved_mask = np.asarray(arr >= threshold_value, dtype=np.bool_) & finite_mask
    return resolved_mask if np.any(resolved_mask) else finite_mask


def _is_close_to_any(value: float, candidates: FloatArray, *, atol: float = 1e-6) -> bool:
    arr = np.asarray(candidates, dtype=FLOAT_DTYPE).reshape(-1)
    return bool(np.any(np.isclose(arr, value, atol=atol, rtol=0.0)))


def _is_grid_corner(point: FloatArray, *, x_edges: FloatArray, y_edges: FloatArray, atol: float = 1e-6) -> bool:
    pt = np.asarray(point, dtype=FLOAT_DTYPE)
    return _is_close_to_any(float(pt[0]), x_edges, atol=atol) and _is_close_to_any(float(pt[1]), y_edges, atol=atol)


def _serialize_kernel(kernel_spec: KernelSpec) -> dict[str, Any]:
    """Serialize a kernel spec, converting any custom kernel array to float32."""
    spec = dict(kernel_spec)
    if spec.get("type") == "custom":
        params = dict(spec.get("params", {}))
        arr = params.get("kernel")
        if arr is not None:
            params["kernel"] = np.asarray(arr, dtype=FLOAT_DTYPE)
        spec["params"] = params
    return spec


def _resolve_kernel(kernel: KernelLike, *, n_dims: int) -> KernelSpec:
    """Resolve a kernel shorthand to a normalized spec dict."""
    if kernel is None:
        return {"type": "identity", "params": {}}
    if isinstance(kernel, str):
        if kernel in {"gaussian", "gauss"}:
            return {"type": "gaussian", "params": {"mode": "reflect"}}
        if kernel in {"none", "identity"}:
            return {"type": "identity", "params": {}}
        raise ValueError(f"Unknown kernel string {kernel!r}. Use 'gaussian', 'identity', or None.")
    if isinstance(kernel, Mapping):
        ktype = str(kernel.get("type", "identity"))
        if ktype not in {"identity", "gaussian", "custom"}:
            raise ValueError(f"Unknown kernel type {ktype!r}. Supported: 'identity', 'gaussian', 'custom'.")
        return {"type": ktype, "params": dict(kernel.get("params", {}))}
    arr = np.asarray(kernel, dtype=FLOAT_DTYPE)
    if arr.ndim not in {1, 2}:
        raise ValueError("Custom kernel array must be 1D or 2D.")
    if arr.ndim != n_dims:
        raise ValueError(f"Custom kernel array dimensionality ({arr.ndim}) must match landscape dimensionality ({n_dims}).")
    return {"type": "custom", "params": {"kernel": arr}}


def _restore_kernel_from_state(payload: Mapping[str, Any]) -> KernelSpec:
    """Restore a kernel spec from a landscape state payload."""
    raw = payload.get("kernel", {"type": "identity", "params": {}})
    if not isinstance(raw, Mapping):
        return {"type": "identity", "params": {}}
    spec = dict(raw)
    if spec.get("type") == "custom":
        params = dict(spec.get("params", {}))
        arr = params.get("kernel")
        if arr is not None:
            params["kernel"] = np.asarray(arr, dtype=FLOAT_DTYPE)
        spec["params"] = params
    return spec


def _apply_kernel(
    values: np.ndarray,
    *,
    kernel_spec: KernelSpec,
) -> FloatArray:
    """Apply a kernel to a grid of values.

    Parameters
    ----------
    values
        The grid array to transform (counts or density).
    kernel_spec
        Normalized kernel spec dict with ``type`` in ``{'identity', 'gaussian', 'custom'}``.
        ``params`` is passed directly to the underlying function (``gaussian_filter`` or
        ``convolve``), with the exception that ``custom`` extracts and normalizes
        ``params["kernel"]`` before forwarding the remaining kwargs.
    """
    arr = np.asarray(values, dtype=FLOAT_DTYPE)
    ktype = kernel_spec.get("type", "identity")
    if ktype == "identity":
        return arr
    if ktype == "gaussian":
        params = dict(kernel_spec.get("params", {}))
        # Builder-only control for auto sigma estimation; never forwarded to scipy.
        params.pop("sigma_adjust", None)
        sigma_raw = params.pop("sigma")
        sigma_arg: float | list[float] = float(sigma_raw[0]) if len(sigma_raw) == 1 else [float(v) for v in sigma_raw]
        params.setdefault("mode", "reflect")
        return np.asarray(gaussian_filter(arr, sigma=sigma_arg, **params), dtype=FLOAT_DTYPE)
    if ktype == "custom":
        params = dict(kernel_spec.get("params", {}))
        k = np.asarray(params.pop("kernel"), dtype=FLOAT_DTYPE)
        if k.ndim != arr.ndim:
            raise ValueError(
                f"Custom kernel dimensionality ({k.ndim}) must match the input grid ({arr.ndim})."
            )
        k_sum = float(np.sum(k))
        if np.isclose(k_sum, 0.0):
            raise ValueError("Custom kernel weights sum to zero.")
        params.setdefault("mode", "nearest")
        return np.asarray(convolve(arr, k / k_sum, **params), dtype=FLOAT_DTYPE)
    raise ValueError(f"Unknown kernel type {ktype!r}.")


def _compute_landscape_derivatives(landscape: "DensityLandscape") -> "dict[str, Any]":
    """Compute gradient and Hessian of the density estimate from a Gaussian-smoothed landscape.

    Differentiation is performed on the normalised histogram
    ``H_norm = counts / (n * prod(step_sizes))`` rather than on
    ``landscape.density``.  ``landscape.density`` equals ``G_σ * H_norm``
    (already smoothed); re-applying the derivative Gaussian filter to ``H_norm``
    with the original σ yields estimates with the correct effective bandwidth.
    Differentiating ``landscape.density`` would effectively use σ·√2.

    Only ``RegularGridLandscape1D`` and ``RegularGridLandscape2D`` with a
    Gaussian kernel are supported.  Other landscape types raise ``TypeError``
    and non-Gaussian kernels raise ``ValueError``.

    Parameters
    ----------
    landscape
        A fitted regular-grid density landscape with a Gaussian kernel.

    Returns
    -------
    dict with keys:

    * ``"gradient"``: tuple of FLOAT_DTYPE arrays (one per axis), physical units
      (density per data unit).
    * ``"hessian"``: tuple of FLOAT_DTYPE arrays.
      1D: ``(f2,)``; 2D: ``(Hxx, Hxy, Hyy)``.
    * ``"eigenvalues"``: tuple of FLOAT_DTYPE arrays.
      1D: ``(f2,)`` (the only curvature is the second derivative itself);
      2D: ``(l_min, l_max)`` — smaller and larger eigenvalue of the Hessian.
    """
    if not isinstance(landscape, (RegularGridLandscape1D, RegularGridLandscape2D)):
        raise TypeError(
            "_compute_landscape_derivatives only supports RegularGridLandscape1D and "
            f"RegularGridLandscape2D, got {type(landscape).__name__!r}."
        )
    kernel = landscape.kernel
    if kernel.get("type") != "gaussian":
        raise ValueError(
            "_compute_landscape_derivatives only supports Gaussian kernels, "
            f"got {kernel.get('type')!r}."
        )
    params = dict(kernel.get("params", {}))
    sigma_px = params["sigma"]  # list: 1D → [σ_px]; 2D → [σ_y_px, σ_x_px]
    mode: str = params.get("mode", "reflect")  # type: ignore[assignment]

    counts = landscape.counts
    n = float(counts.sum())
    if n <= 0.0:
        raise ValueError("Landscape has no events (counts sum to zero).")

    if isinstance(landscape, RegularGridLandscape1D):
        dx = float(landscape.step_size)
        sigma_1d = float(sigma_px[0])
        H_norm = counts / (n * dx)
        gx = gaussian_filter(H_norm, sigma=sigma_1d, mode=mode, order=1) / dx
        f2 = gaussian_filter(H_norm, sigma=sigma_1d, mode=mode, order=2) / (dx * dx)
        return {
            "gradient": (gx,),
            "hessian": (f2,),
            "eigenvalues": (f2,),
        }

    # RegularGridLandscape2D:
    #   counts.shape = (n_y, n_x)  — axis 0 = y, axis 1 = x
    #   sigma_px     = [σ_y_px, σ_x_px]
    #   step_size    = (step_x, step_y)
    dx, dy = landscape.step_size
    dx, dy = float(dx), float(dy)
    sigma_arr = [float(sigma_px[0]), float(sigma_px[1])]  # [σ_y_px, σ_x_px]
    H_norm = counts / (n * dx * dy)
    kw: dict[str, Any] = {"sigma": sigma_arr, "mode": mode}
    # Gradient (axis 1 = x, axis 0 = y)
    grad_x = gaussian_filter(H_norm, order=(0, 1), **kw) / dx  # type: ignore[call-arg]
    grad_y = gaussian_filter(H_norm, order=(1, 0), **kw) / dy  # type: ignore[call-arg]
    # Hessian
    Hxx = gaussian_filter(H_norm, order=(0, 2), **kw) / (dx * dx)  # type: ignore[call-arg]
    Hyy = gaussian_filter(H_norm, order=(2, 0), **kw) / (dy * dy)  # type: ignore[call-arg]
    Hxy = gaussian_filter(H_norm, order=(1, 1), **kw) / (dx * dy)  # type: ignore[call-arg]
    # Eigenvalues (analytical via trace/determinant)
    tr = Hxx + Hyy
    disc = np.maximum(tr * tr - 4.0 * (Hxx * Hyy - Hxy * Hxy), 0.0)
    sqrt_disc = np.sqrt(disc)
    l_min = (tr - sqrt_disc) * 0.5
    l_max = (tr + sqrt_disc) * 0.5
    return {
        "gradient": (grad_x, grad_y),
        "hessian": (Hxx, Hxy, Hyy),
        "eigenvalues": (l_min, l_max),
    }


@_register_component(_LANDSCAPE_BUILDERS, "histogram1d")
class HistogramLandscape1DBuilder(LandscapeBuilder):
    component_type = "histogram1d"
    n_dims = 1

    def __init__(
        self,
        grid_size: int = 256,
        pad_factor: float = 0.05,
        kernel: KernelLike = "gaussian",
    ) -> None:
        """Configure a 1D histogram-based density landscape builder.

        Parameters
        ----------
        grid_size
            Number of histogram bins used to discretize the axis.
        pad_factor
            Extra range padding added on both sides before histogramming, expressed as
            a fraction of an IQR-based standard-deviation estimate
            ``(Q3 - Q1) / 1.3489795``.
        kernel
            Kernel applied to the histogram grid to convert counts to density.
            Supports ``"gaussian"``, ``None``/``"identity"``, a custom kernel array,
            or a normalized spec dict ``{"type": ..., "params": {...}}``.
            For ``"gaussian"``, include ``"sigma"`` in ``params`` to fix the bandwidth
            in pixel units; omit it to auto-compute a data-driven default.
            Optionally include ``"sigma_adjust"`` (default ``1.0``) to scale only the
            auto-computed sigma.
        """
        self.grid_size = int(grid_size)
        self.pad_factor = float(pad_factor)
        self.kernel = _resolve_kernel(kernel, n_dims=1)
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8 for HistogramLandscape1DBuilder.")

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "grid_size": self.grid_size,
                "pad_factor": self.pad_factor,
                "kernel": _serialize_kernel(self.kernel),
            },
        }

    def build(self, data: FloatArray) -> Landscape1D:
        values = np.asarray(data, dtype=FLOAT_DTYPE).reshape(-1)
        if values.size == 0:
            raise ValueError("Cannot build a density landscape from empty data.")
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        iqr_std = float((np.percentile(values, 75.0) - np.percentile(values, 25.0)) / 1.3489795)
        pad_scale = max(iqr_std, 1e-6)
        pad = pad_scale * self.pad_factor
        edges = np.linspace(vmin - pad, vmax + pad, self.grid_size + 1, dtype=FLOAT_DTYPE)
        counts, _ = np.histogram(values, bins=edges)
        counts_f = counts.astype(FLOAT_DTYPE)
        step_size = float(edges[1] - edges[0])
        kernel_spec: KernelSpec = dict(self.kernel)
        if kernel_spec["type"] == "gaussian":
            params = dict(kernel_spec.get("params", {}))
            sigma_adjust = float(params.get("sigma_adjust", 1.0))
            if sigma_adjust <= 0.0:
                raise ValueError("Gaussian kernel params.sigma_adjust must be > 0.")
            if "sigma" not in params:
                sigma_px = _dflt_bandwidth_px(values.reshape(-1, 1), np.array([step_size])) * sigma_adjust
                params["sigma"] = [float(sigma_px[0])]
            params.setdefault("mode", "reflect")
            kernel_spec = {"type": "gaussian", "params": params}
        smoothed = _apply_kernel(counts_f, kernel_spec=kernel_spec)
        total_events = float(np.sum(counts_f))
        density = smoothed / (total_events * step_size)
        return RegularGridLandscape1D(
            start=float(edges[0]),
            step_size=float(edges[1] - edges[0]),
            step_count=int(self.grid_size),
            density=density,
            counts=counts.astype(np.int_),
            kernel=kernel_spec,
        )


@_register_component(_LANDSCAPE_BUILDERS, "histogram2d")
class HistogramLandscape2DBuilder(LandscapeBuilder):
    component_type = "histogram2d"
    n_dims = 2

    def __init__(
        self,
        grid_size: int | Sequence[int] = 256,
        pad_factor: float = 0.05,
        kernel: KernelLike = "gaussian",
    ) -> None:
        """Configure a 2D histogram-based density landscape builder.

        Parameters
        ----------
        grid_size
            Number of histogram bins per axis. A scalar uses the same resolution on
            both axes; a length-2 sequence sets ``x`` and ``y`` resolutions separately.
        pad_factor
            Extra range padding added around the observed data before histogramming,
            expressed as a fraction of an IQR-based standard-deviation estimate on
            each axis: ``(Q3 - Q1) / 1.3489795``.
        kernel
            Kernel applied to the 2D histogram grid to convert counts to density.
            Supports ``"gaussian"``, ``None``/``"identity"``, a custom 2D kernel
            array, or a normalized spec dict.
            For ``"gaussian"``, include ``"sigma"`` in ``params`` as a length-2 list
            ``[sigma_y, sigma_x]`` (array-axis order) to fix the bandwidth in pixel
            units; omit it to auto-compute a data-driven default per axis.
            Optionally include ``"sigma_adjust"`` (default ``1.0``) to scale only the
            auto-computed sigma.
        """
        if isinstance(grid_size, Sequence) and not isinstance(grid_size, (str, bytes)):
            grid_arr = np.asarray(grid_size, dtype=np.int_)
            if grid_arr.shape != (2,):
                raise ValueError("grid_size must be an int or length-2 sequence.")
            self.grid_size = (int(grid_arr[0]), int(grid_arr[1]))
        else:
            self.grid_size = (int(grid_size), int(grid_size))
        if min(self.grid_size) < 8:
            raise ValueError("grid_size must be at least 8 in both dimensions for HistogramLandscape2DBuilder.")
        self.pad_factor = float(pad_factor)
        self.kernel = _resolve_kernel(kernel, n_dims=2)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "grid_size": list(self.grid_size),
                "pad_factor": self.pad_factor,
                "kernel": _serialize_kernel(self.kernel),
            },
        }

    def build(self, data: FloatArray) -> Landscape2D:
        coords = np.asarray(data, dtype=FLOAT_DTYPE)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("HistogramLandscape2DBuilder expects shape (n_events, 2).")
        if coords.shape[0] == 0:
            raise ValueError("Cannot build a density landscape from empty data.")

        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        iqr_std = (
            np.percentile(coords, 75.0, axis=0) - np.percentile(coords, 25.0, axis=0)
        ) / 1.3489795
        pad_scale = np.maximum(iqr_std, 1e-6)
        pad = pad_scale * self.pad_factor
        x_edges = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], self.grid_size[0] + 1, dtype=FLOAT_DTYPE)
        y_edges = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], self.grid_size[1] + 1, dtype=FLOAT_DTYPE)

        counts_xy, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])
        counts = counts_xy.T.astype(np.int_)
        step_x = float(x_edges[1] - x_edges[0])
        step_y = float(y_edges[1] - y_edges[0])
        kernel_spec: KernelSpec = dict(self.kernel)
        if kernel_spec["type"] == "gaussian":
            params = dict(kernel_spec.get("params", {}))
            sigma_adjust = float(params.get("sigma_adjust", 1.0))
            if sigma_adjust <= 0.0:
                raise ValueError("Gaussian kernel params.sigma_adjust must be > 0.")
            if "sigma" not in params:
                # Auto-compute sigma in pixel units; stored in array-axis (y, x) order.
                sigma_px = _dflt_bandwidth_px(coords, np.array([step_x, step_y])) * sigma_adjust
                params["sigma"] = [float(sigma_px[1]), float(sigma_px[0])]  # [sigma_y, sigma_x]
            params.setdefault("mode", "reflect")
            kernel_spec = {"type": "gaussian", "params": params}
        smoothed = _apply_kernel(counts, kernel_spec=kernel_spec)
        total_events = float(np.sum(counts))
        density = smoothed / (total_events * step_x * step_y)
        return RegularGridLandscape2D(
            start=(float(x_edges[0]), float(y_edges[0])),
            step_size=(float(x_edges[1] - x_edges[0]), float(y_edges[1] - y_edges[0])),
            step_count=(int(self.grid_size[0]), int(self.grid_size[1])),
            density=density,
            counts=counts,
            kernel=kernel_spec,
        )


@_register_component(_FEATURE_DETECTORS, "mode_detector")
class ModeDetector(FeatureDetector):
    component_type = "mode_detector"

    def __init__(
        self,
        min_distance: int = 5,
        peak_threshold_spec: MaskSpecLike = None,
        exclude_border: bool | int = False,
        mask_spec: MaskSpecLike = None,
        min_marker_component_size: int = 10,
        structure: StructureLike = "centrosymmetric",
    ) -> None:
        """Configure mode detection on a fitted density landscape.

        Parameters
        ----------
        min_distance
            Minimum separation between detected modes in grid cells.
        peak_threshold_spec
            Threshold strategy used to suppress weak local maxima before peak
            selection. Supports any global threshold method from MaskSpec
            (for example ``"otsu"``, ``"yen"``, ``"li"``, ``"isodata"``,
            ``"triangle"``, ``"mean"``), ``"quantile"``, ``"absolute"``,
            ``"relative"``, or ``None`` for no threshold.
        exclude_border
            Forwarded to ``skimage.feature.peak_local_max`` for 2D mode detection.
        mask_spec
            Mask configuration used to limit where peaks can be detected.
            Accepts KernelSpec-like dictionaries, threshold method strings, or ``None``.
        min_marker_component_size
            Minimum connected-component size allowed in the peak mask before a region
            can contribute peak markers.
        structure
            Connectivity structure used for peak-mask connected components. Supports
            ``"centrosymmetric"``, ``"cross"``, or a custom 2D boolean array.
        """
        super().__init__(mask_spec=mask_spec)
        self.min_distance = int(min_distance)
        self.peak_threshold_spec = _normalize_mask_spec(peak_threshold_spec)
        self.exclude_border = exclude_border
        self.min_marker_component_size = int(min_marker_component_size)
        self.structure = self._restore_structure(structure)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "min_distance": self.min_distance,
                "peak_threshold_spec": _serialize_mask_spec(self.peak_threshold_spec),
                "exclude_border": self.exclude_border,
                "mask_spec": self._serialized_mask_spec(),
                "min_marker_component_size": self.min_marker_component_size,
                "structure": self._serialize_structure(),
            },
        }

    def _serialize_structure(self) -> str | list[list[bool]]:
        structure = self.structure
        if isinstance(structure, str):
            return structure
        arr = np.asarray(structure, dtype=np.bool_)
        if arr.ndim != 2:
            raise ValueError("Marker structure must be a 2D array.")
        return arr.astype(bool).tolist()

    @staticmethod
    def _restore_structure(structure: StructureLike) -> BooleanArray:
        if isinstance(structure, str):
            if structure == "centrosymmetric":
                return np.ones((3, 3), dtype=np.bool_)
            if structure == "cross":
                return np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_)
            raise ValueError("Unsupported structure string. Use 'centrosymmetric' or 'cross'.")
        arr = np.asarray(structure, dtype=np.bool_)
        if arr.ndim != 2:
            raise ValueError("Marker structure must be a 2D array.")
        return arr

    def detect(self, landscape: DensityLandscape) -> list[ModeFeature]:
        if isinstance(landscape, Landscape1D):
            return self._detect_1d(landscape, mask_spec=self.mask_spec)
        if isinstance(landscape, Landscape2D):
            return self._detect_2d(landscape, mask_spec=self.mask_spec)
        raise TypeError(f"Unsupported landscape type {type(landscape)!r}.")

    @staticmethod
    def _to_detection_density(density: FloatArray) -> FloatArray:
        arr = np.asarray(density, dtype=FLOAT_DTYPE)
        return np.log(np.maximum(arr, 0.0) + 1e-9)

    def _detect_1d(self, landscape: Landscape1D, *, mask_spec: MaskSpecLike) -> list[ModeFeature]:
        density_raw = np.asarray(landscape.density, dtype=FLOAT_DTYPE).reshape(-1)
        density = self._to_detection_density(density_raw)
        if density_raw.size == 0:
            return []
        peak_mask = _resolve_density_mask(density, mask_spec)
        threshold = _resolve_scalar_threshold(density[peak_mask], self.peak_threshold_spec)
        candidate_idx, peak_properties = find_peaks(
            density,
            height=threshold,
            distance=max(self.min_distance, 1),
            prominence=0.0,
            width=1.0,
        )
        candidate_idx = candidate_idx[np.asarray(peak_mask[candidate_idx], dtype=np.bool_)]
        if candidate_idx.size == 0:
            # Fallback to global maximum if no peaks are detected
            search_density = np.where(peak_mask, density, -np.inf)
            if not np.isfinite(search_density).any():
                search_density = density
            imax = int(np.argmax(search_density))
            start, end = 0, density.size - 1
            candidate_idx = np.asarray([imax], dtype=np.int_)
            peak_properties = {
                "peak_heights": np.asarray([density_raw[imax]], dtype=FLOAT_DTYPE),
                "left_bases": np.asarray([start], dtype=np.int_),
                "right_bases": np.asarray([end], dtype=np.int_),
                "left_ips": np.asarray([start], dtype=FLOAT_DTYPE),
                "right_ips": np.asarray([end], dtype=FLOAT_DTYPE),
            }

        mode_features: list[dict[str, Any]] = []
        for k, idx in enumerate(candidate_idx):
            props = {
                "point": [float(landscape.axis[idx])],
                "grid_index": [int(idx)],
                "peak_density": float(density_raw[idx]),
                "properties": {
                    "left_base": int(peak_properties["left_bases"][k]),
                    "right_base": int(peak_properties["right_bases"][k]),
                    "left_ip": float(peak_properties["left_ips"][k]),
                    "right_ip": float(peak_properties["right_ips"][k]),
                },
            }
            mode_features.append(props)
        mode_features.sort(key=lambda item: float(item["peak_density"]), reverse=True)

        return [
            ModeFeature(
                id=f"mode_{rank}",
                rank=rank,
                point=mode["point"],
                grid_index=mode["grid_index"],
                peak_density=mode["peak_density"],
                properties=mode["properties"],
            )
            for rank, mode in enumerate(mode_features, start=1)
        ]

    def _marker_peak_mask_2d(self, density: FloatArray, *, mask_spec: MaskSpecLike) -> BooleanArray:
        peak_mask = _resolve_density_mask(density, mask_spec)
        finite_mask = np.asarray(np.isfinite(density), dtype=np.bool_)
        if self.min_marker_component_size > 1:
            component_labels, _ = nd_label(peak_mask, structure=self.structure) # pyright: ignore[reportGeneralTypeIssues]
            sizes = np.bincount(component_labels.ravel())
            valid = np.zeros_like(sizes, dtype=np.bool_)
            valid[sizes >= self.min_marker_component_size] = True
            valid[0] = False
            peak_mask = valid[component_labels] # pyright: ignore[reportArgumentType, reportCallIssue]
        if not np.any(peak_mask):
            return finite_mask
        return peak_mask

    def _detect_2d(
        self,
        landscape: Landscape2D,
        *,
        mask_spec: MaskSpecLike,
    ) -> list[ModeFeature]:
        density_raw = np.asarray(landscape.density, dtype=FLOAT_DTYPE)
        density = self._to_detection_density(density_raw)
        peak_mask = self._marker_peak_mask_2d(density, mask_spec=mask_spec)
        peak_mask_components = None
        if self.min_marker_component_size > 1:
            peak_mask_components, _ = nd_label(peak_mask, structure=self.structure) # pyright: ignore[reportGeneralTypeIssues]
        coords = peak_local_max(
            density,
            min_distance=max(self.min_distance, 1),
            threshold_abs=_resolve_scalar_threshold(density[peak_mask], self.peak_threshold_spec),
            labels=peak_mask.astype(np.uint8),
            exclude_border=self.exclude_border, # pyright: ignore[reportArgumentType]
        )
        if coords.size == 0:
            # Fallback to global maximum if no peaks are detected
            coords = np.asarray([np.unravel_index(int(np.argmax(density)), density.shape)], dtype=np.int_)
        peak_order = sorted(coords.tolist(), key=lambda rc: float(density[rc[0], rc[1]]), reverse=True)
        return [
            ModeFeature(
                id=f"mode_{rank}",
                rank=rank,
                point=[float(landscape.x_centers[col]), float(landscape.y_centers[row])],
                grid_index=[int(row), int(col)],
                peak_density=float(density_raw[row, col]),
                properties={
                    "peak_mask_component": (
                        int(peak_mask_components[row, col]) if peak_mask_components is not None else 0
                    )
                },
            )
            for rank, (row, col) in enumerate(peak_order, start=1)
        ]


@_register_component(_FEATURE_DETECTORS, "curvature_detector")
class CurvatureDetector(FeatureDetector):
    """Detect density modes by locating statistically significant negative-curvature regions.

    The detector implements the curvature significance test from
    Duong & Hazelton (2003) as coded in R's ``feature`` package
    (``SignifFeatureRegion.R``).  For each grid cell the Wald chi-squared
    statistic for the second-order derivative of the density estimate is
    computed, corrected for multiple testing via Hochberg's step-up procedure,
    and cells with significantly negative curvature (local maxima of the
    density) are grouped into connected components.  One ``ModeFeature`` is
    returned per component, located at the density maximum within it.

    Differentiation uses :func:`_compute_landscape_derivatives`, which requires
    a ``RegularGridLandscape1D`` or ``RegularGridLandscape2D`` with a Gaussian
    kernel.  Other landscape types raise ``TypeError`` at detection time.

    Parameters
    ----------
    alpha
        Family-wise error rate for Hochberg's step-up correction.
    ess_min
        Minimum effective sample size (``n · f̂ · prod(h) · (√2π)^d``) for a
        grid cell to be included in the significance test.  Cells below this
        threshold are excluded regardless of their curvature.
    neg_curv_only
        If ``True`` (default), restrict significant cells to those where all
        Hessian eigenvalues are negative (confirmed local mode).  If ``False``,
        any significant curvature region is retained.
    min_component_size
        Minimum number of grid cells a connected significant-curvature component
        must contain to produce a ``ModeFeature``.  Smaller components are
        discarded as noise.
    mask_spec
        Optional density mask applied before significance testing, restricting
        which cells are considered active.  Accepts any ``MaskSpecLike`` value
        (threshold method string, mapping, boolean array, or ``None``).
    exclude_border
        Reserved for future border-exclusion support; currently unused.
    """

    component_type = "curvature_detector"

    def __init__(
        self,
        alpha: float = 0.05,
        ess_min: float = 5.0,
        neg_curv_only: bool = True,
        min_component_size: int = 10,
        mask_spec: MaskSpecLike = None,
        exclude_border: bool = False,
    ) -> None:
        super().__init__(mask_spec=mask_spec)
        self.alpha = float(alpha)
        self.ess_min = float(ess_min)
        self.neg_curv_only = bool(neg_curv_only)
        self.min_component_size = int(min_component_size)
        self.exclude_border = bool(exclude_border)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "alpha": self.alpha,
                "ess_min": self.ess_min,
                "neg_curv_only": self.neg_curv_only,
                "min_component_size": self.min_component_size,
                "mask_spec": self._serialized_mask_spec(),
                "exclude_border": self.exclude_border,
            },
        }

    def detect(
        self,
        landscape: DensityLandscape,
    ) -> list[ModeFeature]:
        derivs = _compute_landscape_derivatives(landscape)
        if isinstance(landscape, Landscape1D):
            return self._detect_1d(landscape, derivs, mask_spec=self.mask_spec)
        if isinstance(landscape, Landscape2D):
            return self._detect_2d(landscape, derivs, mask_spec=self.mask_spec)
        raise TypeError(f"Unsupported landscape type {type(landscape)!r}.")

    @staticmethod
    def _fdr_reject(pval_flat: "np.ndarray", alpha: float) -> "np.ndarray":
        """Benjamini-Hochberg FDR correction on a flat p-value array.

        Cells with ``p == 0`` are auto-rejected; ``NaN`` cells are treated as
        non-rejected.
        """
        reject = np.zeros(pval_flat.size, dtype=np.bool_)
        finite_idx = np.where(np.isfinite(pval_flat))[0]
        if finite_idx.size == 0:
            return reject
        p = pval_flat[finite_idx]
        # p == 0: auto-reject
        reject[finite_idx[p <= 0.0]] = True
        pos_mask = p > 0.0
        if not np.any(pos_mask):
            return reject
        pos_idx = finite_idx[pos_mask]
        p_pos = p[pos_mask]
        p_adj = false_discovery_control(p_pos, method="bh")
        reject[pos_idx] = p_adj <= alpha
        return reject

    def _detect_1d(
        self,
        landscape: Landscape1D,
        derivs: "dict[str, Any]",
        *,
        mask_spec: MaskSpecLike,
    ) -> list[ModeFeature]:
        if not isinstance(landscape, RegularGridLandscape1D):
            raise TypeError(
                "CurvatureDetector requires RegularGridLandscape1D for 1D detection."
            )
        D = landscape.density
        n = float(landscape.counts.sum())
        dx = float(landscape.step_size)
        h = float(landscape.kernel["params"]["sigma"][0]) * dx  # bandwidth in data units

        f2: np.ndarray = derivs["hessian"][0]

        active = np.isfinite(D) & (D > 0.0)
        # Effective sample size: n · f̂ · h · √(2π)
        active &= n * D * h * np.sqrt(2.0 * np.pi) >= self.ess_min
        active &= _resolve_density_mask(D, mask_spec)

        # Sig2_scalar = D / (8·√π·n·h)   [R d=1 formula]
        sig2 = D / (8.0 * np.sqrt(np.pi) * n * h)
        wald = np.full(D.shape, np.nan, dtype=FLOAT_DTYPE)
        valid = active & (sig2 > 0.0)
        wald[valid] = (f2[valid] ** 2 * h ** 4) / (3.0 * sig2[valid])

        pval = np.full(D.shape, np.nan, dtype=FLOAT_DTYPE)
        finite = np.isfinite(wald)
        pval[finite] = chi2.sf(wald[finite], df=1)

        p_flat = pval.ravel().copy()
        p_flat[~active.ravel()] = np.nan
        signif = self._fdr_reject(p_flat, self.alpha).reshape(D.shape)

        if self.neg_curv_only:
            signif &= f2 < 0.0

        structure_1d = np.ones(3, dtype=np.bool_)
        lab, n_comp = nd_label(signif, structure=structure_1d)  # pyright: ignore[reportGeneralTypeIssues]
        if n_comp == 0:
            return []
        sizes = np.bincount(lab.ravel())
        axis = np.asarray(landscape.axis, dtype=FLOAT_DTYPE)

        features: list[ModeFeature] = []
        for comp_id in range(1, n_comp + 1):
            if sizes[comp_id] < self.min_component_size:
                continue
            comp_mask = np.asarray(lab == comp_id, dtype=np.bool_)
            D_comp = np.where(comp_mask, D, -np.inf)
            peak_idx = int(np.argmax(D_comp))
            comp_axis = axis[comp_mask]
            features.append(
                ModeFeature(
                    id=f"mode_{comp_id}",
                    rank=0,
                    point=[float(axis[peak_idx])],
                    grid_index=[peak_idx],
                    peak_density=float(landscape.density[peak_idx]),
                    properties={"component_size": int(sizes[comp_id])},
                    footprint={
                        "grid_mask": comp_mask,
                        "bounds": [[float(comp_axis.min())], [float(comp_axis.max())]],
                    },
                )
            )

        features.sort(key=lambda f: f.peak_density, reverse=True)
        for rank, feat in enumerate(features, start=1):
            feat.rank = rank
            feat.id = f"mode_{rank}"
        return features

    def _detect_2d(
        self,
        landscape: Landscape2D,
        derivs: "dict[str, Any]",
        *,
        mask_spec: MaskSpecLike,
    ) -> list[ModeFeature]:
        if not isinstance(landscape, RegularGridLandscape2D):
            raise TypeError(
                "CurvatureDetector requires RegularGridLandscape2D for 2D detection."
            )
        D = landscape.density
        n = float(landscape.counts.sum())
        dx, dy = landscape.step_size  # (step_x, step_y)
        sigma_px = landscape.kernel["params"]["sigma"]  # [σ_y_px, σ_x_px]
        h_x = float(sigma_px[1]) * dx  # bandwidth in x data units
        h_y = float(sigma_px[0]) * dy  # bandwidth in y data units

        Hxx, Hxy, Hyy = derivs["hessian"]
        l_min, l_max = derivs["eigenvalues"]

        # Setup Wald test active mask:
        active = np.isfinite(D) & (D > 0.0)
        # Effective sample size: n · f̂ · h_x · h_y · 2π
        active &= (n * h_x * h_y * 2.0 * np.pi * D) >= self.ess_min
        active &= _resolve_density_mask(D, mask_spec)

        # Sig2_scalar = D / (16·π·n·h_x·h_y)   [R d=2 formula]
        sig2 = D / (16.0 * np.pi * n * h_x * h_y)

        # Wald statistic — vectorized closed-form quadratic from R's Sig2.mat.inv
        # Sig2.mat ordering: [f20, f11, f02] with h[1]=h_x, h[2]=h_y
        a = 3.0 / h_x ** 4
        b = 1.0 / (h_x * h_y) ** 2
        c = 3.0 / h_y ** 4
        det_sub = a * c - b * b  # > 0 by construction for finite h_x, h_y
        wald = np.full(D.shape, np.nan, dtype=FLOAT_DTYPE)
        valid = active & (sig2 > 0.0) & (det_sub > 0.0)
        quad = (
            (c * Hxx[valid] ** 2 + a * Hyy[valid] ** 2 - 2.0 * b * Hxx[valid] * Hyy[valid]) / det_sub
            + Hxy[valid] ** 2 / b
        )
        wald[valid] = quad / sig2[valid]

        pval = np.full(D.shape, np.nan, dtype=FLOAT_DTYPE)
        finite = np.isfinite(wald)
        pval[finite] = chi2.sf(wald[finite], df=3)  # df = d*(d+1)/2 = 3

        p_flat = pval.ravel().copy()
        p_flat[~active.ravel()] = np.nan
        signif = self._fdr_reject(p_flat, self.alpha).reshape(D.shape)

        if self.neg_curv_only:
            signif &= (l_min < 0.0) & (l_max < 0.0)

        structure_2d = np.ones((3, 3), dtype=np.bool_)
        lab, n_comp = nd_label(signif, structure=structure_2d)  # pyright: ignore[reportGeneralTypeIssues]
        if n_comp == 0:
            return []
        sizes = np.bincount(lab.ravel())
        x_centers = np.asarray(landscape.x_centers, dtype=FLOAT_DTYPE)
        y_centers = np.asarray(landscape.y_centers, dtype=FLOAT_DTYPE)

        features: list[ModeFeature] = []
        for comp_id in range(1, n_comp + 1):
            if sizes[comp_id] < self.min_component_size:
                continue
            comp_mask = np.asarray(lab == comp_id, dtype=np.bool_)
            D_comp = np.where(comp_mask, D, -np.inf)
            row, col = np.unravel_index(int(np.argmax(D_comp)), D.shape)
            row, col = int(row), int(col)
            rows_in, cols_in = np.where(comp_mask)
            lo = [float(x_centers[cols_in.min()]), float(y_centers[rows_in.min()])]
            hi = [float(x_centers[cols_in.max()]), float(y_centers[rows_in.max()])]
            features.append(
                ModeFeature(
                    id=f"mode_{comp_id}",
                    rank=0,
                    point=[float(x_centers[col]), float(y_centers[row])],
                    grid_index=[row, col],
                    peak_density=float(landscape.density[row, col]),
                    properties={"component_size": int(sizes[comp_id])},
                    footprint={
                        "grid_mask": comp_mask,
                        "bounds": [lo, hi],
                    },
                )
            )

        features.sort(key=lambda f: f.peak_density, reverse=True)
        for rank, feat in enumerate(features, start=1):
            feat.rank = rank
            feat.id = f"mode_{rank}"
        return features


@_register_component(_FEATURE_SELECTORS, "nearest_mode")
class NearestModeSelector(FeatureSelector):
    component_type = "nearest_mode"

    def __init__(self) -> None:
        """Select the detected mode nearest to ``selection_hint['seed_point']``.

        If ``roi_bounds`` is also provided, it is treated as a safeguard: the selected
        nearest mode must fall within the ROI or the selector raises an error.
        """
        pass

    def to_spec(self) -> dict[str, Any]:
        return {"type": self.component_type, "params": {}}

    def select(
        self,
        features: Sequence[ModeFeature],
        selection_hint: Mapping[str, Any] | None = None,
    ) -> ModeFeature:
        if not features:
            raise ValueError("NearestModeSelector requires at least one feature.")
        hint = dict(selection_hint or {})
        seed_point = self._selection_point(hint)
        coords = np.asarray([feature.point for feature in features], dtype=FLOAT_DTYPE)
        distances = np.linalg.norm(coords - seed_point.reshape(1, -1), axis=1)
        selected = features[int(np.argmin(distances))]
        roi_bounds = hint.get("roi_bounds")
        if roi_bounds is not None and not selected.within_roi(roi_bounds):
            raise ValueError("Nearest mode selected from seed_point falls outside roi_bounds.")
        return selected

    @staticmethod
    def _selection_point(selection_hint: dict[str, Any]) -> FloatArray:
        seed = selection_hint.get("seed_point")
        if seed is None:
            raise ValueError("NearestModeSelector requires 'seed_point' in selection_hint.")
        arr = np.asarray(seed, dtype=FLOAT_DTYPE)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr



@_register_component(_FEATURE_SELECTORS, "highest_density")
class HighestDensitySelector(FeatureSelector):
    component_type = "highest_density"

    def __init__(self) -> None:
        """Select the highest-density mode globally or within ``roi_bounds``."""
        pass

    def to_spec(self) -> dict[str, Any]:
        return {"type": self.component_type, "params": {}}

    def select(
        self,
        features: Sequence[ModeFeature],
        selection_hint: Mapping[str, Any] | None = None,
    ) -> ModeFeature:
        if not features:
            raise ValueError("HighestDensitySelector requires at least one feature.")
        hint = dict(selection_hint or {})
        roi_bounds = hint.get("roi_bounds")
        if roi_bounds is not None:
            candidates = [feature for feature in features if feature.within_roi(roi_bounds)]
            if candidates:
                return max(candidates, key=lambda feature: feature.peak_density)
        return max(features, key=lambda feature: feature.peak_density)


@_register_component(_FEATURE_SELECTORS, "mode_id")
class ModeIdSelector(FeatureSelector):
    component_type = "mode_id"

    def __init__(self) -> None:
        """Select a mode by its detected ``mode_<rank>`` identifier."""
        pass

    def to_spec(self) -> dict[str, Any]:
        return {"type": self.component_type, "params": {}}

    def select(
        self,
        features: Sequence[ModeFeature],
        selection_hint: Mapping[str, Any] | None = None,
    ) -> ModeFeature:
        if not features:
            raise ValueError("ModeIdSelector requires at least one feature.")
        hint = dict(selection_hint or {})
        requested_id = hint.get("target_mode_id")
        if requested_id is None:
            raise ValueError("ModeIdSelector requires 'target_mode_id' in selection_hint.")
        for feature in features:
            if feature.id == requested_id:
                return feature
        raise KeyError(f"Unknown target_mode_id {requested_id!r}.")


@_register_component(_REGION_EXTRACTORS, "prominence_peak")
class ProminencePeakRegionExtractor(RegionExtractor):
    component_type = "prominence_peak"

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "mask_spec": self._serialized_mask_spec(),
            },
        }

    def extract(
        self,
        landscape: DensityLandscape,
        *,
        target: ModeFeature,
        features: Sequence[ModeFeature],
        region_prefix: str,
    ) -> ExtractionResult:
        if not isinstance(landscape, Landscape1D):
            raise TypeError("ProminencePeakRegionExtractor only supports 1D landscapes.")
        if not features:
            raise ValueError("ProminencePeakRegionExtractor requires at least one feature.")
        feature_bases = self._peak_bases_from_feature_properties(landscape, features)
        try:
            left_bin, right_bin = feature_bases[target.id]
        except KeyError as exc:
            raise KeyError(f"Target feature {target.id!r} was not found in the provided features.") from exc

        region_id = f"{region_prefix}_{target.id}"
        min_value, max_value = landscape.interval_bounds(left_bin, right_bin)
        region = IntervalRegion(
            min_value=min_value,
            max_value=max_value,
        )
        label_image = np.zeros(landscape.density.shape[0], dtype=np.int_)
        label_image[left_bin:right_bin + 1] = 1
        return ExtractionResult(
            regions={region_id: region},
            label_image=label_image,
            diagnostics={"n_regions": 1},
        )

    def segment(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region_prefix: str,
    ) -> ExtractionResult:
        if not isinstance(landscape, Landscape1D):
            raise TypeError("ProminencePeakRegionExtractor only supports 1D landscapes.")
        if not features:
            raise ValueError("ProminencePeakRegionExtractor requires at least one feature.")

        ordered = sorted(features, key=lambda feature: int(feature.grid_index[0]))
        density = np.asarray(landscape.density, dtype=FLOAT_DTYPE).reshape(-1)

        separators: list[int] = []
        for idx in range(len(ordered) - 1):
            left_feature = ordered[idx]
            right_feature = ordered[idx + 1]
            left_peak = int(left_feature.grid_index[0])
            right_peak = int(right_feature.grid_index[0])
            if right_peak <= left_peak:
                raise ValueError("ProminencePeakRegionExtractor requires strictly ordered 1D peak indices.")
            valley = left_peak + int(np.argmin(density[left_peak:right_peak + 1]))
            separators.append(valley)

        region_lookup: dict[str, IntervalRegion] = {}
        region_order: list[str] = []
        for idx, feature in enumerate(ordered):
            left_bin = 0 if idx == 0 else separators[idx - 1] + 1
            right_bin = density.shape[0] - 1 if idx == len(ordered) - 1 else separators[idx]
            region_id = f"{region_prefix}_{feature.id}"
            min_value, max_value = landscape.interval_bounds(left_bin, right_bin)
            region_lookup[region_id] = IntervalRegion(
                min_value=min_value,
                max_value=max_value,
            )
            region_order.append(region_id)

        label_image = np.zeros(density.shape[0], dtype=np.int_)
        for idx, region_id in enumerate(region_order, start=1):
            region = region_lookup[region_id]
            left_bin, right_bin = landscape.interval_to_bin_slice(region.min_value, region.max_value)
            label_image[left_bin:right_bin + 1] = idx

        return ExtractionResult(
            regions=region_lookup,
            label_image=label_image,
            diagnostics={"n_regions": len(region_lookup)},
        )

    @staticmethod
    def _peak_bases_from_feature_properties(
        landscape: Landscape1D,
        features: Sequence[ModeFeature],
    ) -> dict[str, tuple[int, int]]:
        density = np.asarray(landscape.density, dtype=FLOAT_DTYPE).reshape(-1)
        if density.size == 0:
            return {}

        peak_indices = np.asarray([int(feature.grid_index[0]) for feature in features], dtype=np.int_)
        if peak_indices.size == 0:
            return {}
        if np.any(peak_indices < 0) or np.any(peak_indices >= density.size):
            raise ValueError("Mode feature grid_index is out of bounds for the 1D landscape.")

        missing_bases = any(
            feature.properties.get("left_base") is None or feature.properties.get("right_base") is None
            for feature in features
        )
        computed_left: IntArray
        computed_right: IntArray
        if missing_bases:
            _, computed_left, computed_right = peak_prominences(density, peak_indices)
        else:
            computed_left = np.asarray([int(feature.properties.get("left_base", 0)) for feature in features], dtype=np.int_)
            computed_right = np.asarray([int(feature.properties.get("right_base", density.size - 1)) for feature in features], dtype=np.int_)

        feature_bases: dict[str, tuple[int, int]] = dict(zip((feature.id for feature in features), zip(computed_left, computed_right)))
        return feature_bases



@_register_component(_REGION_EXTRACTORS, "watershed")
class WatershedRegionExtractor(RegionExtractor):
    component_type = "watershed"

    def __init__(
        self,
        watershed_line: bool = False,
        mask_spec: MaskSpecLike = None,
    ) -> None:
        """Configure watershed-based region extraction.

        Parameters
        ----------
        watershed_line
            Whether to keep watershed ridge lines as label ``0`` in the output
            segmentation. This is forwarded to ``skimage.segmentation.watershed``.
        mask_spec
            Optional masking strategy used before watershed segmentation.
        """
        super().__init__(mask_spec=mask_spec)
        self.watershed_line = bool(watershed_line)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "watershed_line": self.watershed_line,
                "mask_spec": self._serialized_mask_spec(),
            },
        }

    def segment(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region_prefix: str,
    ) -> ExtractionResult:
        if not features:
            raise ValueError("WatershedRegionExtractor requires at least one feature.")
        if not isinstance(landscape, Landscape2D):
            raise TypeError("WatershedRegionExtractor only supports 2D landscapes.")
        density = np.asarray(landscape.density, dtype=FLOAT_DTYPE)
        segmentation_mask = _resolve_density_mask(density, self.mask_spec)
        feature_order = list(features)
        markers = _build_marker_image_2d(feature_order, density.shape)
        labels = watershed(
            -density,
            markers=markers,
            mask=segmentation_mask,
            watershed_line=self.watershed_line,
        )
        region_lookup: dict[str, GateRegion] = {}
        for marker_id, feature in enumerate(feature_order, start=1):
            region_mask = np.asarray(labels == marker_id, dtype=np.bool_)
            region_id = f"{region_prefix}_{feature.id}"
            region_lookup[region_id] = RasterRegion2D(
                x_edges=landscape.x_edges,
                y_edges=landscape.y_edges,
                mask=region_mask,
            )
        # Build renumbered label image so label i corresponds to region i.
        label_image = np.zeros_like(labels, dtype=np.int_)
        for new_id, region in enumerate(region_lookup.values(), start=1):
            label_image[np.asarray(region.mask, dtype=np.bool_)] = new_id  # pyright: ignore[reportAttributeAccessIssue]
        return ExtractionResult(
            regions=region_lookup,
            label_image=label_image,
            diagnostics={"n_regions": len(region_lookup)},
        )


@_register_component(_REGION_EXTRACTORS, "random_walker")
class RandomWalkerRegionExtractor(RegionExtractor):
    component_type = "random_walker"

    def __init__(
        self,
        beta: float = 130.0,
        mode: str = "cg_j",
        tol: float = 0.001,
        prob_tol: float = 0.001,
        spacing: Sequence[float] | None = None,
        normalize_data: bool = True,
        mask_spec: MaskSpecLike = None,
    ) -> None:
        """Configure random-walker-based region extraction.

        Parameters are forwarded to ``skimage.segmentation.random_walker``
        after feature markers and the optional density mask are resolved.
        """
        super().__init__(mask_spec=mask_spec)
        self.beta = float(beta)
        self.mode = str(mode)
        self.tol = float(tol)
        self.prob_tol = float(prob_tol)
        self.spacing = None if spacing is None else [float(v) for v in spacing]
        self.normalize_data = bool(normalize_data)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "beta": self.beta,
                "mode": self.mode,
                "tol": self.tol,
                "prob_tol": self.prob_tol,
                "spacing": None if self.spacing is None else list(self.spacing),
                "normalize_data": self.normalize_data,
                "mask_spec": self._serialized_mask_spec(),
            },
        }

    def segment(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region_prefix: str,
    ) -> ExtractionResult:
        if not features:
            raise ValueError("RandomWalkerRegionExtractor requires at least one feature.")
        if not isinstance(landscape, Landscape2D):
            raise TypeError("RandomWalkerRegionExtractor only supports 2D landscapes.")
        density = np.asarray(landscape.density, dtype=FLOAT_DTYPE)
        segmentation_mask = _resolve_density_mask(density, self.mask_spec)
        feature_order = list(features)
        markers = _build_marker_image_2d(feature_order, density.shape)
        walker_markers = np.asarray(markers, dtype=np.int_)
        walker_markers[~segmentation_mask] = -1

        walker_data = self._prepare_walker_data(density)
        labels = random_walker(
            walker_data,
            walker_markers,
            beta=self.beta,
            mode=self.mode,
            tol=self.tol,
            prob_tol=self.prob_tol,
            spacing=self.spacing,
            copy=True,
        )

        region_lookup: dict[str, GateRegion] = {}
        for marker_id, feature in enumerate(feature_order, start=1):
            region_mask = np.asarray(labels == marker_id, dtype=np.bool_)
            region_id = f"{region_prefix}_{feature.id}"
            region_lookup[region_id] = RasterRegion2D(
                x_edges=landscape.x_edges,
                y_edges=landscape.y_edges,
                mask=region_mask,
            )

        label_image = np.zeros_like(labels, dtype=np.int_)
        for new_id, region in enumerate(region_lookup.values(), start=1):
            label_image[np.asarray(region.mask, dtype=np.bool_)] = new_id  # pyright: ignore[reportAttributeAccessIssue]
        return ExtractionResult(
            regions=region_lookup,
            label_image=label_image,
            diagnostics={"n_regions": len(region_lookup)},
        )

    def _prepare_walker_data(self, density: FloatArray) -> FloatArray:
        data = np.asarray(density, dtype=FLOAT_DTYPE)
        finite_mask = np.isfinite(data)
        if not np.any(finite_mask):
            return np.zeros_like(data, dtype=FLOAT_DTYPE)

        finite_values = data[finite_mask]
        fill_value = float(np.min(finite_values))
        prepared = np.where(finite_mask, data, fill_value).astype(FLOAT_DTYPE, copy=False)
        if not self.normalize_data:
            return prepared

        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        value_range = max_value - min_value
        if value_range <= 0.0:
            return np.zeros_like(prepared, dtype=FLOAT_DTYPE)
        return np.asarray((prepared - min_value) / value_range, dtype=FLOAT_DTYPE)


@_register_component(_REGION_EXTRACTORS, "level_set_cc")
class LevelSetConnectedComponentExtractor(RegionExtractor):
    component_type = "level_set_cc"

    def __init__(
        self,
        threshold_rel: float = 0.5,
        threshold_abs: float | None = None,
        connectivity: int = 1,
        mask_spec: MaskSpecLike = None,
    ) -> None:
        """Configure level-set connected-component extraction.

        Parameters
        ----------
        threshold_rel
            Relative threshold, measured between the landscape minimum and the selected
            mode peak, used when ``threshold_abs`` is not provided.
        threshold_abs
            Absolute density threshold used to define the active set before connected
            components are computed.
        connectivity
            Pixel connectivity passed to connected-component labeling in 2D.
        mask_spec
            Optional masking strategy used before connected-component extraction.
        """
        super().__init__(mask_spec=mask_spec)
        self.threshold_rel = float(threshold_rel)
        self.threshold_abs = None if threshold_abs is None else float(threshold_abs)
        self.connectivity = int(connectivity)

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {
                "threshold_rel": self.threshold_rel,
                "threshold_abs": self.threshold_abs,
                "connectivity": self.connectivity,
                "mask_spec": self._serialized_mask_spec(),
            },
        }

    def _with_threshold_abs(self, threshold_abs: float) -> "LevelSetConnectedComponentExtractor":
        return type(self)(
            threshold_rel=self.threshold_rel,
            threshold_abs=threshold_abs,
            connectivity=self.connectivity,
            mask_spec=self.mask_spec,
        )

    def extract(
        self,
        landscape: DensityLandscape,
        *,
        target: ModeFeature,
        features: Sequence[ModeFeature],
        region_prefix: str,
    ) -> ExtractionResult:
        threshold_abs = self._threshold_for_feature(target, landscape)
        extractor = self._with_threshold_abs(threshold_abs)
        extraction = super(LevelSetConnectedComponentExtractor, extractor).extract( # pyright: ignore[reportAttributeAccessIssue]
            landscape,
            target=target,
            features=features,
            region_prefix=region_prefix,
        )
        extraction.diagnostics = dict(extraction.diagnostics) | {"threshold_rel": self.threshold_rel}
        selected_id = f"{region_prefix}_{target.id}"
        selected = extraction.regions.get(selected_id)
        region_size: int
        if isinstance(selected, IntervalRegion) and isinstance(landscape, Landscape1D):
            left_bin, right_bin = landscape.interval_to_bin_slice(selected.min_value, selected.max_value)
            region_size = right_bin - left_bin + 1
        elif isinstance(selected, RasterRegion2D):
            region_size = int(np.count_nonzero(selected.mask))
        else:
            region_size = 0  # unknown region type — do not filter
        if region_size == 0:
            raise KeyError(
                f"Region for target {target.id!r} was removed due to unknown region type."
            )
        return extraction

    def segment(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region_prefix: str,
    ) -> ExtractionResult:
        if isinstance(landscape, Landscape1D):
            return self._extract_1d_components(
                landscape,
                features=features,
                region_prefix=region_prefix,
                mask_spec=self.mask_spec,
            )
        if isinstance(landscape, Landscape2D):
            return self._extract_2d_components(
                landscape,
                features=features,
                region_prefix=region_prefix,
                mask_spec=self.mask_spec,
            )
        raise TypeError(f"Unsupported landscape type {type(landscape)!r}.")

    def _threshold_for_feature(self, feature: ModeFeature, landscape: DensityLandscape) -> float:
        if self.threshold_abs is not None:
            return self.threshold_abs
        baseline = 0.0
        if isinstance(landscape, (Landscape1D, Landscape2D)):
            baseline = float(np.min(landscape.density))
        return baseline + (feature.peak_density - baseline) * self.threshold_rel

    def _global_threshold(self, features: Sequence[ModeFeature], landscape: DensityLandscape) -> float:
        if self.threshold_abs is not None:
            return self.threshold_abs
        baseline = 0.0
        if isinstance(landscape, (Landscape1D, Landscape2D)):
            baseline = float(np.min(landscape.density))
        peak_max = max(feature.peak_density for feature in features)
        return baseline + (peak_max - baseline) * self.threshold_rel

    def _extract_1d_components(
        self,
        landscape: Landscape1D,
        *,
        features: Sequence[ModeFeature],
        region_prefix: str,
        mask_spec: MaskSpecLike,
    ) -> ExtractionResult:
        threshold = self._global_threshold(features, landscape)
        support_mask = _resolve_density_mask(np.asarray(landscape.density, dtype=FLOAT_DTYPE), mask_spec)
        active = np.asarray(landscape.density >= threshold, dtype=np.bool_) & support_mask
        labels = self._component_labels_1d(active)
        regions: dict[str, GateRegion] = {}
        labels_out = np.zeros_like(labels, dtype=np.int_)
        used_components: set[int] = set()
        next_label = 1

        for feature in features:
            component = int(labels[feature.grid_index[0]])
            if component == 0 or component in used_components:
                continue
            used_components.add(component)
            component_mask = labels == component
            idxs = np.where(component_mask)[0]
            region_id = f"{region_prefix}_{feature.id}"
            min_value, max_value = landscape.interval_bounds(int(idxs[0]), int(idxs[-1]))
            regions[region_id] = IntervalRegion(
                min_value=min_value,
                max_value=max_value,
            )
            labels_out[component_mask] = next_label
            next_label += 1

        return ExtractionResult(
            regions=regions,
            label_image=labels_out,
            diagnostics={"n_regions": len(regions), "threshold_rel": self.threshold_rel},
        )

    @staticmethod
    def _component_labels_1d(mask: BooleanArray) -> IntArray:
        active = np.asarray(mask, dtype=np.bool_).reshape(-1)
        if active.size == 0:
            return np.zeros(0, dtype=np.int_)
        starts = active & ~np.r_[False, active[:-1]]
        labels = np.cumsum(starts, dtype=np.int_)
        labels[~active] = 0
        return labels

    def _extract_2d_components(
        self,
        landscape: Landscape2D,
        *,
        features: Sequence[ModeFeature],
        region_prefix: str,
        mask_spec: MaskSpecLike,
    ) -> ExtractionResult:
        threshold = self._global_threshold(features, landscape)
        support_mask = _resolve_density_mask(np.asarray(landscape.density, dtype=FLOAT_DTYPE), mask_spec)
        active = np.asarray(landscape.density >= threshold, dtype=np.bool_) & support_mask
        labels: np.ndarray = label(active, connectivity=max(self.connectivity, 1)) # pyright: ignore[reportAssignmentType]
        regions: dict[str, GateRegion] = {}
        labels_out = np.zeros_like(landscape.density, dtype=np.int_)
        used_components: set[int] = set()
        next_label = 1

        for feature in features:
            component = int(labels[feature.grid_index[0], feature.grid_index[1]])
            if component == 0 or component in used_components:
                continue
            used_components.add(component)
            component_mask = labels == component
            region_id = f"{region_prefix}_{feature.id}"
            regions[region_id] = RasterRegion2D(
                x_edges=landscape.x_edges,
                y_edges=landscape.y_edges,
                mask=np.asarray(component_mask, dtype=np.bool_),
            )
            labels_out[component_mask] = next_label
            next_label += 1

        return ExtractionResult(
            regions=regions,
            label_image=labels_out,
            diagnostics={"n_regions": len(regions), "threshold_rel": self.threshold_rel},
        )


@_register_component(_BOUNDARY_BUILDERS, "interval")
class IntervalBoundaryBuilder(BoundaryBuilder):
    component_type = "interval"

    def to_spec(self) -> dict[str, Any]:
        return {"type": self.component_type, "params": {}}

    def build(self, region: GateRegion) -> dict[str, Any]:
        if not isinstance(region, IntervalRegion):
            raise TypeError("IntervalBoundaryBuilder can only build boundaries for IntervalRegion.")
        return {
            "type": "interval",
            "bounds": [float(region.min_value), float(region.max_value)],
        }


@_register_component(_BOUNDARY_BUILDERS, "contour")
class ContourBoundaryBuilder(BoundaryBuilder):
    component_type = "contour"

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "params": {},
        }

    def build(self, region: GateRegion) -> dict[str, Any]:
        if not isinstance(region, RasterRegion2D):
            raise TypeError("ContourBoundaryBuilder can only build boundaries for RasterRegion2D.")
        serialized: list[list[list[float]]] = []
        for contour_xy in self._trace_raster_region_boundaries(region):
            if contour_xy.shape[0] < 2:
                continue
            serialized.append(contour_xy.tolist())
        return {"type": "contour", "contours": serialized}

    def _trace_raster_region_boundaries(self, region: RasterRegion2D) -> list[np.ndarray]:
        mask = np.asarray(region.mask, dtype=np.bool_)
        if mask.ndim != 2 or not np.any(mask):
            return []
        x_edges = np.asarray(region.x_edges, dtype=FLOAT_DTYPE)
        y_edges = np.asarray(region.y_edges, dtype=FLOAT_DTYPE)
        loops: list[np.ndarray] = []
        for contour_vertices in self.find_raster_contours(mask):
            if contour_vertices.shape[0] < 3:
                continue
            coords = np.column_stack(
                [
                    x_edges[np.asarray(contour_vertices[:, 0], dtype=np.int_)],
                    y_edges[np.asarray(contour_vertices[:, 1], dtype=np.int_)],
                ]
            ).astype(FLOAT_DTYPE, copy=False)
            coords = _simplify_polyline(coords, closed=True)
            if coords.shape[0] >= 3:
                loops.append(coords)
        return loops

    @staticmethod
    def find_raster_contours(mask: BooleanArray) -> list[np.ndarray]:
        cells = np.asarray(mask, dtype=np.bool_)
        if cells.ndim != 2 or not np.any(cells):
            return []

        rows, cols = cells.shape
        below = np.pad(cells[:-1, :], ((1, 0), (0, 0)), constant_values=False)
        above = np.pad(cells[1:, :], ((0, 1), (0, 0)), constant_values=False)
        left = np.pad(cells[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        right = np.pad(cells[:, 1:], ((0, 0), (0, 1)), constant_values=False)

        bottom_edges = cells & ~below
        top_edges = cells & ~above
        left_edges = cells & ~left
        right_edges = cells & ~right

        segments: list[np.ndarray] = []

        row_idx, col_idx = np.nonzero(bottom_edges)
        if row_idx.size:
            segments.append(
                np.column_stack([col_idx, row_idx, col_idx + 1, row_idx]).astype(np.int_, copy=False)
            )

        row_idx, col_idx = np.nonzero(right_edges)
        if row_idx.size:
            segments.append(
                np.column_stack([col_idx + 1, row_idx, col_idx + 1, row_idx + 1]).astype(np.int_, copy=False)
            )

        row_idx, col_idx = np.nonzero(top_edges)
        if row_idx.size:
            segments.append(
                np.column_stack([col_idx + 1, row_idx + 1, col_idx, row_idx + 1]).astype(np.int_, copy=False)
            )

        row_idx, col_idx = np.nonzero(left_edges)
        if row_idx.size:
            segments.append(
                np.column_stack([col_idx, row_idx + 1, col_idx, row_idx]).astype(np.int_, copy=False)
            )

        if not segments:
            return []

        segment_array = np.vstack(segments)
        starts: list[tuple[int, int]] = [tuple(row[:2]) for row in segment_array]
        ends: list[tuple[int, int]] = [tuple(row[2:]) for row in segment_array]

        start_map: dict[tuple[int, int], list[int]] = {}
        for idx, start in enumerate(starts):
            start_map.setdefault(start, []).append(idx) # pyright: ignore[reportArgumentType]

        direction_order = {
            (1, 0): [(0, 1), (1, 0), (0, -1), (-1, 0)],
            (0, 1): [(-1, 0), (0, 1), (1, 0), (0, -1)],
            (-1, 0): [(0, -1), (-1, 0), (0, 1), (1, 0)],
            (0, -1): [(1, 0), (0, -1), (-1, 0), (0, 1)],
        }

        contours: list[np.ndarray] = []
        used: set[int] = set()

        for start_idx in range(len(starts)):
            if start_idx in used:
                continue

            used.add(start_idx)
            start_vertex = starts[start_idx]
            current_vertex = ends[start_idx]
            prev_direction = (
                current_vertex[0] - start_vertex[0],
                current_vertex[1] - start_vertex[1],
            )
            loop: list[tuple[int, int]] = [start_vertex, current_vertex]

            while current_vertex != start_vertex:
                candidates = [
                    idx for idx in start_map.get(current_vertex, [])
                    if idx not in used
                ]
                if not candidates:
                    break

                next_idx = candidates[0]
                if len(candidates) > 1:
                    preferred_dirs = direction_order.get(prev_direction, list(direction_order))
                    candidate_by_dir = {
                        (
                            ends[idx][0] - current_vertex[0],
                            ends[idx][1] - current_vertex[1],
                        ): idx
                        for idx in candidates
                    }
                    for direction in preferred_dirs:
                        if direction in candidate_by_dir:
                            next_idx = candidate_by_dir[direction]
                            break

                used.add(next_idx)
                next_vertex = ends[next_idx]
                prev_direction = (
                    next_vertex[0] - current_vertex[0],
                    next_vertex[1] - current_vertex[1],
                )
                current_vertex = next_vertex
                loop.append(current_vertex)

            if len(loop) >= 4 and loop[-1] == loop[0]:
                contours.append(np.asarray(loop[:-1], dtype=np.int_))

        return contours

    @staticmethod
    def _insert_grid_corners(
        contour_xy: FloatArray,
        *,
        x_edges: FloatArray,
        y_edges: FloatArray,
        atol: float = 1e-6,
    ) -> np.ndarray:
        coords = np.asarray(contour_xy, dtype=FLOAT_DTYPE)
        if coords.shape[0] < 2:
            return coords

        closed = _points_close(coords[0], coords[-1], atol=atol)
        points: list[np.ndarray] = []

        segment_count = coords.shape[0] - 1 if closed else coords.shape[0]
        for idx in range(segment_count):
            current = coords[idx]
            nxt = coords[(idx + 1) % coords.shape[0]]
            points.append(current)

            dx = float(nxt[0] - current[0])
            dy = float(nxt[1] - current[1])
            if np.isclose(dx, 0.0, atol=atol) or np.isclose(dy, 0.0, atol=atol):
                continue

            corner: np.ndarray | None = None
            if _is_close_to_any(float(current[0]), x_edges, atol=atol) and _is_close_to_any(float(nxt[1]), y_edges, atol=atol):
                corner = np.asarray([current[0], nxt[1]], dtype=FLOAT_DTYPE)
            elif _is_close_to_any(float(nxt[0]), x_edges, atol=atol) and _is_close_to_any(float(current[1]), y_edges, atol=atol):
                corner = np.asarray([nxt[0], current[1]], dtype=FLOAT_DTYPE)

            if corner is not None and not _points_close(corner, current, atol=atol) and not _points_close(corner, nxt, atol=atol):
                points.append(corner)

        if not closed:
            points.append(coords[-1])
        elif points:
            points.append(points[0])
        return np.asarray(points, dtype=FLOAT_DTYPE)

    @staticmethod
    def _trim_wrapped_edge_midpoints(
        contour_xy: FloatArray,
        *,
        x_edges: FloatArray,
        y_edges: FloatArray,
        atol: float = 1e-6,
    ) -> np.ndarray:
        coords = np.asarray(contour_xy, dtype=FLOAT_DTYPE)
        while coords.shape[0] >= 4:
            first = coords[0]
            last = coords[-1]
            if _is_grid_corner(first, x_edges=x_edges, y_edges=y_edges, atol=atol) or _is_grid_corner(last, x_edges=x_edges, y_edges=y_edges, atol=atol):
                break

            corner_candidates = [
                np.asarray([first[0], last[1]], dtype=FLOAT_DTYPE),
                np.asarray([last[0], first[1]], dtype=FLOAT_DTYPE),
            ]
            should_trim = False
            for corner in corner_candidates:
                if not _is_grid_corner(corner, x_edges=x_edges, y_edges=y_edges, atol=atol):
                    continue
                if any(_points_close(point, corner, atol=atol) for point in coords):
                    should_trim = True
                    break
            if not should_trim:
                break
            coords = coords[1:-1]
        return coords

    @staticmethod
    def _close_open_contour_to_frame(
        contour_xy: FloatArray,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        x_tol: float,
        y_tol: float,
        atol: float = 1e-9,
    ) -> np.ndarray:
        coords = np.asarray(contour_xy, dtype=FLOAT_DTYPE)
        if coords.shape[0] < 2:
            return coords
        start = coords[0]
        end = coords[-1]
        if _points_close(start, end, atol=atol):
            return coords[:-1]

        snapped_start = _snap_point_to_frame(start, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, x_tol=x_tol, y_tol=y_tol)
        snapped_end = _snap_point_to_frame(end, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, x_tol=x_tol, y_tol=y_tol)
        if snapped_start is None or snapped_end is None:
            return coords

        closed = coords.copy()
        closed[0] = snapped_start
        closed[-1] = snapped_end
        frame_path = _frame_path_between(snapped_end, snapped_start, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, atol=atol)
        if frame_path.shape[0] > 1:
            closed = np.vstack([closed, frame_path[1:]])
        return _simplify_polyline(closed, closed=True, atol=atol)



def _default_boundary_builder_for_region(region: GateRegion) -> BoundaryBuilder | None:
    if isinstance(region, IntervalRegion):
        return IntervalBoundaryBuilder()
    if isinstance(region, RasterRegion2D):
        return ContourBoundaryBuilder()
    return None


def _build_boundary_for_region(
    region: GateRegion,
    *,
    builder: BoundaryBuilder | None = None,
) -> dict[str, Any] | None:
    resolved_builder = builder if builder is not None else _default_boundary_builder_for_region(region)
    if resolved_builder is None:
        return None
    return resolved_builder.build(region)

class BaseDensityGate(Gate, ABC):
    default_tunable = True
    density_state_arrays_artifact: ClassVar[str] = "density_state_arrays.npz"

    @classmethod
    def _serialize_density_state_arrays(
        cls,
        density_state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
        arrays: dict[str, np.ndarray] = {}
        payload = cls.externalize_numpy_arrays(dict(density_state), prefix="density_state", arrays=arrays)
        if not arrays:
            return payload, None
        return payload, arrays

    @classmethod
    def _deserialize_density_state_arrays(
        cls,
        density_state: Mapping[str, Any],
        raw_arrays: Mapping[str, Any],
    ) -> dict[str, Any]:
        arrays = {str(key): np.asarray(value) for key, value in raw_arrays.items()}
        restored = cls.restore_numpy_arrays(dict(density_state), arrays=arrays)
        if not isinstance(restored, Mapping):
            raise ValueError("Restored density state must be a mapping.")
        return dict(restored)

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        *,
        landscape_builder: LandscapeBuilder | Mapping[str, Any] | None = None,
        detector: FeatureDetector | Mapping[str, Any] | None = None,
        selector: FeatureSelector | Mapping[str, Any] | None = None,
        extractor: RegionExtractor | Mapping[str, Any] | None = None,
        boundary_builder: BoundaryBuilder | Mapping[str, Any] | None = None,
        selection_hint: Mapping[str, Any] | None = None,
        region_prefix: str | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a modular density gate orchestrator.

        Parameters
        ----------
        gate_name
            User-facing gate name used in returned mask keys and persisted strategy
            nodes.
        dimensions
            One or two channel names used to build the density landscape.
        landscape_builder
            Builder instance or serialized spec that converts raw events into a fitted
            density landscape.
        detector
            Feature detector instance or serialized spec used to detect candidate modes
            on the landscape.
        selector
            Optional feature selector instance or serialized spec. If omitted in
            segmentation mode, all extracted regions are retained.
        extractor
            Region extractor instance or serialized spec that converts detected modes
            into one or more fitted regions.
        boundary_builder
            Optional boundary builder used to derive lightweight plotting/export
            boundaries from extracted regions.
        selection_hint
            JSON-friendly hints used by the selector, such as ``seed_point``,
            ``target_mode_id``, or ``roi_bounds``.
        region_prefix
            Prefix used when naming extracted region IDs.
        hyperparams
            Additional hyperparameters merged into the persisted gate configuration.
        use_as_complement
            Whether ``apply()`` should return the complement of the selected region.
            Only meaningful for single-region gates.
        tunable
            Override for the base gate ``tunable`` flag.
        **kwargs
            Additional persisted hyperparameters stored alongside the normalized
            component specs.
        """
        if len(dimensions) not in {1, 2}:
            raise ValueError(f"{self.__class__.__name__} supports exactly 1D or 2D gating.")

        builder_default: LandscapeBuilder
        extractor_default: RegionExtractor
        boundary_default: BoundaryBuilder | None
        if len(dimensions) == 1:
            builder_default = HistogramLandscape1DBuilder()
            extractor_default = ProminencePeakRegionExtractor()
            boundary_default = IntervalBoundaryBuilder()
        else:
            builder_default = HistogramLandscape2DBuilder()
            extractor_default = WatershedRegionExtractor()
            boundary_default = ContourBoundaryBuilder()

        landscape_builder = _load_component(
            landscape_builder if landscape_builder is not None else None,
            registry=_LANDSCAPE_BUILDERS,
            fallback=builder_default,
        )
        if not isinstance(landscape_builder, LandscapeBuilder):
            raise ValueError("landscape_builder must be a LandscapeBuilder instance or valid spec.")
        self.landscape_builder = landscape_builder

        detector = _load_component(
            detector if detector is not None else None,
            registry=_FEATURE_DETECTORS,
            fallback=ModeDetector(),
        )
        if not isinstance(detector, FeatureDetector):
            raise ValueError("detector must be a FeatureDetector instance or valid spec.")
        self.detector = detector

        self.selector = _load_component(
            selector if selector is not None else None,
            registry=_FEATURE_SELECTORS,
            fallback=None,
        )
        extractor = _load_component(
            extractor if extractor is not None else None,
            registry=_REGION_EXTRACTORS,
            fallback=extractor_default,
        )
        if not isinstance(extractor, RegionExtractor):
            raise ValueError("extractor must be a RegionExtractor instance or valid spec.")
        self.extractor = extractor

        boundary_builder = _load_component(
            boundary_builder if boundary_builder is not None else None,
            registry=_BOUNDARY_BUILDERS,
            fallback=boundary_default,
        )
        if not isinstance(boundary_builder, BoundaryBuilder) and boundary_builder is not None:
            raise ValueError("boundary_builder must be a BoundaryBuilder instance or valid spec.")
        self.boundary_builder = boundary_builder

        if self.landscape_builder.n_dims != len(dimensions):
            raise ValueError(
                f"{self.landscape_builder.__class__.__name__} expects {self.landscape_builder.n_dims} dimensions, "
                f"got {len(dimensions)}."
            )

        removed_mask_kwargs = {"detector_mask_spec", "extractor_mask_spec"} & set(kwargs)
        if removed_mask_kwargs:
            names = ", ".join(sorted(removed_mask_kwargs))
            raise TypeError(f"{names} must be configured on detector/extractor mask_spec.")

        normalized_hyperparams = dict(hyperparams or {})
        removed_mask_hyperparams = {"detector_mask_spec", "extractor_mask_spec"} & set(normalized_hyperparams)
        if removed_mask_hyperparams:
            names = ", ".join(sorted(removed_mask_hyperparams))
            raise TypeError(f"{names} must be configured on detector/extractor mask_spec.")
        normalized_hyperparams.update(kwargs)
        normalized_hyperparams.update(
            {
                "landscape_builder": _serialize_component(self.landscape_builder),
                "detector": _serialize_component(self.detector),
                "selector": _serialize_component(self.selector),
                "extractor": _serialize_component(self.extractor),
                "boundary_builder": _serialize_component(self.boundary_builder),
                "selection_hint": dict(selection_hint or {}),
                "region_prefix": region_prefix or self._sanitize_identifier(gate_name),
            }
        )
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=normalized_hyperparams,
            use_as_complement=use_as_complement,
            tunable=tunable,
        )

    @classmethod
    def _sanitize_identifier(cls, value: str) -> str:
        normalized = re.sub(r"\W+", "_", value).strip("_")
        if not normalized:
            normalized = "density"
        if not re.match(r"^[A-Za-z_]", normalized):
            normalized = f"_{normalized}"
        return normalized

    @property
    def selection_hint(self) -> dict[str, Any]:
        return dict(self.hyperparams.get("selection_hint", {}))

    @property
    def region_prefix(self) -> str:
        return str(self.hyperparams.get("region_prefix", self._sanitize_identifier(self.gate_name)))

    def with_selection_hint(self, **hint: Any) -> "BaseDensityGate":
        gate = self.copy()
        existing = dict(gate.hyperparams.get("selection_hint", {}))
        existing.update(hint)
        gate._hyperparams["selection_hint"] = existing
        return gate # pyright: ignore[reportReturnType]

    def get_features(self) -> list[dict[str, Any]]:
        features = self._density_state().get("features", [])
        if isinstance(features, list):
            return [dict(feature) for feature in features if isinstance(feature, Mapping)]
        return []

    def _features_as_objects(self) -> list[ModeFeature]:
        return [ModeFeature.from_dict(feature) for feature in self.get_features()]

    def _target_feature(self) -> ModeFeature | None:
        payload = self._density_state().get("target_feature")
        if not isinstance(payload, Mapping):
            return None
        return ModeFeature.from_dict(payload)

    def _region_specs(self) -> dict[str, Mapping[str, Any]]:
        region_specs = self._density_state().get("regions")
        if isinstance(region_specs, Mapping):
            return {str(key): value for key, value in region_specs.items() if isinstance(value, Mapping)}
        return {}

    def _density_state(self) -> dict[str, Any]:
        payload = self.state.get("density_state")
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    @classmethod
    def from_node(cls, node: GateNode, sample_id: str | None = None) -> "BaseDensityGate":
        assert node.gate_type == cls.gate_type, f"Gate type mismatch: expected {cls.gate_type}, got {node.gate_type}"
        merged_params = node.get_params_for_sample(sample_id)
        hyperparams = merged_params.get("hyperparams", {})
        diagnostics = merged_params.get("diagnostics", {})
        gate = cls(
            gate_name=node.id,
            dimensions=node.dimensions,
            use_as_complement=node.use_as_complement,
            **hyperparams,
        )
        gate.diagnostics = dict(diagnostics)
        gate._import_json_state_payload(merged_params.get("state", {}))
        return gate

    def get_region(self, region_id: str | None = None) -> GateRegion:
        landscape = self._landscape_from_state()
        if landscape is None:
            raise ValueError("Density gate has not been fitted yet.")
        region_specs = self._region_specs()
        if region_id is None:
            target_region_id = self._density_state().get("target_region_id")
            if isinstance(target_region_id, str):
                region_id = target_region_id
            elif len(region_specs) == 1:
                region_id = next(iter(region_specs))
            else:
                raise ValueError("Multiple regions are available; please provide region_id.")
        try:
            region_spec = region_specs[region_id]
        except KeyError as exc:
            raise KeyError(f"Unknown density region {region_id!r}.") from exc
        return GateRegion.from_spec(region_spec, grid_spec=landscape.grid_spec)

    def get_boundary(self, region_id: str | None = None) -> dict[str, Any] | None:
        try:
            region = self.get_region(region_id)
        except (KeyError, ValueError):
            return None
        return _build_boundary_for_region(region, builder=self.boundary_builder)

    def _fit_density_pipeline(self, events_slice: DataFrame) -> tuple[DensityLandscape, list[ModeFeature]]:
        data = events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False)
        landscape = self.landscape_builder.build(data)
        features = self.detector.detect(landscape)
        if not features:
            raise ValueError("Density gate did not detect any modes.")
        return landscape, features

    def export_state(self) -> dict[str, Any]:
        bundle = super().export_state()
        manifest = dict(bundle["manifest"])
        artifacts = dict(bundle["artifacts"])
        artifact_map = dict(manifest.get("artifacts", {}))

        state_artifact_name = str(artifact_map.get("state", "state.json"))
        raw_state = artifacts.get(state_artifact_name)
        if isinstance(raw_state, Mapping):
            raw_state = dict(raw_state)
            density_state = raw_state.get("state", {}).get("density_state")
            if isinstance(density_state, Mapping):
                compact_state, raw_arrays = self._serialize_density_state_arrays(density_state)
                raw_state["state"] = dict(raw_state.get("state", {}))
                raw_state["state"]["density_state"] = compact_state
                artifacts[state_artifact_name] = raw_state
                if raw_arrays is not None:
                    artifact_map["density_state_arrays"] = self.density_state_arrays_artifact
                    artifacts[self.density_state_arrays_artifact] = raw_arrays
                    manifest["serializer"] = "hybrid"

        manifest["artifacts"] = artifact_map
        return {"manifest": manifest, "artifacts": artifacts}

    def import_state(
        self,
        manifest: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> None:
        super().import_state(manifest, artifacts)
        artifact_map = manifest.get("artifacts", {})
        if not isinstance(artifact_map, Mapping):
            return
        artifact_name = str(artifact_map.get("density_state_arrays", self.density_state_arrays_artifact))
        raw = artifacts.get(artifact_name)
        if raw is None:
            raw = artifacts.get("density_state_arrays")
        if raw is None:
            return
        if not isinstance(raw, Mapping):
            raise ValueError(f"Density state artifact '{artifact_name}' must be an array mapping.")
        density_state = self.state.get("density_state")
        if not isinstance(density_state, Mapping):
            raise ValueError("Density state metadata is missing from the JSON state artifact.")
        self.state["density_state"] = self._deserialize_density_state_arrays(density_state, raw)

    def _common_diagnostics(
        self,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        *,
        region: GateRegion | None = None,
        target_feature: ModeFeature | None = None,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "n_modes": float(len(features)),
        }
        if isinstance(landscape, Landscape1D):
            diagnostics["grid_size_x"] = float(landscape.density.shape[0])
        elif isinstance(landscape, Landscape2D):
            diagnostics["grid_size_x"] = float(landscape.density.shape[1])
            diagnostics["grid_size_y"] = float(landscape.density.shape[0])
        if target_feature is not None:
            diagnostics["selected_mode_rank"] = float(target_feature.rank)
            diagnostics["selected_peak_density"] = float(target_feature.peak_density)
        if region is not None:
            if isinstance(region, IntervalRegion) and isinstance(landscape, Landscape1D):
                mask = region.contains(landscape.axis.reshape(-1, 1))
                diagnostics["region_area_fraction"] = float(np.mean(mask))
                diagnostics["region_mass_fraction"] = float(
                    np.sum(landscape.counts[mask]) / max(np.sum(landscape.counts), 1e-9)
                )
            elif isinstance(region, RasterRegion2D) and isinstance(landscape, Landscape2D):
                mask = np.asarray(region.mask, dtype=np.bool_)
                diagnostics["region_area_fraction"] = float(np.mean(mask))
                diagnostics["region_mass_fraction"] = float(
                    np.sum(landscape.counts[mask]) / max(np.sum(landscape.counts), 1e-9)
                )
        return diagnostics

    def _store_density_state(
        self,
        *,
        landscape: DensityLandscape,
        features: Sequence[ModeFeature],
        extraction: ExtractionResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        density_state = {
            "landscape": landscape.to_state(),
            "features": [feature.to_dict() for feature in features],
            "label_image": None if extraction.label_image is None else np.asarray(extraction.label_image),
            "regions": {region_id: region.to_spec() for region_id, region in extraction.regions.items()},
        }
        if metadata:
            density_state.update(dict(metadata))
        self.state = {"density_state": density_state}

    def _landscape_from_state(self) -> DensityLandscape | None:
        payload = self._density_state().get("landscape")
        if not isinstance(payload, Mapping):
            return None
        return DensityLandscape.from_state(payload)

    def _single_region_overlay(
        self,
        fig: go.Figure,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        **kwargs: Any,
    ) -> None:
        boundary = self.get_boundary()
        self._add_boundary_trace(fig, boundary, row=row, col=col, showlegend=showlegend, **kwargs)

    def _all_region_boundaries(self) -> dict[str, dict[str, Any]]:
        region_ids = list(self._region_specs().keys())
        if not region_ids:
            return {}
        computed: dict[str, dict[str, Any]] = {}
        for region_id in region_ids:
            boundary = self.get_boundary(region_id)
            if isinstance(boundary, Mapping):
                computed[region_id] = dict(boundary)
        return computed

    def _add_boundary_trace(
        self,
        fig: go.Figure,
        boundary: Mapping[str, Any] | None,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(boundary, Mapping):
            return
        boundary_type = str(boundary.get("type"))
        if boundary_type == "interval":
            bounds = boundary.get("bounds", [None, None])
            if isinstance(bounds, Sequence) and len(bounds) == 2:
                for x in bounds:
                    fig.add_vline(
                        x=float(x),
                        line_color=str(kwargs.get("gate_line_color", "red")),
                        line_width=int(kwargs.get("gate_line_width", 2)),
                        line_dash=str(kwargs.get("gate_line_dash", "dash")),
                        row=row,  # pyright: ignore[reportArgumentType]
                        col=col,  # pyright: ignore[reportArgumentType]
                    )
            return

        if boundary_type != "contour":
            return
        for contour in boundary.get("contours", []):
            coords = np.asarray(contour, dtype=FLOAT_DTYPE)
            if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
                continue
            trace = go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="lines",
                line=dict(
                    color=str(kwargs.get("gate_line_color", "red")),
                    width=int(kwargs.get("gate_line_width", 2)),
                    dash=str(kwargs.get("gate_line_dash", "dash")),
                ),
                name=name or self.gate_name,
                showlegend=showlegend,
                fill="toself" if bool(kwargs.get("gate_fill", False)) else None,
                fillcolor=str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)")),
            )
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)

    def _add_feature_markers(
        self,
        fig: go.Figure,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
    ) -> None:
        features = self._features_as_objects()
        if not features:
            return
        if len(self.dimensions) == 1:
            trace = go.Scatter(
                x=[feature.point[0] for feature in features],
                y=[feature.peak_density for feature in features],
                mode="markers",
                marker=dict(color="white", size=8, line=dict(color="black", width=1)),
                name="Modes",
                showlegend=showlegend,
            )
        else:
            trace = go.Scatter(
                x=[feature.point[0] for feature in features],
                y=[feature.point[1] for feature in features],
                mode="markers+text",
                marker=dict(color="white", size=9, line=dict(color="black", width=1)),
                text=[feature.id for feature in features],
                textposition="top center",
                name="Modes",
                showlegend=showlegend,
            )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    def _plot_ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        target = self._target_feature()
        if target is not None:
            return float(target.point[0])
        return super()._plot_ratio_position_1d(dim, data, **kwargs)

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        target = self._target_feature()
        if target is not None and len(target.point) == 2:
            return float(target.point[0]), float(target.point[1])
        return super()._plot_ratio_position_2d(plot_dims, x_data, y_data, **kwargs)

    def decorate_plot_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        del dim, data
        plot_view = str(kwargs.get("plot_view", "regions"))
        if plot_view in {"features", "regions"}:
            self._single_region_overlay(fig, row=row, col=col, **kwargs)
            self._add_feature_markers(fig, row=row, col=col, showlegend=False if row is not None else True)

    def decorate_plot_2d(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        **kwargs: Any,
    ) -> None:
        del plot_dims, x_data, y_data
        plot_view = str(kwargs.get("plot_view", "regions"))
        if plot_view in {"features", "regions"}:
            self._single_region_overlay(fig, row=row, col=col, showlegend=showlegend, **kwargs)
            self._add_feature_markers(fig, row=row, col=col, showlegend=False if row is not None else showlegend)

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray] | None = None,
        dimensions: Sequence[str] | None = None,
        *,
        plot_view: str = "regions",
        **kwargs: Any,
    ) -> go.Figure:
        if plot_view == "landscape":
            return self._plot_landscape(dimensions=dimensions, **kwargs)
        return super().plot(events=events, mask=mask, dimensions=dimensions, plot_view=plot_view, **kwargs)

    def _plot_landscape(
        self,
        dimensions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        landscape = self._landscape_from_state()
        if landscape is None:
            raise ValueError("Density landscape is unavailable. Fit the gate before requesting plot_view='landscape'.")
        plot_dims = self._resolve_plot_dimensions(dimensions, available_dimensions=self.dimensions)
        title = kwargs.get("title") or f"{self.__class__.__name__}: {self.gate_name} Landscape"
        width = int(kwargs.get("width", 900))
        height = int(kwargs.get("height", 650))

        if isinstance(landscape, Landscape1D):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=landscape.axis,
                    y=landscape.density,
                    mode="lines",
                    name="Density",
                    line=dict(color="royalblue", width=2),
                )
            )
            self.decorate_plot_1d(fig, plot_dims[0], landscape.axis, plot_view="features", **kwargs)
            fig.update_xaxes(title=plot_dims[0])
            fig.update_yaxes(title="Density")
            fig.update_layout(title=title, width=width, height=height)
            return fig

        if isinstance(landscape, Landscape2D):
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=landscape.x_centers,
                    y=landscape.y_centers,
                    z=landscape.density,
                    colorscale=str(kwargs.get("colorscale", "Viridis")),
                    colorbar=dict(title="Density"),
                    name="Density",
                )
            )
            for region_id, boundary in self._all_region_boundaries().items():
                self._add_boundary_trace(fig, boundary, showlegend=False, name=region_id, **kwargs)
            self._add_feature_markers(fig, showlegend=False)
            fig.update_xaxes(title=plot_dims[0])
            fig.update_yaxes(title=plot_dims[1])
            fig.update_layout(title=title, width=width, height=height)
            return fig

        raise TypeError(f"Unsupported landscape type {type(landscape)!r}.")


@GateRegistry.register("density")
class DensityGate(BaseDensityGate):
    gate_type = "density"

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        landscape_builder: LandscapeBuilder | Mapping[str, Any] | None = None,
        detector: FeatureDetector | Mapping[str, Any] | None = None,
        selector: FeatureSelector | Mapping[str, Any] | None = None,
        extractor: RegionExtractor | Mapping[str, Any] | None = None,
        boundary_builder: BoundaryBuilder | Mapping[str, Any] | None = None,
        selection_hint: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a single-target density gate.

        Parameters
        ----------
        gate_name
            User-facing gate name used in returned mask keys and persisted nodes.
        dimensions
            One or two channel names used to fit the density landscape.
        landscape_builder, detector, selector, extractor, boundary_builder
            Pipeline components, provided either as live instances or serialized specs.
            If ``selector`` is omitted, ``NearestModeSelector`` is used by default.
        selection_hint
            JSON-friendly target-selection hints such as ``seed_point``,
            ``target_mode_id``, or ``roi_bounds``.
        use_as_complement
            Whether ``apply()`` should invert the fitted region membership.
        tunable
            Override for the base gate ``tunable`` flag.
        **kwargs
            Additional hyperparameters persisted with the gate.
        """
        selector = selector if selector is not None else NearestModeSelector()
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            landscape_builder=landscape_builder,
            detector=detector,
            selector=selector,
            extractor=extractor,
            boundary_builder=boundary_builder,
            selection_hint=selection_hint,
            use_as_complement=use_as_complement,
            tunable=tunable,
            **kwargs,
        )

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def _fit_gate(self, events_slice: DataFrame) -> None:
        if self.selector is None:
            raise ValueError("DensityGate requires a selector.")
        landscape, features = self._fit_density_pipeline(events_slice)
        target = self.selector.select(features, self.selection_hint)
        extraction = self.extractor.extract(
            landscape,
            target=target,
            features=features,
            region_prefix=self.region_prefix,
        )
        if len(extraction.regions) != 1:
            raise ValueError("DensityGate expects exactly one selected region.")
        region_id, region = next(iter(extraction.regions.items()))

        self.diagnostics = self._common_diagnostics(landscape, features, region=region, target_feature=target) | {
            key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
            for key, value in extraction.diagnostics.items()
        }
        self.params = {}
        self._store_density_state(
            landscape=landscape,
            features=features,
            extraction=extraction,
            metadata={
                "target_feature": target.to_dict(),
                "target_region_id": region_id,
            },
        )

    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        region = self.get_region()
        mask = np.asarray(region.contains(data), dtype=np.bool_)
        if self.use_as_complement:
            mask = ~mask
        return {self.gate_name: mask}

    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        scores = np.asarray(self.get_region().score(data), dtype=FLOAT_DTYPE)
        if self.use_as_complement:
            scores = -scores
        return {self.gate_name: scores}


@GateRegistry.register("density_segmentation")
class DensitySegmentationGate(BaseDensityGate):
    gate_type = "density_segmentation"

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        landscape_builder: LandscapeBuilder | Mapping[str, Any] | None = None,
        detector: FeatureDetector | Mapping[str, Any] | None = None,
        extractor: RegionExtractor | Mapping[str, Any] | None = None,
        boundary_builder: BoundaryBuilder | Mapping[str, Any] | None = None,
        selection_hint: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a multi-region density segmentation gate.

        Parameters
        ----------
        gate_name
            User-facing gate name used for the union mask and persisted parent node.
        dimensions
            One or two channel names used to fit the density landscape.
        landscape_builder, detector, extractor, boundary_builder
            Pipeline components, provided either as live instances or serialized specs.
            Segmentation gates do not take a selector because all extracted regions are
            retained.
        selection_hint
            Persisted hint payload for future compatibility. It is stored with the gate
            configuration even though segmentation keeps all regions in the current
            implementation.
        use_as_complement
            Unsupported for segmentation gates.
        tunable
            Override for the base gate ``tunable`` flag.
        **kwargs
            Additional hyperparameters persisted with the gate.
        """
        if use_as_complement:
            raise ValueError("DensitySegmentationGate does not support use_as_complement.")
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            landscape_builder=landscape_builder,
            detector=detector,
            selector=None,
            extractor=extractor,
            boundary_builder=boundary_builder,
            selection_hint=selection_hint,
            use_as_complement=False,
            tunable=tunable,
            **kwargs,
        )

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def _fit_gate(self, events_slice: DataFrame) -> None:
        landscape, features = self._fit_density_pipeline(events_slice)
        extraction = self.extractor.segment(
            landscape,
            features,
            region_prefix=self.region_prefix,
        )
        feature_by_id = {feature.id: feature for feature in features}
        target_features = {
            region_id: feature_by_id[region_id.removeprefix(f"{self.region_prefix}_")].to_dict()
            for region_id in extraction.regions
            if region_id.removeprefix(f"{self.region_prefix}_") in feature_by_id
        }

        self.diagnostics = self._common_diagnostics(landscape, features) | {
            key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
            for key, value in extraction.diagnostics.items()
        }
        self.params = {}
        self._store_density_state(
            landscape=landscape,
            features=features,
            extraction=extraction,
            metadata={
                "region_names": {
                    region_id: f"{self.gate_name} {region_id.removeprefix(f'{self.region_prefix}_')}"
                    for region_id in extraction.regions
                },
                "target_features": target_features,
            },
        )

    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        results: dict[str, np.ndarray] = {}
        union_mask = np.zeros(data.shape[0], dtype=np.bool_)
        for region_id in self._region_specs().keys():
            region = self.get_region(region_id)
            mask = np.asarray(region.contains(data), dtype=np.bool_)
            results[region_id] = mask
            union_mask |= mask
        results[self.gate_name] = union_mask
        return results

    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        results: dict[str, np.ndarray] = {}
        union_scores: list[np.ndarray] = []
        for region_id in self._region_specs().keys():
            region = self.get_region(region_id)
            scores = np.asarray(region.score(data), dtype=np.float32)
            results[region_id] = scores
            union_scores.append(scores)
        results[self.gate_name] = np.maximum.reduce(union_scores) if union_scores else np.zeros(data.shape[0], dtype=np.float32)
        return results

    def generate_offsprings(
        self,
        *,
        parent_node: GateNode,
        sample_overrides: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[GateNode]:
        region_specs = self._region_specs()
        if not region_specs:
            return []

        density_state = self._density_state()
        region_names = density_state.get("region_names", {})
        target_features = density_state.get("target_features", {})
        sample_overrides = sample_overrides or {}
        child_nodes: list[GateNode] = []
        boundary_builder_spec = self.hyperparams.get("boundary_builder")
        batch_landscape = density_state.get("landscape")

        for region_id, region_spec in region_specs.items():
            custom_gates: dict[str, dict[str, Any]] = {}
            for sample_id, sample_params in sample_overrides.items():
                sample_density_state = sample_params.get("state", {}).get("density_state", {})
                sample_region_specs = sample_density_state.get("regions", {})
                if region_id not in sample_region_specs:
                    continue
                custom_gates[sample_id] = {
                    "hyperparams": {
                        **({"boundary_builder": boundary_builder_spec} if isinstance(boundary_builder_spec, Mapping) else {}),
                    },
                    "params": {},
                    "diagnostics": {},
                    "state": {
                        "density_state": {
                            "landscape": dict(sample_density_state.get("landscape", {})),
                            "regions": {region_id: dict(sample_region_specs[region_id])},
                            "target_feature": dict(sample_density_state.get("target_features", {}).get(region_id, {})),
                            "target_region_id": region_id,
                        }
                    },
                }

            child_nodes.append(
                GateNode(
                    id=region_id,
                    gate_type="density_region",
                    dimensions=list(self.dimensions),
                    layer=parent_node.layer,
                    name=str(region_names.get(region_id, region_id)),
                    parent_ids=[parent_node.id],
                    use_as_complement=False,
                    params={
                        "hyperparams": {
                            **({"boundary_builder": boundary_builder_spec} if isinstance(boundary_builder_spec, Mapping) else {}),
                        },
                        "params": {},
                        "diagnostics": {},
                        "state": {
                            "density_state": {
                                "landscape": dict(batch_landscape) if isinstance(batch_landscape, Mapping) else {},
                                "regions": {region_id: dict(region_spec)},
                                "target_feature": dict(target_features.get(region_id, {})),
                                "target_region_id": region_id,
                            }
                        },
                    },
                    custom_gates=custom_gates,
                )
            )

        return child_nodes

    def decorate_plot_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        del dim, data
        for region_id, boundary in self._all_region_boundaries().items():
            self._add_boundary_trace(fig, boundary, row=row, col=col, showlegend=False, name=region_id, **kwargs)
        self._add_feature_markers(fig, row=row, col=col, showlegend=False if row is not None else True)

    def decorate_plot_2d(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        **kwargs: Any,
    ) -> None:
        del plot_dims, x_data, y_data
        for region_id, boundary in self._all_region_boundaries().items():
            self._add_boundary_trace(fig, boundary, row=row, col=col, showlegend=False, name=region_id, **kwargs)
        self._add_feature_markers(fig, row=row, col=col, showlegend=False if row is not None else showlegend)


@GateRegistry.register("density_region")
class DensityRegionGate(Gate):
    gate_type = "density_region"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        density_state: Mapping[str, Any] | None = None,
        boundary_builder: BoundaryBuilder | Mapping[str, Any] | None = None,
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a persisted single-region replay gate.

        Parameters
        ----------
        gate_name
            User-facing gate name used in returned mask keys.
        dimensions
            Channel names the stored region applies to.
        density_state
            Serialized density replay payload containing the landscape and region specs.
        boundary_builder
            Optional boundary builder spec used to derive plotting boundaries on demand.
        hyperparams
            Additional persisted hyperparameters for the gate node.
        use_as_complement
            Unsupported for replay region gates.
        tunable
            Override for the base gate ``tunable`` flag.
        **kwargs
            Additional persisted hyperparameters merged into ``hyperparams``.
        """
        if use_as_complement:
            raise ValueError("DensityRegionGate does not support use_as_complement.")
        normalized_hyperparams = dict(hyperparams or {})
        normalized_hyperparams.update(kwargs)
        if density_state is not None:
            normalized_hyperparams["density_state"] = dict(density_state)
        if boundary_builder is not None:
            normalized_hyperparams["boundary_builder"] = _serialize_component(boundary_builder)
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            hyperparams=normalized_hyperparams,
            use_as_complement=False,
            tunable=tunable,
        )
        self.boundary_builder = _load_component(
            boundary_builder if boundary_builder is not None else self.hyperparams.get("boundary_builder"),
            registry=_BOUNDARY_BUILDERS,
            fallback=None,
        )
        if not isinstance(self.boundary_builder, BoundaryBuilder) and self.boundary_builder is not None:
            raise ValueError("boundary_builder must be a BoundaryBuilder instance or valid spec.")
        if density_state is not None:
            self.state = {"density_state": dict(density_state)}

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def export_state(self) -> dict[str, Any]:
        bundle = super().export_state()
        manifest = dict(bundle["manifest"])
        artifacts = dict(bundle["artifacts"])
        artifact_map = dict(manifest.get("artifacts", {}))

        state_artifact_name = str(artifact_map.get("state", "state.json"))
        raw_state = artifacts.get(state_artifact_name)
        if isinstance(raw_state, Mapping):
            raw_state = dict(raw_state)
            density_state = raw_state.get("state", {}).get("density_state")
            if isinstance(density_state, Mapping):
                compact_state, raw_arrays = BaseDensityGate._serialize_density_state_arrays(density_state)
                raw_state["state"] = dict(raw_state.get("state", {}))
                raw_state["state"]["density_state"] = compact_state
                artifacts[state_artifact_name] = raw_state
                if raw_arrays is not None:
                    artifact_map["density_state_arrays"] = BaseDensityGate.density_state_arrays_artifact
                    artifacts[BaseDensityGate.density_state_arrays_artifact] = raw_arrays
                    manifest["serializer"] = "hybrid"

        manifest["artifacts"] = artifact_map
        return {"manifest": manifest, "artifacts": artifacts}

    def import_state(
        self,
        manifest: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> None:
        super().import_state(manifest, artifacts)
        artifact_map = manifest.get("artifacts", {})
        if not isinstance(artifact_map, Mapping):
            return
        artifact_name = str(artifact_map.get("density_state_arrays", BaseDensityGate.density_state_arrays_artifact))
        raw = artifacts.get(artifact_name)
        if raw is None:
            raw = artifacts.get("density_state_arrays")
        if raw is None:
            return
        if not isinstance(raw, Mapping):
            raise ValueError(f"Density state artifact '{artifact_name}' must be an array mapping.")
        density_state = self.state.get("density_state")
        if not isinstance(density_state, Mapping):
            raise ValueError("Density state metadata is missing from the JSON state artifact.")
        self.state["density_state"] = BaseDensityGate._deserialize_density_state_arrays(density_state, raw)

    @classmethod
    def from_node(cls, node: GateNode, sample_id: str | None = None) -> "DensityRegionGate":
        assert node.gate_type == cls.gate_type, f"Gate type mismatch: expected {cls.gate_type}, got {node.gate_type}"
        merged_params = node.get_params_for_sample(sample_id)
        hyperparams = merged_params.get("hyperparams", {})
        diagnostics = merged_params.get("diagnostics", {})
        gate = cls(
            gate_name=node.id,
            dimensions=node.dimensions,
            use_as_complement=node.use_as_complement,
            **hyperparams,
        )
        gate.diagnostics = dict(diagnostics)
        gate._import_json_state_payload(merged_params.get("state", {}))
        return gate

    def _fit_gate(self, events_slice: DataFrame) -> None:
        del events_slice
        if self._density_state():
            return
        raise RuntimeError(f"{self.__class__.__name__} cannot be fit independently without a region spec.")

    def _density_state(self) -> dict[str, Any]:
        payload = self.state.get("density_state")
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    def _region(self) -> GateRegion:
        density_state = self._density_state()
        landscape = self._landscape_from_state()
        if landscape is None:
            raise ValueError("DensityRegionGate is missing fitted density state.")
        region_specs = density_state.get("regions", {})
        if not isinstance(region_specs, Mapping):
            raise ValueError("DensityRegionGate is missing fitted region parameters.")
        region_id = density_state.get("target_region_id", self.gate_name)
        region_spec = region_specs.get(region_id)
        if not isinstance(region_spec, Mapping):
            raise ValueError("DensityRegionGate is missing fitted region parameters.")
        return GateRegion.from_spec(region_spec, grid_spec=landscape.grid_spec)

    def _landscape_from_state(self) -> DensityLandscape | None:
        payload = self._density_state().get("landscape")
        if not isinstance(payload, Mapping):
            return None
        return DensityLandscape.from_state(payload)

    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        return {self.gate_name: np.asarray(self._region().contains(data), dtype=np.bool_)}

    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        data = np.asarray(events_slice.to_numpy(dtype=FLOAT_DTYPE, copy=False), dtype=FLOAT_DTYPE)
        return {self.gate_name: np.asarray(self._region().score(data), dtype=np.float32)}

    def get_region(self, region_id: str | None = None) -> GateRegion:
        del region_id
        return self._region()

    def get_boundary(self, region_id: str | None = None) -> dict[str, Any] | None:
        del region_id
        return _build_boundary_for_region(self._region(), builder=self.boundary_builder)

    def _plot_ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        del dim, data, kwargs
        target = self._density_state().get("target_feature")
        if isinstance(target, Mapping):
            point = np.asarray(target.get("point", []), dtype=FLOAT_DTYPE)
            if point.size >= 1:
                return float(point[0])
        return None

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        del plot_dims, x_data, y_data, kwargs
        target = self._density_state().get("target_feature")
        if isinstance(target, Mapping):
            point = np.asarray(target.get("point", []), dtype=FLOAT_DTYPE)
            if point.size >= 2:
                return float(point[0]), float(point[1])
        return None

    def decorate_plot_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        del dim, data
        boundary = self.get_boundary()
        if not isinstance(boundary, Mapping):
            return
        bounds = boundary.get("bounds", [])
        if isinstance(bounds, Sequence) and len(bounds) == 2:
            for x in bounds:
                fig.add_vline(
                    x=float(x),
                    line_color=str(kwargs.get("gate_line_color", "red")),
                    line_width=int(kwargs.get("gate_line_width", 2)),
                    line_dash=str(kwargs.get("gate_line_dash", "dash")),
                    row=row,  # pyright: ignore[reportArgumentType]
                    col=col,  # pyright: ignore[reportArgumentType]
                )

    def decorate_plot_2d(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        **kwargs: Any,
    ) -> None:
        del plot_dims, x_data, y_data
        boundary = self.get_boundary()
        if not isinstance(boundary, Mapping) or boundary.get("type") != "contour":
            return
        for contour in boundary.get("contours", []):
            coords = np.asarray(contour, dtype=FLOAT_DTYPE)
            if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
                continue
            trace = go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="lines",
                line=dict(
                    color=str(kwargs.get("gate_line_color", "red")),
                    width=int(kwargs.get("gate_line_width", 2)),
                    dash=str(kwargs.get("gate_line_dash", "dash")),
                ),
                name=self.gate_name,
                showlegend=showlegend,
                fill="toself" if bool(kwargs.get("gate_fill", False)) else None,
                fillcolor=str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)")),
            )
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)
