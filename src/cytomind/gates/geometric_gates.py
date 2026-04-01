from __future__ import annotations

from bisect import bisect_right
from typing import Any, Hashable, Sequence, Mapping, TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Gate
from . import GateRegistry
from cytomind.domain.gates import GateNode
from cytomind.visualization.gates import (
    _create_rectangle_trace,
    _create_polygon_trace,
    _create_ellipse_trace,
)

if TYPE_CHECKING:
    from cytomind.domain.constants import BooleanArray, FloatArray
else:
    BooleanArray = object
    FloatArray = object

__all__ = [
    "RootGate",
    "RectangleGate",
    "PolygonGate",
    "EllipsoidGate",
    "QuadrantGate",
    "BooleanGate",
]



@GateRegistry.register("Root")
class RootGate(Gate):
    """
    Trivial root gate that passes all events through.

    Used as the entry point for gating strategies to avoid special-casing
    the "root" node when loading masks etc. This gate has no dimensions,
    no hyperparameters, requires no fitting, and apply() always passes
    all events through.
    """

    gate_type = "Root"
    default_tunable = False

    def __init__(
        self,
        gate_name: str = "root",
        use_as_complement: bool = False,
        tunable: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name (defaults to "root")
        use_as_complement : bool
            Must be False for Root gate (raises ValueError if True)

        Raises
        ------
        ValueError
            If use_as_complement is True
        """
        if use_as_complement:
            raise ValueError("RootGate does not support use_as_complement")
        if tunable:
            raise ValueError("RootGate does not support tunable=True")
        super().__init__(gate_name, [], {}, use_as_complement, tunable=tunable)

    def _param_key(self) -> Hashable:
        return ()

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """No fitting needed for Root gate."""
        pass

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        return True

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Root gate passes all events through."""
        mask = np.ones(len(events_slice), dtype=np.bool_)
        return {self.gate_name: mask}

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        scores = np.ones(len(events_slice), dtype=np.float32)
        return {self.gate_name: scores}

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray] | None = None,
        dimensions: Sequence[str] | None = None,
        *,
        show_ratio: bool = True,
        **kwargs: Any,
    ) -> go.Figure:
        """Root gate has no meaningful visualization."""
        if dimensions is not None:
            self._resolve_plot_dimensions(dimensions)
        fig = go.Figure()
        fig.add_annotation(text="Root Gate - passes all events through")
        try:
            if show_ratio:
                ratio = self._compute_ratio(events, mask)
                if ratio is not None:
                    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=f"{ratio*100:.1f}%", showarrow=False, bgcolor="white", opacity=0.8)
        except Exception:
            pass
        return fig


@GateRegistry.register("Rectangle")
class RectangleGate(Gate):
    """
    Rectangle gate with per-dimension min/max bounds.

    No fitting required; parameters are provided at initialization.
    """

    gate_type = "Rectangle"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        min_vals: Mapping[str, float] | None = None,
        max_vals: Mapping[str, float] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : Sequence[str]
            Dimension IDs to operate on (must have 1 or more)
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """
        hyperparams = {"min_vals": dict(min_vals or {}), "max_vals": dict(max_vals or {})}
        if tunable:
            raise ValueError("RectangleGate does not support tunable=True")
        super().__init__(gate_name, dimensions, hyperparams, use_as_complement, tunable=tunable)
        self._parse_hyperparams()

    def _parse_hyperparams(self) -> None:
        """Parse and validate hyperparameters, setting them in params."""
        min_vals: dict[str, float] = self.hyperparams["min_vals"]
        max_vals: dict[str, float] = self.hyperparams["max_vals"]

        # Validate min_vals
        self._check_thresholds(min_vals)

        # Validate max_vals
        self._check_thresholds(max_vals)

        # Validate that min_vals < max_vals for each dimension
        for dim in set(min_vals.keys()) & set(max_vals.keys()):
            if min_vals[dim] > max_vals[dim]:
                raise ValueError(
                    f"min_val {min_vals[dim]} for dimension '{dim}' exceeds max_val {max_vals[dim]}"
                )

        # Store boundaries in the unified params format expected by QC collectors:
        #   boundaries = [[min_per_dim], [max_per_dim]]
        # Missing boundaries are represented as None.
        # NOTE: for testing workflows, we may later replace None with explicit
        # max/min allowed values to force fully bounded ranges.
        boundaries_min = [min_vals.get(dim, None) for dim in self.dimensions]
        boundaries_max = [max_vals.get(dim, None) for dim in self.dimensions]
        self.params["boundaries"] = [boundaries_min, boundaries_max]

    def _param_key(self) -> Hashable:
        boundaries_min, boundaries_max = self.params["boundaries"]
        return tuple(boundaries_min + boundaries_max)

    def _check_thresholds(self, thresholds: Mapping[str, float]) -> None:
        """Validate threshold dictionary."""
        dim_set = set(self.dimensions)
        for dim, val in thresholds.items():
            if dim not in dim_set:
                raise ValueError(f"Threshold dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise ValueError(f"Threshold value for dimension '{dim}' must be numeric")

    def _boundaries(self) -> tuple[list[float | None], list[float | None]]:
        """Return validated [mins, maxs] boundary lists aligned to dimensions."""
        try:
            boundaries = self.params["boundaries"]
        except KeyError:
            raise ValueError("RectangleGate boundaries have not been set. Please fit the gate first.")
        if not isinstance(boundaries, list) or len(boundaries) != 2:
            raise ValueError("RectangleGate boundaries must be a [mins, maxs] list")
        mins = boundaries[0]
        maxs = boundaries[1]
        if not isinstance(mins, list) or len(mins) != len(self.dimensions):
            raise ValueError("RectangleGate boundaries[0] must match gate dimensions")
        if not isinstance(maxs, list) or len(maxs) != len(self.dimensions):
            raise ValueError("RectangleGate boundaries[1] must match gate dimensions")
        return mins, maxs

    def validate_params(self, params: Mapping[str, Any]) -> bool:

        # Parsability and type checks
        try:
            boundaries = params["boundaries"]
            mins = boundaries[0]
            maxs = boundaries[1]
        except Exception:
            return False

        # Structural checks
        if not isinstance(mins, list) or len(mins) != len(self.dimensions):
            return False
        if not isinstance(maxs, list) or len(maxs) != len(self.dimensions):
            return False

        return True

    @property
    def min_vals(self) -> dict[str, float | None]:
        """Return per-dimension minimum thresholds as {dim_id: value}."""
        mins, _ = self._boundaries()
        return {dim: mins[idx] for idx, dim in enumerate(self.dimensions)}

    @property
    def max_vals(self) -> dict[str, float | None]:
        """Return per-dimension maximum thresholds as {dim_id: value}."""
        _, maxs = self._boundaries()
        return {dim: maxs[idx] for idx, dim in enumerate(self.dimensions)}

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """For RectangleGate, fit just copies hyperparams to params."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Apply rectangular bounds to events."""
        mask = np.ones(len(events_slice), dtype=np.bool_)
        mins, maxs = self._boundaries()

        for idx, dim in enumerate(self.dimensions):
            col_vals = np.asarray(events_slice[dim].values)
            min_val = mins[idx]
            max_val = maxs[idx]
            if min_val is not None:
                mask &= col_vals >= float(min_val)
            if max_val is not None:
                mask &= col_vals < float(max_val)

        # Apply complement if requested
        if self.use_as_complement:
            np.logical_not(mask, out=mask)

        return {self.gate_name: mask}

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        mins, maxs = self._boundaries()
        scores = np.full(len(events_slice), np.inf, dtype=np.float32)
        has_active_boundary = False

        for idx, dim in enumerate(self.dimensions):
            col_vals = np.asarray(events_slice[dim].values, dtype=np.float32)
            min_val, max_val = mins[idx], maxs[idx]

            if min_val is None and max_val is None:
                continue
            elif min_val is None:
                dim_scores = max_val - col_vals # pyright: ignore[reportOptionalOperand]
            elif max_val is None:
                dim_scores = col_vals - min_val
            else:
                dim_scores = np.minimum(col_vals - min_val, max_val - col_vals)

            has_active_boundary = True
            scores = np.minimum(scores, dim_scores)

        if not has_active_boundary:
            scores.fill(1.0)

        if self.use_as_complement:
            scores = -scores
        return {self.gate_name: scores}

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        if len(plot_dims) > 2:
            return f"RectangleGate: {self.gate_name} (Pair Plot)"
        return f"RectangleGate: {self.gate_name}"

    def _plot_ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        min_val = self.min_vals[dim]
        max_val = self.max_vals[dim]
        if min_val is not None or max_val is not None:
            start = float(np.min(data)) if min_val is None else float(min_val)
            end = float(np.max(data)) if max_val is None else float(max_val)
            return (start + end) / 2.0
        return super()._plot_ratio_position_1d(dim, data, **kwargs)

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        x_dim, y_dim = plot_dims
        x_min = self.min_vals[x_dim]
        x_max = self.max_vals[x_dim]
        y_min = self.min_vals[y_dim]
        y_max = self.max_vals[y_dim]
        cx = float(np.mean(x_data)) if x_min is None or x_max is None else (float(x_min) + float(x_max)) / 2.0
        cy = float(np.mean(y_data)) if y_min is None or y_max is None else (float(y_min) + float(y_max)) / 2.0
        return cx, cy

    def _add_plot_overlays_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))
        min_val = self.min_vals[dim]
        max_val = self.max_vals[dim]
        if min_val is not None:
            fig.add_vline(
                x=float(min_val),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="min",
                row=row, col=col # pyright: ignore[reportArgumentType]
            )
        if max_val is not None:
            fig.add_vline(
                x=float(max_val),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="max",
                row=row, col=col # pyright: ignore[reportArgumentType]
            )

    def _add_plot_overlays_2d(
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
        x_dim, y_dim = plot_dims
        min_vals = self.min_vals
        max_vals = self.max_vals
        trace = _create_rectangle_trace(
            x_min=min_vals[x_dim],
            x_max=max_vals[x_dim],
            y_min=min_vals[y_dim],
            y_max=max_vals[y_dim],
            data_x_range=(float(np.min(x_data)), float(np.max(x_data))),
            data_y_range=(float(np.min(y_data)), float(np.max(y_data))),
            line_color=str(kwargs.get("gate_line_color", "red")),
            line_width=int(kwargs.get("gate_line_width", 2)),
            name=self.gate_name if showlegend else None,
            use_gl=bool(kwargs.get("use_gl", False)),
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)


@GateRegistry.register("Polygon")
class PolygonGate(Gate):
    """
    2D polygon gate using winding number algorithm.

    Must have exactly 2 dimensions and at least 3 vertices.
    No fitting required; polygon vertices are provided at initialization.
    """

    gate_type = "Polygon"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        vertices: list[list[float]],
        use_as_complement: bool = False,
        tunable: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : Sequence[str]
            Exactly 2 dimension IDs (x, y)
        vertices : list[list[float]]
            Ordered list of (x, y) coordinates defining polygon boundary
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """
        if len(dimensions) != 2:
            raise ValueError(f"PolygonGate requires exactly 2 dimensions, got {len(dimensions)}")

        if tunable:
            raise ValueError("PolygonGate does not support tunable=True")
        super().__init__(gate_name, dimensions, {"vertices": vertices}, use_as_complement, tunable=tunable)
        self.vertices = self._hyperparams["vertices"]

    @property
    def vertices(self) -> FloatArray:
        """Access vertices hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["vertices"], dtype=np.float32)
        except KeyError:
            raise ValueError("PolygonGate vertices have not been set. Please fit the gate first.")

    @vertices.setter
    def vertices(self, value: Sequence[Sequence[float]]) -> None:
        """Set vertices hyperparameter."""
        try:
            coords = np.asarray(value, dtype=np.float32)
        except Exception:
            raise TypeError("vertices must be convertible to a numpy array of floats")

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("vertices must be a list of (x, y) coordinate pairs")
        if coords.shape[0] < 3:
            raise ValueError(f"PolygonGate requires at least 3 vertices, got {coords.shape[0]}")
        if not np.isfinite(coords).all():
            raise ValueError("All vertex coordinates must be finite numbers")
        if np.isnan(coords).any():
            raise ValueError("Vertex coordinates cannot be NaN")

        # Fix vertex order to be consistent (counterclockwise) by sorting based on angle from centroid
        centered = coords - coords.mean(axis=0)
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        sorted_indices = np.argsort(angles)
        coords = coords[sorted_indices]

        self.params["vertices"] = coords

    def validate_params(self, params: Mapping[str, Any]) -> bool:

        # Parsability and type checks
        try:
            vertices = params["vertices"]
            coords = np.asarray(vertices, dtype=np.float32)
        except Exception:
            return False

        # Shape checks
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 3:
            return False

        # Content checks
        if not np.isfinite(coords).all() or np.isnan(coords).any():
            return False

        return True

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """For PolygonGate, fit just copies hyperparams to params."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Apply polygon gate using winding number algorithm."""

        from flowutils import gating
        coords = events_slice.to_numpy(dtype=np.float32)
        mask: BooleanArray = gating.points_in_polygon(self.vertices, coords)

        # Apply complement if requested
        if self.use_as_complement:
            np.logical_not(mask, out=mask)

        return {self.gate_name: mask}

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        from flowutils import gating

        coords = events_slice.to_numpy(dtype=np.float32)
        vertices = self.vertices
        inside_mask = np.asarray(gating.points_in_polygon(vertices, coords), dtype=bool)

        edge_starts = vertices
        edge_ends = np.roll(vertices, -1, axis=0)
        edge_vectors = edge_ends - edge_starts
        edge_lengths_sq = np.sum(edge_vectors * edge_vectors, axis=1)

        min_dist_sq = np.full(coords.shape[0], np.inf, dtype=np.float32)
        for start, vector, length_sq in zip(edge_starts, edge_vectors, edge_lengths_sq, strict=False):
            if length_sq <= 0.0:
                diff = coords - start
            else:
                rel = coords - start
                projection = np.sum(rel * vector, axis=1) / length_sq
                projection = np.clip(projection, 0.0, 1.0)
                nearest = start + projection[:, None] * vector
                diff = coords - nearest

            dist_sq = np.sum(diff * diff, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        distances = np.sqrt(min_dist_sq)
        scores = np.where(inside_mask, distances, -distances)
        if self.use_as_complement:
            scores = -scores
        return {self.gate_name: scores}

    def _param_key(self) -> Hashable:
        return tuple(self.vertices.flatten().tolist())

    def _resolve_plot_dimensions(self, requested_dimensions: Sequence[str] | None, available_dimensions: Sequence[str] | None = None, *, min_dims: int = 1, max_dims: int | None = None) -> list[str]:
        return super()._resolve_plot_dimensions(requested_dimensions, available_dimensions, min_dims=2, max_dims=2)

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        return f"PolygonGate: {self.gate_name}"

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        base_dims = list(self.dimensions)
        base_indices = [base_dims.index(plot_dims[0]), base_dims.index(plot_dims[1])]
        vertices_for_dims = self.vertices[:, base_indices]
        return float(vertices_for_dims[:, 0].mean()), float(vertices_for_dims[:, 1].mean())

    def _add_plot_overlays_2d(
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
        base_dims = list(self.dimensions)
        base_indices = [base_dims.index(plot_dims[0]), base_dims.index(plot_dims[1])]
        trace = _create_polygon_trace(
            self.vertices[:, base_indices],
            line_color=str(kwargs.get("gate_line_color", "red")),
            line_width=int(kwargs.get("gate_line_width", 2)),
            fill=bool(kwargs.get("gate_fill", True)),
            fill_color=str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)")),
            name=self.gate_name if showlegend else None,
            use_gl=bool(kwargs.get("use_gl", False)),
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["params"]["vertices"] = self.vertices.tolist()
        return base


@GateRegistry.register("Ellipsoid")
class EllipsoidGate(Gate):
    """
    Ellipsoid gate using Mahalanobis distance.

    Requires fitting to learn covariance matrix and center from data.
    """

    gate_type = "Ellipsoid"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        center: Sequence[float] | FloatArray,
        covariance_matrix: Sequence[Sequence[float]] | FloatArray,
        distance_square: float,
        use_as_complement: bool = False,
        tunable: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : Sequence[str]
            Dimension IDs (must have 2 or more)
        center : Sequence[float] | NDArray
            Center of the ellipsoid in each dimension
        covariance_matrix : Sequence[Sequence[float]] | NDArray
            Covariance matrix defining ellipsoid shape
        distance_square : float
            Square of Mahalanobis distance threshold.
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """
        # Set attributes before calling super().__init__
        hyperparams = {
            "center": center,
            "covariance_matrix": covariance_matrix,
            "distance_square": distance_square
        }
        if tunable:
            raise ValueError("EllipsoidGate does not support tunable=True")
        super().__init__(gate_name, dimensions, hyperparams, use_as_complement, tunable=tunable)

        if len(self.dimensions) < 2:
            raise ValueError(f"EllipsoidGate requires at least 2 dimensions, got {len(self.dimensions)}")

        self.center = self.hyperparams["center"]
        self.covariance_matrix = self.hyperparams["covariance_matrix"]
        self.distance_square = self.hyperparams["distance_square"]

    @property
    def center(self) -> FloatArray:
        """Access center hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["center"], dtype=np.float32)
        except KeyError:
            raise ValueError("EllipsoidGate center has not been set. Please fit the gate first.")

    @center.setter
    def center(self, value: FloatArray) -> None:
        """Set center hyperparameter."""
        try:
            value = np.asarray(value, dtype=np.float32)
        except Exception:
            raise TypeError("center must be convertible to a numpy array of floats")

        D = len(self.dimensions)
        if value.shape != (D,):
            raise ValueError(f"center must have shape ({D},), got {value.shape}")
        self.params["center"] = value

    @property
    def covariance_matrix(self) -> FloatArray:
        """Access covariance_matrix hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["covariance_matrix"], dtype=np.float32)
        except KeyError:
            raise ValueError("EllipsoidGate covariance_matrix has not been set. Please fit the gate first.")

    @covariance_matrix.setter
    def covariance_matrix(self, value: FloatArray) -> None:
        """Set covariance_matrix hyperparameter."""
        try:
            value = np.asarray(value, dtype=np.float32)
        except Exception:
            raise TypeError("covariance_matrix must be convertible to a numpy array of floats")

        # check shape
        D = len(self.dimensions)
        if value.shape != (D, D):
            raise ValueError(f"covariance_matrix must have shape ({D},{D}), got {value.shape}")

        # check symmetry
        if not np.allclose(value, value.T):
            raise ValueError("covariance_matrix must be symmetric")

        # check positive definiteness
        atol, rtol = 1e-8, 1e-5
        w = np.linalg.eigvalsh(value)
        if not w.min() >= -max(atol, rtol * np.abs(w).max(initial=0.0)):
            raise ValueError("covariance_matrix must be positive definite")

        self.params["covariance_matrix"] = value

    @property
    def distance_square(self) -> float:
        """Access distance_square hyperparameter or fitted params."""
        try:
            return float(self.params["distance_square"])
        except KeyError:
            raise ValueError("EllipsoidGate distance_square has not been set. Please fit the gate first.")

    @distance_square.setter
    def distance_square(self, value: float) -> None:
        """Set distance_square hyperparameter."""
        try:
            value = float(value)
        except Exception:
            raise TypeError("distance_square must be convertible to float")

        if value <= 0.:
            raise ValueError("distance_square must be positive")

        self.params["distance_square"] = value

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        # Parsability and type checks
        try:
            center = np.asarray(params["center"], dtype=np.float32)
            covariance_matrix = np.asarray(params["covariance_matrix"], dtype=np.float32)
            distance_square = float(params["distance_square"])
        except Exception:
            return False

        # Shape
        D = len(self.dimensions)
        if center.shape != (D,) or covariance_matrix.shape != (D, D):
            return False

        # Semidefiniteness and symmetry of covariance matrix
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            return False
        atol, rtol = 1e-8, 1e-5
        w = np.linalg.eigvalsh(covariance_matrix)
        if not w.min() >= -max(atol, rtol * np.abs(w).max(initial=0.0)):
            return False

        # Non-negativity of distance_square
        if distance_square <= 0.:
            return False

        return True

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Apply ellipsoid gate using Mahalanobis distance."""

        try:
            center = self.center
            cov_matrix = self.covariance_matrix
            distance_sq = self.distance_square
        except KeyError as e:
            raise ValueError(f"EllipsoidGate missing fitted parameter: {e}")
        coords = events_slice.values

        from flowutils import gating
        mask: BooleanArray = gating.points_in_ellipsoid(cov_matrix, center, distance_sq, coords)

        if self.use_as_complement:
            np.logical_not(mask, out=mask)

        return {self.gate_name: mask}

    def _squared_mahalanobis_distances(self, points: FloatArray) -> FloatArray:
        """
        Mahalanobis distance squared from the center for each point, using the covariance matrix.
        The distance is computed as: d^2 = (x - center)^T @ inv(covariance_matrix) @ (x - center)
        It is proportional to the negative log-likelihood under the Gaussian defined by the center and covariance.

        Args:
            points (FloatArray): Array of points with shape (N, D), where N is the number of points and D is the dimensionality.

        Raises:
            ValueError: If the input points do not have the correct shape.

        Returns:
            FloatArray: Array of squared Mahalanobis distances for each point to the center.
        """
        if points.ndim != 2 or points.shape[1] != len(self.dimensions):
            raise ValueError(f"Input points must have shape (N, {len(self.dimensions)}), got {points.shape}")
        centered = np.asarray(points, dtype=np.float32) - self.center
        solved = np.linalg.solve(self.covariance_matrix, centered.T).T
        d2 = np.einsum("ij,ij->i", centered, solved)
        return np.maximum(d2, 0.)

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        coords = events_slice.to_numpy(dtype=np.float32)
        # score ~ normalized log-likelihood under the Gaussian defined by the center and covariance
        scores = self._squared_mahalanobis_distances(coords) / self.distance_square
        # center boundary at 0
        scores = scores - 1. if self.use_as_complement else 1. - scores
        return {self.gate_name: scores}

    def _param_key(self) -> Hashable:
        center = self.center.tolist()
        cov_flat = self.covariance_matrix.flatten().tolist()
        return tuple(center + cov_flat + [self.distance_square])

    def _resolve_plot_dimensions(self, requested_dimensions: Sequence[str] | None, available_dimensions: Sequence[str] | None = None, *, min_dims: int = 2, max_dims: int | None = None) -> list[str]:
        return super()._resolve_plot_dimensions(requested_dimensions, available_dimensions, min_dims=2, max_dims=max_dims)

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        if len(plot_dims) > 2:
            return f"EllipsoidGate: {self.gate_name} (Pair Plot)"
        return f"EllipsoidGate: {self.gate_name}"

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        idx_x = list(self.dimensions).index(plot_dims[0])
        idx_y = list(self.dimensions).index(plot_dims[1])
        return float(self.center[idx_x]), float(self.center[idx_y])

    def _add_plot_overlays_2d(
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
        full_indices = [list(self.dimensions).index(plot_dims[0]), list(self.dimensions).index(plot_dims[1])]
        trace = _create_ellipse_trace(
            self.center[full_indices],
            self.covariance_matrix[np.ix_(full_indices, full_indices)],
            self.distance_square,
            line_color=str(kwargs.get("gate_line_color", "red")),
            line_width=int(kwargs.get("gate_line_width", 2)),
            fill=bool(kwargs.get("gate_fill", True)),
            fill_color=str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)")),
            name=self.gate_name if showlegend else None,
            use_gl=bool(kwargs.get("use_gl", False)),
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Convert numpy arrays to lists for JSON serialization
        for p in ("hyperparams", "params"):
            for k in ("center", "covariance_matrix"):
                if k in base[p] and not isinstance(base[p][k], list):
                    base[p][k] = base[p][k].tolist()
        return base


# TODO: add a "Quadrant" gate for each individual quadrant for completeness
@GateRegistry.register("Quadrant")
class QuadrantGate(Gate):
    """
    Quadrant gate following GatingML 2.0 specification.

    A QuadrantGate divides an n-dimensional space using divider dimensions,
    creating axis-aligned regions (quadrants or hyperrectangular regions).
    Each quadrant is identified by a location point (one per divider dimension).
    """

    gate_type = "Quadrant"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        dividers: Mapping[str, Sequence[float]],
        quadrants: Mapping[str, Sequence[tuple[str, float]]],
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dividers : Mapping[str, Iterable[float]]
            Dictionary mapping dimension IDs to sorted list of division points.
            Example: {'CD4': [100, 200], 'CD8': [50, 150]}
        quadrants : Mapping[str, Iterable[tuple[str, float]]]
            Dictionary mapping quadrant_id to list of (dimension_id, location) tuples.
            The location identifies which segment of each divider the quadrant is in.
            Missing dimensions mean "merged" (include all segments of that dimension).
            Example:
                {
                    'Q1': [('CD4', 150), ('CD8', 100)],  # High CD4, High CD8
                    'Q2': [('CD4', 50), ('CD8', 100)],   # Low CD4, High CD8
                    'Q3': [('CD4', 50)],                 # Low CD4, all CD8 (merged)
                }
        dimensions : list[str]
            Argument ignored; dimensions are inferred from dividers.
        """
        # Validate dividers
        dividers = dict(dividers)
        for dim_id, points in dividers.items():
            if not isinstance(points, Sequence):
                raise TypeError(f"Division points for '{dim_id}' must be a list/tuple, got {type(points)}")
            points_list = sorted(list(points))
            if len(points_list) != len(set(points_list)):
                raise ValueError(f"Division points for '{dim_id}' contain duplicates")
            dividers[dim_id] = points_list

        # Validate quadrants
        quadrants = dict(quadrants)
        for quad_id, locations in quadrants.items():
            for dim_id, loc in locations:
                if dim_id not in dividers:
                    raise ValueError(
                        f"Quadrant '{quad_id}' references dimension '{dim_id}' "
                        f"not in dividers {list(dividers.keys())}"
                    )
                if not isinstance(loc, (int, float)):
                    raise TypeError(f"Location for dimension '{dim_id}' in quadrant '{quad_id}' must be numeric")

        dimensions = list(dividers.keys())
        hyperparams = {
            "dividers": dividers,
            "quadrants": quadrants
        }
        if tunable:
            raise ValueError("QuadrantGate does not support tunable=True")
        super().__init__(gate_name, dimensions, hyperparams, tunable=tunable)

        if len(self.dimensions) < 1:
            raise ValueError("QuadrantGate requires at least 1 divider dimension")

        self._compute_quadrants()

    @property
    def dividers(self) -> dict[str, list[float]]:
        """Access dividers hyperparameter."""
        return self._hyperparams["dividers"]

    @property
    def locations(self) -> dict[str, dict[str, float]]:
        """Access original quadrant locations from hyperparams."""
        quad_locs_input = self._hyperparams["quadrants"]
        return {quad_id: dict(loc_list) for quad_id, loc_list in quad_locs_input.items()}

    @property
    def quadrants(self) -> dict[str, dict[str, tuple[float | None, float | None]]]:
        """Access computed quadrant definitions from params (with min/max ranges)."""
        try:
            return self.params["quadrants"]
        except KeyError:
            raise ValueError("QuadrantGate quadrants have not been computed. Please fit the gate first.")

    def _compute_quadrants(self) -> None:
        """
        Convert GatingML 2.0 dividers and locations to computed quadrants with borders.

        For each quadrant, determines the min/max range for each divider dimension
        based on which segment the location point falls into using binary search.
        """

        # Compute quadrants with min/max ranges
        # Format: {quadrant_id: {dimension_id: (min_val, max_val)}}
        computed_quadrants: dict[str, dict[str, tuple[float | None, float | None]]] = {}

        for quad_id, loc_dict in self.locations.items():
            # Quadrant definition: {dim_id: (min_val, max_val)}
            quad_def: dict[str, tuple[float | None, float | None]] = {}
            for dim_id in self.dimensions:
                try:
                    location = loc_dict[dim_id]
                except KeyError:
                    # Dimension not specified: merged (include all segments)
                    quad_def[dim_id] = (None, None)
                    continue

                # This dimension is specified for this quadrant
                division_points = self.dividers[dim_id]

                # Find which segment this location falls into using binary search
                # bisect_right returns the index where location would be inserted
                pos = bisect_right(division_points, location)

                if pos == 0:
                    # Location is before first division point
                    min_val = None
                    max_val = division_points[0]
                elif pos == len(division_points):
                    # Location is at or after last division point
                    min_val = division_points[-1]
                    max_val = None
                else:
                    # Location is between division_points[pos-1] and division_points[pos]
                    min_val = division_points[pos - 1]
                    max_val = division_points[pos]

                quad_def[dim_id] = (min_val, max_val)

            computed_quadrants[quad_id] = quad_def

        # Store computed quadrants
        self.params["quadrants"] = computed_quadrants

    @classmethod
    def from_node(cls, node, sample_id: str | None = None):
        merged_params = node.get_params_for_sample(sample_id)
        params = merged_params.get("params", {})
        boundaries = params.get("boundaries")
        if isinstance(boundaries, Mapping):
            min_vals = {dim: bounds[0] for dim, bounds in boundaries.items() if bounds[0] is not None}
            max_vals = {dim: bounds[1] for dim, bounds in boundaries.items() if bounds[1] is not None}
            gate = RectangleGate(
                gate_name=node.name or node.id,
                dimensions=node.dimensions,
                min_vals=min_vals,
                max_vals=max_vals,
                use_as_complement=node.use_as_complement,
            )
            gate.diagnostics = dict(merged_params.get("diagnostics", {}))
            return gate
        return super().from_node(node, sample_id=sample_id)

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        try:
            quadrants = params["quadrants"]
        except KeyError:
            return False

        if not isinstance(quadrants, dict):
            return False

        for quad in quadrants.values():
            if not isinstance(quad, dict):
                return False
            for dim_id, (min_val, max_val) in quad.items():
                if dim_id not in self.dimensions:
                    return False
                if min_val is not None and not isinstance(min_val, (int, float)):
                    return False
                if max_val is not None and not isinstance(max_val, (int, float)):
                    return False
                if min_val is not None and max_val is not None and min_val >= max_val:
                    return False

        return True

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        return

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """
        Apply quadrant gate to generate masks for each quadrant.

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping quadrant_id to boolean mask for that quadrant.
            Each mask indicates which events fall within the quadrant's bounds.
        """
        results: dict[str, BooleanArray] = {
            self.gate_name: np.ones(len(events_slice), dtype=np.bool_)
        }

        for quad_id, quad_def in self.quadrants.items():
            # Start with all events True for this quadrant
            quad_mask = np.ones(len(events_slice), dtype=np.bool_)

            # Apply each divider's range restrictions
            for div_id, (min_val, max_val) in quad_def.items():
                col_vals = np.asarray(events_slice[div_id].values)

                if min_val is not None:
                    quad_mask &= col_vals >= min_val
                if max_val is not None:
                    quad_mask &= col_vals < max_val

            results[quad_id] = quad_mask

        return results

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        masks = self._apply_gate(events_slice)
        return {
            region_id: np.where(np.asarray(region_mask, dtype=bool), 1.0, -1.0)
            for region_id, region_mask in masks.items()
        }

    @staticmethod
    def _quadrant_to_child_params(
        quadrant_definition: Mapping[str, tuple[float | None, float | None]],
    ) -> dict[str, Any]:
        return {
            "hyperparams": {},
            "params": {
                "boundaries": {
                    dim_id: tuple(bounds)
                    for dim_id, bounds in quadrant_definition.items()
                }
            },
            "diagnostics": {},
            "state": {},
        }

    def generate_offsprings(
        self,
        *,
        parent_node,
        sample_overrides: Mapping[str, dict[str, Any]] | None = None,
    ):
        sample_overrides = sample_overrides or {}
        child_nodes = []

        for quadrant_id, quadrant_definition in self.quadrants.items():
            custom_gates: dict[str, dict[str, Any]] = {}
            for sample_id, sample_params in sample_overrides.items():
                sample_quadrants = sample_params.get("params", {}).get("quadrants", {})
                if quadrant_id in sample_quadrants:
                    custom_gates[sample_id] = self._quadrant_to_child_params(sample_quadrants[quadrant_id])

            child_nodes.append(
                GateNode(
                    id=quadrant_id,
                    gate_type="Quadrant",
                    dimensions=list(self.dimensions),
                    layer=parent_node.layer,
                    name=quadrant_id,
                    parent_ids=[parent_node.id],
                    use_as_complement=False,
                    params=self._quadrant_to_child_params(quadrant_definition),
                    custom_gates=custom_gates,
                )
            )

        return child_nodes

    def _param_key(self) -> Hashable:
        quadrants_tuple = tuple(
            (quad_id, tuple((dim, locations[dim]) for dim in self.dimensions if dim in locations))
            for quad_id, locations in sorted(self.locations.items())
        )
        return quadrants_tuple

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray] | None = None,
        dimensions: Sequence[str] | None = None,
        *,
        quadrant_label_font_size: int = 12,
        quadrant_label_color: str | None = None,
        quadrant_label_bgcolor: str | None = None,
        quadrant_label_opacity: float = 0.8,
        **kwargs: Any,
    ) -> go.Figure:
        return super().plot(
            events,
            mask=mask,
            dimensions=dimensions,
            quadrant_label_font_size=quadrant_label_font_size,
            quadrant_label_color=kwargs.get("gate_line_color", "red") if quadrant_label_color is None else quadrant_label_color,
            quadrant_label_bgcolor=quadrant_label_bgcolor,
            quadrant_label_opacity=quadrant_label_opacity,
            **kwargs,
        )

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        if len(plot_dims) > 2:
            return f"QuadrantGate: {self.gate_name} (Pair Plot)"
        return f"QuadrantGate: {self.gate_name}"

    def _add_plot_overlays_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        for divider_val in self.dividers.get(dim, []):
            fig.add_vline(
                x=float(divider_val),
                line_color=str(kwargs.get("gate_line_color", "red")),
                line_width=int(kwargs.get("gate_line_width", 2)),
                line_dash=str(kwargs.get("gate_line_dash", "dash")),
                row=row, col=col,  # pyright: ignore[reportArgumentType]
            )

    def _add_plot_overlays_2d(
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
        del showlegend
        self._add_quadrant_divider_segments(
            fig=fig,
            x_dim=plot_dims[0],
            y_dim=plot_dims[1],
            x_min=float(np.min(x_data)),
            x_max=float(np.max(x_data)),
            y_min=float(np.min(y_data)),
            y_max=float(np.max(y_data)),
            line_color=str(kwargs.get("gate_line_color", "red")),
            line_width=int(kwargs.get("gate_line_width", 2)),
            line_dash=str(kwargs.get("gate_line_dash", "dash")),
            row=row,
            col=col,
        )

    @staticmethod
    def _merge_intervals(intervals: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda v: (v[0], v[1]))
        merged: list[tuple[float, float]] = [sorted_intervals[0]]
        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + tol:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _intersect_intervals(
        left_intervals: list[tuple[float, float]],
        right_intervals: list[tuple[float, float]],
        tol: float = 1e-9,
    ) -> list[tuple[float, float]]:
        intersections: list[tuple[float, float]] = []
        for left_start, left_end in left_intervals:
            for right_start, right_end in right_intervals:
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if end - start > tol:
                    intersections.append((start, end))
        if not intersections:
            return []

        sorted_intervals = sorted(intersections, key=lambda v: (v[0], v[1]))
        merged: list[tuple[float, float]] = [sorted_intervals[0]]
        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + tol:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _add_quadrant_divider_segments(
        self,
        fig: go.Figure,
        x_dim: str,
        y_dim: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        line_color: str,
        line_width: int,
        line_dash: str,
        row: int | None = None,
        col: int | None = None,
    ) -> None:
        tol = 1e-9

        def _x_bounds(quad_def: dict[str, tuple[float | None, float | None]]) -> tuple[float, float]:
            x_lower, x_upper = quad_def[x_dim]
            return (
                x_min if x_lower is None else float(x_lower),
                x_max if x_upper is None else float(x_upper),
            )

        def _y_bounds(quad_def: dict[str, tuple[float | None, float | None]]) -> tuple[float, float]:
            y_lower, y_upper = quad_def[y_dim]
            return (
                y_min if y_lower is None else float(y_lower),
                y_max if y_upper is None else float(y_upper),
            )

        for divider_val in self.dividers.get(x_dim, []):
            boundary = float(divider_val)
            left_intervals: list[tuple[float, float]] = []
            right_intervals: list[tuple[float, float]] = []

            for quad_def in self.quadrants.values():
                x_start, x_end = _x_bounds(quad_def)
                y_start, y_end = _y_bounds(quad_def)
                if abs(x_end - boundary) <= tol:
                    left_intervals.append((y_start, y_end))
                if abs(x_start - boundary) <= tol:
                    right_intervals.append((y_start, y_end))

            for y_start, y_end in self._intersect_intervals(left_intervals, right_intervals, tol=tol):
                if row is None or col is None:
                    fig.add_shape(
                        type="line",
                        x0=boundary,
                        x1=boundary,
                        y0=y_start,
                        y1=y_end,
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                    )
                else:
                    fig.add_shape(
                        type="line",
                        x0=boundary,
                        x1=boundary,
                        y0=y_start,
                        y1=y_end,
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        row=row,
                        col=col,
                    )

        for divider_val in self.dividers.get(y_dim, []):
            boundary = float(divider_val)
            lower_intervals: list[tuple[float, float]] = []
            upper_intervals: list[tuple[float, float]] = []

            for quad_def in self.quadrants.values():
                x_start, x_end = _x_bounds(quad_def)
                y_start, y_end = _y_bounds(quad_def)
                if abs(y_end - boundary) <= tol:
                    lower_intervals.append((x_start, x_end))
                if abs(y_start - boundary) <= tol:
                    upper_intervals.append((x_start, x_end))

            for x_start, x_end in self._intersect_intervals(lower_intervals, upper_intervals, tol=tol):
                if row is None or col is None:
                    fig.add_shape(
                        type="line",
                        x0=x_start,
                        x1=x_end,
                        y0=boundary,
                        y1=boundary,
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                    )
                else:
                    fig.add_shape(
                        type="line",
                        x0=x_start,
                        x1=x_end,
                        y0=boundary,
                        y1=boundary,
                        line=dict(color=line_color, width=line_width, dash=line_dash),
                        row=row,
                        col=col,
                    )

@GateRegistry.register("Boolean")
class BooleanGate(Gate):
    """
    Boolean gate that combines input masks using a boolean expression.

    Dynamic gate that doesn't operate on event data directly, but instead
    evaluates a boolean expression over masks from parent gates using pandas.eval().

    Example:
        expr = "CD4_gate_pos & CD8_gate_neg"
        gate = BooleanGate('CD4_CD8_double_neg', expression=expr)

        Then in apply:
            masks = {
                'CD4_gate_pos': np.array([T, F, T, F]),
                'CD8_gate_neg': np.array([T, T, F, F]),
            }
            result = gate.apply(events, mask=masks)
            # Returns: {'CD4_CD8_double_neg.pos': np.array([T, F, F, F])}
    """

    gate_type = "Boolean"
    default_tunable = False

    def __init__(
        self,
        gate_name: str,
        expression: str,
        dimensions: tuple[str, ...] = (),
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name for the gate
        expression : str
            Boolean expression as a string using bitwise operators.
            Variables are mask names (e.g., 'CD4_gate_pos').
            Operators: & (AND), | (OR), ~ (NOT)
            Example: "(CD4_gate_pos & CD8_gate_pos) | ~CD3_neg"
        dimensions : list[str]
            List of dimensions for plotting. Should be the union of dimensions from parent gates used in the expression.
        use_as_complement : bool
            If True, returns complement (negative) of the gate result
        """
        if tunable:
            raise ValueError("BooleanGate does not support tunable=True")
        super().__init__(gate_name, dimensions, {"expression": expression}, use_as_complement, tunable=tunable)
        self._parse_expression()

    @staticmethod
    def _extract_variables(expression) -> set[str]:
        """Extract variable names from the boolean expression."""
        import re
        # Match valid Python identifiers: must start with letter or underscore,
        # followed by letters, digits, or underscores (no dots or colons)
        pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
        matches = re.findall(pattern, expression)
        operators = set(('and', 'or', 'not', 'True', 'False', 'AND', 'OR', 'NOT', '&', '|', '~', '(', ')'))
        vars = set(matches) - operators
        # check that all "words" in the expression are either variables or operators
        all_words = set(re.findall(r'\b\w+\b', expression))
        invalid_words = all_words - vars - operators
        if invalid_words:
            raise ValueError(f"Invalid tokens in boolean expression: {invalid_words}")

        return vars

    @property
    def variables(self) -> set[str]:
        """Get the set of variable names used in the expression."""
        try:
            return set(self.params["variables"])
        except KeyError:
            raise ValueError("BooleanGate variables have not been set. Please fit the gate first.")

    @property
    def expression(self) -> str:
        """Get the boolean expression."""
        try:
            return self.params["expression"]
        except KeyError:
            raise ValueError("BooleanGate expression has not been set. Please fit the gate first.")

    def _parse_expression(self):
        """
        Parses the provided boolean expression.
        BooleanGate doesn't require fitting (no event data to learn from).
        """
        expression = self._hyperparams.get("expression")

        if not isinstance(expression, str):
            raise TypeError("BooleanGate expression must be a string")
        variables = self._extract_variables(expression)
        dummy_vals = {var: False for var in variables}
        try:
            res = pd.eval(expression, local_dict=dummy_vals, global_dict={})
        except Exception as e:
            raise ValueError(f"Invalid boolean expression '{expression}': {e}")
        if not isinstance(res, int) or (res != 0 and res != 1):
            raise ValueError(f"BooleanGate expression must evaluate to a boolean value, got {res}")

        self.params["expression"] = expression
        self.params["variables"] = variables

    def validate_params(self, params: Mapping[str, Any]) -> bool:
        try:
            dummy_vals = {var: False for var in params["variables"]}
            expression = params["expression"]
            res = pd.eval(expression, local_dict=dummy_vals, global_dict={})
        except Exception:
            return False

        if not isinstance(res, int) or (res != 0 and res != 1):
            return False

        return True

    def _param_key(self) -> Hashable:
        # this is not really an invariant but reducing is NP-hard...
        if self.use_as_complement:
            return f"~({self.expression})"
        return self.expression

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        return

    def apply(self, events: ad.AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, BooleanArray]:
        """
        Apply boolean expression to masks from parent gates.

        BooleanGate completely overrides the standard gate apply behavior.
        It works only with masks from parent gates; event data is not used.

        Parameters
        ----------
        events : ad.AnnData
            Not used for BooleanGate, but required for interface compatibility
        mask : dict[str, BooleanArray]
            Dictionary of masks from parent gates, mapping variable names to boolean arrays.
            Must contain all variables referenced in the expression.

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary with single key self.gate_name containing the result of the expression.
            Mask size equals the size of the input mask arrays.
        """
        if mask is None:
            raise ValueError("BooleanGate requires a mask dictionary with parent gate masks for evaluation.")

        # Ensure we have all needed variables
        provided_vars = set(mask.keys())
        missing_vars = self.variables - provided_vars
        if missing_vars:
            raise ValueError(
                f"BooleanGate '{self.gate_name}' missing required masks: {missing_vars}. "
                f"Provided masks: {provided_vars}"
            )

        # Evaluate the expression with the provided masks using pandas.
        # BooleanGate does not depend on event matrix content.
        mask_df = pd.DataFrame(mask)
        try:
            result = mask_df.eval(self.expression, local_dict={}, global_dict={})
        except Exception as e:
            raise ValueError(f"Error evaluating boolean expression '{self.expression}': {e}")

        result = np.asarray(result, dtype=bool)

        if self.use_as_complement:
            np.logical_not(result, out=result)

        return {self.gate_name: result}

    def score(self, events: ad.AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, FloatArray]:
        if mask is None:
            raise ValueError("BooleanGate requires a mask dictionary with parent gate masks for evaluation.")

        result_masks = self.apply(events, mask)
        return {
            region_id: np.where(np.asarray(region_mask, dtype=bool), 1.0, -1.0)
            for region_id, region_mask in result_masks.items()
        }

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        raise NotImplementedError("BooleanGate uses apply() override instead of _apply_gate")

    def _score_gate(self, events_slice: pd.DataFrame) -> dict[str, FloatArray]:
        raise NotImplementedError("BooleanGate overrides score() instead of _score_gate")
