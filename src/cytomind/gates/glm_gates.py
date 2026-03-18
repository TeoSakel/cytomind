from __future__ import annotations

from bisect import bisect_right
import warnings
from typing import Any, Hashable, Sequence, Mapping, TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Gate
from . import GateRegistry
from cytomind.visualization.gates import (
    _compute_density_colors,
    _create_scatter_trace,
    _create_rectangle_trace,
    _create_polygon_trace,
    _create_ellipse_trace,
    _create_subplot_grid,
    _format_gate_plot,
)

if TYPE_CHECKING:
    from cytomind.domain.constants import BooleanArray, FloatArray
else:
    BooleanArray = object
    FloatArray = object


def _downsample_indices(n_points: int, max_points: int, seed: int) -> np.ndarray | None:
    """Return random row indices for downsampling, or None when no downsampling is needed."""
    if max_points <= 0 or n_points <= max_points:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(n_points, size=max_points, replace=False)


def _warn_unused_plot_kwargs(gate_cls_name: str, kwargs: Mapping[str, Any]) -> None:
    """Warn when unexpected plotting kwargs are provided."""
    if not kwargs:
        return
    unused = ", ".join(sorted(kwargs.keys()))
    warnings.warn(
        f"{gate_cls_name}.plot() got unexpected keyword argument(s): {unused}",
        UserWarning,
        stacklevel=3,
    )


def _resolve_plot_dimensions(
    requested_dimensions: Sequence[str] | None,
    gate_dimensions: Sequence[str],
    *,
    gate_cls_name: str,
    min_dims: int = 1,
    max_dims: int | None = None,
) -> list[str]:
    """Resolve and validate dimensions used for plotting."""
    if requested_dimensions is None:
        dims = list(gate_dimensions)
    else:
        dims = list(requested_dimensions)

    if not dims:
        raise ValueError(f"{gate_cls_name}.plot() requires at least one dimension")

    unknown = [dim for dim in dims if dim not in gate_dimensions]
    if unknown:
        raise ValueError(
            f"{gate_cls_name}.plot() received dimensions not in gate dimensions: {unknown}. "
            f"Available dimensions: {list(gate_dimensions)}"
        )

    if len(set(dims)) != len(dims):
        raise ValueError(f"{gate_cls_name}.plot() dimensions must be unique")

    if len(dims) < min_dims:
        raise ValueError(f"{gate_cls_name}.plot() requires at least {min_dims} dimensions")
    if max_dims is not None and len(dims) > max_dims:
        raise ValueError(f"{gate_cls_name}.plot() supports at most {max_dims} dimensions")

    return dims


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
    glm_type = None
    tunable = False

    def __init__(self, gate_name: str = "root", use_as_complement: bool = False) -> None:
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
        super().__init__(gate_name, [], {}, use_as_complement)

    def __param_key(self) -> Hashable:
        return ()

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """No fitting needed for Root gate."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Root gate passes all events through."""
        mask = np.ones(len(events_slice), dtype=np.bool_)
        return {self.gate_name: mask}

    def apply(self, events: ad.AnnData, mask: dict[str, BooleanArray] = {}) -> dict[str, BooleanArray]:
        """
        Apply root gate - passes all events through.

        Parameters
        ----------
        events : ad.AnnData
            Event data
        mask : dict[str, BooleanArray], default {}
            Parent gate masks (ignored for Root gate, can be empty)

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping root gate name to boolean mask of all True
        """
        if events.isbacked:
            events = events.to_memory()

        # Root gate passes all events through
        all_mask = np.ones(events.n_obs, dtype=np.bool_)
        return {self.gate_name: all_mask}

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Root gate has no meaningful visualization."""
        if dimensions is not None:
            _resolve_plot_dimensions(dimensions, self.dimensions, gate_cls_name=self.__class__.__name__)
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        fig = go.Figure()
        fig.add_annotation(text="Root Gate - passes all events through")
        return fig


@GateRegistry.register("Rectangle")
class RectangleGate(Gate):
    """
    Rectangle gate with per-dimension min/max bounds.

    No fitting required; parameters are provided at initialization.
    """

    gate_type = "Rectangle"
    glm_type = "RectangleGate"
    tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        min_vals: Mapping[str, float] = {},
        max_vals: Mapping[str, float] = {},
        use_as_complement: bool = False,
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
        hyperparams = {"min_vals": dict(min_vals), "max_vals": dict(max_vals)}
        super().__init__(gate_name, dimensions, hyperparams, use_as_complement)
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

    def __param_key(self) -> Hashable:
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

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: Sequence[str] | None = None,
        *,
        hist_nbins: int = 100,
        hist_color: str = "rgba(100, 100, 200, 0.6)",
        histnorm: str = "probability",
        density_nbins: int = 50,
        density_log_scale: bool = True,
        marker_size: int = 3,
        colorscale: str = "Viridis",
        use_gl: bool = True,
        max_points: int = 50000,
        downsample_seed: int = 0,
        gate_line_color: str = "red",
        gate_line_width: int = 2,
        gate_line_dash: str = "dash",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot events with rectangular gate boundaries.

        For 1D gates: creates a histogram with vertical threshold lines.
        For 2D gates: creates a scatter plot with rectangle overlay.
        For N-D gates: creates pairwise projection grid.

        Parameters
        ----------
        events : ad.AnnData
            Event data (pre-filtered by parent mask)
        mask : dict[str, BooleanArray]
            Parent gate mask (required but not used for plotting)

        Returns
        -------
        go.Figure
            Plotly figure with events and gate boundaries

        Examples
        --------
        >>> gate = RectangleGate("live", ["FSC-A"], min_vals={"FSC-A": 1000})
        >>> fig = gate.plot(events, {"root": root_mask})
        >>> fig.show()
        """
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        plot_dims = _resolve_plot_dimensions(dimensions, self.dimensions, gate_cls_name=self.__class__.__name__)
        n_dims = len(plot_dims)
        plot_kwargs: dict[str, Any] = {
            "hist_nbins": hist_nbins,
            "hist_color": hist_color,
            "histnorm": histnorm,
            "density_nbins": density_nbins,
            "density_log_scale": density_log_scale,
            "marker_size": marker_size,
            "colorscale": colorscale,
            "use_gl": use_gl,
            "max_points": max_points,
            "downsample_seed": downsample_seed,
            "gate_line_color": gate_line_color,
            "gate_line_width": gate_line_width,
            "gate_line_dash": gate_line_dash,
            "title": title,
            "width": width,
            "height": height,
        }
        plot_fn = self._plot_1D if n_dims == 1 else self._plot_2D if n_dims == 2 else self._plot_nD
        return plot_fn(events, plot_dims, **plot_kwargs)

    def _plot_1D(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        dim = plot_dims[0]
        data = np.asarray(events[:, dim].X).ravel()
        nbins = int(kwargs.get("hist_nbins", 100))
        hist_color = str(kwargs.get("hist_color", "rgba(100, 100, 200, 0.6)"))
        histnorm = str(kwargs.get("histnorm", "probability"))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"RectangleGate: {self.gate_name}"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=nbins,
            name="Events",
            marker=dict(color=hist_color),
            histnorm=histnorm,
        ))

        min_val = self.min_vals[dim]
        max_val = self.max_vals[dim]

        if min_val is not None:
            fig.add_vline(
                x=float(min_val),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="min",
            )
        if max_val is not None:
            fig.add_vline(
                x=float(max_val),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="max",
            )

        yaxis_title = histnorm.title() if histnorm else "Count"
        fig.update_xaxes(title=dim)
        fig.update_yaxes(title=yaxis_title)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_2D(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        x_dim, y_dim = plot_dims
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"RectangleGate: {self.gate_name}"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        downsample_idx = _downsample_indices(x_data.shape[0], max_points, downsample_seed)
        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]

        density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)

        fig = go.Figure()
        fig.add_trace(_create_scatter_trace(
            x_data,
            y_data,
            density,
            marker_size=marker_size,
            colorscale=colorscale,
            use_gl=use_gl,
        ))

        min_vals = self.min_vals
        max_vals = self.max_vals
        x_min_raw = min_vals[x_dim]
        x_max_raw = max_vals[x_dim]
        y_min_raw = min_vals[y_dim]
        y_max_raw = max_vals[y_dim]
        x_min = float(x_min_raw) if x_min_raw is not None else None
        x_max = float(x_max_raw) if x_max_raw is not None else None
        y_min = float(y_min_raw) if y_min_raw is not None else None
        y_max = float(y_max_raw) if y_max_raw is not None else None

        data_x_range = (x_data.min(), x_data.max())
        data_y_range = (y_data.min(), y_data.max())

        fig.add_trace(_create_rectangle_trace(
            x_min, x_max, y_min, y_max,
            data_x_range, data_y_range,
            line_color=gate_line_color,
            line_width=gate_line_width,
            name=self.gate_name,
            use_gl=use_gl,
        ))

        fig.update_xaxes(title=x_dim)
        fig.update_yaxes(title=y_dim)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_nD(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        fig, pairs = _create_subplot_grid(len(plot_dims))
        n_cols = int(np.ceil(np.sqrt(len(pairs))))
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"RectangleGate: {self.gate_name} (Pairwise Projections)"
        downsample_idx = _downsample_indices(events.n_obs, max_points, downsample_seed)
        min_vals = self.min_vals
        max_vals = self.max_vals

        for idx, (i, j) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            x_dim = plot_dims[i]
            y_dim = plot_dims[j]
            x_data = np.asarray(events[:, x_dim].X).ravel()
            y_data = np.asarray(events[:, y_dim].X).ravel()
            if downsample_idx is not None:
                x_data = x_data[downsample_idx]
                y_data = y_data[downsample_idx]

            density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)
            fig.add_trace(
                _create_scatter_trace(
                    x_data,
                    y_data,
                    density,
                    marker_size=marker_size,
                    colorscale=colorscale,
                    use_gl=use_gl,
                    showlegend=False,
                ),
                row=row, col=col,
            )

            x_min_raw = min_vals[x_dim]
            x_max_raw = max_vals[x_dim]
            y_min_raw = min_vals[y_dim]
            y_max_raw = max_vals[y_dim]
            x_min = float(x_min_raw) if x_min_raw is not None else None
            x_max = float(x_max_raw) if x_max_raw is not None else None
            y_min = float(y_min_raw) if y_min_raw is not None else None
            y_max = float(y_max_raw) if y_max_raw is not None else None

            data_x_range = (x_data.min(), x_data.max())
            data_y_range = (y_data.min(), y_data.max())

            fig.add_trace(
                _create_rectangle_trace(
                    x_min, x_max, y_min, y_max,
                    data_x_range, data_y_range,
                    line_color=gate_line_color,
                    line_width=gate_line_width,
                    use_gl=use_gl,
                    name=self.gate_name if idx == 0 else None,
                ),
                row=row, col=col,
            )

            fig.update_xaxes(title=x_dim, row=row, col=col)
            fig.update_yaxes(title=y_dim, row=row, col=col)

        n_plots = len(pairs)
        n_rows = int(np.ceil(n_plots / n_cols))
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_height = int(height) if height is not None else 400 * n_rows
        plot_width = int(width) if width is not None else 400 * n_cols

        return _format_gate_plot(
            fig,
            title=plot_title,
            width=plot_width,
            height=plot_height,
        )


@GateRegistry.register("Polygon")
class PolygonGate(Gate):
    """
    2D polygon gate using winding number algorithm.

    Must have exactly 2 dimensions and at least 3 vertices.
    No fitting required; polygon vertices are provided at initialization.
    """

    gate_type = "Polygon"
    glm_type = "PolygonGate"
    tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        vertices: list[list[float]],
        use_as_complement: bool = False,
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

        super().__init__(gate_name, dimensions, {"vertices": vertices}, use_as_complement)
        self.vertices = self._hyperparams["vertices"]

    @property
    def vertices(self) -> FloatArray:
        """Access vertices hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["vertices"])
        except KeyError:
            raise ValueError("PolygonGate vertices have not been set. Please fit the gate first.")

    @vertices.setter
    def vertices(self, value: Sequence[Sequence[float]]) -> None:
        """Set vertices hyperparameter."""
        try:
            coords = np.asarray(value, dtype=np.float64)
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

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """For PolygonGate, fit just copies hyperparams to params."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        """Apply polygon gate using winding number algorithm."""

        from flowutils import gating
        coords = events_slice.values
        mask: BooleanArray = gating.points_in_polygon(self.vertices, coords)

        # Apply complement if requested
        if self.use_as_complement:
            np.logical_not(mask, out=mask)

        return {self.gate_name: mask}

    def __param_key(self) -> Hashable:
        return tuple(self.vertices.flatten().tolist())

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: Sequence[str] | None = None,
        *,
        density_nbins: int = 50,
        density_log_scale: bool = True,
        marker_size: int = 3,
        colorscale: str = "Viridis",
        use_gl: bool = False,
        max_points: int = 50000,
        downsample_seed: int = 0,
        gate_line_color: str = "red",
        gate_line_width: int = 2,
        gate_fill: bool = True,
        gate_fill_color: str = "rgba(255, 0, 0, 0.1)",
        title: str | None = None,
        width: int = 800,
        height: int = 600,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot events with polygon gate boundary.

        Creates a 2D scatter plot with polygon boundary overlaid.
        The polygon interior is filled with semi-transparent color.

        Parameters
        ----------
        events : ad.AnnData
            Event data (pre-filtered by parent mask)
        mask : dict[str, BooleanArray]
            Parent gate mask (required but not used for plotting)

        Returns
        -------
        go.Figure
            Plotly figure with events and polygon boundary

        Examples
        --------
        >>> gate = PolygonGate("lymph", ["FSC-A", "SSC-A"], vertices=[[0, 0], [1, 1], [0, 1]])
        >>> fig = gate.plot(events, {"root": root_mask})
        >>> fig.show()
        """
        plot_dims = _resolve_plot_dimensions(
            dimensions,
            self.dimensions,
            gate_cls_name=self.__class__.__name__,
            min_dims=2,
            max_dims=2,
        )
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        x_dim, y_dim = plot_dims
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        plot_title = title if title is not None else f"PolygonGate: {self.gate_name}"

        base_dims = list(self.dimensions)
        base_indices = [base_dims.index(x_dim), base_dims.index(y_dim)]
        vertices_for_dims = self.vertices[:, base_indices]

        # Downsample large clouds to keep interactive plotting responsive.
        n_points = x_data.shape[0]
        if max_points > 0 and n_points > max_points:
            rng = np.random.default_rng(downsample_seed)
            idx = rng.choice(n_points, size=max_points, replace=False)
            x_data = x_data[idx]
            y_data = y_data[idx]

        # Compute density colors
        density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)

        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(_create_scatter_trace(
            x_data,
            y_data,
            density,
            marker_size=marker_size,
            colorscale=colorscale,
            use_gl=use_gl,
        ))

        # Add polygon boundary
        fig.add_trace(_create_polygon_trace(
            vertices_for_dims,
            line_color=gate_line_color,
            line_width=gate_line_width,
            fill=gate_fill,
            fill_color=gate_fill_color,
            name=self.gate_name,
            use_gl=use_gl,
        ))

        fig.update_xaxes(title=x_dim)
        fig.update_yaxes(title=y_dim)
        return _format_gate_plot(fig, title=plot_title, width=width, height=height)

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
    glm_type = "EllipsoidGate"
    tunable = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        center: Sequence[float] | FloatArray,
        covariance_matrix: Sequence[Sequence[float]] | FloatArray,
        distance_square: float,
        use_as_complement: bool = False,
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
        super().__init__(gate_name, dimensions, hyperparams, use_as_complement)

        if len(self.dimensions) < 2:
            raise ValueError(f"EllipsoidGate requires at least 2 dimensions, got {len(self.dimensions)}")

        self.center = self.hyperparams["center"]
        self.covariance_matrix = self.hyperparams["covariance_matrix"]
        self.distance_square = self.hyperparams["distance_square"]

    @property
    def center(self) -> FloatArray:
        """Access center hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["center"])
        except KeyError:
            raise ValueError("EllipsoidGate center has not been set. Please fit the gate first.")

    @center.setter
    def center(self, value: FloatArray) -> None:
        """Set center hyperparameter."""
        try:
            value = np.asarray(value, dtype=np.float64)
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
            return np.asarray(self.params["covariance_matrix"])
        except KeyError:
            raise ValueError("EllipsoidGate covariance_matrix has not been set. Please fit the gate first.")

    @covariance_matrix.setter
    def covariance_matrix(self, value: FloatArray) -> None:
        """Set covariance_matrix hyperparameter."""
        try:
            value = np.asarray(value, dtype=np.float64)
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

    def __param_key(self) -> Hashable:
        center = self.center.tolist()
        cov_flat = self.covariance_matrix.flatten().tolist()
        return tuple(center + cov_flat + [self.distance_square])

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: Sequence[str] | None = None,
        *,
        density_nbins: int = 50,
        density_log_scale: bool = True,
        marker_size: int = 3,
        colorscale: str = "Viridis",
        use_gl: bool = True,
        max_points: int = 50000,
        downsample_seed: int = 0,
        gate_line_color: str = "red",
        gate_line_width: int = 2,
        gate_fill: bool = True,
        gate_fill_color: str = "rgba(255, 0, 0, 0.1)",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot events with ellipsoid gate boundary.

        For 2D gates: creates a scatter plot with ellipse overlay.
        For N-D gates: creates pairwise projection grid showing ellipsoid projections.

        Parameters
        ----------
        events : ad.AnnData
            Event data (pre-filtered by parent mask)
        mask : dict[str, BooleanArray]
            Parent gate mask (required but not used for plotting)

        Returns
        -------
        go.Figure
            Plotly figure with events and ellipsoid boundary

        Examples
        --------
        >>> gate = EllipsoidGate("lymph", ["FSC-A", "SSC-A"], center=[50000, 30000],
        ...                      covariance_matrix=[[1e8, 0], [0, 1e7]], distance_square=9)
        >>> fig = gate.plot(events, {"root": root_mask})
        >>> fig.show()
        """
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        plot_dims = _resolve_plot_dimensions(
            dimensions,
            self.dimensions,
            gate_cls_name=self.__class__.__name__,
            min_dims=2,
        )
        plot_kwargs: dict[str, Any] = {
            "density_nbins": density_nbins,
            "density_log_scale": density_log_scale,
            "marker_size": marker_size,
            "colorscale": colorscale,
            "use_gl": use_gl,
            "max_points": max_points,
            "downsample_seed": downsample_seed,
            "gate_line_color": gate_line_color,
            "gate_line_width": gate_line_width,
            "gate_fill": gate_fill,
            "gate_fill_color": gate_fill_color,
            "title": title,
            "width": width,
            "height": height,
        }
        plot_fn = self._plot_2D if len(plot_dims) == 2 else self._plot_nD
        return plot_fn(events, plot_dims, **plot_kwargs)

    def _plot_2D(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        x_dim, y_dim = plot_dims
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_fill = bool(kwargs.get("gate_fill", True))
        gate_fill_color = str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"EllipsoidGate: {self.gate_name}"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        downsample_idx = _downsample_indices(x_data.shape[0], max_points, downsample_seed)
        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]

        density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)

        fig = go.Figure()
        fig.add_trace(_create_scatter_trace(
            x_data,
            y_data,
            density,
            marker_size=marker_size,
            colorscale=colorscale,
            use_gl=use_gl,
        ))
        fig.add_trace(_create_ellipse_trace(
            self.center[[list(self.dimensions).index(x_dim), list(self.dimensions).index(y_dim)]],
            self.covariance_matrix[np.ix_(
                [list(self.dimensions).index(x_dim), list(self.dimensions).index(y_dim)],
                [list(self.dimensions).index(x_dim), list(self.dimensions).index(y_dim)],
            )],
            self.distance_square,
            line_color=gate_line_color,
            line_width=gate_line_width,
            fill=gate_fill,
            fill_color=gate_fill_color,
            name=self.gate_name,
            use_gl=use_gl,
        ))

        fig.update_xaxes(title=x_dim)
        fig.update_yaxes(title=y_dim)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_nD(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        fig, pairs = _create_subplot_grid(len(plot_dims))
        n_cols = int(np.ceil(np.sqrt(len(pairs))))
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_fill = bool(kwargs.get("gate_fill", True))
        gate_fill_color = str(kwargs.get("gate_fill_color", "rgba(255, 0, 0, 0.1)"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"EllipsoidGate: {self.gate_name} (Pairwise Projections)"
        downsample_idx = _downsample_indices(events.n_obs, max_points, downsample_seed)

        for idx, (i, j) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            x_dim = plot_dims[i]
            y_dim = plot_dims[j]
            x_data = np.asarray(events[:, x_dim].X).ravel()
            y_data = np.asarray(events[:, y_dim].X).ravel()
            if downsample_idx is not None:
                x_data = x_data[downsample_idx]
                y_data = y_data[downsample_idx]

            density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)
            fig.add_trace(
                _create_scatter_trace(
                    x_data,
                    y_data,
                    density,
                    marker_size=marker_size,
                    colorscale=colorscale,
                    use_gl=use_gl,
                    showlegend=False,
                ),
                row=row, col=col,
            )

            full_indices = [list(self.dimensions).index(plot_dims[i]), list(self.dimensions).index(plot_dims[j])]
            center_2d = self.center[full_indices]
            cov_2d = self.covariance_matrix[np.ix_(full_indices, full_indices)]

            fig.add_trace(
                _create_ellipse_trace(
                    center_2d,
                    cov_2d,
                    self.distance_square,
                    line_color=gate_line_color,
                    line_width=gate_line_width,
                    fill=gate_fill,
                    fill_color=gate_fill_color,
                    use_gl=use_gl,
                    name=self.gate_name if idx == 0 else None,
                ),
                row=row, col=col,
            )

            fig.update_xaxes(title=x_dim, row=row, col=col)
            fig.update_yaxes(title=y_dim, row=row, col=col)

        n_plots = len(pairs)
        n_rows = int(np.ceil(n_plots / n_cols))
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_height = int(height) if height is not None else 400 * n_rows
        plot_width = int(width) if width is not None else 400 * n_cols

        return _format_gate_plot(
            fig,
            title=plot_title,
            width=plot_width,
            height=plot_height,
        )

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
    glm_type = "QuadrantGate"
    tunable = False

    def __init__(
        self,
        gate_name: str,
        dividers: Mapping[str, Sequence[float]],
        quadrants: Mapping[str, Sequence[tuple[str, float]]],
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
        super().__init__(gate_name, dimensions, hyperparams)

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

    def __param_key(self) -> Hashable:
        quadrants_tuple = tuple(
            (quad_id, tuple((dim, locations[dim]) for dim in self.dimensions if dim in locations))
            for quad_id, locations in sorted(self.locations.items())
        )
        return quadrants_tuple

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: Sequence[str] | None = None,
        *,
        hist_nbins: int = 100,
        hist_color: str = "rgba(100, 100, 200, 0.6)",
        histnorm: str = "probability",
        density_nbins: int = 50,
        density_log_scale: bool = True,
        marker_size: int = 3,
        colorscale: str = "Viridis",
        use_gl: bool = True,
        max_points: int = 50000,
        downsample_seed: int = 0,
        gate_line_color: str = "red",
        gate_line_width: int = 2,
        gate_line_dash: str = "dash",
        quadrant_label_font_size: int = 12,
        quadrant_label_color: str | None = None,
        quadrant_label_bgcolor: str | None = None,
        quadrant_label_opacity: float = 0.8,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot events with quadrant divider lines.

        For 2D gates: creates a scatter plot with divider lines creating a grid.
        For N-D gates: creates pairwise projection grid showing dividers.

        Parameters
        ----------
        events : ad.AnnData
            Event data (pre-filtered by parent mask)
        mask : dict[str, BooleanArray]
            Parent gate mask (required but not used for plotting)

        Returns
        -------
        go.Figure
            Plotly figure with events and quadrant dividers

        Examples
        --------
        >>> gate = QuadrantGate("quads", dividers={"CD4": [100], "CD8": [100]},
        ...                     quadrants={"Q1": [("CD4", 150), ("CD8", 150)]})
        >>> fig = gate.plot(events, {"root": root_mask})
        >>> fig.show()
        """
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        plot_dims = _resolve_plot_dimensions(dimensions, self.dimensions, gate_cls_name=self.__class__.__name__)
        label_color = gate_line_color if quadrant_label_color is None else quadrant_label_color
        n_dims = len(plot_dims)
        plot_kwargs: dict[str, Any] = {
            "hist_nbins": hist_nbins,
            "hist_color": hist_color,
            "histnorm": histnorm,
            "density_nbins": density_nbins,
            "density_log_scale": density_log_scale,
            "marker_size": marker_size,
            "colorscale": colorscale,
            "use_gl": use_gl,
            "max_points": max_points,
            "downsample_seed": downsample_seed,
            "gate_line_color": gate_line_color,
            "gate_line_width": gate_line_width,
            "gate_line_dash": gate_line_dash,
            "quadrant_label_font_size": quadrant_label_font_size,
            "quadrant_label_color": label_color,
            "quadrant_label_bgcolor": quadrant_label_bgcolor,
            "quadrant_label_opacity": quadrant_label_opacity,
            "title": title,
            "width": width,
            "height": height,
        }
        plot_fn = self._plot_1D if n_dims == 1 else self._plot_2D if n_dims == 2 else self._plot_nD
        return plot_fn(events, plot_dims, **plot_kwargs)

    def _plot_1D(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        dim = plot_dims[0]
        data = np.asarray(events[:, dim].X).ravel()
        nbins = int(kwargs.get("hist_nbins", 100))
        hist_color = str(kwargs.get("hist_color", "rgba(100, 100, 200, 0.6)"))
        histnorm = str(kwargs.get("histnorm", "probability"))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"QuadrantGate: {self.gate_name}"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=nbins,
            name="Events",
            marker=dict(color=hist_color),
            histnorm=histnorm,
        ))

        for divider_val in self.dividers[dim]:
            fig.add_vline(
                x=divider_val,
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
            )

        label_font_size = int(kwargs.get("quadrant_label_font_size", 12))
        label_font_color = str(kwargs.get("quadrant_label_color", gate_line_color))
        label_bg_color = kwargs.get("quadrant_label_bgcolor")
        label_opacity = float(kwargs.get("quadrant_label_opacity", 0.8))
        dim_min, dim_max = float(np.min(data)), float(np.max(data))

        for quad_id, quad_def in self.quadrants.items():
            dim_lower, dim_upper = quad_def[dim]

            # Convert open-ended quadrant bounds into finite plot-space bounds.
            dim_start = dim_min if dim_lower is None else float(dim_lower)
            dim_end = dim_max if dim_upper is None else float(dim_upper)
            dim_center = (dim_start + dim_end) / 2.0

            annotation_kwargs: dict[str, Any] = {
                "x": dim_center,
                "y": 0.5,
                "yref": "paper",
                "text": quad_id,
                "showarrow": False,
                "font": {"size": label_font_size, "color": label_font_color},
                "opacity": label_opacity,
            }
            if label_bg_color is not None:
                annotation_kwargs["bgcolor"] = label_bg_color

            fig.add_annotation(**annotation_kwargs)

        yaxis_title = histnorm.title() if histnorm else "Count"
        fig.update_xaxes(title=dim)
        fig.update_yaxes(title=yaxis_title)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_2D(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        x_dim, y_dim = plot_dims
        x_all = np.asarray(events[:, x_dim].X).ravel()
        y_all = np.asarray(events[:, y_dim].X).ravel()
        x_data = x_all
        y_data = y_all
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"QuadrantGate: {self.gate_name}"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        downsample_idx = _downsample_indices(x_data.shape[0], max_points, downsample_seed)
        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]

        density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)

        fig = go.Figure()
        fig.add_trace(_create_scatter_trace(
            x_data,
            y_data,
            density,
            marker_size=marker_size,
            colorscale=colorscale,
            use_gl=use_gl,
        ))

        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        self._add_quadrant_divider_segments(
            fig=fig,
            x_dim=x_dim,
            y_dim=y_dim,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            line_color=gate_line_color,
            line_width=gate_line_width,
            line_dash=gate_line_dash,
        )

        label_font_size = int(kwargs.get("quadrant_label_font_size", 12))
        label_font_color = str(kwargs.get("quadrant_label_color", gate_line_color))
        label_bg_color = kwargs.get("quadrant_label_bgcolor")
        label_opacity = float(kwargs.get("quadrant_label_opacity", 0.8))
        for quad_id, quad_def in self.quadrants.items():
            x_lower, x_upper = quad_def[x_dim]
            y_lower, y_upper = quad_def[y_dim]

            # Convert open-ended quadrant bounds into finite plot-space bounds.
            x_start = x_min if x_lower is None else float(x_lower)
            x_end = x_max if x_upper is None else float(x_upper)
            y_start = y_min if y_lower is None else float(y_lower)
            y_end = y_max if y_upper is None else float(y_upper)

            x_center = (x_start + x_end) / 2.0
            y_center = (y_start + y_end) / 2.0

            annotation_kwargs: dict[str, Any] = {
                "x": x_center,
                "y": y_center,
                "text": quad_id,
                "showarrow": False,
                "font": {"size": label_font_size, "color": label_font_color},
                "opacity": label_opacity,
            }
            if label_bg_color is not None:
                annotation_kwargs["bgcolor"] = label_bg_color

            fig.add_annotation(**annotation_kwargs)

        fig.update_xaxes(title=x_dim)
        fig.update_yaxes(title=y_dim)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_nD(self, events: ad.AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        fig, pairs = _create_subplot_grid(len(plot_dims))
        n_cols = int(np.ceil(np.sqrt(len(pairs))))
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = str(kwargs.get("colorscale", "Viridis"))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))
        label_font_size = int(kwargs.get("quadrant_label_font_size", 12))
        label_font_color = str(kwargs.get("quadrant_label_color", gate_line_color))
        label_bg_color = kwargs.get("quadrant_label_bgcolor")
        label_opacity = float(kwargs.get("quadrant_label_opacity", 0.8))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"QuadrantGate: {self.gate_name} (Pairwise Projections)"
        downsample_idx = _downsample_indices(events.n_obs, max_points, downsample_seed)

        for idx, (i, j) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            x_dim = plot_dims[i]
            y_dim = plot_dims[j]
            x_all = np.asarray(events[:, x_dim].X).ravel()
            y_all = np.asarray(events[:, y_dim].X).ravel()
            x_data = x_all
            y_data = y_all
            if downsample_idx is not None:
                x_data = x_data[downsample_idx]
                y_data = y_data[downsample_idx]

            density = _compute_density_colors(x_data, y_data, nbins=density_nbins, log_scale=density_log_scale)
            fig.add_trace(
                _create_scatter_trace(
                    x_data,
                    y_data,
                    density,
                    marker_size=marker_size,
                    colorscale=colorscale,
                    use_gl=use_gl,
                    showlegend=False,
                ),
                row=row, col=col,
            )

            x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
            y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
            self._add_quadrant_divider_segments(
                fig=fig,
                x_dim=x_dim,
                y_dim=y_dim,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                row=row,
                col=col,
            )

            for quad_id, quad_def in self.quadrants.items():
                x_lower, x_upper = quad_def[x_dim]
                y_lower, y_upper = quad_def[y_dim]

                # Convert open-ended quadrant bounds into finite plot-space bounds.
                x_start = x_min if x_lower is None else float(x_lower)
                x_end = x_max if x_upper is None else float(x_upper)
                y_start = y_min if y_lower is None else float(y_lower)
                y_end = y_max if y_upper is None else float(y_upper)

                x_center = (x_start + x_end) / 2.0
                y_center = (y_start + y_end) / 2.0

                annotation_kwargs: dict[str, Any] = {
                    "x": x_center,
                    "y": y_center,
                    "text": quad_id,
                    "showarrow": False,
                    "font": {"size": label_font_size, "color": label_font_color},
                    "opacity": label_opacity,
                    "row": row,
                    "col": col,
                }
                if label_bg_color is not None:
                    annotation_kwargs["bgcolor"] = label_bg_color

                fig.add_annotation(**annotation_kwargs)

            fig.update_xaxes(title=x_dim, row=row, col=col)
            fig.update_yaxes(title=y_dim, row=row, col=col)

        n_plots = len(pairs)
        n_rows = int(np.ceil(n_plots / n_cols))
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_height = int(height) if height is not None else 400 * n_rows
        plot_width = int(width) if width is not None else 400 * n_cols

        return _format_gate_plot(
            fig,
            title=plot_title,
            width=plot_width,
            height=plot_height,
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
    glm_type = "BooleanGate"
    tunable = False

    def __init__(
        self,
        gate_name: str,
        expression: str,
        dimensions: list[str] = [],
        use_as_complement: bool = False,
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
        super().__init__(gate_name, dimensions, {"expression": expression}, use_as_complement)
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

    def __param_key(self) -> Hashable:
        # this is not really an invariant but reducing is NP-hard...
        return self.expression

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        return

    def apply(self, events: ad.AnnData, mask: dict[str, BooleanArray]) -> dict[str, BooleanArray]:
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

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, BooleanArray]:
        raise NotImplementedError("BooleanGate uses apply() override instead of _apply_gate")

    def plot(
        self,
        events: ad.AnnData,
        mask: dict[str, BooleanArray],
        dimensions: list[str] | None = None,
        *,
        hist_nbins: int = 100,
        histnorm: str = "probability",
        marker_size: int = 3,
        use_gl: bool = True,
        max_points: int = 50000,
        downsample_seed: int = 0,
        fail_color: str = "rgba(255, 0, 0, 0.3)",
        pass_color: str = "rgba(0, 255, 0, 0.5)",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Plot events colored by boolean gate result.

        For 1D gates: creates a pass/fail overlaid histogram.
        For 2D gates: creates a pass/fail scatter plot.
        For N-D gates: creates pairwise projection grid.
        """
        _warn_unused_plot_kwargs(self.__class__.__name__, kwargs)
        if self.dimensions:
            plot_dims = _resolve_plot_dimensions(dimensions, self.dimensions, gate_cls_name=self.__class__.__name__)
        else:
            event_dims = [str(name) for name in events.var_names]
            if dimensions is not None:
                plot_dims = _resolve_plot_dimensions(dimensions, event_dims, gate_cls_name=self.__class__.__name__)
            else:
                if events.n_vars == 0:
                    raise ValueError("BooleanGate plot requires at least one dimension")
                # Keep the previous fallback behavior when no plotting dims are configured.
                plot_dims = [str(events.var_names[0])]
                if events.n_vars > 1:
                    plot_dims.append(str(events.var_names[1]))

        gate_result = np.asarray(self.apply(events, mask)[self.gate_name], dtype=bool)

        n_dims = len(plot_dims)
        plot_kwargs: dict[str, Any] = {
            "hist_nbins": hist_nbins,
            "histnorm": histnorm,
            "marker_size": marker_size,
            "use_gl": use_gl,
            "max_points": max_points,
            "downsample_seed": downsample_seed,
            "fail_color": fail_color,
            "pass_color": pass_color,
            "title": title,
            "width": width,
            "height": height,
        }
        plot_fn = self._plot_1D if n_dims == 1 else self._plot_2D if n_dims == 2 else self._plot_nD
        return plot_fn(events, plot_dims, gate_result, **plot_kwargs)

    def _plot_1D(self, events: ad.AnnData, dimensions: list[str], gate_result: np.ndarray, **kwargs: Any) -> go.Figure:
        dim = dimensions[0]
        data = np.asarray(events[:, dim].X).ravel()
        pass_mask = gate_result.astype(bool)
        fail_mask = ~pass_mask

        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = _downsample_indices(data.shape[0], max_points, downsample_seed)
        if downsample_idx is not None:
            data = data[downsample_idx]
            pass_mask = pass_mask[downsample_idx]
            fail_mask = fail_mask[downsample_idx]

        nbins = int(kwargs.get("hist_nbins", 100))
        histnorm = str(kwargs.get("histnorm", "probability"))
        fail_color = str(kwargs.get("fail_color", "rgba(255, 0, 0, 0.35)"))
        pass_color = str(kwargs.get("pass_color", "rgba(0, 255, 0, 0.45)"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"BooleanGate: {self.gate_name} ({self.expression})"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        fig = go.Figure()
        if fail_mask.any():
            fig.add_trace(go.Histogram(
                x=data[fail_mask],
                nbinsx=nbins,
                name="Fail",
                marker=dict(color=fail_color),
                opacity=0.7,
                histnorm=histnorm,
            ))
        if pass_mask.any():
            fig.add_trace(go.Histogram(
                x=data[pass_mask],
                nbinsx=nbins,
                name="Pass",
                marker=dict(color=pass_color),
                opacity=0.8,
                histnorm=histnorm,
            ))

        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=dim)
        fig.update_yaxes(title=histnorm.title() if histnorm else "Count")
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_2D(self, events: ad.AnnData, dimensions: list[str], gate_result: np.ndarray, **kwargs: Any) -> go.Figure:
        x_dim, y_dim = dimensions
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        pass_mask = gate_result.astype(bool)
        fail_mask = ~pass_mask

        marker_size = int(kwargs.get("marker_size", 3))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        fail_color = str(kwargs.get("fail_color", "rgba(255, 0, 0, 0.3)"))
        pass_color = str(kwargs.get("pass_color", "rgba(0, 255, 0, 0.5)"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"BooleanGate: {self.gate_name} ({self.expression})"
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)

        downsample_idx = _downsample_indices(x_data.shape[0], max_points, downsample_seed)
        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]
            pass_mask = pass_mask[downsample_idx]
            fail_mask = fail_mask[downsample_idx]

        fig = go.Figure()
        trace_cls = go.Scattergl if use_gl else go.Scatter

        if fail_mask.any():
            fig.add_trace(trace_cls(
                x=x_data[fail_mask],
                y=y_data[fail_mask],
                mode="markers",
                marker=dict(size=marker_size, color=fail_color, line=dict(width=0)),
                name="Fail",
                hoverinfo="x+y",
            ))

        if pass_mask.any():
            fig.add_trace(trace_cls(
                x=x_data[pass_mask],
                y=y_data[pass_mask],
                mode="markers",
                marker=dict(size=marker_size, color=pass_color, line=dict(width=0)),
                name="Pass",
                hoverinfo="x+y",
            ))

        fig.update_xaxes(title=x_dim)
        fig.update_yaxes(title=y_dim)
        return _format_gate_plot(fig, title=plot_title, width=plot_width, height=plot_height)

    def _plot_nD(self, events: ad.AnnData, dimensions: list[str], gate_result: np.ndarray, **kwargs: Any) -> go.Figure:
        fig, pairs = _create_subplot_grid(len(dimensions))
        n_cols = int(np.ceil(np.sqrt(len(pairs))))

        marker_size = int(kwargs.get("marker_size", 3))
        use_gl = bool(kwargs.get("use_gl", True))
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        fail_color = str(kwargs.get("fail_color", "rgba(255, 0, 0, 0.25)"))
        pass_color = str(kwargs.get("pass_color", "rgba(0, 255, 0, 0.4)"))
        title = kwargs.get("title")
        plot_title = title if title is not None else f"BooleanGate: {self.gate_name} ({self.expression}) (Pairwise Projections)"

        pass_mask = gate_result.astype(bool)
        fail_mask = ~pass_mask
        downsample_idx = _downsample_indices(events.n_obs, max_points, downsample_seed)
        if downsample_idx is not None:
            pass_mask = pass_mask[downsample_idx]
            fail_mask = fail_mask[downsample_idx]

        trace_cls = go.Scattergl if use_gl else go.Scatter
        for idx, (i, j) in enumerate(pairs):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            x_dim = dimensions[i]
            y_dim = dimensions[j]
            x_data = np.asarray(events[:, x_dim].X).ravel()
            y_data = np.asarray(events[:, y_dim].X).ravel()
            if downsample_idx is not None:
                x_data = x_data[downsample_idx]
                y_data = y_data[downsample_idx]

            if fail_mask.any():
                fig.add_trace(
                    trace_cls(
                        x=x_data[fail_mask],
                        y=y_data[fail_mask],
                        mode="markers",
                        marker=dict(size=marker_size, color=fail_color, line=dict(width=0)),
                        name="Fail",
                        hoverinfo="x+y",
                        showlegend=idx == 0,
                    ),
                    row=row, col=col,
                )
            if pass_mask.any():
                fig.add_trace(
                    trace_cls(
                        x=x_data[pass_mask],
                        y=y_data[pass_mask],
                        mode="markers",
                        marker=dict(size=marker_size, color=pass_color, line=dict(width=0)),
                        name="Pass",
                        hoverinfo="x+y",
                        showlegend=idx == 0,
                    ),
                    row=row, col=col,
                )

            fig.update_xaxes(title=x_dim, row=row, col=col)
            fig.update_yaxes(title=y_dim, row=row, col=col)

        n_plots = len(pairs)
        n_rows = int(np.ceil(n_plots / n_cols))
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_height = int(height) if height is not None else 400 * n_rows
        plot_width = int(width) if width is not None else 400 * n_cols

        return _format_gate_plot(
            fig,
            title=plot_title,
            width=plot_width,
            height=plot_height,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize gate to dictionary."""
        base = super().to_dict()
        base["params"]["variables"] = list(self.variables)  # Add for reference
        return base
