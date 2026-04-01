"""Shared plotting utilities for gate visualization."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from cytomind.domain.constants import FloatArray
else:
    FloatArray = object


def _compute_density_colors(
    x: np.ndarray,
    y: np.ndarray,
    nbins: int = 50,
    log_scale: bool = True,
) -> np.ndarray:
    """Compute density-based colors for scatter plot points."""
    if len(x) == 0:
        return np.array([])

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=nbins)
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, nbins - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, nbins - 1)
    density = hist[x_idx, y_idx]

    if log_scale:
        density = np.log10(density + 1)

    return density


def _create_scatter_trace(
    x: np.ndarray,
    y: np.ndarray,
    density_colors: np.ndarray | None = None,
    marker_size: int = 3,
    colorscale: str = "Viridis",
    color_midpoint: float | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
    showlegend: bool = False,
    name: str | None = "Events",
    use_gl: bool = True,
) -> go.Scatter | go.Scattergl:
    """Create a scatter trace with optional density-based coloring."""
    if density_colors is not None and len(density_colors) > 0:
        sort_idx = np.argsort(density_colors)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        colors_sorted = density_colors[sort_idx]

        marker = dict(
            size=marker_size,
            color=colors_sorted,
            colorscale=colorscale,
            cmid=color_midpoint,
            cmin=color_min,
            cmax=color_max,
            showscale=False,
            line=dict(width=0),
        )
    else:
        x_sorted = x
        y_sorted = y
        marker = dict(
            size=marker_size,
            color="rgba(100, 100, 100, 0.5)",
            line=dict(width=0),
        )

    trace_cls = go.Scattergl if use_gl else go.Scatter
    return trace_cls(
        x=x_sorted,
        y=y_sorted,
        mode="markers",
        marker=marker,
        showlegend=showlegend,
        name=name if name is not None else "Events",
        hoverinfo="x+y",
    )


def _create_rectangle_trace(
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    data_x_range: tuple[float, float],
    data_y_range: tuple[float, float],
    line_color: str = "red",
    line_width: int = 2,
    name: str | None = "Gate",
    use_gl: bool = False,
) -> go.Scatter | go.Scattergl:
    """Create a rectangle boundary trace."""
    x_low = x_min if x_min is not None else data_x_range[0]
    x_high = x_max if x_max is not None else data_x_range[1]
    y_low = y_min if y_min is not None else data_y_range[0]
    y_high = y_max if y_max is not None else data_y_range[1]

    x_coords = [x_low, x_high, x_high, x_low, x_low]
    y_coords = [y_low, y_low, y_high, y_high, y_low]

    trace_cls = go.Scattergl if use_gl else go.Scatter
    return trace_cls(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        showlegend=True if name is not None else False,
        name=name if name is not None else "Gate",
        hoverinfo="skip",
    )


def _create_polygon_trace(
    vertices: FloatArray,
    line_color: str = "red",
    line_width: int = 2,
    fill: bool = True,
    fill_color: str = "rgba(255, 0, 0, 0.1)",
    name: str | None = "Gate",
    use_gl: bool = False,
) -> go.Scatter | go.Scattergl:
    """Create a polygon boundary trace."""
    x_coords = np.append(vertices[:, 0], vertices[0, 0])
    y_coords = np.append(vertices[:, 1], vertices[0, 1])

    trace_cls = go.Scattergl if use_gl else go.Scatter
    trace = trace_cls(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        showlegend=True if name is not None else False,
        name=name if name is not None else "Gate",
        hoverinfo="skip",
    )

    if fill:
        trace.update(fill="toself", fillcolor=fill_color)

    return trace


def _create_ellipse_trace(
    center: FloatArray,
    cov_matrix: FloatArray,
    distance_square: float,
    n_points: int = 100,
    line_color: str = "red",
    line_width: int = 2,
    fill: bool = True,
    fill_color: str = "rgba(255, 0, 0, 0.1)",
    name: str | None = "Gate",
    use_gl: bool = False,
) -> go.Scatter | go.Scattergl:
    """Create an ellipse boundary trace from covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    a = np.sqrt(eigenvalues[0] * distance_square)
    b = np.sqrt(eigenvalues[1] * distance_square)

    theta = np.linspace(0, 2 * np.pi, n_points)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)

    ellipse_points = np.column_stack([ellipse_x, ellipse_y])
    rotated_points = ellipse_points @ eigenvectors.T
    x_coords = rotated_points[:, 0] + center[0]
    y_coords = rotated_points[:, 1] + center[1]

    trace_cls = go.Scattergl if use_gl else go.Scatter
    trace = trace_cls(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        showlegend=True if name is not None else False,
        name=name if name is not None else "Gate",
        hoverinfo="skip",
    )

    if fill:
        trace.update(fill="toself", fillcolor=fill_color)

    return trace


def _create_subplot_grid(n_dims: int) -> tuple[go.Figure, list[tuple[int, int]]]:
    """Create a subplot grid for pairwise dimension projections."""
    pairs = [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]
    n_plots = len(pairs)

    if n_plots == 0:
        raise ValueError("Need at least 2 dimensions for pairwise projections")

    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Dims {i} vs {j}" for i, j in pairs],
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    return fig, pairs


def _format_gate_plot(
    fig: go.Figure,
    title: str | None = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Apply consistent styling to a gate plot."""
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template="plotly_white",
        hovermode="closest",
    )
    return fig
