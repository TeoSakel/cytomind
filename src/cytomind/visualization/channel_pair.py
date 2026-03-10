from __future__ import annotations
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .transforms import apply_transform


def build_histogram2d_with_marginals(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int = 128,
    range: np.ndarray | None = None,
    transformation: str = "identity",
    coloraxis_log: bool = False,
    colorscale: Any = "viridis",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int = 750,
    height: int = 750,
) -> go.Figure:
    """Build a 2D density heatmap with 1D marginals using plotly express.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of equal length; already masked and transformed.
    nbins : int
        Number of histogram bins for both axes and marginals.
    colorscale : Any
        Plotly colorscale.
    range : Optional[np.ndarray]
        Axis ranges [[y_min, y_max], [x_min, x_max]] for the axes.
    transformation : str
        Transformation to apply to the data.
    coloraxis_log : bool
        Whether to use log scale for the coloraxis.
    title, xaxis_title, yaxis_title : Optional[str]
        Layout labels.
    width, height : int
        Figure size in pixels.

    Returns
    -------
    go.Figure
        Figure with 2D density heatmap and marginal distributions.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    if x.size == 0:
        raise ValueError("x and y must be non-empty arrays")

    # Apply per-channel transform
    x_t = apply_transform(x, transformation=transformation)
    y_t = apply_transform(y, transformation=transformation)

    # Create density heatmap with marginals
    fig = px.density_heatmap(
        x=x_t,
        y=y_t,
        nbinsx=nbins,
        nbinsy=nbins,
        marginal_x="histogram",
        marginal_y="histogram",
        title=title,
        labels={"x": xaxis_title, "y": yaxis_title},
    )

    # Apply log scale to z-axis if requested
    if coloraxis_log:
        fig.update_layout(coloraxis=dict(colorscale=colorscale, type="log"))
    else:
        fig.update_layout(coloraxis=dict(colorscale=colorscale))

    # Set axis ranges if provided
    if range is not None:
        range_y = range[0]
        range_x = range[1]
        fig.update_xaxes(range=range_x)
        fig.update_yaxes(range=range_y)

    # Update layout
    fig.update_layout(
        width=width,
        height=height,
    )

    return fig


def build_scatter2d_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int = 50,
    nbins_marginal: int = 50,
    transformation: str = "identity",
    coloraxis_log: bool = False,
    colorscale: Any = "viridis",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int = 800,
    height: int = 700,
    marker_size: int = 5,
) -> go.Figure:
    """Build a 2D scatter plot with grid-based density coloring and marginal histograms.

    Points are colored by the normalized number of points in their grid bin, providing
    a simple and fast density visualization.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of equal length with transformed coordinates.
    nbins : int
        Number of bins for 2D grid density calculation (default: 50)
    nbins_marginal : int
        Number of bins for marginal histograms (default: 50)
    transformation : str
        Transformation to apply to the data (default: "identity")
    coloraxis_log : bool
        Whether to apply a log transform to the density values (default: False)
    colorscale : Any
        Plotly colorscale (default: "viridis")
    title : str | None
        Figure title
    xaxis_title : str | None
        X-axis label
    yaxis_title : str | None
        Y-axis label
    width : int
        Figure width in pixels (default: 800)
    height : int
        Figure height in pixels (default: 700)
    marker_size : int
        Marker size (default: 5)

    Returns
    -------
    go.Figure
        Plotly figure with scatter plot and marginals
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    if x.size == 0:
        raise ValueError("x and y must be non-empty arrays")

    # Apply per-channel transform
    x_t = apply_transform(x, transformation=transformation)
    y_t = apply_transform(y, transformation=transformation)

    # Create 2D histogram for density coloring
    h, x_edges, y_edges = np.histogram2d(y_t, x_t, bins=nbins)

    # Assign each point to its bin and get the density value
    x_indices = np.digitize(x_t, x_edges) - 1
    y_indices = np.digitize(y_t, y_edges) - 1

    # Clip indices to valid range
    x_indices = np.clip(x_indices, 0, h.shape[1] - 1)
    y_indices = np.clip(y_indices, 0, h.shape[0] - 1)

    # Get density (bin count) for coloring
    density_counts = h[y_indices, x_indices]
    max_count = float(np.max(density_counts)) if density_counts.size else 0.0
    if max_count <= 0.0:
        density_norm = np.zeros_like(density_counts, dtype=float)
    else:
        if coloraxis_log:
            density_norm = np.log10(density_counts + 1.0) / np.log10(max_count + 1.0)
        else:
            density_norm = density_counts / max_count

    # Plot lower density first so higher density sits on top
    sort_idx = np.argsort(density_norm)
    x_t = x_t[sort_idx]
    y_t = y_t[sort_idx]
    density_norm = density_norm[sort_idx]

    # Create figure with marginals using subplot approach
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.2, 0.8],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Main scatter plot (bottom-left)
    fig.add_trace(
        go.Scattergl(
            x=x_t,
            y=y_t,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=density_norm,
                colorscale=colorscale,
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(
                    title="Density",
                    x=1.12,
                    len=0.6,
                    y=0.4,
                ),
                line=dict(width=0),
                showscale=True,
            ),
            name="Points",
            showlegend=False,
            hovertemplate=(
                f"{xaxis_title or 'X'}: %{{x:.3g}}<br>"
                f"{yaxis_title or 'Y'}: %{{y:.3g}}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    histnorm = kwargs.get("histnorm", "probability")

    # Top marginal histogram (X-axis)
    fig.add_trace(
        go.Histogram(
            x=x_t,
            nbinsx=nbins_marginal,
            marker=dict(color="steelblue", line=dict(color="black", width=0.5)),
            name="X marginal",
            showlegend=False,
            histnorm=histnorm,
            hovertemplate="Frequency: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Right marginal histogram (Y-axis)
    fig.add_trace(
        go.Histogram(
            y=y_t,
            nbinsy=nbins_marginal,
            marker=dict(color="steelblue", line=dict(color="black", width=0.5)),
            name="Y marginal",
            showlegend=False,
            histnorm=histnorm,
            hovertemplate="Frequency: %{x}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text=xaxis_title, row=2, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="", row=2, col=2, showticklabels=False)

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
        hovermode="closest",
    )

    return fig
