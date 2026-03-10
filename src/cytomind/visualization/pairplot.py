"""Pairplot visualization for multi-channel flow cytometry data."""
from __future__ import annotations
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .transforms import apply_transform


def build_pairplot(
    data: np.ndarray,
    channel_names: list[str],
    *,
    transformation: str = "logicle",
    nbins: int = 50,
    colorscale: Any = "viridis",
    title: str | None = None,
    width: int = 1200,
    height: int = 1200,
) -> go.Figure:
    """Build a pairplot with scatter plots in lower triangle and histograms on diagonal.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_events, n_channels) with event data.
    channel_names : list[str]
        List of channel names matching the columns of data.
    transformation : str
        Transformation to apply: "logicle", "identity", "asinh", etc. (default: "logicle")
    nbins : int
        Number of bins for histograms (default: 50)
    colorscale : Any
        Plotly colorscale for density scatter plots (default: "viridis")
    title : str | None
        Figure title
    width : int
        Figure width in pixels (default: 1200)
    height : int
        Figure height in pixels (default: 1200)

    Returns
    -------
    go.Figure
        Plotly figure with pairplot layout
    """

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if data.shape[1] != len(channel_names):
        raise ValueError("Number of channel_names must match the number of columns in data")
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one event")

    n_channels = len(channel_names)

    # Create subplots for pairplot
    # Size of grid is n_channels x n_channels
    # We'll use only diagonal and lower triangle, leaving upper triangle empty
    fig = make_subplots(
        rows=n_channels,
        cols=n_channels,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
    )

    # Apply transformation to all channels
    data_transformed = np.zeros_like(data, dtype=float)
    for i in range(n_channels):
        data_transformed[:, i] = apply_transform(data[:, i], transformation=transformation)

    # Add traces to the figure
    for i in range(n_channels):
        for j in range(n_channels):
            row = i + 1
            col = j + 1

            # Diagonal: histograms
            if i == j:
                channel_data = data_transformed[:, i]
                histnorm = kwargs.get("histnorm", "probability")

                fig.add_trace(
                    go.Histogram(
                        x=channel_data,
                        nbinsx=nbins,
                        name=channel_names[i],
                        marker=dict(color="steelblue", line=dict(color="black", width=0.5)),
                        showlegend=False,
                        histnorm=histnorm,
                        hovertemplate=f"{channel_names[i]}: %{{x:.3g}}<br>Frequency: %{{y}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

                # Add axis labels only for diagonal
                yaxis_title = histnorm.title() if histnorm else "Count"
                fig.update_xaxes(title_text=channel_names[i], row=row, col=col)
                fig.update_yaxes(title_text=yaxis_title, row=row, col=col)

            # Lower triangle: scatter plots with density coloring
            elif i > j:
                x_data = data_transformed[:, j]
                y_data = data_transformed[:, i]

                # Create 2D histogram for density coloring
                h, x_edges, y_edges = np.histogram2d(y_data, x_data, bins=nbins)

                # Find density value for each point by binning
                x_indices = np.digitize(x_data, x_edges) - 1
                y_indices = np.digitize(y_data, y_edges) - 1

                # Clip indices to valid range
                x_indices = np.clip(x_indices, 0, h.shape[1] - 1)
                y_indices = np.clip(y_indices, 0, h.shape[0] - 1)

                # Get density values for coloring
                densities = h[y_indices, x_indices]

                fig.add_trace(
                    go.Scattergl(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=densities,
                            colorscale=colorscale,
                            showscale=False,
                            line=dict(width=0),
                        ),
                        name=f"{channel_names[j]} vs {channel_names[i]}",
                        showlegend=False,
                        hovertemplate=(
                            f"{channel_names[j]}: %{{x:.3g}}<br>"
                            f"{channel_names[i]}: %{{y:.3g}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

                # Add axis labels for lower triangle
                if j == 0:  # Left edge
                    fig.update_yaxes(title_text=channel_names[i], row=row, col=col)
                if i == n_channels - 1:  # Bottom edge
                    fig.update_xaxes(title_text=channel_names[j], row=row, col=col)

    # Update layout
    fig.update_layout(
        title=title or "Pairplot",
        width=width,
        height=height,
        showlegend=False,
        hovermode="closest",
    )

    # Hide upper triangle and empty cells
    for i in range(n_channels):
        for j in range(n_channels):
            if i < j:  # Upper triangle
                fig.update_xaxes(visible=False, row=i + 1, col=j + 1)
                fig.update_yaxes(visible=False, row=i + 1, col=j + 1)

    return fig
