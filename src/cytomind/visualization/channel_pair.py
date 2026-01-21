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
    log_scale: bool = True,
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
    log_scale : bool
        Whether to use log scale for the color axis.
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

    # Update colorscale via coloraxis in layout
    fig.update_layout(
        coloraxis=dict(colorscale=colorscale)
    )

    # Apply log scale to z-axis if requested
    if log_scale or coloraxis_log:
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
