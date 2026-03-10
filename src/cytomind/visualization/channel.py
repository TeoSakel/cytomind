from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .transforms import apply_transform


def build_histogram1d(
    values: np.ndarray,
    *,
    nbins: int = 128,
    color: str | None = None,
    transformation: str = "identity",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = "Frequency",
    histnorm: str | None = "probability",
    width: int = 700,
    height: int = 500,
) -> go.Figure:
    """Build a simple 1D histogram figure.

    Parameters
    ----------
    values : np.ndarray
        1D array; already masked/transformed.
    nbins : int
        Number of bins.
    color : str | None
        Bar color.
    title, xaxis_title, yaxis_title : str | None
        Layout labels.
    histnorm : str | None
        Histogram normalization mode. Default is "probability" for frequency display.
        Use None for absolute counts.

    Returns
    -------
    go.Figure
        Plotly figure with a single histogram trace.
    """

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    data = apply_transform(values, transformation=transformation)

    if yaxis_title is None:
        yaxis_title = "Frequency"

    trace = go.Histogram(x=data, nbinsx=nbins, histnorm=histnorm)
    if color:
        trace.marker = dict(color=color, line=dict(color="black", width=1))
    else:
        trace.marker = dict(line=dict(color="black", width=1))

    fig = go.Figure(data=trace)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
    )
    return fig
