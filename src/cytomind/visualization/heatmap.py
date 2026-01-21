from __future__ import annotations
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_matrix_heatmap(
    matrix: np.ndarray | pd.DataFrame,
    *,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    colorscale: Any = "RdBu",
    zmid: float = 0.0,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int = 700,
    height: int = 700,
) -> go.Figure:
    """Build a matrix heatmap figure (diverging by default).

    Parameters
    ----------
    matrix : np.ndarray
        2D array of values.
    x_labels, y_labels : Optional[Sequence[str]]
        Axis labels for columns/rows respectively.
    colorscale : Any
        Plotly colorscale; default diverging.
    zmid : Optional[float]
        Center value for diverging scale. Use None for non-diverging.
    title, xaxis_title, yaxis_title : Optional[str]
        Layout labels.

    Returns
    -------
    go.Figure
        Plotly figure containing the heatmap.
    """

    if isinstance(matrix, pd.DataFrame):
        if x_labels is None:
            x_labels = matrix.columns.tolist()
        if y_labels is None:
            y_labels = matrix.index.tolist()
        matrix = matrix.values

    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("matrix must be a 2D numpy array or pandas DataFrame")

    z = matrix
    # Symmetric limits around zero when zmid provided
    if zmid is not None:
        max_abs = float(np.nanmax(np.abs(z))) if z.size else 1.0
        zmin, zmax = -max_abs, max_abs
    else:
        zmin = float(np.nanmin(z)) if z.size else 0.0
        zmax = float(np.nanmax(z)) if z.size else 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            text=z,
            texttemplate="%{z:.2f}",
            hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
    )

    return fig
