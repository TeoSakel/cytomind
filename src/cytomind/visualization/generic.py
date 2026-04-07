from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.colors import get_colorscale, sample_colorscale
from plotly.exceptions import PlotlyError
from plotly.subplots import make_subplots

from .gates import (
    _compute_density_colors,
    _create_scatter_trace,
    _format_gate_plot,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from cytomind.domain.constants import BooleanArray, FloatArray
else:
    AnnData = object
    BooleanArray = object
    FloatArray = object


class GenericPlotEngine:
    """Stateless generic plotting engine used by gates and pipeline APIs."""

    @staticmethod
    def _facet_grid_size(n_panels: int) -> tuple[int, int]:
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        return n_rows, n_cols

    @staticmethod
    def _legend_marker_size(marker_size: int) -> int:
        return max(int(marker_size) + 4, 8)

    @staticmethod
    def _opaque_color(color: str) -> str:
        match = re.fullmatch(
            r"rgba\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*[0-9.]+\s*\)",
            color.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            return color
        red, green, blue = match.groups()
        return f"rgb({red}, {green}, {blue})"

    def _add_scatter_legend_proxy(
        self,
        fig: go.Figure,
        *,
        name: str,
        color: str,
        marker_size: int,
        row: int | None = None,
        col: int | None = None,
        legendgroup: str | None = None,
    ) -> None:
        trace = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=name,
            showlegend=True,
            hoverinfo="skip",
            legendgroup=legendgroup or name,
            marker=dict(
                size=self._legend_marker_size(marker_size),
                color=self._opaque_color(color),
                opacity=1.0,
                line=dict(width=0),
            ),
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    @staticmethod
    def resolve_plot_dimensions(
        requested_dimensions: Sequence[str] | None,
        *,
        available_dimensions: Sequence[str],
        owner_name: str = "plot",
        min_dims: int = 1,
        max_dims: int | None = None,
    ) -> list[str]:
        dims = list(available_dimensions) if requested_dimensions is None else list(requested_dimensions)
        if not dims:
            raise ValueError(f"{owner_name} requires at least one dimension")
        dim_set = set(dims)
        if len(dim_set) != len(dims):
            raise ValueError(f"{owner_name} dimensions must be unique")
        if len(dims) < min_dims:
            raise ValueError(f"{owner_name} requires at least {min_dims} dimensions")
        if max_dims is not None and len(dims) > max_dims:
            raise ValueError(f"{owner_name} supports at most {max_dims} dimensions")
        unknown = dim_set - set(available_dimensions)
        if unknown:
            raise ValueError(
                f"{owner_name} received dimensions not in available dimensions: {unknown}. "
                f"Available dimensions: {list(available_dimensions)}"
            )
        return dims

    def plot(
        self,
        *,
        events: AnnData,
        plot_dims: Sequence[str],
        mask: dict[str, BooleanArray] | None = None,
        primary_mask_values: np.ndarray | None = None,
        score_values: np.ndarray | None = None,
        decorator: Any | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Dispatch to the dimension-specific plotting entrypoint.

        Use this as the main public API when the number of plot dimensions is
        only known at runtime. For the concrete argument surface of each plot
        shape, see `plot_1d()`, `plot_2d()`, `plot_nd()`, and
        `plot_faceted()`.

        Shared kwargs used by the specialized methods include:
        - `color_by`: `"density"` for density-colored scatter, `"mask"` for
          discrete pass/fail or group coloring, `"score"` for continuous
          score-based coloring.
        - `use_gl`: when `True`, use Plotly WebGL scatter traces
          (`Scattergl`), which scale better to large point clouds; when
          `False`, use regular SVG scatter traces.
        - `max_points` and `downsample_seed`: control optional random
          downsampling before plotting.
        - `title`, `width`, and `height`: override the default figure layout.
        """
        n_dims = len(plot_dims)
        if n_dims == 1:
            return self.plot_1d(
                events=events,
                plot_dims=plot_dims,
                mask=mask,
                primary_mask_values=primary_mask_values,
                score_values=score_values,
                decorator=decorator,
                **kwargs,
            )
        if n_dims == 2:
            return self.plot_2d(
                events=events,
                plot_dims=plot_dims,
                mask=mask,
                primary_mask_values=primary_mask_values,
                score_values=score_values,
                decorator=decorator,
                **kwargs,
            )
        return self.plot_nd(
            events=events,
            plot_dims=plot_dims,
            mask=mask,
            primary_mask_values=primary_mask_values,
            score_values=score_values,
            decorator=decorator,
            **kwargs,
        )

    def plot_faceted(
        self,
        *,
        panels: Sequence[Mapping[str, Any]],
        plot_dims: Sequence[str],
        **layout_kwargs: Any,
    ) -> go.Figure:
        """Render multiple preloaded panels into a shared faceted figure.

        The pipeline is expected to resolve samples, filtering, and optional
        gate-derived payloads first, then pass one panel payload per facet.

        Faceted plots currently support 1D and 2D panels. Axes are shared
        across facets, and continuous colorbars are normalized across panels.
        `marginals=True` is intentionally rejected here for now.
        """
        if not panels:
            raise ValueError("plot_faceted() requires at least one panel.")
        if bool(layout_kwargs.get("marginals", False)):
            raise ValueError("marginals=True is not yet supported for faceted multi-sample plots.")

        n_rows, n_cols = self._facet_grid_size(len(panels))
        titles = [str(panel.get("title", f"Panel {idx + 1}")) for idx, panel in enumerate(panels)]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles,
            shared_xaxes="all", shared_yaxes="all" # pyright: ignore[reportArgumentType]
        )

        showlegend = False
        colorbar_traces: list[go.BaseTraceType] = [] # pyright: ignore[reportAttributeAccessIssue]
        x_ranges: list[tuple[float, float]] = []
        y_ranges: list[tuple[float, float]] = []
        for idx, panel in enumerate(panels, start=1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            panel_events = panel["events"]
            if len(plot_dims) >= 1:
                panel_x = np.asarray(panel_events[:, plot_dims[0]].X).ravel()
                if panel_x.size:
                    x_ranges.append((float(np.min(panel_x)), float(np.max(panel_x))))
            if len(plot_dims) >= 2:
                panel_y = np.asarray(panel_events[:, plot_dims[1]].X).ravel()
                if panel_y.size:
                    y_ranges.append((float(np.min(panel_y)), float(np.max(panel_y))))
            panel_fig = self.plot(
                events=panel_events,
                plot_dims=plot_dims,
                mask=panel.get("mask"),
                primary_mask_values=panel.get("primary_mask_values"),
                score_values=panel.get("score_values"),
                decorator=panel.get("decorator"),
                color_by=panel.get("color_by", layout_kwargs.get("color_by", "density")),
                **layout_kwargs,
            )
            showlegend = showlegend or any(getattr(trace, "showlegend", False) for trace in panel_fig.data)
            for trace in panel_fig.data:
                fig.add_trace(trace, row=row, col=col)
                marker = getattr(trace, "marker", None)
                if marker is not None and getattr(marker, "showscale", False):
                    colorbar_traces.append(fig.data[-1])

            if len(plot_dims) == 1:
                fig.update_xaxes(title_text=plot_dims[0], row=row, col=col)
                yaxis_title = getattr(panel_fig.layout.yaxis.title, "text", None) or "Count"
                fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
            else:
                fig.update_xaxes(title_text=plot_dims[0], row=row, col=col)
                fig.update_yaxes(title_text=plot_dims[1], row=row, col=col)

        if colorbar_traces:
            all_colors: list[np.ndarray] = []
            colorbar_title: str | None = None
            for trace in colorbar_traces:
                marker = getattr(trace, "marker", None)
                if marker is None:
                    continue
                colors = getattr(marker, "color", None)
                if colors is None:
                    continue
                arr = np.asarray(colors, dtype=float).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    all_colors.append(arr)
                colorbar = getattr(marker, "colorbar", None)
                if colorbar_title is None and colorbar is not None:
                    title_obj = getattr(colorbar, "title", None)
                    title_text = getattr(title_obj, "text", None)
                    if title_text:
                        colorbar_title = str(title_text)
            if all_colors:
                finite_colors = np.concatenate(all_colors)
                cmin = float(np.min(finite_colors))
                cmax = float(np.max(finite_colors))
                cmid: float | None = None
                if any(panel.get("color_by") == "score" for panel in panels):
                    max_abs = max(abs(cmin), abs(cmax))
                    cmin = -max_abs
                    cmax = max_abs
                    cmid = 0.0
                for trace_idx, trace in enumerate(colorbar_traces):
                    marker = getattr(trace, "marker", None)
                    if marker is None:
                        continue
                    marker.cmin = cmin
                    marker.cmax = cmax
                    marker.cmid = cmid
                    marker.showscale = trace_idx == len(colorbar_traces) - 1
                    if marker.showscale:
                        marker.colorbar = dict(
                            title=colorbar_title,
                            x=1.02,
                            y=0.5,
                            len=0.9,
                        )

        if len(plot_dims) >= 1 and x_ranges:
            global_x_min = min(start for start, _ in x_ranges)
            global_x_max = max(end for _, end in x_ranges)
            x_pad = 0.01 * max(global_x_max - global_x_min, 1e-9)
            fig.update_xaxes(range=[global_x_min - x_pad, global_x_max + x_pad])
        if len(plot_dims) >= 2 and y_ranges:
            global_y_min = min(start for start, _ in y_ranges)
            global_y_max = max(end for _, end in y_ranges)
            y_pad = 0.01 * max(global_y_max - global_y_min, 1e-9)
            fig.update_yaxes(range=[global_y_min - y_pad, global_y_max + y_pad])

        fig.update_layout(
            title=layout_kwargs.get("title"),
            width=layout_kwargs.get("width"),
            height=layout_kwargs.get("height"),
            showlegend=showlegend,
        )
        return fig

    def _decorator_title(self, decorator: Any | None, plot_dims: Sequence[str]) -> str:
        if decorator is not None and hasattr(decorator, "default_plot_title"):
            return str(decorator.default_plot_title(plot_dims))
        if len(plot_dims) > 2:
            return "Plot (Pair Plot)"
        return "Plot"

    def _decorator_ratio_position_1d(
        self,
        decorator: Any | None,
        dim: str,
        data: FloatArray,
        **kwargs: Any,
    ) -> float | None:
        if decorator is not None and hasattr(decorator, "ratio_position_1d"):
            return decorator.ratio_position_1d(dim, data, **kwargs)
        if np.asarray(data).size == 0:
            return None
        return float(np.mean(data))

    def _decorator_ratio_position_2d(
        self,
        decorator: Any | None,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        if decorator is not None and hasattr(decorator, "ratio_position_2d"):
            return decorator.ratio_position_2d(plot_dims, x_data, y_data, **kwargs)
        if np.asarray(x_data).size == 0 or np.asarray(y_data).size == 0:
            return None
        return float(np.mean(x_data)), float(np.mean(y_data))

    def _decorate_plot_1d(
        self,
        decorator: Any | None,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        **kwargs: Any,
    ) -> None:
        if decorator is not None and hasattr(decorator, "decorate_plot_1d"):
            decorator.decorate_plot_1d(fig, dim, data, **kwargs)

    def _decorate_plot_2d(
        self,
        decorator: Any | None,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> None:
        if decorator is not None and hasattr(decorator, "decorate_plot_2d"):
            decorator.decorate_plot_2d(fig, plot_dims, x_data, y_data, **kwargs)

    def _decorate_plot_2d_marginals(
        self,
        decorator: Any | None,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> None:
        if decorator is not None and hasattr(decorator, "decorate_plot_2d_marginals"):
            decorator.decorate_plot_2d_marginals(fig, plot_dims, x_data, y_data, **kwargs)

    def _compute_ratio(
        self,
        *,
        primary_mask_values: np.ndarray | None,
    ) -> float | None:
        if primary_mask_values is None:
            return None
        arr = np.asarray(primary_mask_values, dtype=bool).ravel()
        if arr.size == 0:
            return None
        return float(arr.mean())

    @staticmethod
    def _downsample_indices(n_points: int, max_points: int, seed: int) -> np.ndarray | None:
        if max_points <= 0 or n_points <= max_points:
            return None
        rng = np.random.default_rng(seed)
        return rng.choice(n_points, size=max_points, replace=False)

    @staticmethod
    def _normalize_histogram(
        counts: np.ndarray,
        widths: np.ndarray,
        histnorm: str,
        total: int,
    ) -> FloatArray:
        values = counts.astype(float, copy=False)
        if total <= 0:
            return np.zeros_like(values, dtype=float)
        if histnorm in ("density", "probability density"):
            denom = widths * float(total)
            return np.divide(values, denom, out=np.zeros_like(values, dtype=float), where=denom > 0)
        if histnorm == "percent":
            return values / float(total) * 100.0
        if histnorm == "probability":
            return values / float(total)
        return values

    def _stacked_histogram_profile(
        self,
        values: FloatArray,
        pass_mask: BooleanArray,
        fail_mask: BooleanArray,
        nbins: int | str,
        histnorm: str,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float | None]:
        if np.asarray(values).size == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), None
        edges = self._resolve_histogram_bin_edges(values, nbins)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0
        pass_counts, _ = np.histogram(values[pass_mask], bins=edges)
        fail_counts, _ = np.histogram(values[fail_mask], bins=edges)
        pass_heights = self._normalize_histogram(pass_counts, widths, histnorm, int(np.asarray(values).size))
        fail_heights = self._normalize_histogram(fail_counts, widths, histnorm, int(np.asarray(values).size))
        total_heights = pass_heights + fail_heights
        max_height = float(np.max(total_heights)) * 1.15 if total_heights.size else None
        return centers, widths, pass_heights, fail_heights, max_height

    @staticmethod
    def _density_curve(
        values: FloatArray,
        *,
        n_points: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.asarray([]), np.asarray([])
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            return np.asarray([vmin]), np.asarray([1.0])
        try:
            from scipy.stats import gaussian_kde

            grid = np.linspace(vmin, vmax, n_points)
            density = gaussian_kde(values)(grid)
            return grid, density
        except Exception:
            counts, edges = np.histogram(values, bins=min(64, max(8, values.size // 50)), density=True)
            centers = edges[:-1] + np.diff(edges) / 2.0
            return centers, counts

    def _add_stacked_histogram_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        pass_mask: BooleanArray,
        fail_mask: BooleanArray,
        *,
        nbins: int | str,
        histnorm: str,
        row: int | None,
        col: int | None,
        orientation: str,
        fail_color: str,
        pass_color: str,
    ) -> float | None:
        centers, widths, pass_heights, fail_heights, axis_max = self._stacked_histogram_profile(
            values, pass_mask, fail_mask, nbins, histnorm
        )
        if centers.size == 0:
            return axis_max
        if orientation == "h":
            fail_trace = go.Bar(x=fail_heights, y=centers, width=widths, orientation="h", name="Fail", marker=dict(color=fail_color), opacity=0.65, showlegend=False)
            pass_trace = go.Bar(x=pass_heights, y=centers, width=widths, orientation="h", name="Pass", marker=dict(color=pass_color), opacity=0.8, showlegend=False)
        else:
            fail_trace = go.Bar(x=centers, y=fail_heights, width=widths, name="Fail", marker=dict(color=fail_color), opacity=0.65, showlegend=False)
            pass_trace = go.Bar(x=centers, y=pass_heights, width=widths, name="Pass", marker=dict(color=pass_color), opacity=0.8, showlegend=False)
        if row is None or col is None:
            fig.add_trace(fail_trace)
            fig.add_trace(pass_trace)
        else:
            fig.add_trace(fail_trace, row=row, col=col)
            fig.add_trace(pass_trace, row=row, col=col)
        return axis_max

    def _add_density_marginal_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        pass_mask: BooleanArray,
        fail_mask: BooleanArray,
        *,
        row: int | None,
        col: int | None,
        orientation: str,
        fail_color: str,
        pass_color: str,
    ) -> float | None:
        axis_max: float | None = None
        for current_mask, color, name in (
            (fail_mask, fail_color, "Fail"),
            (pass_mask, pass_color, "Pass"),
        ):
            grid, density = self._density_curve(values[current_mask])
            if grid.size == 0:
                continue
            axis_max = max(axis_max or 0.0, float(np.max(density)))
            if orientation == "h":
                trace = go.Scatter(x=density, y=grid, mode="lines", line=dict(color=color), name=name, showlegend=False)
            else:
                trace = go.Scatter(x=grid, y=density, mode="lines", line=dict(color=color), name=name, showlegend=False)
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)
        return None if axis_max is None else axis_max * 1.15

    @staticmethod
    def _resolve_plot_mask_groups(
        mask: dict[str, BooleanArray] | None,
        n_obs: int,
        indices: np.ndarray | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        if mask is None:
            return []
        groups: list[tuple[str, np.ndarray]] = []
        for key, value in mask.items():
            arr = np.asarray(value, dtype=bool).ravel()
            if arr.shape[0] != n_obs:
                continue
            groups.append((key, arr if indices is None else arr[indices]))
        return groups

    def _resolve_discrete_mask_colors(
        self,
        n_masks: int,
        colorscale: str | Sequence[str] | None,
    ) -> list[str]:
        if n_masks <= 0:
            return []
        scale: str | Sequence[str] = "plotly3" if colorscale is None else colorscale
        if isinstance(scale, str):
            scale = self._resolve_named_colorscale(scale)
            if n_masks == 1:
                return [sample_colorscale(get_colorscale(scale), [0.5])[0]]  # pyright: ignore[reportReturnType]
            sample_points = np.linspace(0.0, 1.0, n_masks).tolist()
            return list(sample_colorscale(get_colorscale(scale), sample_points))  # pyright: ignore[reportReturnType]
        palette = [str(color) for color in scale]
        if len(palette) != n_masks:
            raise ValueError(f"Expected {n_masks} mask colors, got {len(palette)}")
        return palette

    @staticmethod
    def _resolve_named_colorscale(colorscale: str) -> str:
        scale = str(colorscale).strip()
        if not scale:
            raise ValueError("Received an empty colorscale name")
        aliases = {"plotly": "plotly3"}
        resolved = aliases.get(scale.casefold(), scale)
        try:
            get_colorscale(resolved)
        except PlotlyError as exc:
            raise ValueError(f"Received unsupported colorscale {scale!r}") from exc
        return resolved

    @staticmethod
    def _resolve_histogram_bin_edges(values: FloatArray, hist_nbins: int | str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.asarray([0.0, 1.0])
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if np.isclose(vmin, vmax):
            delta = 0.5 if np.isclose(vmin, 0.0) else max(abs(vmin) * 0.05, 1e-9)
            return np.asarray([vmin - delta, vmax + delta], dtype=float)
        return np.asarray(np.histogram_bin_edges(arr, bins=hist_nbins), dtype=float)

    def _histogram_bin_params(
        self,
        values: FloatArray,
        hist_nbins: int | str,
        *,
        axis: str,
    ) -> dict[str, dict[str, float]]:
        edges = self._resolve_histogram_bin_edges(values, hist_nbins)
        widths = np.diff(edges)
        size = float(widths[0]) if widths.size else 1.0
        if widths.size > 1 and not np.allclose(widths, size):
            size = float(np.mean(widths))
        return {axis: {"start": float(edges[0]), "end": float(edges[-1]), "size": size}}

    def _add_multi_1d_distribution_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        *,
        plot_type: str,
        hist_nbins: int | str,
        histnorm: str,
        mask_groups: Sequence[tuple[str, np.ndarray]],
        mask_colors: Sequence[str],
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
    ) -> str:
        histogram_bins = self._histogram_bin_params(values, hist_nbins, axis="xbins") if plot_type != "density" else {}
        for (mask_name, mask_values), color in zip(mask_groups, mask_colors, strict=False):
            selected = np.asarray(values)[mask_values]
            if selected.size == 0:
                continue
            if plot_type == "density":
                grid, density = self._density_curve(selected)
                trace = go.Scatter(x=grid, y=density, mode="lines", name=mask_name, line=dict(color=color), showlegend=showlegend)
            else:
                trace = go.Histogram(x=selected, name=mask_name, marker=dict(color=color), histnorm=histnorm, opacity=0.85, showlegend=showlegend, **histogram_bins)
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)
        if plot_type != "density":
            fig.update_layout(barmode="stack")
        return "Density" if plot_type == "density" else histnorm.title() if histnorm else "Count"

    def _add_multi_mask_scatter_traces(
        self,
        fig: go.Figure,
        x_data: np.ndarray,
        y_data: np.ndarray,
        *,
        mask_groups: Sequence[tuple[str, np.ndarray]],
        mask_colors: Sequence[str],
        marker_size: int,
        use_gl: bool,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
    ) -> None:
        for (mask_name, mask_values), color in zip(mask_groups, mask_colors, strict=False):
            if not np.any(mask_values):
                continue
            trace = _create_scatter_trace(
                x_data[mask_values],
                y_data[mask_values],
                None,
                marker_size=marker_size,
                showlegend=False,
                name=mask_name,
                use_gl=use_gl,
            )
            trace.legendgroup = mask_name
            trace.marker.color = color  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            trace.marker.opacity = 0.7  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)
            if showlegend:
                self._add_scatter_legend_proxy(
                    fig,
                    name=mask_name,
                    color=color,
                    marker_size=marker_size,
                    row=row,
                    col=col,
                    legendgroup=mask_name,
                )

    def _add_mask_scatter_traces(
        self,
        fig: go.Figure,
        *,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fail_mask: np.ndarray,
        pass_mask: np.ndarray,
        fail_color: str,
        pass_color: str,
        marker_size: int,
        use_gl: bool,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
    ) -> None:
        for name, current_mask, color in (
            ("Fail", fail_mask, fail_color),
            ("Pass", pass_mask, pass_color),
        ):
            if not np.any(current_mask):
                continue
            trace = _create_scatter_trace(
                x_data[current_mask],
                y_data[current_mask],
                None,
                marker_size=marker_size,
                use_gl=use_gl,
                showlegend=False,
                name=name,
            )
            trace.legendgroup = name
            trace.marker.color = color  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)
            if showlegend:
                self._add_scatter_legend_proxy(
                    fig,
                    name=name,
                    color=color,
                    marker_size=marker_size,
                    row=row,
                    col=col,
                    legendgroup=name,
                )

    def _add_multi_mask_marginal_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        *,
        mask_groups: Sequence[tuple[str, np.ndarray]],
        mask_colors: Sequence[str],
        plot_type: str,
        hist_nbins: int | str,
        histnorm: str,
        row: int,
        col: int,
        orientation: str,
    ) -> None:
        histogram_bins = self._histogram_bin_params(
            values,
            hist_nbins,
            axis="ybins" if orientation == "h" else "xbins",
        ) if plot_type != "density" else {}
        for (mask_name, mask_values), color in zip(mask_groups, mask_colors, strict=False):
            selected = np.asarray(values)[mask_values]
            if selected.size == 0:
                continue
            if plot_type == "density":
                grid, density = self._density_curve(selected)
                if orientation == "h":
                    trace = go.Scatter(x=density, y=grid, mode="lines", line=dict(color=color), name=mask_name, showlegend=False)
                else:
                    trace = go.Scatter(x=grid, y=density, mode="lines", line=dict(color=color), name=mask_name, showlegend=False)
            else:
                if orientation == "h":
                    trace = go.Histogram(y=selected, orientation="h", name=mask_name, marker=dict(color=color), histnorm=histnorm, opacity=0.85, showlegend=False, **histogram_bins)
                else:
                    trace = go.Histogram(x=selected, name=mask_name, marker=dict(color=color), histnorm=histnorm, opacity=0.85, showlegend=False, **histogram_bins)
            fig.add_trace(trace, row=row, col=col)
        if plot_type != "density":
            fig.update_layout(barmode="stack")

    def _add_single_marginal_trace(
        self,
        fig: go.Figure,
        values: FloatArray,
        *,
        plot_type: str,
        hist_nbins: int | str,
        histnorm: str,
        hist_color: str,
        row: int,
        col: int,
        orientation: str,
    ) -> float | None:
        selected = np.asarray(values)
        if selected.size == 0:
            return None
        if plot_type == "density":
            grid, density = self._density_curve(selected)
            if grid.size == 0:
                return None
            axis_max = float(np.max(density)) * 1.15
            if orientation == "h":
                trace = go.Scatter(
                    x=density,
                    y=grid,
                    mode="lines",
                    line=dict(color=hist_color),
                    name="Density",
                    showlegend=False,
                )
            else:
                trace = go.Scatter(
                    x=grid,
                    y=density,
                    mode="lines",
                    line=dict(color=hist_color),
                    name="Density",
                    showlegend=False,
                )
        else:
            histogram_bins = self._histogram_bin_params(
                selected,
                hist_nbins,
                axis="ybins" if orientation == "h" else "xbins",
            )
            axis_max = None
            if orientation == "h":
                trace = go.Histogram(
                    y=selected,
                    orientation="h",
                    name="Events",
                    marker=dict(color=hist_color),
                    histnorm=histnorm,
                    showlegend=False,
                    **histogram_bins,
                )
            else:
                trace = go.Histogram(
                    x=selected,
                    name="Events",
                    marker=dict(color=hist_color),
                    histnorm=histnorm,
                    showlegend=False,
                    **histogram_bins,
                )
        fig.add_trace(trace, row=row, col=col)
        return axis_max

    def _add_1d_distribution_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        *,
        plot_type: str,
        hist_nbins: int | str,
        hist_color: str,
        histnorm: str,
        pass_mask: np.ndarray | None,
        fail_mask: np.ndarray | None,
        fail_color: str,
        pass_color: str,
        row: int | None = None,
        col: int | None = None,
    ) -> tuple[str, float | None]:
        if plot_type == "density":
            if pass_mask is not None and fail_mask is not None:
                axis_max = self._add_density_marginal_traces(
                    fig,
                    values,
                    pass_mask,
                    fail_mask,
                    row=row,
                    col=col,
                    orientation="v",
                    fail_color=fail_color,
                    pass_color=pass_color,
                )
            else:
                grid, density = self._density_curve(values)
                axis_max = float(np.max(density)) * 1.15 if density.size else None
                trace = go.Scatter(x=grid, y=density, mode="lines", name="Density", line=dict(color=hist_color), showlegend=False)
                if row is None or col is None:
                    fig.add_trace(trace)
                else:
                    fig.add_trace(trace, row=row, col=col)
            return "Density", axis_max
        if pass_mask is not None and fail_mask is not None:
            axis_max = self._add_stacked_histogram_traces(
                fig,
                values,
                pass_mask,
                fail_mask,
                nbins=hist_nbins,
                histnorm=histnorm,
                row=row,
                col=col,
                orientation="v",
                fail_color=fail_color,
                pass_color=pass_color,
            )
            fig.update_layout(barmode="stack")
            return histnorm.title() if histnorm else "Count", axis_max
        trace = go.Histogram(
            x=values,
            name="Events",
            marker=dict(color=hist_color),
            histnorm=histnorm,
            showlegend=False,
            **self._histogram_bin_params(values, hist_nbins, axis="xbins"),
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)
        return histnorm.title() if histnorm else "Count", None

    def _scatter_color_style(
        self,
        color_values: np.ndarray | None,
        *,
        color_by: str,
        colorscale: str | None,
    ) -> dict[str, Any]:
        resolved_colorscale = "balance" if color_by == "score" else "Viridis"
        if colorscale is not None:
            resolved_colorscale = self._resolve_named_colorscale(colorscale)
        style: dict[str, Any] = {
            "colorscale": resolved_colorscale,
            "color_midpoint": None,
            "color_min": None,
            "color_max": None,
        }
        if color_by != "score" or color_values is None:
            return style
        finite_values = np.asarray(color_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return style
        max_abs = float(np.max(np.abs(finite_values)))
        if max_abs <= 0.0:
            return style
        style["color_midpoint"] = 0.0
        style["color_min"] = -max_abs
        style["color_max"] = max_abs
        return style

    def plot_1d(
        self,
        *,
        events: AnnData,
        plot_dims: Sequence[str],
        mask: dict[str, BooleanArray] | None,
        primary_mask_values: np.ndarray | None,
        score_values: np.ndarray | None,
        decorator: Any | None,
        **kwargs: Any,
    ) -> go.Figure:
        """Render a 1D distribution plot.

        Relevant kwargs:
        - `plot_type`: `"histogram"` for a binned histogram or `"density"` for
          a smoothed density curve.
        - `color_by`: usually `"density"` for a single ungrouped distribution
          or `"mask"` when `mask` contains multiple named groups.
        - `hist_nbins`: histogram bin count or any value accepted by
          `numpy.histogram_bin_edges`.
        - `histnorm`: Plotly histogram normalization such as `"count"`,
          `"probability"`, `"percent"`, `"density"`, or
          `"probability density"`.
        - `hist_color`, `fail_hist_color`, `pass_hist_color`: trace colors for
          single-distribution and pass/fail grouped rendering.
        - `show_ratio`: when `True`, annotate the fraction of events in
          `primary_mask_values`.
        - `max_points` and `downsample_seed`: optionally downsample before
          plotting.
        """
        del score_values
        dim = plot_dims[0]
        data = np.asarray(events[:, dim].X).ravel()
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = self._downsample_indices(data.shape[0], max_points, downsample_seed)
        mask_groups = self._resolve_plot_mask_groups(mask, events.n_obs, downsample_idx)
        color_by = str(kwargs.get("color_by", "density"))
        use_multi_mask_coloring = color_by == "mask" and len(mask_groups) > 1
        pass_mask: np.ndarray | None = None
        fail_mask: np.ndarray | None = None
        primary_mask = primary_mask_values
        if primary_mask is not None:
            primary_mask = np.asarray(primary_mask, dtype=bool).ravel()
        if primary_mask is not None and not use_multi_mask_coloring:
            pass_mask = primary_mask if downsample_idx is None else primary_mask[downsample_idx]
            fail_mask = ~pass_mask
        if downsample_idx is not None:
            data = data[downsample_idx]
        plot_type = str(kwargs.get("plot_type", kwargs.get("diag_plot_type", "histogram")))
        hist_nbins = kwargs.get("hist_nbins", 100)
        hist_color = str(kwargs.get("hist_color", "rgba(100, 100, 200, 0.6)"))
        histnorm = str(kwargs.get("histnorm", "probability"))
        fail_color = str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)"))
        pass_color = str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)"))
        title = kwargs.get("title")
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 800 if width is None else int(width)
        plot_height = 600 if height is None else int(height)
        fig = go.Figure()
        if use_multi_mask_coloring:
            mask_colors = self._resolve_discrete_mask_colors(len(mask_groups), kwargs.get("colorscale"))
            yaxis_title = self._add_multi_1d_distribution_traces(
                fig,
                data,
                plot_type=plot_type,
                hist_nbins=hist_nbins,
                histnorm=histnorm,
                mask_groups=mask_groups,
                mask_colors=mask_colors,
            )
            yaxis_max = None
        else:
            yaxis_title, yaxis_max = self._add_1d_distribution_traces(
                fig,
                data,
                plot_type=plot_type,
                hist_nbins=hist_nbins,
                hist_color=hist_color,
                histnorm=histnorm,
                pass_mask=pass_mask,
                fail_mask=fail_mask,
                fail_color=fail_color,
                pass_color=pass_color,
            )
        self._decorate_plot_1d(decorator, fig, dim, data, **kwargs)
        fig.update_xaxes(title=dim)
        fig.update_yaxes(title=yaxis_title)
        if yaxis_max is not None:
            fig.update_yaxes(range=[0.0, yaxis_max])
        try:
            if bool(kwargs.get("show_ratio", True)):
                ratio = self._compute_ratio(primary_mask_values=primary_mask)
                x_pos = self._decorator_ratio_position_1d(decorator, dim, data, **kwargs)
                if ratio is not None and x_pos is not None:
                    fig.add_annotation(x=x_pos, y=0.95, yref="paper", text=f"{ratio*100:.1f}%", showarrow=False)
        except Exception:
            pass
        return _format_gate_plot(fig, title=title if title is not None else self._decorator_title(decorator, plot_dims), width=plot_width, height=plot_height)

    def plot_2d(
        self,
        *,
        events: AnnData,
        plot_dims: Sequence[str],
        mask: dict[str, BooleanArray] | None,
        primary_mask_values: np.ndarray | None,
        score_values: np.ndarray | None,
        decorator: Any | None,
        **kwargs: Any,
    ) -> go.Figure:
        """Render a 2D scatter plot with optional marginals.

        Relevant kwargs:
        - `color_by`: `"density"` colors each point by local 2D density,
          `"mask"` uses discrete group or pass/fail coloring, `"score"` uses
          `score_values` as a continuous color scale.
        - `marginals`: when `True`, add top/right marginal plots for the x and
          y distributions.
        - `marginal_plot_type`: `"histogram"` or `"density"` for those
          marginals.
        - `density_nbins` and `density_log_scale`: control the density estimate
          used for `"density"` coloring.
        - `marker_size`: point size in the scatter panel.
        - `colorscale`: Plotly colorscale name for continuous or discrete
          coloring.
        - `show_colorbar`: whether to render the continuous colorbar for
          `"density"` or `"score"` coloring.
        - `use_gl`: when `True`, use `Scattergl` for better performance on
          large event sets.
        - `margin_pad_scale`: expand the main scatter axis ranges slightly past
          the observed data bounds.
        - `show_ratio`: when `True`, annotate the fraction of events in
          `primary_mask_values`.
        """
        x_dim, y_dim = plot_dims
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = self._downsample_indices(x_data.shape[0], max_points, downsample_seed)
        mask_groups = self._resolve_plot_mask_groups(mask, events.n_obs, downsample_idx)
        pass_mask: np.ndarray | None = None
        fail_mask: np.ndarray | None = None
        primary_mask = primary_mask_values
        if primary_mask is not None:
            primary_mask = np.asarray(primary_mask, dtype=bool).ravel()
        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]
        if primary_mask is not None and bool(kwargs.get("marginals", False)) and not (str(kwargs.get("color_by", "density")) == "mask" and len(mask_groups) > 1):
            pass_mask = primary_mask if downsample_idx is None else primary_mask[downsample_idx]
            fail_mask = ~pass_mask
        score_subset = None if score_values is None else (np.asarray(score_values, dtype=float) if downsample_idx is None else np.asarray(score_values, dtype=float)[downsample_idx])
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = kwargs.get("colorscale")
        if colorscale is not None and isinstance(colorscale, str):
            colorscale = str(colorscale)
        color_by = str(kwargs.get("color_by", "density"))
        use_multi_mask_coloring = color_by == "mask" and len(mask_groups) > 1
        show_colorbar = bool(kwargs.get("show_colorbar", True))
        use_gl = bool(kwargs.get("use_gl", True))
        title = kwargs.get("title")
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 900 if width is None else int(width)
        plot_height = 700 if height is None else int(height)
        uses_marginal_subplots = False
        if use_multi_mask_coloring and bool(kwargs.get("marginals", False)):
            uses_marginal_subplots = True
            mask_colors = self._resolve_discrete_mask_colors(len(mask_groups), kwargs.get("colorscale"))
            fig = make_subplots(
                rows=2,
                cols=2,
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
            )
            self._add_multi_mask_scatter_traces(fig, x_data, y_data, mask_groups=mask_groups, mask_colors=mask_colors, marker_size=marker_size, use_gl=use_gl, row=2, col=1)
            self._decorate_plot_2d(decorator, fig, plot_dims, x_data, y_data, row=2, col=1, **kwargs)
            marginal_plot_type = str(kwargs.get("marginal_plot_type", "histogram"))
            hist_nbins = kwargs.get("hist_nbins", 100)
            histnorm = str(kwargs.get("histnorm", "probability"))
            self._add_multi_mask_marginal_traces(fig, x_data, mask_groups=mask_groups, mask_colors=mask_colors, plot_type=marginal_plot_type, hist_nbins=hist_nbins, histnorm=histnorm, row=1, col=1, orientation="v")
            self._add_multi_mask_marginal_traces(fig, y_data, mask_groups=mask_groups, mask_colors=mask_colors, plot_type=marginal_plot_type, hist_nbins=hist_nbins, histnorm=histnorm, row=2, col=2, orientation="h")
            data_x_range = (float(np.min(x_data)), float(np.max(x_data)))
            data_y_range = (float(np.min(y_data)), float(np.max(y_data)))
            margin_pad_scale = float(kwargs.get("margin_pad_scale", 0.01))
            margin_x = margin_pad_scale * max(data_x_range[1] - data_x_range[0], 1e-9)
            margin_y = margin_pad_scale * max(data_y_range[1] - data_y_range[0], 1e-9)
            padded_range_x = [data_x_range[0] - margin_x, data_x_range[1] + margin_x]
            padded_range_y = [data_y_range[0] - margin_y, data_y_range[1] + margin_y]
            fig.update_xaxes(title_text=x_dim, row=2, col=1, range=padded_range_x)
            fig.update_yaxes(title_text=y_dim, row=2, col=1, range=padded_range_y)
            fig.update_xaxes(range=padded_range_x, row=1, col=1, showticklabels=False)
            fig.update_yaxes(range=padded_range_y, row=2, col=2, showticklabels=False)
            self._decorate_plot_2d_marginals(decorator, fig, plot_dims, x_data, y_data, **kwargs)
        elif pass_mask is not None and fail_mask is not None:
            uses_marginal_subplots = True
            fig = make_subplots(
                rows=2,
                cols=2,
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
            )
            fail_color = str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)"))
            pass_color = str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)"))
            if color_by == "mask":
                self._add_mask_scatter_traces(
                    fig,
                    x_data=x_data,
                    y_data=y_data,
                    fail_mask=fail_mask,
                    pass_mask=pass_mask,
                    fail_color=fail_color,
                    pass_color=pass_color,
                    marker_size=marker_size,
                    use_gl=use_gl,
                    row=2,
                    col=1,
                    showlegend=True,
                )
            else:
                colors = score_subset if color_by == "score" else _compute_density_colors(x_data.astype(float), y_data.astype(float), nbins=density_nbins, log_scale=density_log_scale)
                color_style = self._scatter_color_style(colors, color_by=color_by, colorscale=colorscale)
                colorbar_title = "Score" if color_by == "score" else "Density"
                fig.add_trace(
                    _create_scatter_trace(x_data, y_data, colors, marker_size=marker_size, use_gl=use_gl, show_colorbar=show_colorbar, colorbar_title=colorbar_title, **color_style),
                    row=2,
                    col=1,
                )
            self._decorate_plot_2d(decorator, fig, plot_dims, x_data, y_data, row=2, col=1, **kwargs)
            marginal_plot_type = str(kwargs.get("marginal_plot_type", "histogram"))
            hist_nbins = kwargs.get("hist_nbins", 100)
            histnorm = str(kwargs.get("histnorm", "probability"))
            if marginal_plot_type == "density":
                top_y_max = self._add_density_marginal_traces(fig, x_data, pass_mask, fail_mask, row=1, col=1, orientation="v", fail_color=fail_color, pass_color=pass_color)
                right_x_max = self._add_density_marginal_traces(fig, y_data, pass_mask, fail_mask, row=2, col=2, orientation="h", fail_color=fail_color, pass_color=pass_color)
            else:
                top_y_max = self._add_stacked_histogram_traces(fig, x_data, pass_mask, fail_mask, nbins=hist_nbins, histnorm=histnorm, row=1, col=1, orientation="v", fail_color=fail_color, pass_color=pass_color)
                right_x_max = self._add_stacked_histogram_traces(fig, y_data, pass_mask, fail_mask, nbins=hist_nbins, histnorm=histnorm, row=2, col=2, orientation="h", fail_color=fail_color, pass_color=pass_color)
                fig.update_layout(barmode="stack")
            data_x_range = (float(np.min(x_data)), float(np.max(x_data)))
            data_y_range = (float(np.min(y_data)), float(np.max(y_data)))
            margin_pad_scale = float(kwargs.get("margin_pad_scale", 0.01))
            margin_x = margin_pad_scale * max(data_x_range[1] - data_x_range[0], 1e-9)
            margin_y = margin_pad_scale * max(data_y_range[1] - data_y_range[0], 1e-9)
            padded_range_x = [data_x_range[0] - margin_x, data_x_range[1] + margin_x]
            padded_range_y = [data_y_range[0] - margin_y, data_y_range[1] + margin_y]
            fig.update_xaxes(title_text=x_dim, row=2, col=1, range=padded_range_x)
            fig.update_yaxes(title_text=y_dim, row=2, col=1, range=padded_range_y)
            fig.update_xaxes(range=padded_range_x, row=1, col=1, showticklabels=False)
            fig.update_yaxes(range=padded_range_y, row=2, col=2, showticklabels=False)
            if top_y_max is not None:
                fig.update_yaxes(range=[0.0, top_y_max], row=1, col=1)
            if right_x_max is not None:
                fig.update_xaxes(range=[0.0, right_x_max], row=2, col=2)
            self._decorate_plot_2d_marginals(decorator, fig, plot_dims, x_data, y_data, **kwargs)
        elif bool(kwargs.get("marginals", False)):
            uses_marginal_subplots = True
            fig = make_subplots(
                rows=2,
                cols=2,
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
            )
            colors = score_subset if color_by == "score" else _compute_density_colors(
                x_data.astype(float),
                y_data.astype(float),
                nbins=density_nbins,
                log_scale=density_log_scale,
            )
            color_style = self._scatter_color_style(colors, color_by=color_by, colorscale=colorscale)
            colorbar_title = "Score" if color_by == "score" else "Density"
            fig.add_trace(
                _create_scatter_trace(
                    x_data,
                    y_data,
                    colors,
                    marker_size=marker_size,
                    use_gl=use_gl,
                    show_colorbar=show_colorbar,
                    colorbar_title=colorbar_title,
                    **color_style,
                ),
                row=2,
                col=1,
            )
            self._decorate_plot_2d(decorator, fig, plot_dims, x_data, y_data, row=2, col=1, **kwargs)
            marginal_plot_type = str(kwargs.get("marginal_plot_type", "histogram"))
            hist_nbins = kwargs.get("hist_nbins", 100)
            histnorm = str(kwargs.get("histnorm", "probability"))
            hist_color = str(kwargs.get("hist_color", "rgba(100, 100, 200, 0.6)"))
            top_y_max = self._add_single_marginal_trace(
                fig,
                x_data,
                plot_type=marginal_plot_type,
                hist_nbins=hist_nbins,
                histnorm=histnorm,
                hist_color=hist_color,
                row=1,
                col=1,
                orientation="v",
            )
            right_x_max = self._add_single_marginal_trace(
                fig,
                y_data,
                plot_type=marginal_plot_type,
                hist_nbins=hist_nbins,
                histnorm=histnorm,
                hist_color=hist_color,
                row=2,
                col=2,
                orientation="h",
            )
            data_x_range = (float(np.min(x_data)), float(np.max(x_data)))
            data_y_range = (float(np.min(y_data)), float(np.max(y_data)))
            margin_pad_scale = float(kwargs.get("margin_pad_scale", 0.01))
            margin_x = margin_pad_scale * max(data_x_range[1] - data_x_range[0], 1e-9)
            margin_y = margin_pad_scale * max(data_y_range[1] - data_y_range[0], 1e-9)
            padded_range_x = [data_x_range[0] - margin_x, data_x_range[1] + margin_x]
            padded_range_y = [data_y_range[0] - margin_y, data_y_range[1] + margin_y]
            fig.update_xaxes(title_text=x_dim, row=2, col=1, range=padded_range_x)
            fig.update_yaxes(title_text=y_dim, row=2, col=1, range=padded_range_y)
            fig.update_xaxes(range=padded_range_x, row=1, col=1, showticklabels=False)
            fig.update_yaxes(range=padded_range_y, row=2, col=2, showticklabels=False)
            if top_y_max is not None:
                fig.update_yaxes(range=[0.0, top_y_max], row=1, col=1)
            if right_x_max is not None:
                fig.update_xaxes(range=[0.0, right_x_max], row=2, col=2)
            self._decorate_plot_2d_marginals(decorator, fig, plot_dims, x_data, y_data, **kwargs)
        else:
            fig = go.Figure()
            if len(mask_groups) > 1:
                mask_colors = self._resolve_discrete_mask_colors(len(mask_groups), kwargs.get("colorscale"))
                self._add_multi_mask_scatter_traces(fig, x_data, y_data, mask_groups=mask_groups, mask_colors=mask_colors, marker_size=marker_size, use_gl=use_gl)
            else:
                colors = score_subset if color_by == "score" else _compute_density_colors(x_data.astype(float), y_data.astype(float), nbins=density_nbins, log_scale=density_log_scale)
                color_style = self._scatter_color_style(colors, color_by=color_by, colorscale=colorscale)
                colorbar_title = "Score" if color_by == "score" else "Density"
                fig.add_trace(_create_scatter_trace(x_data, y_data, colors, marker_size=marker_size, use_gl=use_gl, show_colorbar=show_colorbar, colorbar_title=colorbar_title, **color_style))
            self._decorate_plot_2d(decorator, fig, plot_dims, x_data, y_data, **kwargs)
            fig.update_xaxes(title=x_dim)
            fig.update_yaxes(title=y_dim)
        try:
            if bool(kwargs.get("show_ratio", True)):
                ratio = self._compute_ratio(primary_mask_values=primary_mask)
                ratio_xy = self._decorator_ratio_position_2d(decorator, plot_dims, x_data, y_data, **kwargs)
                if ratio is not None and ratio_xy is not None:
                    xref = "x3" if uses_marginal_subplots else "x"
                    yref = "y3" if uses_marginal_subplots else "y"
                    fig.add_annotation(x=float(ratio_xy[0]), y=float(ratio_xy[1]), xref=xref, yref=yref, text=f"{ratio*100:.1f}%", showarrow=False, bgcolor="white", opacity=0.8)
        except Exception:
            pass
        return _format_gate_plot(fig, title=title if title is not None else self._decorator_title(decorator, plot_dims), width=plot_width, height=plot_height)

    def plot_nd(
        self,
        *,
        events: AnnData,
        plot_dims: Sequence[str],
        mask: dict[str, BooleanArray] | None,
        primary_mask_values: np.ndarray | None,
        score_values: np.ndarray | None,
        decorator: Any | None,
        **kwargs: Any,
    ) -> go.Figure:
        """Render an N-dimensional pair plot.

        Diagonal panels use the 1D distribution settings, while off-diagonal
        panels use the 2D scatter settings. This entrypoint supports the same
        generic coloring and styling controls as `plot_1d()` and `plot_2d()`,
        plus pair-plot layout kwargs such as:
        - `diag_plot_type`: `"histogram"` or `"density"` for the diagonal.
        - `horizontal_spacing` and `vertical_spacing`: subplot spacing.
        - `use_gl`: use WebGL scatter traces in the off-diagonal panels.
        """
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = self._downsample_indices(events.n_obs, max_points, downsample_seed)
        mask_groups = self._resolve_plot_mask_groups(mask, events.n_obs, downsample_idx)
        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = kwargs.get("colorscale")
        if colorscale is not None and isinstance(colorscale, str):
            colorscale = str(colorscale)
        color_by = str(kwargs.get("color_by", "density"))
        use_gl = bool(kwargs.get("use_gl", True))
        diag_plot_type = str(kwargs.get("diag_plot_type", kwargs.get("plot_type", "histogram")))
        hist_nbins = kwargs.get("hist_nbins", 100)
        hist_color = str(kwargs.get("hist_color", "rgba(100, 100, 200, 0.6)"))
        horizontal_spacing = float(kwargs.get("horizontal_spacing", 0.03))
        vertical_spacing = float(kwargs.get("vertical_spacing", 0.03))
        n_dims = len(plot_dims)
        fig = make_subplots(rows=n_dims, cols=n_dims, shared_xaxes=True, shared_yaxes=False, horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing)
        use_multi_mask_coloring = color_by == "mask" and len(mask_groups) > 1
        primary_mask = None if primary_mask_values is None else np.asarray(primary_mask_values, dtype=bool).ravel()
        nd_pass_mask: np.ndarray | None = None
        nd_fail_mask: np.ndarray | None = None
        if primary_mask is not None and not use_multi_mask_coloring:
            nd_pass_mask = primary_mask if downsample_idx is None else primary_mask[downsample_idx]
            nd_fail_mask = ~nd_pass_mask
        scatter_color_values = None if use_multi_mask_coloring else (None if score_values is None else (np.asarray(score_values, dtype=float) if downsample_idx is None else np.asarray(score_values, dtype=float)[downsample_idx]))
        scatter_color_style = self._scatter_color_style(scatter_color_values, color_by=color_by, colorscale=colorscale) if not use_multi_mask_coloring else {}
        mask_colors = self._resolve_discrete_mask_colors(len(mask_groups), kwargs.get("colorscale")) if use_multi_mask_coloring else []
        for row_idx, y_dim in enumerate(plot_dims, start=1):
            y_values = np.asarray(events[:, y_dim].X).ravel()
            if downsample_idx is not None:
                y_values = y_values[downsample_idx]
            for col_idx, x_dim in enumerate(plot_dims, start=1):
                x_values = np.asarray(events[:, x_dim].X).ravel()
                if downsample_idx is not None:
                    x_values = x_values[downsample_idx]
                if row_idx == col_idx:
                    if use_multi_mask_coloring:
                        yaxis_title = self._add_multi_1d_distribution_traces(fig, x_values, plot_type=diag_plot_type, hist_nbins=hist_nbins, histnorm=str(kwargs.get("histnorm", "probability")), mask_groups=mask_groups, mask_colors=mask_colors, row=row_idx, col=col_idx, showlegend=(row_idx == 1 and col_idx == 1))
                        yaxis_max = None
                    else:
                        yaxis_title, yaxis_max = self._add_1d_distribution_traces(fig, x_values, plot_type=diag_plot_type, hist_nbins=hist_nbins, hist_color=hist_color, histnorm=str(kwargs.get("histnorm", "probability")), pass_mask=nd_pass_mask, fail_mask=nd_fail_mask, fail_color=str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)")), pass_color=str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)")), row=row_idx, col=col_idx)
                    self._decorate_plot_1d(decorator, fig, x_dim, x_values, row=row_idx, col=col_idx, **kwargs)
                    if col_idx == 1:
                        fig.update_yaxes(title=yaxis_title, row=row_idx, col=col_idx)
                    if yaxis_max is not None:
                        fig.update_yaxes(range=[0.0, yaxis_max], row=row_idx, col=col_idx)
                else:
                    if use_multi_mask_coloring:
                        self._add_multi_mask_scatter_traces(fig, x_values, y_values, mask_groups=mask_groups, mask_colors=mask_colors, marker_size=marker_size, use_gl=use_gl, row=row_idx, col=col_idx, showlegend=(row_idx == 1 and col_idx == 2))
                    else:
                        if color_by == "mask" and nd_pass_mask is not None and nd_fail_mask is not None:
                            self._add_mask_scatter_traces(
                                fig,
                                x_data=x_values,
                                y_data=y_values,
                                fail_mask=nd_fail_mask,
                                pass_mask=nd_pass_mask,
                                fail_color=str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)")),
                                pass_color=str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)")),
                                marker_size=marker_size,
                                use_gl=use_gl,
                                row=row_idx,
                                col=col_idx,
                                showlegend=(row_idx == 1 and col_idx == 2),
                            )
                        else:
                            color_values = scatter_color_values if scatter_color_values is not None else _compute_density_colors(x_values.astype(float), y_values.astype(float), nbins=density_nbins, log_scale=density_log_scale)
                            color_style = scatter_color_style if scatter_color_values is not None else self._scatter_color_style(color_values, color_by="density", colorscale=colorscale)
                            fig.add_trace(_create_scatter_trace(x_values, y_values, color_values, marker_size=marker_size, use_gl=use_gl, showlegend=False, **color_style), row=row_idx, col=col_idx)
                    self._decorate_plot_2d(decorator, fig, (x_dim, y_dim), x_values, y_values, row=row_idx, col=col_idx, showlegend=False, **kwargs)
                if row_idx == n_dims:
                    fig.update_xaxes(title=x_dim, row=row_idx, col=col_idx)
                if col_idx == 1 and row_idx != col_idx:
                    fig.update_yaxes(title=y_dim, row=row_idx, col=col_idx)
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = int(width) if width is not None else max(350 * n_dims, 700)
        plot_height = int(height) if height is not None else max(350 * n_dims, 700)
        try:
            if bool(kwargs.get("show_ratio", True)):
                ratio = self._compute_ratio(primary_mask_values=primary_mask)
                if ratio is not None:
                    fig.add_annotation(x=0.5, y=1.02, xref="paper", yref="paper", text=f"{ratio*100:.1f}%", showarrow=False, bgcolor="white", opacity=0.8)
        except Exception:
            pass
        return _format_gate_plot(fig, title=kwargs.get("title") or self._decorator_title(decorator, plot_dims), width=plot_width, height=plot_height)
