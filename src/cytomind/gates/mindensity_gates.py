from __future__ import annotations
from typing import Any, Mapping, Sequence, TYPE_CHECKING
from itertools import product
import warnings

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline

from .geometric_gates import RectangleGate, QuadrantGate
from . import GateRegistry

if TYPE_CHECKING:
    from cytomind.domain.constants import FloatArray, IntArray
    from pandas import DataFrame
else:
    FloatArray = object
    IntArray = object
    DataFrame = object

@GateRegistry.register("min_density")
class MinDensityGate(RectangleGate):
    """
    Min-density gate that finds cut points on one or more dimensions.

    For each dimension, fits a min-density cut. The `combo` vector specifies
    which side of each cut point to keep:
    - combo[i] = 1: keep values >= cut (high side)
    - combo[i] = -1: keep values < cut (low side)

    Example (2D gate):
        dimensions = ['CD4', 'CD8']
        combo = [1, -1]  # CD4 high AND CD8 low
    """

    gate_type = "min_density"
    default_tunable = True

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        directions: Sequence[int] | None = None,
        min_ranges: Mapping[str, float] | None = None,
        max_ranges: Mapping[str, float] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : list[str]
            One or more dimension IDs to gate on
        directions : list[int] | None
            Direction for each dimension: 1 (keep high) or -1 (keep low).
            If None, defaults to all 1s (keep high side for all dims).
        min_ranges : Mapping[str, float] | None
            Optional per-dimension minimum values for density fitting
        max_ranges : Mapping[str, float] | None
            Optional per-dimension maximum values for density fitting
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """

        if len(dimensions) < 1:
            raise ValueError("MinDensityGate requires at least 1 dimension.")

        min_ranges = dict(min_ranges or {})
        max_ranges = dict(max_ranges or {})

        for dim, val in min_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"min_range dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise TypeError(f"min_range for dimension '{dim}' must be numeric")

        for dim, val in max_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"max_range dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise TypeError(f"max_range for dimension '{dim}' must be numeric")

        if directions is None:
            directions = [1] * len(dimensions)
        else:
            directions = list(directions)

        if len(directions) != len(dimensions):
            raise ValueError(
                f"directions length {len(directions)} does not match dimensions length {len(dimensions)}"
            )

        for i, val in enumerate(directions):
            if val not in (-1, 1):
                raise ValueError(f"directions[{i}] must be -1 or 1, got {val}")

        super().__init__(gate_name=gate_name, dimensions=dimensions, use_as_complement=use_as_complement)
        self.tunable = type(self).default_tunable if tunable is None else bool(tunable)
        self._hyperparams = {
            "directions": directions,
            "min_ranges": min_ranges,
            "max_ranges": max_ranges,
        }

    def _fit_gate(self, events_slice: DataFrame) -> None:
        directions = self.hyperparams["directions"]
        min_vals: dict[str, float] = {}
        max_vals: dict[str, float] = {}

        for dim, direction in zip(self.dimensions, directions):
            vals = events_slice[dim].to_numpy(dtype=np.float64)
            min_range = self.hyperparams["min_ranges"].get(dim)
            max_range = self.hyperparams["max_ranges"].get(dim)
            res = improved_mindensity(vals, min_range, max_range)
            cut: float = res.pop("cut_point")

            if not np.isfinite(cut):
                raise ValueError(f"Computed cut point for dimension '{dim}' is not finite")

            if direction == 1:
                # Keep high side: values >= cut
                min_vals[dim] = cut
                res["mass"] = res["mass"][1]  # mass on the high
            else:  # direction == -1
                # Keep low side: values < cut
                max_vals[dim] = cut
                res["mass"] = res["mass"][0]  # mass on the low
            self.diagnostics[dim] = res

        self.params["boundaries"] = [
            [min_vals.get(dim, None) for dim in self.dimensions],
            [max_vals.get(dim, None) for dim in self.dimensions]
        ]

    def _add_plot_overlays_2d_marginals(
        self,
        fig,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> None:
        del x_data, y_data
        min_vals = self.min_vals # pyright: ignore[reportAttributeAccessIssue]
        max_vals = self.max_vals # pyright: ignore[reportAttributeAccessIssue]
        x_dim, y_dim = plot_dims
        gate_line_color = str(kwargs.get("gate_line_color", "red"))
        gate_line_width = int(kwargs.get("gate_line_width", 2))
        gate_line_dash = str(kwargs.get("gate_line_dash", "dash"))

        if min_vals[x_dim] is not None:
            fig.add_vline(
                x=float(min_vals[x_dim]),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="min",
                row=1, col=1  # pyright: ignore[reportArgumentType]
            )
        if max_vals[x_dim] is not None:
            fig.add_vline(
                x=float(max_vals[x_dim]),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="max",
                row=1, col=1  # pyright: ignore[reportArgumentType]
            )
        if min_vals[y_dim] is not None:
            fig.add_hline(
                y=float(min_vals[y_dim]),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="min",
                row=2, col=2  # pyright: ignore[reportArgumentType]
            )
        if max_vals[y_dim] is not None:
            fig.add_hline(
                y=float(max_vals[y_dim]),
                line_color=gate_line_color,
                line_width=gate_line_width,
                line_dash=gate_line_dash,
                annotation_text="max",
                row=2, col=2  # pyright: ignore[reportArgumentType]
            )


@GateRegistry.register("min_density_quadrant")
class MinDensityQuadrantGate(QuadrantGate):
    """
    Quadrant-like gate with one min-density divider per dimension.

    Fits a MinDensityGate on each dimension to find a single cut point and then
    constructs 2^N quadrant regions (low/high for each dimension).
    """

    gate_type = "min_density_quadrant"
    default_tunable = True

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        min_ranges: Mapping[str, float] | None = None,
        max_ranges: Mapping[str, float] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if len(dimensions) < 1:
            raise ValueError("MinDensityQuadrantGate requires at least 1 dimension")

        if use_as_complement:
            raise ValueError("MinDensityQuadrantGate does not support use_as_complement")

        min_ranges = dict(min_ranges or {})
        max_ranges = dict(max_ranges or {})

        for dim, val in min_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"min_range dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise TypeError(f"min_range for dimension '{dim}' must be numeric")

        for dim, val in max_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"max_range dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise TypeError(f"max_range for dimension '{dim}' must be numeric")

        placeholder_dividers = {dim: [] for dim in dimensions}
        super().__init__(
            gate_name=gate_name,
            dividers=placeholder_dividers,
            quadrants={},
            **kwargs,
        )
        self.tunable = type(self).default_tunable if tunable is None else bool(tunable)

        self._hyperparams["min_ranges"] = min_ranges
        self._hyperparams["max_ranges"] = max_ranges

    def _fit_gate(self, events_slice: DataFrame) -> None:
        cut_points: dict[str, float] = {}

        for dim in self.dimensions:
            vals = events_slice[dim].to_numpy(dtype=np.float64)
            min_range = self.hyperparams["min_ranges"].get(dim)
            max_range = self.hyperparams["max_ranges"].get(dim)
            res = improved_mindensity(vals, min_range, max_range)
            cut: float = res.pop("cut_point")
            cut_points[dim] = cut
            self.diagnostics[dim] = res

        dividers = {dim: [cut] for dim, cut in cut_points.items()}
        locations = self._build_locations(cut_points)

        # Construct QuadrantGate with the computed dividers and locations
        self._hyperparams["dividers"] = dividers
        self._hyperparams["quadrants"] = locations
        # populate self.params["quadrants"] as well
        self._compute_quadrants()  # pyright: ignore[reportAttributeAccessIssue]

    def _build_locations(self, cut_points: Mapping[str, float]) -> dict[str, list[tuple[str, float]]]:
        locations: dict[str, list[tuple[str, float]]] = {}
        dims = self.dimensions

        for combo in product([0, 1], repeat=len(dims)):
            quad_id = "_".join(f"{dim}_{'high' if side == 1 else 'low'}" for dim, side in zip(dims, combo))
            locs: list[tuple[str, float]] = []
            for dim, side in zip(dims, combo):
                cut = cut_points[dim]
                loc = np.nextafter(cut, np.inf) if side == 1 else np.nextafter(cut, -np.inf)
                locs.append((dim, loc))
            locations[quad_id] = locs

        return locations

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Remove computed "hyperparameters" of QuadrantGate
        base["hyperparams"].pop("dividers", None)
        base["hyperparams"].pop("quadrants", None)
        return base


def improved_mindensity(vals: FloatArray, low_bd: float | None = None, up_bd: float | None = None) -> dict[str, Any]:
    """
    Determines a cutpoint as the minimum point of a kernel density estimate between two peaks.
    Based on R/Bioconductor package openCyto::gate_mindensity2 implementation.

    Args:
        vals (FloatArray): Input data values.
        low_bd (float | None, optional): Lower bound for cut point search. Default (None) uses the minimum value in `vals`.
        up_bd (float | None, optional): Upper bound for cut point search. Default (None) uses the maximum value in `vals`.

    Raises:
        ValueError: If `low_bd` is greater than `up_bd`.
        ValueError: If the specified range is outside the data range.

    Returns:
        dict[str, Any]: Dictionary containing the cut point and related metrics.
    """

    # Step 1: Parse extreme values
    min_val, max_val = float(np.min(vals)), float(np.max(vals))
    low_bd = max(low_bd, min_val) if low_bd is not None else min_val
    up_bd = min(up_bd, max_val) if up_bd is not None else max_val
    if low_bd > up_bd:
        raise ValueError(f"low_bd {low_bd} cannot be greater than up_bd {up_bd}")

    # Degenerate case: all values are the same or range is a single point
    if np.allclose(low_bd, up_bd):
        return {
            'cut_point': low_bd,
            'mass': [.5, .5],
            'minima': [(low_bd, 1.0)],
            'maxima': [],
            'shoulders': [],
            'range': [low_bd, up_bd],
        }

    if len(vals) < 100:
        # Warn if very few points, as KDE may be unreliable
        warnings.warn(f"Only {len(vals)} data points provided; density estimation may be unreliable")

    # Step 2: Fit Density Function
    kde = gaussian_kde(vals)
    x_vals = np.linspace(low_bd, up_bd, 512)
    y_vals: FloatArray = kde(x_vals)
    y_vals /= y_vals.max() # normalize to max of 1 for stability/interpretability
    sp = UnivariateSpline(x_vals, y_vals, s=1, k=3)

    # Step 3: Get Critical Points
    def zero_crossings(y: FloatArray, direction=0) -> IntArray:
        # direction: 0 all zero-crossings, -1 positive to negative, +1 negative to positive
        signs = np.sign(y)
        flips = np.where(signs[1:] * signs[:-1] < 0)[0] + 1
        if direction == 0:
            return flips
        return flips[signs[flips] * direction >= 0]

    # Get minima and maxima and shoulder points
    d1: FloatArray = sp.derivative(n=1)(x_vals) # pyright: ignore[reportAssignmentType]
    zx = zero_crossings(d1)
    min_idx = zx[d1[zx] > 0]  # slope goes from negative to positive
    max_idx = zx[d1[zx] < 0]  # slope goes from positive to negative

    d3: FloatArray = sp.derivative(n=3)(x_vals) # pyright: ignore[reportAssignmentType]
    sld_idx = zero_crossings(d3, direction=-1)  # maximum of the second derivative (shoulder points)

    # Step 4: Determine final cut point
    if len(min_idx) >= 1:
        # Case 1: minima exists -> use the lowest/deepest minimum as the cut point
        min_x, min_y = x_vals[min_idx], y_vals[min_idx]
        pt = min_x[np.argmin(min_y)]
    elif len(max_idx) == 1:
        # Case2: no minima but single peak -> try to find a point where the density starts falling
        peak = max_idx[0]
        post_sld = sld_idx[sld_idx > peak]
        d2: FloatArray = sp.derivative(n=2)(x_vals) # pyright: ignore[reportAssignmentType]
        inf_idx = zero_crossings(-d2, direction=1)
        post_inf = inf_idx[inf_idx > peak]
        if post_sld.size > 0:
            pt = x_vals[int(np.median(post_sld))]
        elif post_inf.size > 0:
            pt = x_vals[int(np.median(post_inf))]
        elif sld_idx.size > 0:
            pt = x_vals[int(np.median(sld_idx))]
        else:
            pt = x_vals[np.argmin(y_vals)]
    else:
        # Case 3: multiple maxima or no minima -> use global features
        pt = x_vals[int(np.median(sld_idx))] if sld_idx.size > 0 else x_vals[np.argmin(y_vals)]

    mass = np.array([kde.integrate_box_1d(low_bd, pt), kde.integrate_box_1d(pt, up_bd)])
    mass /= mass.sum()  # normalize to sum to 1
    return {
        'cut_point': float(pt),
        'mass': mass.tolist(),
        'minima': list(zip(x_vals[min_idx].tolist(), y_vals[min_idx].tolist())),
        'maxima': list(zip(x_vals[max_idx].tolist(), y_vals[max_idx].tolist())),
        'shoulders': list(zip(x_vals[sld_idx].tolist(), y_vals[sld_idx].tolist())),
        'range': [x_vals[0], x_vals[-1]],
    }
