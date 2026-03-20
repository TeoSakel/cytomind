from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from . import GateRegistry
from .glm_gates import RectangleGate, QuadrantGate

if TYPE_CHECKING:
    from pandas import DataFrame
else :
    DataFrame = object


@GateRegistry.register("percentile_rectangle")
class PercentileRectangleGate(RectangleGate):
    """
    Percentile gate that computes cut points from percentiles per-dimension.

    For each dimension a percentile (in range [0, 1]) is specified and the
    corresponding cut point is computed using `np.percentile`. The `directions`
    vector controls which side is kept (1 -> keep >= cut, -1 -> keep < cut).
    """

    gate_type = "percentile_rectangle"
    tunable = True

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        min_ranges: Mapping[str, float] | None = None,
        max_ranges: Mapping[str, float] | None = None,
        use_as_complement: bool = False,
    ):
        if len(dimensions) < 1:
            raise ValueError("PercentileGate requires at least 1 dimension.")
        min_ranges = dict(min_ranges or {})
        max_ranges = dict(max_ranges or {})

        # validate keys and values; None means "no threshold" and is allowed
        for dim, p in min_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"min_range dimension '{dim}' not in gate dimensions")
            if p is None:
                continue
            if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"min_range percentile for dimension '{dim}' must be numeric in [0,1]")

        for dim, p in max_ranges.items():
            if dim not in dimensions:
                raise ValueError(f"max_range dimension '{dim}' not in gate dimensions")
            if p is None:
                continue
            if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"max_range percentile for dimension '{dim}' must be numeric in [0,1]")

        super().__init__(gate_name=gate_name, dimensions=dimensions, use_as_complement=use_as_complement)
        self._hyperparams = {
            "min_ranges": min_ranges,
            "max_ranges": max_ranges,
        }

    def _fit_gate(self, events_slice: DataFrame) -> None:
        min_ranges: Mapping[str, float] = self.hyperparams.get("min_ranges", {})
        max_ranges: Mapping[str, float] = self.hyperparams.get("max_ranges", {})
        min_vals: dict[str, float] = {}
        max_vals: dict[str, float] = {}

        for dim in self.dimensions:
            vals = events_slice[dim].to_numpy(dtype=np.float32)
            if dim in min_ranges and min_ranges[dim] is not None:
                p = float(min_ranges[dim])
                cut = float(np.percentile(vals, 100.0 * p))
                if np.isfinite(cut):
                    min_vals[dim] = cut
            if dim in max_ranges and max_ranges[dim] is not None:
                p = float(max_ranges[dim])
                cut = float(np.percentile(vals, 100.0 * p))
                if np.isfinite(cut):
                    max_vals[dim] = cut

        self.params["boundaries"] = [
            [min_vals.get(dim, None) for dim in self.dimensions],
            [max_vals.get(dim, None) for dim in self.dimensions],
        ]


@GateRegistry.register("percentile_quadrant")
class PercentileQuadrantGate(QuadrantGate):
    """
    Quadrant gate where dividers and quadrant locations are specified as
    percentiles (values in [0,1]). At fit time the gate converts those
    percentiles into absolute divider and location values using the data
    in `events_slice` and then behaves like a standard `QuadrantGate`.
    """

    gate_type = "percentile_quadrant"
    tunable = True

    def __init__(
        self,
        gate_name: str,
        dividers: Mapping[str, Sequence[float]],
        quadrants: Mapping[str, Sequence[tuple[str, float]]],
        **kwargs: Any,
    ) -> None:
        # Validate percentile dividers and quadrants (values must be numeric in [0,1])
        dividers_pct = {dim: list(points) for dim, points in dict(dividers).items()}
        for dim, pts in dividers_pct.items():
            if not isinstance(pts, Sequence):
                raise TypeError(f"Division percentiles for '{dim}' must be a list/tuple")
            for p in pts:
                if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
                    raise ValueError(f"Divider percentile for '{dim}' must be numeric in [0,1]")

        quadrants_pct = dict(quadrants)
        for quad_id, locations in quadrants_pct.items():
            for dim_id, loc in locations:
                if dim_id not in dividers_pct:
                    raise ValueError(
                        f"Quadrant '{quad_id}' references dimension '{dim_id}' not in dividers {list(dividers_pct.keys())}"
                    )
                if not isinstance(loc, (int, float)) or not (0.0 <= float(loc) <= 1.0):
                    raise TypeError(f"Location percentile for dimension '{dim_id}' in quadrant '{quad_id}' must be numeric in [0,1]")

        # create placeholder absolute dividers (QuadrantGate requires a dividers mapping at init)
        placeholder_dividers = {dim: [] for dim in dividers_pct}
        super().__init__(gate_name=gate_name, dividers=placeholder_dividers, quadrants={}, **kwargs)

        # store percentile hyperparams; actual absolute dividers/locations computed at fit
        self._hyperparams["dividers_percent"] = {dim: sorted(points) for dim, points in dividers_pct.items()}
        self._hyperparams["quadrants_percent"] = quadrants_pct

    def _fit_gate(self, events_slice: DataFrame) -> None:
        # Read percentiles
        dividers_pct: Mapping[str, Sequence[float]] = self.hyperparams.get("dividers_percent", {})
        quadrants_pct: Mapping[str, Sequence[tuple[str, float]]] = self.hyperparams.get("quadrants_percent", {})

        # Compute absolute divider points per-dimension
        dividers_abs: dict[str, list[float]] = {}
        for dim, pts in dividers_pct.items():
            vals = events_slice[dim].to_numpy(dtype=np.float32)
            # TODO: check for duplicate divider values after conversion ?
            dividers_abs[dim] = np.percentile(vals, 100. * np.asarray(pts, dtype=np.float32)).tolist()

        # Convert quadrant percentile locations to absolute locations
        quadrants_abs: dict[str, list[tuple[str, float]]] = {}
        for quad_id, locs in quadrants_pct.items():
            abs_locs: list[tuple[str, float]] = []
            for dim_id, loc_pct in locs:
                vals = events_slice[dim_id].to_numpy(dtype=np.float32)
                loc_val = float(np.percentile(vals, 100.0 * float(loc_pct)))
                abs_locs.append((dim_id, loc_val))
            quadrants_abs[quad_id] = abs_locs

        # Store computed absolute dividers and quadrant locations into hyperparams
        self._hyperparams["dividers"] = dividers_abs
        self._hyperparams["quadrants"] = quadrants_abs

        # Now compute quadrant min/max ranges as QuadrantGate expects
        self._compute_quadrants() # pyright: ignore[reportAttributeAccessIssue]

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Remove computed "dividers" and "quadrants" from serialized hyperparams
        base["hyperparams"].pop("dividers", None)
        base["hyperparams"].pop("quadrants", None)
        return base
