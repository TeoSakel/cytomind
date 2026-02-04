from bisect import bisect_right
from typing import Any, Iterable, Mapping
from numpy.typing import NDArray

import anndata as ad
import numpy as np
import pandas as pd

from .base import Gate
from . import GateRegistry


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
        dimensions: list[str],
        min_vals: Mapping[str, float] = {},
        max_vals: Mapping[str, float] = {},
        use_as_complement: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : list[str]
            Dimension IDs to operate on (must have 1 or more)
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """
        super().__init__(gate_name, dimensions, {"min_vals": min_vals, "max_vals": max_vals}, use_as_complement)
        self._parse_hyperparams()

    def _parse_hyperparams(self) -> None:
        """Parse and validate hyperparameters, setting them in params."""
        min_vals = self.hyperparams["min_vals"]
        max_vals = self.hyperparams["max_vals"]

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

        # Set params
        self.params["min_vals"] = dict(min_vals)
        self.params["max_vals"] = dict(max_vals)

    def _check_thresholds(self, thresholds: Mapping[str, float]) -> None:
        """Validate threshold dictionary."""
        dim_set = set(self.dimensions)
        for dim, val in thresholds.items():
            if dim not in dim_set:
                raise ValueError(f"Threshold dimension '{dim}' not in gate dimensions")
            if not isinstance(val, (int, float)):
                raise ValueError(f"Threshold value for dimension '{dim}' must be numeric")

    @property
    def min_vals(self) -> dict[str, float]:
        """Access min_vals hyperparameter or fitted params."""
        try:
            return self.params["min_vals"]
        except KeyError:
            raise ValueError("RectangleGate min_vals have not been set. Please fit the gate first.")

    @property
    def max_vals(self) -> dict[str, float]:
        """Access max_vals hyperparameter or fitted params."""
        try:
            return self.params["max_vals"]
        except KeyError:
            raise ValueError("RectangleGate max_vals have not been set. Please fit the gate first.")

    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "RectangleGate":
        """
        RectangleGate doesn't require fitting (no event data to learn from).
        Just copies hyperparams to params.
        """
        return self

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """For RectangleGate, fit just copies hyperparams to params."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        """Apply rectangular bounds to events."""
        mask = np.ones(len(events_slice), dtype=np.bool_)

        for dim in self.dimensions:
            col_vals = np.asarray(events_slice[dim].values)
            try:
                np.bitwise_and(mask, col_vals >= self.min_vals[dim], out=mask)
            except KeyError:
                pass
            try:
                np.bitwise_and(mask, col_vals < self.max_vals[dim], out=mask)
            except KeyError:
                pass

        # Apply complement if requested
        if self.use_as_complement:
            mask = ~mask

        return {self.gate_name: mask}

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Convert numpy arrays to lists for JSON serialization
        for p in ("hyperparams", "params"):
            for k in ("center", "covariance_matrix"):
                if k in base[p] and not isinstance(base[p][k], list):
                    base[p][k] = base[p][k].tolist()
        return base


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
        dimensions: list[str],
        vertices: list[tuple[float, float]],
        use_as_complement: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : list[str]
            Exactly 2 dimension IDs (x, y)
        vertices : list[tuple[float, float]]
            Ordered list of (x, y) coordinates defining polygon boundary
        use_as_complement : bool
            If True, returns complement (negative) of the gate
        """
        super().__init__(gate_name, dimensions, {"vertices": vertices}, use_as_complement)
        if len(self.dimensions) != 2:
            raise ValueError(f"PolygonGate requires exactly 2 dimensions, got {len(self.dimensions)}")
        self.vertices = self._hyperparams["vertices"]

    @property
    def vertices(self) -> NDArray[np.float64]:
        """Access vertices hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["vertices"])
        except KeyError:
            raise ValueError("PolygonGate vertices have not been set. Please fit the gate first.")

    @vertices.setter
    def vertices(self, value: Iterable[Iterable[float]]) -> None:
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

        self.params["vertices"] = coords


    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "PolygonGate":
        return self

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        """For PolygonGate, fit just copies hyperparams to params."""
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        """Apply polygon gate using winding number algorithm."""

        from flowutils import gating
        coords = events_slice[self.dimensions].values
        mask: NDArray[np.bool_] = gating.points_in_polygon(self.vertices, coords)

        # Apply complement if requested
        if self.use_as_complement:
            mask = ~mask

        return {self.gate_name: mask}

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
        dimensions: list[str],
        center: list[float] | NDArray,
        covariance_matrix: list[list[float]] | NDArray,
        distance_square: float,
        use_as_complement: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        gate_name : str
            Human-readable name
        dimensions : list[str]
            Dimension IDs (must have 2 or more)
        center : list[float] | NDArray
            Center of the ellipsoid in each dimension
        covariance_matrix : list[list[float]] | NDArray
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
    def center(self) -> NDArray[np.float64]:
        """Access center hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["center"])
        except KeyError:
            raise ValueError("EllipsoidGate center has not been set. Please fit the gate first.")

    @center.setter
    def center(self, value: NDArray[np.float64]) -> None:
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
    def covariance_matrix(self) -> NDArray[np.float64]:
        """Access covariance_matrix hyperparameter or fitted params."""
        try:
            return np.asarray(self.params["covariance_matrix"])
        except KeyError:
            raise ValueError("EllipsoidGate covariance_matrix has not been set. Please fit the gate first.")

    @covariance_matrix.setter
    def covariance_matrix(self, value: NDArray[np.float64]) -> None:
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

    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "EllipsoidGate":
        """
        EllipsoidGate requires fitting to learn parameters from event data.
        """
        return self

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        pass

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        """Apply ellipsoid gate using Mahalanobis distance."""

        try:
            center = self.center
            cov_matrix = self.covariance_matrix
            distance_sq = self.distance_square
        except KeyError as e:
            raise ValueError(f"EllipsoidGate missing fitted parameter: {e}")
        coords = events_slice.values

        from flowutils import gating
        mask: NDArray[np.bool_] = gating.points_in_ellipsoid(cov_matrix, center, distance_sq, coords)

        # Apply complement if requested
        if self.use_as_complement:
            mask = ~mask

        return {self.gate_name: mask}

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # Convert numpy arrays to lists for JSON serialization
        for p in ("hyperparams", "params"):
            for k in ("center", "covariance_matrix"):
                if k in base[p] and not isinstance(base[p][k], list):
                    base[p][k] = base[p][k].tolist()
        return base


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
        dividers: Mapping[str, Iterable[float]],
        quadrants: Mapping[str, Iterable[tuple[str, float]]],
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
            if not isinstance(points, Iterable):
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

    def _compute_quadrants(self):
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

    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "QuadrantGate":
        return self

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        return

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        """
        Apply quadrant gate to generate masks for each quadrant.

        Returns
        -------
        dict[str, NDArray[np.bool_]]
            Dictionary mapping quadrant_id to boolean mask for that quadrant.
            Each mask indicates which events fall within the quadrant's bounds.
        """
        results: dict[str, NDArray[np.bool_]] = {}

        for quad_id, quad_def in self.quadrants.items():
            # Start with all events True for this quadrant
            quad_mask = np.ones(len(events_slice), dtype=np.bool_)

            # Apply each divider's range restrictions
            for div_id, (min_val, max_val) in quad_def.items():
                col_vals = np.asarray(events_slice[div_id].values)

                if min_val is not None:
                    np.bitwise_and(quad_mask, col_vals >= min_val, out=quad_mask)
                if max_val is not None:
                    np.bitwise_and(quad_mask, col_vals < max_val, out=quad_mask)

            results[quad_id] = quad_mask

        return results

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
        use_as_complement: bool = False,
        **kwargs: Any,
    ) -> None:
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
        use_as_complement : bool
            If True, returns complement (negative) of the gate result
        """
        super().__init__(gate_name, [], {"expression": expression}, use_as_complement)
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

    def _parse_expression(self) -> "BooleanGate":
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

        return self

    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "BooleanGate":
        """
        Parses the provided boolean expression.
        BooleanGate doesn't require fitting (no event data to learn from).
        """
        return self

    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
        return

    def apply(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> dict[str, NDArray[np.bool_]]:
        """
        Apply boolean expression to masks from parent gates.

        Parameters
        ----------
        events : ad.AnnData
            Not used for BooleanGate, but accepted for interface consistency
        mask : dict[str, NDArray[np.bool_]]
            Dictionary of masks from parent gates, mapping variable names to boolean arrays.
            Must contain all variables referenced in the expression.

        Returns
        -------
        dict[str, NDArray[np.bool_]]
            Dictionary with single key self.mask_key containing the result of the expression
        """
        # Ensure we have all needed variables
        provided_vars = set(mask.keys())
        missing_vars = self.variables - provided_vars
        if missing_vars:
            raise ValueError(
                f"BooleanGate '{self.gate_name}' missing required masks: {missing_vars}. "
                f"Provided masks: {provided_vars}"
            )

        # Evaluate the expression with the provided masks using pandas
        mask_df = pd.DataFrame(mask)
        if len(mask_df) != events.n_obs:
            raise ValueError(
                f"BooleanGate '{self.gate_name}' received masks of length {len(mask_df)}, "
                f"but events AnnData has length {events.n_obs}"
            )
        try:
            result = mask_df.eval(self.expression, local_dict={}, global_dict={})
        except Exception as e:
            raise ValueError(f"Error evaluating boolean expression '{self.expression}': {e}")

        result = np.asarray(result, dtype=bool)

        # Apply complement if requested
        if self.use_as_complement:
            result = ~result

        return {self.gate_name: result}

    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        raise NotImplementedError("BooleanGate uses apply() override instead of _apply_gate")

    def to_dict(self) -> dict[str, Any]:
        """Serialize gate to dictionary."""
        base = super().to_dict()
        base["params"]["variables"] = list(self.variables)  # Add for reference
        return base
