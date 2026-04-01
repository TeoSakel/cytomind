from abc import ABC, abstractmethod
from typing import Any, Hashable, Sequence, Mapping, TYPE_CHECKING
from copy import deepcopy
import json
import hashlib
from cytomind.domain.pipeline import NumpyEncoder

import numpy as np
import plotly.graph_objects as go
from plotly.colors import get_colorscale, sample_colorscale
from plotly.subplots import make_subplots
from cytomind.visualization.gates import (
    _compute_density_colors,
    _create_scatter_trace,
    _format_gate_plot,
)

if TYPE_CHECKING:
    from cytomind.domain.gates import GateNode
    from cytomind.domain.constants import BooleanArray, FloatArray
    from anndata import AnnData
    from pandas import DataFrame
else:
    GateNode = object
    BooleanArray = object
    FloatArray = object
    AnnData = object
    DataFrame = object

class Gate(ABC):
    """
    Abstract base class for flow cytometry gates following scikit-learn conventions.

    Gates separate hyperparameters (user-configured at initialization) from
    fitted params and runtime state. Gates always operate on event arrays, but
    the fitted runtime state can be stored and reloaded independently of fitting.

    Parameter Model
    ---------------
    A Gate maintains four distinct state sets:

    1. **hyperparams**: User-provided configuration at initialization.
       - Static; do not change after __init__
       - Examples: n_clusters=5, distance_square=2.5, vertices=[[0, 0], [1, 1]]
       - Stored in _hyperparams (read-only via .hyperparams property)

    2. **params**: Apply-time fitted values required for apply()/score().
       - For parameter-only gates (tunable=False): copied from hyperparams during fit()
       - For learnable gates (tunable=True): computed during fit() (e.g., fitted vertices, boundaries)
       - Used by apply() to generate masks
       - Mutable; updated by fit() and can be set by from_node() when reconstructing

    3. **diagnostics**: Metadata from fitting; not needed for apply().
       - Used for QC validation and analysis
       - Examples: silhouette_score, n_iterations_to_convergence
       - Optional; gates can leave empty if not applicable

    4. **state**: Runtime-only in-memory state persisted separately from GateNode params.
       - Used to decouple fit() from later apply()/score() calls
       - Examples: fitted sklearn/statsmodels objects, cached decompositions
       - Stored to disk by export_state()/import_state()
       - Mutable; may contain non-JSON Python objects

    GateNode Integration
    --------------------
    When persisting to GateNode (for the gating strategy graph), use the param_dict():
    - GateNode.params: stores batch-level param_dict() (hyperparams + params + diagnostics + state)
    - GateNode.custom_gates[sample_id]: stores sample-specific param_dict() overrides

    To reconstruct a Gate from GateNode, use from_node(node, sample_id=None):
    - Automatically fetches params from batch-level or sample-specific storage
    - Initializes a new Gate with proper hyperparams and applies saved params

    Key methods:
    - fit(events, mask=None): Learn gate parameters from events (optional for parameter-only gates)
    - apply(events, mask=None): Generate boolean mask for events passing through gate
    - from_node(node, sample_id=None): Reconstruct Gate from persisted GateNode (classmethod)
    - to_node_params(): Extract param structure for GateNode storage
    - export_state()/import_state(): Pure serialization hooks for dataloader-owned artifact I/O

    Both fit/apply work on AnnData objects; gates internally select their dimensions.
    """

    gate_type: str
    default_tunable: bool = False

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] | None = None,
        use_as_complement: bool = False,
        tunable: bool | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize gate with hyperparameters (user-provided configuration).

        Parameters
        ----------
        gate_name : str
            Human-readable name for the gate
        dimensions : Sequence[str]
            List of dimension/channel IDs that this gate operates on
        hyperparams : Mapping[str, Any]
            Dictionary of hyperparameters for the gate used during fitting (e.g., number of clusters for a clustering gate).
            These are static user-provided settings, not learned during fit().
        use_as_complement : bool
            If False, mask key is "{gate_name}.pos" (default)
            If True, mask key is "{gate_name}.neg" (complement of the gate)

        Notes
        -----
        - self.params is initially empty; populated during fit() or set via from_node()
        - self.diagnostics is initially empty; may be populated during fit() by subclasses
        - self.state is initially empty; may be populated during fit() by subclasses
        - For subclasses: do NOT override __init__; put initialization logic in _fit_gate()
        """
        self.gate_name = gate_name
        self.dimensions = list(dimensions)
        self.use_as_complement = use_as_complement
        self.tunable = type(self).default_tunable if tunable is None else bool(tunable)
        self._hyperparams: dict[str, Any] = dict(hyperparams or {})
        self._hyperparams.update(kwargs)
        self.params: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}
        self.state: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.gate_name}, dims={self.dimensions})"

    @property
    def hyperparams(self) -> dict[str, Any]:
        """Read-only access to hyperparameters."""
        return self._hyperparams

    def to_dict(self) -> dict[str, Any]:
        """Serialize gate to dictionary."""
        return {
            "gate_class": self.__class__.__name__,
            "gate_name": self.gate_name,
            "dimensions": self.dimensions,
            "use_as_complement": self.use_as_complement,
            "hyperparams": self._hyperparams,
            "params": self.params,
            "diagnostics": self.diagnostics,
            "state": self._export_json_state_payload(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gate":
        """Deserialize gate from full dictionary representation (includes all fields).

        Use from_node() to reconstruct a Gate from a GateNode, as it handles the
        batch/sample parameter structure automatically.

        Parameters
        ----------
        data : Mapping[str, Any]
            Dictionary with keys: gate_name, dimensions, use_as_complement, hyperparams, params, diagnostics

        Returns
        -------
        Gate
            Reconstructed gate instance with params and diagnostics restored
        """
        gate = cls(
            gate_name=data["gate_name"],
            dimensions=data["dimensions"],
            use_as_complement=data.get("use_as_complement", False),
            **data.get("hyperparams", {}),
        )
        gate.params = dict(data.get("params", {}))
        gate.diagnostics = dict(data.get("diagnostics", {}))
        gate._import_json_state_payload(data.get("state", {}))
        return gate

    @classmethod
    def from_node(cls, node: GateNode, sample_id: str | None = None) -> "Gate":
        """Reconstruct a Gate instance from a persisted GateNode.

        Handles the parameter precedence logic:
        - If sample_id is provided: use sample-specific params from node.custom_gates[sample_id],
          falling back to batch-level node.params
        - If sample_id is None: use batch-level node.params

        Parameters
        ----------
        node : GateNode
            The GateNode to load from (typically from a gating strategy)
        sample_id : str, optional
            Sample identifier for looking up sample-specific parameter overrides.
            If None, uses batch-level parameters.
            If provided but not in custom_gates, falls back to batch-level params.

        Returns
        -------
        Gate
            New Gate instance initialized with node's properties and params.
            Ready for apply() (or refit if tunable=True)

        Examples
        --------
        Load a gate with batch-level params:
        >>> gate = Gate.from_node(node)
        >>> masks = gate.apply(adata)

        Load a gate with sample-specific param overrides:
        >>> gate = Gate.from_node(node, sample_id="sample_123")
        >>> masks = gate.apply(adata)
        """

        assert node.gate_type == cls.gate_type, f"Gate type mismatch: expected {cls.gate_type}, got {node.gate_type}"

        merged_params = node.get_params_for_sample(sample_id)

        # merged_params has structure: {"hyperparams": {...}, "params": {...}, "diagnostics": {...}, "state": {...}}
        hyperparams = merged_params.get("hyperparams", {})
        params = merged_params.get("params", {})
        diagnostics = merged_params.get("diagnostics", {})

        # Reconstruct gate with node's base properties and hyperparams
        # Note: Unpack hyperparams as **kwargs to match subclass __init__ signatures
        gate = cls(
            gate_name=node.name or node.id,
            dimensions=node.dimensions,
            use_as_complement=node.use_as_complement,
            **hyperparams,
        )

        # Apply saved params and diagnostics.
        # If persisted params are empty, keep constructor-initialized params
        # (important for parameter-only gates initialized from hyperparams).
        if params and gate.validate_params(params):
            gate.params = dict(params)
            gate.diagnostics = dict(diagnostics)

        return gate

    def _export_json_state_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable subset of ``state`` for default persistence."""
        payload = deepcopy(self.state)
        payload["tunable"] = self.tunable
        return payload

    def _import_json_state_payload(self, payload: Mapping[str, Any] | None) -> None:
        """Restore the JSON-serializable subset of ``state`` from persistence."""
        restored_state = deepcopy(dict(payload or {}))
        self.tunable = bool(restored_state.get("tunable", self.tunable))
        self.state = restored_state

    @abstractmethod
    def validate_params(self, params: Mapping[str, Any]) -> bool:
        """Validate the gate's parameters.

        Returns
        -------
        bool
            True if parameters are valid, False otherwise.
        """
        pass

    def copy(self) -> "Gate":
        """Create a deep copy of the gate (includes current params and diagnostics).

        Creates a new gate instance with all state from the current gate:
        - Same hyperparameters (user-provided configuration)
        - Same params (learned values or fitted state)
        - Same diagnostics (QC metadata from fitting)

        Useful for creating independent copies when you need to refit a gate
        on different data without affecting the original instance. The returned
        gate is a completely independent object—modifying its params or fitting
        will not affect the source gate.

        Returns
        -------
        Gate
            New gate instance with deep copies of params and diagnostics.
            Ready for fit() or apply() independently.

        Examples
        --------
        >>> batch_gate = original_gate.copy()
        >>> batch_gate.fit(sample_events)  # Won't affect original_gate
        """
        return deepcopy(self)

    def export_state(self) -> dict[str, Any]:
        """Return a pure-serialization state bundle for dataloader-managed persistence."""
        return {
            "manifest": {
                "serializer": "json",
                "version": 1,
                "artifacts": {
                    "state": "state.json",
                },
            },
            "artifacts": {
                "state.json": {
                    "params": deepcopy(self.params),
                    "diagnostics": deepcopy(self.diagnostics),
                    "state": self._export_json_state_payload(),
                },
            },
        }

    def import_state(
        self,
        manifest: Mapping[str, Any],
        artifacts: Mapping[str, Any],
    ) -> None:
        """Restore runtime state from a dataloader-loaded state bundle."""
        artifact_name = "state.json"
        manifest_artifacts = manifest.get("artifacts", {})
        if isinstance(manifest_artifacts, Mapping):
            artifact_name = str(manifest_artifacts.get("state", artifact_name))

        raw_state = artifacts.get(artifact_name)
        if raw_state is None and artifact_name.endswith(".json"):
            raw_state = artifacts.get(artifact_name.removesuffix(".json"))
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"State artifact '{artifact_name}' is required to restore gate state.")

        params = raw_state.get("params", {})
        diagnostics = raw_state.get("diagnostics", {})
        self._import_json_state_payload(raw_state.get("state", {}))
        if params and self.validate_params(params):
            self.params = dict(params)
        elif not params:
            self.params = {}
        self.diagnostics = dict(diagnostics)

    def to_node_params(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Extract params in the structure used by GateNode for persistence.

        Extracts all three parameter sets (hyperparams, params, diagnostics) in the structure
        for GateNode storage (as opposed to other serialization formats).

        Returns: {"hyperparams": {...}, "params": {...}, "diagnostics": {...}, "state": {...}}

        Use this when storing fitted gate results back to GateNode:
        - For batch-level: node.params = gate.to_node_params()
        - For sample-specific: node.custom_gates[sample_id] = gate.to_node_params()

        Returns
        -------
        dict[str, Any]
            Dictionary suitable for storing in GateNode.
        """
        gate_data = self.to_dict()
        return {
            "hyperparams": gate_data["hyperparams"],
            "params": gate_data["params"],
            "diagnostics": gate_data["diagnostics"],
            "state": {
                **dict(gate_data["state"]),
                **dict(state or {}),
            },
        }

    def generate_offsprings(
        self,
        *,
        parent_node: GateNode,
        sample_overrides: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[GateNode]:
        """Return child gate nodes generated by this gate after fitting.

        Most gates do not emit child gates and should therefore keep the
        default empty implementation.
        """
        del parent_node, sample_overrides
        return []

    def fit(self, events: AnnData) -> "Gate":
        """
        Fit gate parameters from events (optional for parameter-only gates).

        This is the main extension point for subclasses. Gates without real learning
        should copy hyperparams to params. Gates with learning (e.g., ellipsoids)
        compute and store learned values in params.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names

        Returns
        -------
        Gate
            Self, for method chaining
        """

        # TODO: Consider adding optional mask parameter for fitting on pre-filtered events (e.g., fit only on parent gate's positive events).
        if events.isbacked:
            events = events.to_memory()
        events_adata = events[:, self.dimensions]
        self._fit_gate(events_adata.to_df())
        return self

    @abstractmethod
    def _fit_gate(self, events_slice: DataFrame) -> None:
        """
        Internal fit implementation for subclasses to override.

        Subclasses implement learning logic here and populate self.params.
        For parameter-only gates, simply copy hyperparams to params.

        Parameters
        ----------
        events_slice : pd.DataFrame
            Events filtered to only this gate's dimensions
        """
        pass

    def apply(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, BooleanArray]:
        """
        Apply gate to pre-filtered events and expand result to parent-mask length.

        Parameters
        ----------
        events : ad.AnnData
            Pre-filtered event data for this gate (already subset by AddGateStep).
        mask : dict[str, BooleanArray] | None, default None
            Parent gate masks (required, must be non-empty).
            Exactly one parent/root mask for standard gates.

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping region/quadrant IDs to boolean masks.
            Output mask size equals the size of the input mask array.

        Raises
        ------
        ValueError
            If mask dict is empty.
        """
        if mask is None or len(mask) == 0:
            mask = {"root": np.ones(events.n_obs, dtype=bool)}

        if len(mask) == 1:
            parent_mask = next(iter(mask.values()))
        else:
            raise ValueError(
                f"Standard gates support single parent mask only, got {len(mask)}. "
                "Multiple parent masks are only supported by BooleanGate."
            )

        parent_mask = np.asarray(parent_mask)
        if parent_mask.dtype != np.bool_:
            raise ValueError("Gate.apply parent mask must be boolean.")

        missing = [dim for dim in self.dimensions if dim not in events.var_names]
        if missing:
            raise ValueError(f"Events is missing required dimension(s): {missing}")

        if events.isbacked:
            events = events.to_memory()
        events_adata = events[:, self.dimensions]

        n_parents = int(parent_mask.sum())
        if n_parents != events_adata.n_obs:
            raise ValueError(
                f"Parent mask selects {n_parents} events, but received {events_adata.n_obs} events. "
                "Ensure that the input events are pre-filtered to match the parent mask."
            )

        # Apply gate logic
        result = {}
        gate_results = self._apply_gate(events_adata.to_df())
        for region_id, region_mask in gate_results.items():
            if len(region_mask) != n_parents:
                raise ValueError(
                    f"Gate '{self.gate_name}' produced {len(region_mask)} rows, "
                    f"but parent mask selects {n_parents} events."
                )
            # Expand result back to the input mask size
            full_mask = np.zeros_like(parent_mask)
            full_mask[parent_mask] = region_mask
            result[region_id] = full_mask
        return result

    @abstractmethod
    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        """
        Internal apply implementation for subclasses to override.

        Parameters
        ----------
        events_slice : pd.DataFrame
            Events filtered to only this gate's dimensions

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping region/quadrant IDs to boolean masks.
            Single-region gates should use a consistent key like "default".
        """
        pass

    def score(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, FloatArray]:
        """
        Score gate affinity on pre-filtered events and expand back to parent-mask length.

        This mirrors ``apply()`` semantics: implementations score only the
        parent-selected event slice, then the result is expanded back to the
        input mask size. Events not selected by the parent mask are filled with
        ``NaN``. Implementations should use the shared soft-membership
        convention where boundary points are 0, positive scores indicate
        in-gate affinity, and negative scores indicate out-of-gate affinity.

        Parameters
        ----------
        events : ad.AnnData
            Pre-filtered event data for this gate (already subset by AddGateStep).
        mask : dict[str, BooleanArray] | None, default None
            Parent gate masks (required, must be non-empty).
            Exactly one parent/root mask for standard gates.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping region IDs to score arrays.
            Output score size equals the size of the input mask array.

        Raises
        ------
        ValueError
            If mask dict is empty.
        """
        if mask is None or len(mask) == 0:
            mask = {"root": np.ones(events.n_obs, dtype=bool)}

        if len(mask) == 1:
            parent_mask = next(iter(mask.values()))
        else:
            raise ValueError(
                f"Standard gates support single parent mask only, got {len(mask)}. "
                "Multiple parent masks are only supported by BooleanGate."
            )

        parent_mask = np.asarray(parent_mask)
        if parent_mask.dtype != np.bool_:
            raise ValueError("Gate.score parent mask must be boolean.")

        missing = [dim for dim in self.dimensions if dim not in events.var_names]
        if missing:
            raise ValueError(f"Events is missing required dimension(s): {missing}")

        if events.isbacked:
            events = events.to_memory()
        events_adata = events[:, self.dimensions]
        n_parents = int(parent_mask.sum())
        if n_parents != events_adata.n_obs:
            raise ValueError(
                f"Parent mask selects {n_parents} events, but received {events_adata.n_obs} events. "
                "Ensure that the input events are pre-filtered to match the parent mask."
            )

        result: dict[str, np.ndarray] = {}
        gate_scores = self._score_gate(events_adata.to_df())
        for region_id, region_scores in gate_scores.items():
            scores = np.asarray(region_scores, dtype=np.float32).ravel()
            if len(scores) != n_parents:
                raise ValueError(
                    f"Gate '{self.gate_name}' produced {len(scores)} score rows, "
                    f"but parent mask selects {n_parents} events."
                )
            full_scores = np.full(parent_mask.shape, np.nan, dtype=np.float32)
            full_scores[parent_mask] = scores
            result[region_id] = full_scores
        return result

    @abstractmethod
    def _score_gate(self, events_slice: DataFrame) -> dict[str, FloatArray]:
        """
        Internal score implementation for subclasses to override.

        Parameters
        ----------
        events_slice : pd.DataFrame
            Events filtered to only this gate's dimensions

        Returns
        -------
        dict[str, FloatArray]
            Dictionary mapping region/quadrant IDs to soft-membership score arrays.
        """
        pass

    def fit_apply(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, BooleanArray]:
        """
        Convenience method to fit and then apply the gate in one step.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, BooleanArray] | None, default None
            Dictionary of boolean masks from parent gates (default: None → treated as no mask, fit/apply on all events)

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping region/quadrant IDs to boolean masks
        """
        return self.fit(events).apply(events, mask)

    def _param_key(self) -> Any:
        """
        Generate a hashable object that uniquely identifies the gate's configuration.

        This is used for hashing and equality checks. Subclasses should implement this
        to include all relevant hyperparameters and learned parameters that define the gate.

        Returns
        -------
        Any
            A hashable object containing all relevant parameters for hashing.
        """
        return dict(self.params)

    def __hash__(self) -> int:
        """Deterministic hash for cross-process stability.

        Uses a canonical JSON serialization of the gate-identifying fields
        and returns an integer derived from SHA-256. This avoids Python's
        per-process hash randomization and yields stable keys across runs.
        """
        key_obj = {
            "gate_type": self.gate_type,
            "dimensions": list(self.dimensions),
            "use_as_complement": bool(self.use_as_complement),
            "params": self._param_key()
        }
        json_s = json.dumps(key_obj, cls=NumpyEncoder, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(json_s.encode("utf-8")).hexdigest()
        # Return a stable integer derived from the digest
        return int(digest, 16)

    def _compute_ratio(self, events: AnnData, mask: dict[str, Any] | None = None, region_id: str | None = None) -> float | None:
        """Compute fraction of events included in this gate/region.

        Strategy:
        1) If `mask` is provided and contains this gate's `gate_name`, use that array.
        2) Otherwise call `self.apply(events, mask=mask)` to obtain masks and compute the fraction.

        Returns fraction in [0,1] or `None` on failure.
        """
        try:
            # 1) mask directly provided for this gate
            if mask is not None and self.gate_name in mask:
                arr = np.asarray(mask[self.gate_name])
                if arr.size == 0:
                    return None
                return float(arr.mean())
        except Exception:
            pass

        # 2) Fallback: apply the gate to compute masks
        try:
            if events is None:
                return None
            if getattr(events, "isbacked", False):
                events = events.to_memory()

            res = self.apply(events, mask=mask)
            if region_id is not None and region_id in res:
                arr = np.asarray(res[region_id])
            elif self.gate_name in res:
                arr = np.asarray(res[self.gate_name])
            else:
                arr = np.asarray(next(iter(res.values())))

            if arr.size == 0:
                return None
            return float(arr.mean())
        except Exception:
            return None

    ### -------- Plotting -------- ###

    def plot(
        self,
        events: AnnData,
        mask: dict[str, BooleanArray] | None = None,
        dimensions: Sequence[str] | None = None,
        *,
        marginals: bool = False,
        plot_type: str = "histogram",
        marginal_plot_type: str = "histogram",
        color_by: str = "density",
        hist_nbins: int | str = 100,
        hist_color: str = "rgba(100, 100, 200, 0.6)",
        histnorm: str = "probability",
        density_nbins: int = 50,
        density_log_scale: bool = True,
        marker_size: int = 3,
        colorscale: str | Sequence[str] | None = None,
        use_gl: bool = True,
        max_points: int = 50000,
        downsample_seed: int = 0,
        gate_line_color: str = "red",
        gate_line_width: int = 2,
        gate_line_dash: str = "dash",
        gate_fill: bool = True,
        gate_fill_color: str = "rgba(255, 0, 0, 0.1)",
        fail_hist_color: str = "rgba(255, 0, 0, 0.35)",
        pass_hist_color: str = "rgba(0, 255, 0, 0.45)",
        margin_pad_scale: float = 0.01,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
        show_ratio: bool = True,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Generate diagnostic plot for the gate.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, BooleanArray] | None, default None
            Dictionary of boolean masks from parent gates (default: None → treated as no mask, fit/apply on all events)
        dimensions : Sequence[str] | None, (default None: use all gate dimensions present in events.var_names)
            Optional list of dimensions to plot (gate implementations may accept/require it)
        marginals : bool, default False
            If True, show marginal panels for 2D plots.
        plot_type : str, default "histogram"
            Diagonal/1D plot style. Supported by the base implementation: ``"histogram"`` or ``"density"``.
        marginal_plot_type : str, default "histogram"
            Marginal plot style for 2D views. Supported by the base implementation: ``"histogram"`` or ``"density"``.
        color_by : str, default "density"
            Scatter coloring mode. Supported by the base implementation: ``"density"`` or ``"score"``.
        hist_nbins, hist_color, histnorm, density_nbins, density_log_scale, marker_size, colorscale,
        use_gl, max_points, downsample_seed, gate_line_color, gate_line_width, gate_line_dash,
        gate_fill, gate_fill_color, fail_hist_color, pass_hist_color, margin_pad_scale, title,
        width, height
            Shared plotting options used by the base geometric plotting pipeline.
            When plotting multiple masks, ``colorscale`` may be either a Plotly colorscale name
            or an explicit sequence of colors whose length matches the number of masks.
            ``hist_nbins`` is passed through to ``numpy.histogram_bin_edges`` and may be
            an integer or any NumPy-supported automatic bin estimator string.
        show_ratio : bool, default True
            If True, annotate the plot with the ratio (fraction) of events included in the gate/mask.
            Placement is best-effort and will be centered roughly on the geometric mask when possible.
        **kwargs : Any
            Optional plot configuration parameters passed through by callers.

        Returns
        -------
        go.Figure
            Plotly figure object for visualization.
        """
        dimensions = dimensions or [dim for dim in self.dimensions if dim in events.var_names]
        plot_dims = self._resolve_plot_dimensions(dimensions, available_dimensions=self.dimensions)
        if not plot_dims:
            raise ValueError(f"{self.__class__.__name__}.plot() requires at least one dimension")

        n_dims = len(plot_dims)
        plot_kwargs = dict(kwargs)
        plot_kwargs.update(
            marginals=marginals,
            plot_type=plot_type,
            marginal_plot_type=marginal_plot_type,
            color_by=color_by,
            hist_nbins=hist_nbins,
            hist_color=hist_color,
            histnorm=histnorm,
            density_nbins=density_nbins,
            density_log_scale=density_log_scale,
            marker_size=marker_size,
            colorscale=colorscale,
            use_gl=use_gl,
            max_points=max_points,
            downsample_seed=downsample_seed,
            gate_line_color=gate_line_color,
            gate_line_width=gate_line_width,
            gate_line_dash=gate_line_dash,
            gate_fill=gate_fill,
            gate_fill_color=gate_fill_color,
            fail_hist_color=fail_hist_color,
            pass_hist_color=pass_hist_color,
            margin_pad_scale=margin_pad_scale,
            title=title,
            width=width,
            height=height,
            show_ratio=show_ratio,
            mask=mask,
        )

        if n_dims == 1:
            return self._plot_base_1d(events, plot_dims, **plot_kwargs)
        if n_dims == 2:
            return self._plot_base_2d(events, plot_dims, **plot_kwargs)
        return self._plot_base_nd(events, plot_dims, **plot_kwargs)

    def _resolve_plot_dimensions(
        self,
        requested_dimensions: Sequence[str] | None,
        available_dimensions: Sequence[str] | None = None,
        *,
        min_dims: int = 1,
        max_dims: int | None = None,
    ) -> list[str]:
        """Resolve and validate dimensions used for plotting.

        Parameters
        - requested_dimensions: explicit dims requested by caller
        - available_dimensions: list of valid dimension ids (defaults to this gate's dimensions)
        - min_dims, max_dims: bounds on number of dimensions
        """
        gate_dims = list(available_dimensions) if available_dimensions is not None else list(self.dimensions)
        gate_cls_name = self.__class__.__name__

        if requested_dimensions is None:
            dims = list(gate_dims)
        else:
            dims = list(requested_dimensions)

        if not dims:
            raise ValueError(f"{gate_cls_name}.plot() requires at least one dimension")

        dim_set = set(dims)

        if len(dim_set) != len(dims):
            raise ValueError(f"{gate_cls_name}.plot() dimensions must be unique")

        if len(dims) < min_dims:
            raise ValueError(f"{gate_cls_name}.plot() requires at least {min_dims} dimensions")
        if max_dims is not None and len(dims) > max_dims:
            raise ValueError(f"{gate_cls_name}.plot() supports at most {max_dims} dimensions")

        unknown = dim_set - set(gate_dims)
        if unknown:
            raise ValueError(
                f"{gate_cls_name}.plot() received dimensions not in available dimensions: {unknown}. "
                f"Available dimensions: {list(gate_dims)}"
            )

        return dims

    def _downsample_indices(self, n_points: int, max_points: int, seed: int) -> np.ndarray | None:
        if max_points <= 0 or n_points <= max_points:
            return None
        rng = np.random.default_rng(seed)
        return rng.choice(n_points, size=max_points, replace=False)

    def _resolve_gate_plot_mask(
        self,
        events: AnnData,
        mask: dict[str, BooleanArray] | None,
    ) -> np.ndarray:
        if mask is not None and self.gate_name in mask:
            arr = np.asarray(mask[self.gate_name], dtype=bool).ravel()
            if arr.shape[0] == events.n_obs:
                return arr

        resolved = self.apply(events, mask=mask)[self.gate_name]
        return np.asarray(resolved, dtype=bool).ravel()

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
        if values.size == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), None

        edges = self._resolve_histogram_bin_edges(values, nbins)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0

        pass_counts, _ = np.histogram(values[pass_mask], bins=edges)
        fail_counts, _ = np.histogram(values[fail_mask], bins=edges)

        pass_heights = self._normalize_histogram(pass_counts, widths, histnorm, int(values.size))
        fail_heights = self._normalize_histogram(fail_counts, widths, histnorm, int(values.size))
        total_heights = pass_heights + fail_heights
        max_height = float(np.max(total_heights)) * 1.15 if total_heights.size else None
        return centers, widths, pass_heights, fail_heights, max_height

    def _density_curve(
        self,
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
        row: int,
        col: int,
        orientation: str,
        fail_color: str,
        pass_color: str,
    ) -> float | None:
        centers, widths, pass_heights, fail_heights, axis_max = self._stacked_histogram_profile(
            values,
            pass_mask,
            fail_mask,
            nbins,
            histnorm,
        )
        if centers.size == 0:
            return axis_max

        if orientation == "h":
            fig.add_trace(go.Bar(x=fail_heights, y=centers, width=widths, orientation="h", name="Fail", marker=dict(color=fail_color), opacity=0.65, showlegend=False), row=row, col=col)
            fig.add_trace(go.Bar(x=pass_heights, y=centers, width=widths, orientation="h", name="Pass", marker=dict(color=pass_color), opacity=0.8, showlegend=False), row=row, col=col)
        else:
            fig.add_trace(go.Bar(x=centers, y=fail_heights, width=widths, name="Fail", marker=dict(color=fail_color), opacity=0.65, showlegend=False), row=row, col=col)
            fig.add_trace(go.Bar(x=centers, y=pass_heights, width=widths, name="Pass", marker=dict(color=pass_color), opacity=0.8, showlegend=False), row=row, col=col)

        return axis_max

    def _add_density_marginal_traces(
        self,
        fig: go.Figure,
        values: FloatArray,
        pass_mask: BooleanArray,
        fail_mask: BooleanArray,
        *,
        row: int,
        col: int,
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
                fig.add_trace(go.Scatter(x=density, y=grid, mode="lines", line=dict(color=color), name=name, showlegend=False), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(x=grid, y=density, mode="lines", line=dict(color=color), name=name, showlegend=False), row=row, col=col)
        return None if axis_max is None else axis_max * 1.15

    def _plot_has_pass_fail_split(self, mask: dict[str, BooleanArray] | None) -> bool:
        return bool(mask) and len(mask) == 1

    def _resolve_plot_mask_groups(
        self,
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

        if colorscale is None:
            scale: str | Sequence[str] = "Plotly"
        else:
            scale = colorscale

        if isinstance(scale, str):
            if n_masks == 1:
                return [sample_colorscale(get_colorscale(scale), [0.5])[0]] # pyright: ignore[reportReturnType]
            sample_points = np.linspace(0.0, 1.0, n_masks).tolist()
            return list(sample_colorscale(get_colorscale(scale), sample_points)) # pyright: ignore[reportReturnType]

        palette = [str(color) for color in scale]
        if len(palette) != n_masks:
            raise ValueError(
                f"{self.__class__.__name__}.plot() expected {n_masks} mask colors, got {len(palette)}"
            )
        return palette

    def _resolve_histogram_bin_edges(self, values: FloatArray, hist_nbins: int | str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.asarray([0.0, 1.0])

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if np.isclose(vmin, vmax):
            delta = 0.5 if np.isclose(vmin, 0.0) else max(abs(vmin) * 0.05, 1e-9)
            return np.asarray([vmin - delta, vmax + delta], dtype=float)

        return np.asarray(
            np.histogram_bin_edges(arr, bins=hist_nbins),
            dtype=float,
        )

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
        return {
            axis: {
                "start": float(edges[0]),
                "end": float(edges[-1]),
                "size": size,
            }
        }

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
            selected = values[mask_values]
            if selected.size == 0:
                continue
            if plot_type == "density":
                grid, density = self._density_curve(selected)
                trace = go.Scatter(
                    x=grid,
                    y=density,
                    mode="lines",
                    name=mask_name,
                    line=dict(color=color),
                    showlegend=showlegend,
                )
            else:
                trace = go.Histogram(
                    x=selected,
                    name=mask_name,
                    marker=dict(color=color),
                    histnorm=histnorm,
                    opacity=0.85,
                    showlegend=showlegend,
                    **histogram_bins,
                )
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
                showlegend=showlegend,
                name=mask_name,
                use_gl=use_gl,
            )
            trace.marker.color = color # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            trace.marker.opacity = 0.7 # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=row, col=col)

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
            selected = values[mask_values]
            if selected.size == 0:
                continue
            if plot_type == "density":
                grid, density = self._density_curve(selected)
                if orientation == "h":
                    trace = go.Scatter(
                        x=density,
                        y=grid,
                        mode="lines",
                        line=dict(color=color),
                        name=mask_name,
                        showlegend=False,
                    )
                else:
                    trace = go.Scatter(
                        x=grid,
                        y=density,
                        mode="lines",
                        line=dict(color=color),
                        name=mask_name,
                        showlegend=False,
                    )
            else:
                if orientation == "h":
                    trace = go.Histogram(
                        y=selected,
                        orientation="h",
                        name=mask_name,
                        marker=dict(color=color),
                        histnorm=histnorm,
                        opacity=0.85,
                        showlegend=False,
                        **histogram_bins,
                    )
                else:
                    trace = go.Histogram(
                        x=selected,
                        name=mask_name,
                        marker=dict(color=color),
                        histnorm=histnorm,
                        opacity=0.85,
                        showlegend=False,
                        **histogram_bins,
                    )
            fig.add_trace(trace, row=row, col=col)

        if plot_type != "density":
            fig.update_layout(barmode="stack")

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
                    row=1 if row is None else row,
                    col=1 if col is None else col,
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
                row=1 if row is None else row,
                col=1 if col is None else col,
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

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        if len(plot_dims) > 2:
            return f"{self.__class__.__name__}: {self.gate_name} (Pair Plot)"
        return f"{self.__class__.__name__}: {self.gate_name}"

    def _plot_ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        if np.asarray(data).size == 0:
            return None
        return float(np.mean(data))

    def _plot_ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        if np.asarray(x_data).size == 0 or np.asarray(y_data).size == 0:
            return None
        return float(np.mean(x_data)), float(np.mean(y_data))

    def _plot_scatter_color_values(
        self,
        x_data: FloatArray,
        y_data: FloatArray,
        *,
        density_nbins: int,
        density_log_scale: bool,
    ) -> np.ndarray | None:
        return _compute_density_colors(
            np.asarray(x_data, dtype=float),
            np.asarray(y_data, dtype=float),
            nbins=density_nbins,
            log_scale=density_log_scale,
        )

    def _subset_plot_mask(
        self,
        mask: dict[str, BooleanArray] | None,
        indices: np.ndarray | None,
        n_obs: int,
    ) -> dict[str, np.ndarray] | None:
        if mask is None:
            return None

        subset_mask: dict[str, np.ndarray] = {}
        for key, value in mask.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                continue
            arr = arr.ravel()
            if arr.shape[0] != n_obs:
                subset_mask[key] = arr
                continue
            subset_mask[key] = arr if indices is None else arr[indices]
        return subset_mask

    def _plot_score_color_values(
        self,
        events: AnnData,
        *,
        downsample_idx: np.ndarray | None,
        mask: dict[str, BooleanArray] | None,
    ) -> np.ndarray | None:
        score_events = events if downsample_idx is None else events[downsample_idx].copy()
        score_mask = self._subset_plot_mask(mask, downsample_idx, events.n_obs)

        scores = self.score(score_events, mask=score_mask)
        score_values = scores.get(self.gate_name)
        if score_values is None and scores:
            score_values = next(iter(scores.values()))
        if score_values is None:
            return None
        return np.asarray(score_values, dtype=float).ravel()

    def _scatter_color_style(
        self,
        color_values: np.ndarray | None,
        *,
        color_by: str,
        colorscale: str | None,
    ) -> dict[str, Any]:
        resolved_colorscale = "balance" if color_by == "score" else "Viridis"
        if colorscale is not None:
            resolved_colorscale = colorscale
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

    def _add_plot_overlays_1d(
        self,
        fig: go.Figure,
        dim: str,
        data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        **kwargs: Any,
    ) -> None:
        return

    def _add_plot_overlays_2d(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        *,
        row: int | None = None,
        col: int | None = None,
        showlegend: bool = True,
        **kwargs: Any,
    ) -> None:
        return

    def _add_plot_overlays_2d_marginals(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> None:
        return

    def _plot_base_1d(self, events: AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        dim = plot_dims[0]
        data = np.asarray(events[:, dim].X).ravel()
        mask = kwargs.get("mask")
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = self._downsample_indices(data.shape[0], max_points, downsample_seed)
        mask_groups = self._resolve_plot_mask_groups(mask, events.n_obs, downsample_idx)
        pass_mask: np.ndarray | None = None
        fail_mask: np.ndarray | None = None

        if self._plot_has_pass_fail_split(mask):
            full_mask = self._resolve_gate_plot_mask(events, mask)
            pass_mask = full_mask if downsample_idx is None else full_mask[downsample_idx]
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
        if len(mask_groups) > 1:
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

        self._add_plot_overlays_1d(fig, dim, data, **kwargs)
        fig.update_xaxes(title=dim)
        fig.update_yaxes(title=yaxis_title)
        if yaxis_max is not None:
            fig.update_yaxes(range=[0.0, yaxis_max])

        try:
            if bool(kwargs.get("show_ratio", True)):
                ratio = self._compute_ratio(events, mask)
                x_pos = self._plot_ratio_position_1d(dim, data, **kwargs)
                if ratio is not None and x_pos is not None:
                    fig.add_annotation(x=x_pos, y=0.95, yref="paper", text=f"{ratio*100:.1f}%", showarrow=False)
        except Exception:
            pass

        return _format_gate_plot(fig, title=title if title is not None else self._plot_default_title(plot_dims), width=plot_width, height=plot_height)

    def _plot_base_2d(self, events: AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        x_dim, y_dim = plot_dims
        x_data = np.asarray(events[:, x_dim].X).ravel()
        y_data = np.asarray(events[:, y_dim].X).ravel()
        mask = kwargs.get("mask")
        max_points = int(kwargs.get("max_points", 50000))
        downsample_seed = int(kwargs.get("downsample_seed", 0))
        downsample_idx = self._downsample_indices(x_data.shape[0], max_points, downsample_seed)
        mask_groups = self._resolve_plot_mask_groups(mask, events.n_obs, downsample_idx)
        pass_mask: np.ndarray | None = None
        fail_mask: np.ndarray | None = None

        if downsample_idx is not None:
            x_data = x_data[downsample_idx]
            y_data = y_data[downsample_idx]

        if bool(kwargs.get("marginals", False)) and len(mask_groups) <= 1:
            full_mask = self._resolve_gate_plot_mask(events, mask)
            pass_mask = full_mask if downsample_idx is None else full_mask[downsample_idx]
            fail_mask = ~pass_mask

        density_nbins = int(kwargs.get("density_nbins", 50))
        density_log_scale = bool(kwargs.get("density_log_scale", True))
        marker_size = int(kwargs.get("marker_size", 3))
        colorscale = kwargs.get("colorscale")
        if colorscale is not None and isinstance(colorscale, str):
            colorscale = str(colorscale)
        color_by = str(kwargs.get("color_by", "density"))
        use_gl = bool(kwargs.get("use_gl", True))
        title = kwargs.get("title")
        width = kwargs.get("width")
        height = kwargs.get("height")
        plot_width = 900 if width is None else int(width)
        plot_height = 700 if height is None else int(height)

        if len(mask_groups) > 1 and bool(kwargs.get("marginals", False)):
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
            self._add_multi_mask_scatter_traces(
                fig,
                x_data,
                y_data,
                mask_groups=mask_groups,
                mask_colors=mask_colors,
                marker_size=marker_size,
                use_gl=use_gl,
                row=2,
                col=1,
            )
            self._add_plot_overlays_2d(fig, plot_dims, x_data, y_data, row=2, col=1, **kwargs)

            marginal_plot_type = str(kwargs.get("marginal_plot_type", "histogram"))
            hist_nbins = kwargs.get("hist_nbins", 100)
            histnorm = str(kwargs.get("histnorm", "probability"))
            self._add_multi_mask_marginal_traces(
                fig,
                x_data,
                mask_groups=mask_groups,
                mask_colors=mask_colors,
                plot_type=marginal_plot_type,
                hist_nbins=hist_nbins,
                histnorm=histnorm,
                row=1,
                col=1,
                orientation="v",
            )
            self._add_multi_mask_marginal_traces(
                fig,
                y_data,
                mask_groups=mask_groups,
                mask_colors=mask_colors,
                plot_type=marginal_plot_type,
                hist_nbins=hist_nbins,
                histnorm=histnorm,
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
            self._add_plot_overlays_2d_marginals(fig, plot_dims, x_data, y_data, **kwargs)
        elif pass_mask is not None and fail_mask is not None:
            colors = (
                self._plot_score_color_values(events, downsample_idx=downsample_idx, mask=mask)
                if color_by == "score"
                else self._plot_scatter_color_values(
                    x_data,
                    y_data,
                    density_nbins=density_nbins,
                    density_log_scale=density_log_scale,
                )
            )
            color_style = self._scatter_color_style(colors, color_by=color_by, colorscale=colorscale)
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
            fig.add_trace(_create_scatter_trace(x_data, y_data, colors, marker_size=marker_size, use_gl=use_gl, **color_style), row=2, col=1)
            self._add_plot_overlays_2d(fig, plot_dims, x_data, y_data, row=2, col=1, **kwargs)

            marginal_plot_type = str(kwargs.get("marginal_plot_type", "histogram"))
            fail_color = str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)"))
            pass_color = str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)"))
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
            self._add_plot_overlays_2d_marginals(fig, plot_dims, x_data, y_data, **kwargs)
        else:
            fig = go.Figure()
            if len(mask_groups) > 1:
                mask_colors = self._resolve_discrete_mask_colors(len(mask_groups), kwargs.get("colorscale"))
                self._add_multi_mask_scatter_traces(
                    fig,
                    x_data,
                    y_data,
                    mask_groups=mask_groups,
                    mask_colors=mask_colors,
                    marker_size=marker_size,
                    use_gl=use_gl,
                )
            else:
                colors = (
                    self._plot_score_color_values(events, downsample_idx=downsample_idx, mask=mask)
                    if color_by == "score"
                    else self._plot_scatter_color_values(
                        x_data,
                        y_data,
                        density_nbins=density_nbins,
                        density_log_scale=density_log_scale,
                    )
                )
                color_style = self._scatter_color_style(colors, color_by=color_by, colorscale=colorscale)
                fig.add_trace(_create_scatter_trace(x_data, y_data, colors, marker_size=marker_size, use_gl=use_gl, **color_style))
            self._add_plot_overlays_2d(fig, plot_dims, x_data, y_data, **kwargs)
            fig.update_xaxes(title=x_dim)
            fig.update_yaxes(title=y_dim)

        try:
            if bool(kwargs.get("show_ratio", True)):
                ratio = self._compute_ratio(events, mask)
                ratio_xy = self._plot_ratio_position_2d(plot_dims, x_data, y_data, **kwargs)
                if ratio is not None and ratio_xy is not None:
                    fig.add_annotation(x=float(ratio_xy[0]), y=float(ratio_xy[1]), xref="x", yref="y", text=f"{ratio*100:.1f}%", showarrow=False, bgcolor="white", opacity=0.8)
        except Exception:
            pass

        return _format_gate_plot(fig, title=title if title is not None else self._plot_default_title(plot_dims), width=plot_width, height=plot_height)

    def _plot_base_nd(self, events: AnnData, plot_dims: Sequence[str], **kwargs: Any) -> go.Figure:
        mask = kwargs.get("mask")
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
        fig = make_subplots(
            rows=n_dims,
            cols=n_dims,
            shared_xaxes=True,
            shared_yaxes=False,  # because of diagonal 1D plots
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing
        )

        use_multi_mask_coloring = len(mask_groups) > 1
        scatter_color_values = None if use_multi_mask_coloring else (
            self._plot_score_color_values(events, downsample_idx=downsample_idx, mask=mask)
            if color_by == "score"
            else None
        )
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
                    diag_pass_mask: np.ndarray | None = None
                    diag_fail_mask: np.ndarray | None = None
                    if use_multi_mask_coloring:
                        yaxis_title = self._add_multi_1d_distribution_traces(
                            fig,
                            x_values,
                            plot_type=diag_plot_type,
                            hist_nbins=hist_nbins,
                            histnorm=str(kwargs.get("histnorm", "probability")),
                            mask_groups=mask_groups,
                            mask_colors=mask_colors,
                            row=row_idx,
                            col=col_idx,
                            showlegend=(row_idx == 1 and col_idx == 1),
                        )
                        yaxis_max = None
                    else:
                        if self._plot_has_pass_fail_split(mask):
                            full_mask = self._resolve_gate_plot_mask(events, mask)
                            diag_pass_mask = full_mask if downsample_idx is None else full_mask[downsample_idx]
                            diag_fail_mask = ~diag_pass_mask
                        yaxis_title, yaxis_max = self._add_1d_distribution_traces(
                            fig,
                            x_values,
                            plot_type=diag_plot_type,
                            hist_nbins=hist_nbins,
                            hist_color=hist_color,
                            histnorm=str(kwargs.get("histnorm", "probability")),
                            pass_mask=diag_pass_mask,
                            fail_mask=diag_fail_mask,
                            fail_color=str(kwargs.get("fail_hist_color", "rgba(255, 0, 0, 0.35)")),
                            pass_color=str(kwargs.get("pass_hist_color", "rgba(0, 255, 0, 0.45)")),
                            row=row_idx,
                            col=col_idx,
                        )
                    self._add_plot_overlays_1d(fig, x_dim, x_values, row=row_idx, col=col_idx, **kwargs)
                    if col_idx == 1:
                        fig.update_yaxes(title=yaxis_title, row=row_idx, col=col_idx)
                    if yaxis_max is not None:
                        fig.update_yaxes(range=[0.0, yaxis_max], row=row_idx, col=col_idx)
                else:
                    if use_multi_mask_coloring:
                        self._add_multi_mask_scatter_traces(
                            fig,
                            x_values,
                            y_values,
                            mask_groups=mask_groups,
                            mask_colors=mask_colors,
                            marker_size=marker_size,
                            use_gl=use_gl,
                            row=row_idx,
                            col=col_idx,
                            showlegend=(row_idx == 1 and col_idx == 2),
                        )
                    else:
                        color_values = (
                            scatter_color_values
                            if scatter_color_values is not None
                            else self._plot_scatter_color_values(
                                x_values,
                                y_values,
                                density_nbins=density_nbins,
                                density_log_scale=density_log_scale,
                            )
                        )
                        color_style = scatter_color_style if scatter_color_values is not None else self._scatter_color_style(color_values, color_by="density", colorscale=colorscale)
                        fig.add_trace(_create_scatter_trace(x_values, y_values, color_values, marker_size=marker_size, use_gl=use_gl, showlegend=False, **color_style), row=row_idx, col=col_idx)
                    self._add_plot_overlays_2d(fig, (x_dim, y_dim), x_values, y_values, row=row_idx, col=col_idx, showlegend=False, **kwargs)

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
                ratio = self._compute_ratio(events, mask)
                if ratio is not None:
                    fig.add_annotation(x=0.5, y=1.02, xref="paper", yref="paper", text=f"{ratio*100:.1f}%", showarrow=False, bgcolor="white", opacity=0.8)
        except Exception:
            pass

        return _format_gate_plot(fig, title=kwargs.get("title") or self._plot_default_title(plot_dims), width=plot_width, height=plot_height)
