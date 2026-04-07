from abc import ABC, abstractmethod
from typing import Any, Hashable, Sequence, Mapping, TYPE_CHECKING
from copy import deepcopy
import json
import hashlib
from cytomind.domain.pipeline import NumpyEncoder

import numpy as np
import plotly.graph_objects as go
from cytomind.visualization import GenericPlotEngine

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
        """

        assert node.gate_type == cls.gate_type, f"Gate type mismatch: expected {cls.gate_type}, got {node.gate_type}"

        merged_params = node.get_params_for_sample(sample_id)
        hyperparams = merged_params.get("hyperparams", {})
        params = merged_params.get("params", {})
        diagnostics = merged_params.get("diagnostics", {})

        gate = cls(
            gate_name=node.id,
            dimensions=node.dimensions,
            use_as_complement=node.use_as_complement,
            **hyperparams,
        )

        if params and gate.validate_params(params):
            gate.params = dict(params)
            gate.diagnostics = dict(diagnostics)

        return gate

    @staticmethod
    def externalize_numpy_arrays(
        value: Any,
        *,
        prefix: str,
        arrays: dict[str, np.ndarray],
    ) -> Any:
        if isinstance(value, np.ndarray):
            arrays[prefix] = np.asarray(value)
            return {"__npz__": prefix}
        if isinstance(value, Mapping):
            return {
                str(key): Gate.externalize_numpy_arrays(val, prefix=f"{prefix}__{key}", arrays=arrays)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [
                Gate.externalize_numpy_arrays(item, prefix=f"{prefix}__{idx}", arrays=arrays)
                for idx, item in enumerate(value)
            ]
        return value

    @staticmethod
    def restore_numpy_arrays(
        value: Any,
        *,
        arrays: Mapping[str, np.ndarray],
    ) -> Any:
        if isinstance(value, Mapping):
            if set(value.keys()) == {"__npz__"}:
                key = str(value["__npz__"])
                try:
                    return np.asarray(arrays[key])
                except KeyError as exc:
                    raise ValueError(f"Missing array artifact entry {key!r}.") from exc
            return {str(key): Gate.restore_numpy_arrays(val, arrays=arrays) for key, val in value.items()}
        if isinstance(value, list):
            return [Gate.restore_numpy_arrays(item, arrays=arrays) for item in value]
        return value

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
        """Validate the gate's parameters."""
        pass

    def copy(self) -> "Gate":
        """Create a deep copy of the gate (includes current params and diagnostics)."""
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
        """Extract params in the structure used by GateNode for persistence."""
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
        """Return child gate nodes generated by this gate after fitting."""
        del parent_node, sample_overrides
        return []

    def fit(self, events: AnnData) -> "Gate":
        """Fit gate parameters from events."""
        if events.isbacked:
            events = events.to_memory()
        events_adata = events[:, self.dimensions]
        self._fit_gate(events_adata.to_df())
        return self

    @abstractmethod
    def _fit_gate(self, events_slice: DataFrame) -> None:
        """Internal fit implementation for subclasses to override."""
        pass

    def apply(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, BooleanArray]:
        """Apply gate to pre-filtered events and expand result to parent-mask length."""
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

        result = {}
        gate_results = self._apply_gate(events_adata.to_df())
        for region_id, region_mask in gate_results.items():
            if len(region_mask) != n_parents:
                raise ValueError(
                    f"Gate '{self.gate_name}' produced {len(region_mask)} rows, "
                    f"but parent mask selects {n_parents} events."
                )
            full_mask = np.zeros_like(parent_mask)
            full_mask[parent_mask] = region_mask
            result[region_id] = full_mask
        return result

    @abstractmethod
    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanArray]:
        """Internal apply implementation for subclasses to override."""
        pass

    def score(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, FloatArray]:
        """Score gate affinity on pre-filtered events and expand back to parent-mask length."""
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
        """Internal score implementation for subclasses to override."""
        pass

    def fit_apply(self, events: AnnData, mask: dict[str, BooleanArray] | None = None) -> dict[str, BooleanArray]:
        """Convenience method to fit and then apply the gate in one step."""
        return self.fit(events).apply(events, mask)

    def _param_key(self) -> Any:
        """Generate a hashable object that uniquely identifies the gate's configuration."""
        return dict(self.params)

    def __hash__(self) -> int:
        """Deterministic hash for cross-process stability."""
        key_obj = {
            "gate_type": self.gate_type,
            "dimensions": list(self.dimensions),
            "use_as_complement": bool(self.use_as_complement),
            "params": self._param_key()
        }
        json_s = json.dumps(key_obj, cls=NumpyEncoder, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(json_s.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _compute_ratio(self, events: AnnData, mask: dict[str, Any] | None = None, region_id: str | None = None) -> float | None:
        """Compute fraction of events included in this gate/region."""
        try:
            if mask is not None and self.gate_name in mask:
                arr = np.asarray(mask[self.gate_name])
                if arr.size == 0:
                    return None
                return float(arr.mean())
        except Exception:
            pass

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
        show_colorbar: bool = True,
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
        dimensions = dimensions or [dim for dim in self.dimensions if dim in events.var_names]
        plot_dims = self._resolve_plot_dimensions(dimensions, available_dimensions=self.dimensions)
        if not plot_dims:
            raise ValueError(f"{self.__class__.__name__}.plot() requires at least one dimension")
        primary_mask_values: np.ndarray | None = None
        if mask is not None:
            try:
                primary_mask_values = self._resolve_gate_plot_mask(events, mask)
            except Exception:
                primary_mask_values = None
        score_values = self.compute_plot_score_values(events, mask=mask) if color_by == "score" else None
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
            show_colorbar=show_colorbar,
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
        )
        return GenericPlotEngine().plot(
            events=events,
            plot_dims=plot_dims,
            mask=mask,
            primary_mask_values=primary_mask_values,
            score_values=score_values,
            decorator=self,
            **plot_kwargs,
        )

    def _resolve_plot_dimensions(
        self,
        requested_dimensions: Sequence[str] | None,
        available_dimensions: Sequence[str] | None = None,
        *,
        min_dims: int = 1,
        max_dims: int | None = None,
    ) -> list[str]:
        """Resolve and validate dimensions used for plotting."""
        gate_dims = list(available_dimensions) if available_dimensions is not None else list(self.dimensions)
        return GenericPlotEngine.resolve_plot_dimensions(
            requested_dimensions,
            available_dimensions=gate_dims,
            owner_name=f"{self.__class__.__name__}.plot()",
            min_dims=min_dims,
            max_dims=max_dims,
        )

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

    def _plot_default_title(self, plot_dims: Sequence[str]) -> str:
        if len(plot_dims) > 2:
            return f"{self.__class__.__name__}: {self.gate_name} (Pair Plot)"
        return f"{self.__class__.__name__}: {self.gate_name}"

    def default_plot_title(self, plot_dims: Sequence[str]) -> str:
        return self._plot_default_title(plot_dims)

    def _plot_ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        if np.asarray(data).size == 0:
            return None
        return float(np.mean(data))

    def ratio_position_1d(self, dim: str, data: FloatArray, **kwargs: Any) -> float | None:
        return self._plot_ratio_position_1d(dim, data, **kwargs)

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

    def ratio_position_2d(
        self,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> tuple[float, float] | None:
        return self._plot_ratio_position_2d(plot_dims, x_data, y_data, **kwargs)

    def compute_plot_score_values(
        self,
        events: AnnData,
        mask: dict[str, BooleanArray] | None = None,
    ) -> np.ndarray | None:
        score_events = events if not getattr(events, "isbacked", False) else events.to_memory()
        if self.gate_type == "Boolean":
            if mask is None:
                return None
            scores = self.score(score_events, mask=mask)
        else:
            scores = self.score(score_events, mask={"root": np.ones(score_events.n_obs, dtype=bool)})
        score_values = scores.get(self.gate_name)
        if score_values is None and scores:
            score_values = next(iter(scores.values()))
        if score_values is None:
            return None
        return np.asarray(score_values, dtype=float).ravel()

    def decorate_plot_1d(
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

    def decorate_plot_2d(
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

    def decorate_plot_2d_marginals(
        self,
        fig: go.Figure,
        plot_dims: Sequence[str],
        x_data: FloatArray,
        y_data: FloatArray,
        **kwargs: Any,
    ) -> None:
        return
