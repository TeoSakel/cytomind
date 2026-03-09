from abc import ABC, abstractmethod
from typing import Any, Sequence, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cytomind.domain.gates import GateNode
    from cytomind.domain.constants import BooleanArray
    from anndata import AnnData
    from pandas import DataFrame
else:
    GateNode = object
    BooleanArray = object
    AnnData = object
    DataFrame = object

class Gate(ABC):
    """
    Abstract base class for flow cytometry gates following scikit-learn conventions.

    Gates separate hyperparameters (user-configured at initialization) from params
    (learned values during fit). Gates are stateless and always operate on event arrays.

    Parameter Model
    ---------------
    A Gate maintains three distinct parameter sets:

    1. **hyperparams**: User-provided configuration at initialization.
       - Static; do not change after __init__
       - Examples: n_clusters=5, distance_square=2.5, vertices=[[0, 0], [1, 1]]
       - Stored in _hyperparams (read-only via .hyperparams property)

    2. **params**: Runtime state; includes hyperparams + learned values + set values.
       - For parameter-only gates (tunable=False): copied from hyperparams during fit()
       - For learnable gates (tunable=True): computed during fit() (e.g., fitted center, covariance)
       - Used by apply() to generate masks
       - Mutable; updated by fit() and can be set by from_node() when reconstructing

    3. **diagnostics**: Metadata from fitting; not needed for apply().
       - Used for QC validation and analysis
       - Examples: silhouette_score, n_iterations_to_convergence
       - Optional; gates can leave empty if not applicable

    GateNode Integration
    --------------------
    When persisting to GateNode (for the gating strategy graph), use the param_dict():
    - GateNode.params: stores batch-level param_dict() (hyperparams + params + diagnostics)
    - GateNode.custom_gates[sample_id]: stores sample-specific param_dict() overrides

    To reconstruct a Gate from GateNode, use from_node(node, sample_id=None):
    - Automatically fetches params from batch-level or sample-specific storage
    - Initializes a new Gate with proper hyperparams and applies saved params

    Key methods:
    - fit(events, mask=None): Learn gate parameters from events (optional for parameter-only gates)
    - apply(events, mask=None): Generate boolean mask for events passing through gate
    - from_node(node, sample_id=None): Reconstruct Gate from persisted GateNode (classmethod)
    - to_node_params(): Extract param structure for GateNode storage

    Both fit/apply work on AnnData objects; gates internally select their dimensions.
    """

    gate_type: str
    glm_type: str | None
    tunable: bool

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        hyperparams: Mapping[str, Any] = {},
        use_as_complement: bool = False,
        **kwargs
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
        - For subclasses: do NOT override __init__; put initialization logic in _fit_gate()
        """
        self.gate_name = gate_name
        self.dimensions = list(dimensions)
        self.use_as_complement = use_as_complement
        self._hyperparams: dict[str, Any] = dict(hyperparams)
        self._hyperparams.update(kwargs)
        self.params: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}

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
        gate.params = data.get("params", {})
        gate.diagnostics = data.get("diagnostics", {})
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
        # Get params: sample-specific override or batch-level fallback
        if sample_id and sample_id in node.custom_gates:
            merged_params = node.custom_gates[sample_id]
        else:
            merged_params = node.params

        # merged_params has structure: {"hyperparams": {...}, "params": {...}, "diagnostics": {...}}
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
        if params:
            gate.params = params
        gate.diagnostics = diagnostics

        return gate

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
        new_gate = self.__class__(
            gate_name=self.gate_name,
            dimensions=self.dimensions.copy(),
            use_as_complement=self.use_as_complement,
            **self._hyperparams,
        )
        new_gate.params = self.params.copy()
        new_gate.diagnostics = self.diagnostics.copy()
        return new_gate

    def update_params(self, params: dict[str, Any]) -> "Gate":
        """Create a new gate with externally-provided parameters.

        Returns a new gate instance initialized with the current hyperparameters
        and the provided params (learned values). The new gate is independent and
        ready for apply() or refit().

        This is useful when reconstructing gates from persisted parameters or
        when you need to apply sample-specific fitted parameters without mutating
        the current gate.

        Parameters
        ----------
        params : dict[str, Any]
            Parameter dictionary to set. Expected structure: {"params": {...}, "diagnostics": {...}}
            or flat params dict. Malformed params may cause apply() to fail.

        Returns
        -------
        Gate
            New gate instance with provided params set.
            Hyperparameters and other state unchanged.

        Raises
        ------
        ValueError
            If params validation fails (subclass-specific checks)

        Examples
        --------
        >>> new_gate = original_gate.update_params({"params": {"center": [1, 2]}})
        >>> masks = new_gate.apply(events)  # Uses new params
        """
        # Create new gate with same configuration
        new_gate = self.copy()

        # Handle both flat params dict and nested structure
        if "params" in params:
            new_gate.params = params["params"].copy()
            if "diagnostics" in params:
                new_gate.diagnostics = params["diagnostics"].copy()
        else:
            new_gate.params = params.copy()

        return new_gate


    def to_node_params(self) -> dict[str, Any]:
        """Extract params in the structure used by GateNode for persistence.

        Extracts all three parameter sets (hyperparams, params, diagnostics) in the structure
        for GateNode storage (as opposed to other serialization formats).

        Returns: {"hyperparams": {...}, "params": {...}, "diagnostics": {...}}

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
        }

    def get_tests(self) -> dict[str, type]:
        """
        Return dictionary of tester classes for this gate.

        Subclasses compose tests via:
            tests = super().get_tests(entity=entity)  # Get parent tests
            tests.update({"new_test": NewTestClass})  # Add own tests
            return tests

        Parameters
        ----------
        entity : Any, optional
            Entity for which to get tests (used by subclasses for entity-specific tests).

        Returns
        -------
        dict[str, type]
            Mapping of test_name → QCTester subclass specific to this gate type
        """
        return {}

    def fit(self, events: AnnData, mask: dict[str, BooleanArray] = {}) -> "Gate":
        """
        Fit gate parameters from events (optional for parameter-only gates).

        This is the main extension point for subclasses. Gates without real learning
        should copy hyperparams to params. Gates with learning (e.g., ellipsoids)
        compute and store learned values in params.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, BooleanArray], default {}
            Dictionary of boolean masks from parent gates (optional).
            - {}: empty dict (default), fit using all events
            - {key: mask_array}: single or multiple entries, fit using masked subset
            For gates with single parent, pass that single mask.
            For multi-parent scenarios (e.g., BooleanGate), pass all parent masks.

        Returns
        -------
        Gate
            Self, for method chaining
        """

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

    def apply(self, events: AnnData, mask: dict[str, BooleanArray],) -> dict[str, BooleanArray]:
        """
        Apply gate to pre-filtered events and expand result to parent-mask length.

        Parameters
        ----------
        events : ad.AnnData
            Pre-filtered event data for this gate (already subset by AddGateStep).
        mask : dict[str, BooleanArray]
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
        if not mask:
            raise ValueError("mask dict is required and cannot be empty. Pass at least one parent mask.")

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

    def fit_apply(self, events: AnnData, mask: dict[str, BooleanArray] = {}) -> dict[str, BooleanArray]:
        """
        Convenience method to fit and then apply the gate in one step.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, BooleanArray]
            Dictionary of boolean masks from parent gates (default: empty dict)

        Returns
        -------
        dict[str, BooleanArray]
            Dictionary mapping region/quadrant IDs to boolean masks
        """
        return self.fit(events, mask).apply(events, mask)

    @abstractmethod
    def plot(self, events: AnnData, mask: dict[str, BooleanArray], **kwargs: Any) -> Any:
        """
        Generate diagnostic plot for the gate.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, BooleanArray]
            Dictionary of boolean masks from parent gates (default: empty dict)
        **kwargs : Any
            Optional plot configuration parameters passed through by callers.

        Returns
        -------
        Any
            Plot object (e.g., matplotlib figure, plotly figure) for visualization.
            Subclasses should define the specific type and content of the plot.
        """
        pass
