from abc import ABC, abstractmethod
from typing import Any, Sequence, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cytomind.qc.base import QCTester
else:
    QCTester = object

if TYPE_CHECKING:
    from cytomind.domain.constants import MaskLike
    from numpy.typing import NDArray
    BooleanMask = NDArray[np.bool_]
    from anndata import AnnData
    from pandas import DataFrame
else:
    MaskLike = object
    BooleanMask = object
    AnnData = object
    DataFrame = object

class Gate(ABC):
    """
    Abstract base class for flow cytometry gates following scikit-learn conventions.

    Gates separate hyperparameters (user-configured at initialization) from params
    (learned values during fit). Gates are stateless and always operate on event arrays.

    Key methods:
    - fit(events, mask=None): Learn gate parameters from events (optional for parameter-only gates)
    - apply(events, mask=None): Generate boolean mask for events passing through gate

    Both methods work on AnnData objects; gates internally select their dimensions.
    """

    gate_type: str
    glm_type: str
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
            Dictionary of hyperparameters for the gate used during fitting (e.g., number of clusters for a clustering gate)
        use_as_complement : bool
            If False, mask key is "{gate_name}.pos" (default)
            If True, mask key is "{gate_name}.neg" (complement of the gate)
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

    def param_dict(self) -> dict[str, Any]:
        """Extract parameters from gate these include:
            - hyperparams: user-configured settings that influence fitting (e.g., number of clusters)
            - params: learned values from fitting (e.g., cluster centers) used to apply the gate
            - diagnostics: any additional info from fitting (e.g., silhouette score) used for QC or analysis but not needed for applying the gate

        Returns:
            dict[str, Any]: Dictionary of gate parameters
        """
        gate_data = self.to_dict()
        return {
            "hyperparams": gate_data["hyperparams"],
            "params": gate_data["params"],
            "diagnostics": gate_data["diagnostics"],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Gate":
        """Deserialize gate from dictionary."""
        gate = cls(
            gate_name=data["gate_name"],
            dimensions=data["dimensions"],
            use_as_complement=data.get("use_as_complement", False),
            **data.get("hyperparams", {}),
        )
        gate.params = data.get("params", {})
        gate.diagnostics = data.get("diagnostics", {})
        return gate

    def copy(self) -> "Gate":
        """Create a deep copy of the gate."""
        new_gate = self.__class__(
            gate_name=self.gate_name,
            dimensions=self.dimensions.copy(),
            use_as_complement=self.use_as_complement,
            **self._hyperparams,
        )
        new_gate.params = self.params.copy()
        new_gate.diagnostics = self.diagnostics.copy()
        return new_gate

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

    def fit(self, events: AnnData, mask: dict[str, MaskLike] = {}) -> "Gate":
        """
        Fit gate parameters from events (optional for parameter-only gates).

        This is the main extension point for subclasses. Gates without real learning
        should copy hyperparams to params. Gates with learning (e.g., ellipsoids)
        compute and store learned values in params.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, MaskLike]
            Dictionary of boolean masks from parent gates.
            - {}: empty dict (default), no masking applied
            - {key: mask_array}: single entry, applies that mask
            - Multiple entries: raises ValueError

        Returns
        -------
        Gate
            Self, for method chaining
        """

        events_slice, _ = self._extract_events_slice(events, mask)
        self._fit_gate(events_slice)
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

    def apply(self, events: AnnData, mask: dict[str, MaskLike] = {}) -> dict[str, BooleanMask]:
        """
        Apply gate to events and generate dictionary of boolean masks.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, MaskLike]
            Dictionary of boolean masks from parent gates.
            - {}: empty dict (default), no masking applied
            - {key: mask_array}: single entry, applies that mask
            - Multiple entries: raises ValueError (ambiguous which parent mask to use)

        Returns
        -------
        dict[str, BooleanMask]
            Dictionary mapping region/quadrant IDs to boolean masks
        """
        events_slice, parent_mask = self._extract_events_slice(events, mask)

        result = {}
        gate_results = self._apply_gate(events_slice)
        for region_id, region_mask in gate_results.items():
            # Expand to full size
            full_mask = np.zeros(events.n_obs, dtype=bool)
            full_mask[parent_mask] = region_mask
            result[region_id] = full_mask
        return result

    @abstractmethod
    def _apply_gate(self, events_slice: DataFrame) -> dict[str, BooleanMask]:
        """
        Internal apply implementation for subclasses to override.

        Parameters
        ----------
        events_slice : pd.DataFrame
            Events filtered to only this gate's dimensions

        Returns
        -------
        dict[str, BooleanMask]
            Dictionary mapping region/quadrant IDs to boolean masks.
            Single-region gates should use a consistent key like "default".
        """
        pass

    def fit_apply(self, events: AnnData, mask: dict[str, MaskLike] = {}) -> dict[str, BooleanMask]:
        """
        Convenience method to fit and then apply the gate in one step.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, MaskLike]
            Dictionary of boolean masks from parent gates (default: empty dict)

        Returns
        -------
        dict[str, BooleanMask]
            Dictionary mapping region/quadrant IDs to boolean masks
        """
        return self.fit(events, mask).apply(events, mask)

    def _extract_events_slice(self, events: AnnData, mask: dict[str, MaskLike]) -> tuple[DataFrame, MaskLike | slice]:
        """
        Helper method to extract event data for the gate's dimensions,
        applying the provided mask if any.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, MaskLike]
            Dictionary of boolean masks from parent gates.

        Returns
        -------
        tuple[pd.DataFrame, MaskLike | slice]
            DataFrame of events filtered to this gate's dimensions and mask
        """
        if not mask:  # empty dict
            events_adata = events[:, self.dimensions]
            parent_mask = slice(None)
        elif len(mask) == 1:  # single entry
            parent_mask = next(iter(mask.values()))
            events_adata = events[parent_mask, self.dimensions]
        else:
            raise ValueError(
                f"Mask dict must have 0 or 1 entries, got {len(mask)}. "
                "Cannot extract with multiple parent masks."
            )

        if events_adata.isbacked:
            events_adata = events_adata.to_memory()

        return events_adata.to_df(), parent_mask

