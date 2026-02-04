from abc import ABC, abstractmethod
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
        dimensions: list[str],
        hyperparams: dict[str, Any] = {},
        use_as_complement: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize gate with hyperparameters (user-provided configuration).

        Parameters
        ----------
        gate_name : str
            Human-readable name for the gate
        dimensions : list[str]
            List of dimension/channel IDs that this gate operates on
        hyperparams : dict[str, Any]
            Dictionary of hyperparameters for the gate
        use_as_complement : bool
            If False, mask key is "{gate_name}.pos" (default)
            If True, mask key is "{gate_name}.neg" (complement of the gate)
        """
        self.gate_name = gate_name
        self.dimensions = dimensions
        self.use_as_complement = use_as_complement
        self._hyperparams: dict[str, Any] = hyperparams.copy()
        self._hyperparams.update(kwargs)
        self.params: dict[str, Any] = {}

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Gate":
        """Deserialize gate from dictionary."""
        gate = cls(
            gate_name=data["gate_name"],
            dimensions=data["dimensions"],
            use_as_complement=data.get("use_as_complement", False),
            **data.get("hyperparams", {}),
        )
        gate.params = data.get("params", {})
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
        return new_gate

    def fit(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> "Gate":
        """
        Fit gate parameters from events (optional for parameter-only gates).

        This is the main extension point for subclasses. Gates without real learning
        should copy hyperparams to params. Gates with learning (e.g., ellipsoids)
        compute and store learned values in params.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, NDArray[np.bool_]]
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
    def _fit_gate(self, events_slice: pd.DataFrame) -> None:
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

    def apply(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> dict[str, NDArray[np.bool_]]:
        """
        Apply gate to events and generate dictionary of boolean masks.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, NDArray[np.bool_]]
            Dictionary of boolean masks from parent gates.
            - {}: empty dict (default), no masking applied
            - {key: mask_array}: single entry, applies that mask
            - Multiple entries: raises ValueError (ambiguous which parent mask to use)

        Returns
        -------
        dict[str, NDArray[np.bool_]]
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
    def _apply_gate(self, events_slice: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
        """
        Internal apply implementation for subclasses to override.

        Parameters
        ----------
        events_slice : pd.DataFrame
            Events filtered to only this gate's dimensions

        Returns
        -------
        dict[str, NDArray[np.bool_]]
            Dictionary mapping region/quadrant IDs to boolean masks.
            Single-region gates should use a consistent key like "default".
        """
        pass

    def fit_apply(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]] = {}) -> dict[str, NDArray[np.bool_]]:
        """
        Convenience method to fit and then apply the gate in one step.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, NDArray[np.bool_]]
            Dictionary of boolean masks from parent gates (default: empty dict)

        Returns
        -------
        dict[str, NDArray[np.bool_]]
            Dictionary mapping region/quadrant IDs to boolean masks
        """
        return self.fit(events, mask).apply(events, mask)

    def _extract_events_slice(self, events: ad.AnnData, mask: dict[str, NDArray[np.bool_]]) -> tuple[pd.DataFrame, NDArray[np.bool_] | slice]:
        """
        Helper method to extract event data for the gate's dimensions,
        applying the provided mask if any.

        Parameters
        ----------
        events : ad.AnnData
            Event data with dimension IDs as var_names
        mask : dict[str, NDArray[np.bool_]]
            Dictionary of boolean masks from parent gates.

        Returns
        -------
        tuple[pd.DataFrame, NDArray[np.bool_] | slice]
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

