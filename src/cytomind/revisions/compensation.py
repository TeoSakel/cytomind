"""
Compensation revision handler with stateless persistence.

Handles iterative refinement of compensation results with lazy visualization
subset materialization and on-demand feedback generation.
"""
from __future__ import annotations
from typing import Any, Iterable, Mapping, TYPE_CHECKING
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go

from cytomind.domain.flow import CompensationRef
from cytomind.domain.pipeline import StepRun
from cytomind.steps.compensation import apply_compensation
from cytomind.revisions import RevisionHandlerRegistry
from cytomind.revisions.base import BaseRevisionHandler
from cytomind.visualization import (
    build_histogram2d_with_marginals,
    build_histogram1d,
    build_matrix_heatmap,
    build_pairplot,
    build_scatter2d_density,
)
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.domain.pipeline import RevisionSession
else:
    RevisionSession = object

@RevisionHandlerRegistry.register("compensation")
class CompensationRevisionHandler(BaseRevisionHandler):
    """
    Compensation revision handler.

    Simplified implementation for iterative refinement of compensation matrices.
    """

    entity_type = "compensation"
    _supported_figures = {
        "comp_heatmap": {
            "description": "Spillover matrix heatmap",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string (default: 'current')",
                "show_markers": "bool (default: False)",
                "colorscale": "any (default: 'RdBu')",
                "kwargs": "additional arguments passed to heatmap",
            }
        },
        "heatmap2d": {
            "description": "2D histogram density with marginals",
            "input_params": {
                "sample_id": "string",
                "donor": "string (channel name)",
                "receiver": "string (channel name)",
                "comp_id": "string (default: 'current')",
                "n_subset": "int (default: 10000)",
                "transformation": "string (default: 'logicle')",
                "show_markers": "bool (default: False)",
                "width": "int (default: 750)",
                "height": "int (default: 750)",
                "kwargs": "additional arguments passed to heatmap2d_with_marginals",
            }
        },
        "scatter2d": {
            "description": "2D scatter plot with grid-based density coloring and marginals",
            "input_params": {
                "sample_id": "string",
                "donor": "string (channel name)",
                "receiver": "string (channel name)",
                "comp_id": "string (default: 'current')",
                "n_subset": "int (default: 10000)",
                "transformation": "string (default: 'logicle')",
                "show_markers": "bool (default: False)",
                "coloraxis_log": "bool (default: False)",
                "nbins": "int bins for 2D grid density (default: 50)",
                "marker_size": "int marker size (default: 5)",
                "width": "int (default: 800)",
                "height": "int (default: 700)",
                "kwargs": "additional arguments passed to scatter2d_density",
            }
        },
        "channel_histogram": {
            "description": "Histogram for a single channel",
            "input_params": {
                "sample_id": "string",
                "channel": "string (channel name)",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
                "n_subset": "int number of events to use",
                "transformation": "string transformation to apply to data (default: 'identity')",
                "show_markers": "bool (default: False)",
                "kwargs": "additional arguments passed to plotly.histogram1d",
            }
        },
        "heatmap2d_tuner": {
            "description": "Interactive 2D histogram with spillover coefficient tuner",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
                "donor": "string: channel name",
                "receiver": "string: channel name",
                "n_subset": "int number of events to use",
                "transformation": "string transformation to apply to data (default: 'identity')",
                "show_markers": "bool (default: False)",
                "coef_min": "float minimum allowed coefficient (default: -0.5)",
                "coef_max": "float maximum allowed coefficient (default: 0.5)",
                "n_steps": "int number of steps (default: 41)",
                "nbins": "int number of bins per dimension (default: 128)",
                "colorscale": "any colorscale (default: 'viridis')",
            }
        },
        "qc_test_plot": {
            "description": "QC test diagnostic plot",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string (default: 'current')",
                "test_key": "hashable: unique test identifier",
                "step_id": "string: optional step ID to narrow down test search",
                "kwargs": "additional arguments passed to test plotter",
            }
        },
        "pairplot": {
            "description": "Pairplot with histograms on diagonal and scatter plots in lower triangle",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string (default: 'current')",
                "n_subset": "int (default: 10000)",
                "transformation": "string (default: 'logicle')",
                "show_markers": "bool (default: False)",
                "nbins": "int (default: 50)",
                "colorscale": "any (default: 'viridis')",
                "width": "int (default: 1200)",
                "height": "int (default: 1200)",
                "kwargs": "additional arguments passed to build_pairplot",
            }
        },
    }

    _supported_tables = {
        "spillover": {
            "description": "Spillover matrix table",
            "input_params": {
                "sample_id": "string | None. If None, `comp_id` must be an explicit compensation ID.",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
        "channel_tests": {
            "description": "QC metrics table for individual channels",
            "input_params": {
                "sample_id": "string(s)",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
        "pairwise_tests": {
            "description": "QC metrics table for channel pairs",
            "input_params": {
                "sample_id": "string(s)",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
    }

    # -- Properties for accessing workspace state ----

    @property
    def comp_dir(self) -> Path:
        """Get or create the compensation storage directory."""
        return self.workspace / "compensations"

    @property
    def samples(self) -> dict[str, dict[str, Any]]:
        """Get the sample metadata dictionary from state."""
        return self.state.get("samples", {})

    @property
    def compensations(self) -> dict[str, dict[str, Any]]:
        """Get the compensation metadata dictionary from state."""
        return self.state.get("compensations", {})

    # -- Data loading with on-the-fly compensation application ----

    def get_or_create_viz_subset(
        self,
        sample_id: str,
        layer: str,
        n_subset: int | None = None,
        seed: int | None = None,
    ) -> ad.AnnData:
        """Override to create raw layer on the fly if missing.

        If raw layer doesn't exist in main repo, it will be created by
        inverting the compensation from the comp layer.
        """
        if n_subset is None:
            n_subset = int(self.state["n_subset"])
        if seed is None:
            seed = int(self.state["seed"])

        # Use default behavior if not raw layer or if raw layer exists
        try:
            return super().get_or_create_viz_subset(sample_id, layer, n_subset, seed)
        except FileNotFoundError as e:
            if layer != "raw":
                raise e

        # Raw layer doesn't exist - need to create it from comp layer
        subset_key = f"{sample_id}:{layer}:{n_subset}"
        print(f"Creating raw viz subset from comp layer: {subset_key}")

        comp_path = self.main_repo.sample_adata_path(sample_id, layer="comp")
        if not comp_path.exists():
            raise ValueError(
                f"Sample {sample_id} has neither raw nor comp data - cannot create viz subset"
            )

        # Sample indices first, then load only the subset
        sample_ref = self.main_repo.load_sample_meta(sample_id)
        rng = np.random.RandomState(seed)
        n_total = sample_ref.n_events
        if n_total <= 0:
            raise ValueError(f"Sample {sample_id} has no events to subset.")
        if n_total <= n_subset:
            indices = slice(None)
        else:
            indices = rng.choice(n_total, n_subset, replace=False)
            indices.sort()

        # Load only the subset from comp layer
        adata_comp_subset = self.main_repo.load_sample_adata(sample_id, layer="comp", mask=indices)

        # Get the compensation matrix that was used
        comp_id = self.samples[sample_id]["active_compensation"]
        try:
            comp_ref = self.get_comp_ref(comp_id)
            adata_raw_subset = apply_compensation(adata_comp_subset, comp_ref, invert=True)
        except KeyError:
            # Compensation not available, assuming identity/raw case
            adata_raw_subset = adata_comp_subset

        # Save using the base class save_viz_object method (without additional subsetting)
        return self.save_viz_object(
            key=subset_key,
            adata=adata_raw_subset,
            seed=seed,
            n_subset=None  # Already subsetted, don't subset again
        )

    def load_viz_data_compensated(self, sample_id: str, comp_id: str, n_subset: int) -> ad.AnnData:
        """Load a visualization subset for a sample with on-the-fly compensation applied.

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Compensation ID to apply
        n_subset : int
            Number of events in the subset

        Returns
        -------
        ad.AnnData
            Compensated visualization subset
        """
        comp_id = self._resolve_compensation(sample_id, comp_id)

        # Load raw subset (creates if doesn't exist) and apply compensation on the fly
        if comp_id == self.samples[sample_id]["active_compensation"]:
            # If the requested compensation is the same as the active one, we can load directly from comp layer
            comp_subset = self.get_or_create_viz_subset(sample_id, "comp", n_subset, self.state["seed"])
        else:
            raw_subset = self.get_or_create_viz_subset(sample_id, "raw", n_subset, self.state["seed"])
            if comp_id == "raw": # Identity compensation, no need to apply
                comp_subset = raw_subset
            else:
                comp_ref = self.get_comp_ref(comp_id)
                comp_subset = apply_compensation(raw_subset, comp_ref, invert=False)

        return comp_subset

    def load_compensated_data(self, sample_id: str, comp_id: str) -> ad.AnnData:
        """Load the full compensated dataset for a sample (no subsetting).

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Compensation ID to apply

        Returns
        -------
        ad.AnnData
            Full compensated dataset
        """
        comp_id = self._resolve_compensation(sample_id, comp_id)

        # Load raw data and apply compensation on the fly
        if comp_id == self.samples[sample_id]["active_compensation"]:
            # If the requested compensation is the same as the active one, we can load directly from comp layer
            return self.main_repo.load_sample_adata(sample_id, layer="comp")

        raw_adata = self.main_repo.load_sample_adata(sample_id, layer="raw")
        if comp_id == "raw": # Identity compensation, no need to apply
            return raw_adata
        else:
            comp_ref = self.get_comp_ref(comp_id)
            return apply_compensation(raw_adata, comp_ref, invert=False)

    # --- Protocol methods (revision lifecycle) ----

    def start_revision(self, input_spec: Mapping[str, Any] = {}) -> RevisionSession:
        """
        Initialize revision workspace for compensation refinement.

        Sets up:
        - Discovers samples from compensation batch
        - Copies compensation matrices from main repo
        - Tracks sample metadata (n_subset, compensation)
        - Identifies fluorescence channels from panel
        - Prepares raw data subsets for visualization
        - Initializes state with seed and n_subset

        Parameters
        ----------
        input_spec : Mapping
            Input specification with optional sample_ids, seed, n_subset.
            If sample_ids not provided, uses compensation batch for sample discovery.

        Returns
        -------
        RevisionSession
            Initialized session with handler state
        """
        if self.session is not None and self.session.state in ("active", "committed"):
            raise RuntimeError("Revision session already active or committed; cannot start a new revision.")

        project = self.main_repo.load_project()

        input_spec = dict(input_spec)
        self.state["n_subset"] = input_spec.pop("n_subset", self.state["n_subset"])
        self.state["seed"] = input_spec.pop("seed", self.state["seed"])

        # Get fluorescence channels from raw panel
        raw_panel = project.dimensions.get("raw", [])
        fluoro_markers = {dim.id: dim.marker for dim in raw_panel if dim.type == "fluorescence"}

        # Get sample info
        samples = {}
        for sid, sample in project.samples.items():
            samples[sid] = {
                "n_events": sample.n_events,
                "active_compensation": sample.compensation or "raw",
                "compensation": sample.compensation or "raw",
            }

        # Copy all available compensation matrices from main repo to use as options
        self.comp_dir.mkdir(parents=True, exist_ok=True)
        compensations = {}
        for comp_id, comp_ref in project.compensations.items():
            compensations[comp_id] = {
                "id": comp_id,
                "name": comp_ref.name,
                "source": comp_ref.source,
                "path": comp_ref.path,
                "parent": None,  # Default compensations have no parent
                "is_new": False,
                "batch": comp_ref.batch.copy(),
            }

        # Initialize handler state
        self.state.update({
            "fluoro_markers": fluoro_markers,
            "samples": samples,
            "compensations": compensations,
        })

        # Update session metadata
        self.session.context = input_spec
        now = now_iso()
        self.session.state = "active"
        self.session.created_at = now
        self.session.updated_at = now
        self.save_session()
        return self.session

    def apply_revision(
        self,
        user_input: Mapping[str, Any] = {},
    ) -> None:
        """
        Internally update the compensation matrix assigned to a sample based on user input.

        Tracks all changes in the session's revision history for reproducibility.

        Supported inputs (sample_id is required as str or list[str]):
          - Existing compensation: {"sample_id": "S1", "comp_id": "comp_123" | "raw" | None}
          - New spillover:        {"sample_id": "S1", "spillover": df, "name": "optional"}

        Notes:
        - comp_id of None or "raw" applies identity compensation.
        """

        # Normalize sample_ids
        try:
            sample_id: str = user_input["sample_id"]
        except KeyError:
            raise ValueError("user_input must contain 'sample_id'")

        # Validate that either comp_id or spillover is provided
        if "comp_id" not in user_input and "spillover" not in user_input:
            raise ValueError("user_input must contain either an existing 'comp_id' or 'spillover'")

        comp_id = self.update_sample_compensation(
                sample_id,
                spill_df=user_input.get("spillover"),
                comp_name=user_input.get("name"),
                comp_id=user_input.get("comp_id"),
        )

        if comp_id is None:
            return  # No change needed

        # Update QC status for the sample under the new compensation
        qc_status = self.load_entity_qc_cache(comp_id)
        if sample_id not in qc_status.sample_qc:
            comp_ref = self.get_comp_ref(comp_id)
            adata = self.load_viz_data_compensated(sample_id, comp_id, n_subset=self.samples[sample_id]["n_events"])
            qc_status = self.qc_evaluator.update_entity_qc(entity=comp_ref, entity_qc=qc_status, sample_data=[(sample_id, adata)])
            self.save_entity_qc_cache(comp_id, qc_status)

        # Update session metadata
        now = now_iso()
        self.session.revision_history.append({
            "timestamp": now,
            "mode": "new_spillover" if "spillover" in user_input else "existing_compensation",
            "update": (sample_id, comp_id),
        })
        self.session.updated_at = now
        self.save_session()

    def _commit(self) -> tuple[dict[str, Any], StepRun | None]:
        """
        Commit revision changes.

        Parameters
        ----------
        session : RevisionSession
            Revision session

        Returns
        -------
        tuple
            (metadata_updates, new_step)
        """

        comp_refs = {
            sid: self.get_comp_ref(sinfo["compensation"])
            for sid, sinfo in self.samples.items()
            if sinfo["compensation"] != sinfo["active_compensation"]
        }

        new_comps = {}
        for ref in comp_refs.values():
            if ref.id in new_comps or not self.compensations.get(ref.id, {}).get("is_new", True):
                continue  # Already committed this compensation
            new_comps[ref.id] = ref

        step_run = StepRun(
            id = f"rev_{self.session.id}",
            step_type="compensate",
            inputs={"sample_ids": list(comp_refs.keys())},
            config={"comp_id": {sid: ref.id for sid, ref in comp_refs.items()}},
            created_at=now_iso(),
        )

        self.session.state = "committed"
        self.session.updated_at = now_iso()
        self.save_session()

        return {"compensations": list(new_comps.values())}, step_run

    def cleanup_workspace(self) -> None:
        super().cleanup_workspace()
        rmtree(self.comp_dir)

    def update_sample_compensation(self, sample_id: str, spill_df: pd.DataFrame | None = None, comp_name: str | None = None, comp_id: str | None = None) -> str:
        """Update the compensation assigned to a sample, either by selecting an existing comp or adding a new one from spill_df.

        Parameters
        ----------
        sample_id : str
            Sample ID to update
        spill_df : pd.DataFrame, optional
            New spillover matrix to add as a compensation (if comp_id not provided)
        comp_name : str, optional
            Name for the new compensation (if spill_df provided)
        comp_id : str, optional
            Existing compensation ID to assign (if spill_df not provided)

        Returns
        -------
        str
            The compensation ID that is now assigned to the sample
        """
        current_comp_id = self.current_comp(sample_id)

        if spill_df is not None:
            if not comp_name:
                comp_name = self._make_name_from_parent(current_comp_id, sample_id)
            elif any(c["name"] == comp_name for c in self.compensations.values()):
                raise ValueError(f"Compensation name '{comp_name}' already exists in workspace; please choose a unique name")

            new_comp_id = self.add_compensation(spill_df, comp_name)
            self.compensations[new_comp_id]["parent"] = current_comp_id
            self.compensations[new_comp_id]["batch"].append(sample_id)
            if comp_id and comp_id != new_comp_id:
                raise ValueError(f"Cannot specify both spill_df and comp_id; new comp_id {new_comp_id} does not match provided comp_id {comp_id}")
            self.samples[sample_id]["compensation"] = new_comp_id
            return new_comp_id

        elif comp_id is not None:
            # Assign existing compensation
            if comp_id not in self.compensations and comp_id != "raw":
                raise KeyError(f"Compensation {comp_id} not found in workspace")
            if sample_id not in self.compensations[comp_id]["batch"]:
                self.compensations[comp_id]["batch"].append(sample_id)
            self.samples[sample_id]["compensation"] = comp_id
            return comp_id

        else:
            raise ValueError("Either spill_df or comp_id must be provided to update sample compensation.")

    # ---- Compensation accessors ----

    def current_comp(self, sample_id: str) -> str:
        """Get the current compensation id mapped to a sample in the workspace."""
        sample_info = self.samples.get(sample_id)
        if not sample_info:
            raise KeyError(f"Sample {sample_id} not found in workspace")
        return sample_info.get("compensation", "raw")

    def spillover_path(self, comp_id: str) -> Path:
        return self.comp_dir / f"{comp_id}.csv"

    def get_comp_ref(self, comp_id: str) -> CompensationRef:
        """Load compensation from workspace (not main repo)."""
        try:
            comp_info = self.state["compensations"][comp_id]
        except KeyError:
            if comp_id == "raw":
                return self._get_identity_compensation()
            raise KeyError(f"Compensation {comp_id} not found in workspace")

        spill_df = pd.read_csv(comp_info["path"], index_col=False)
        spill_df.index = spill_df.columns

        return CompensationRef(
            id=comp_info["id"],
            name=comp_info["name"],
            source=comp_info["source"],
            batch=comp_info["batch"],
            _spill=spill_df,
        )

    def add_compensation(self, spill_df: pd.DataFrame, name: str | None) -> str:
        """Add a new compensation to the workspace from a spillover matrix.

        Validates the spillover matrix and creates a new compensation entry in the workspace.
        Does not assign the compensation to any sample.

        Parameters
        ----------
        spill_df : pd.DataFrame
            Spillover matrix with channels as rows and columns

        Returns
        -------
        str
            The ID of the newly added compensation
        """

        if spill_df.shape[1] == spill_df.shape[0] + 1 and spill_df.dtypes[0] == object:
            # First column is index (detector names) and rest are spill
            spill_df = spill_df.set_index(spill_df.columns[0])

        comp_id = CompensationRef.generate_id(spill_df)
        if comp_id in self.compensations:
            raise ValueError(f"Compensation with ID '{comp_id}' already exists in workspace")

        comp_path = self.spillover_path(comp_id)
        spill_df.to_csv(comp_path, index=False)

        self.compensations[comp_id] = {
            "id": comp_id,
            "name": name or comp_id,
            "source": "user",
            "path": comp_path.as_posix(),
            "parent": None,
            "is_new": True,
            "batch": [],
        }

        return comp_id

    def _make_name_from_parent(self, parent_comp_id: str, sample_id: str) -> str:
        parent_info = self.compensations[parent_comp_id]
        parent_name = parent_info.get("name", "")
        comp_name = f"{parent_name}_{sample_id}" if parent_name else f"comp_{sample_id}"
        n_previous_revisions = sum(1 for c in self.compensations.values() if c["name"].startswith(comp_name))
        if n_previous_revisions > 0:
            comp_name += f"_{n_previous_revisions + 1}"
        return comp_name

    def _resolve_compensation(self, sample_id: str, comp_id: str = "current") -> str:
        """Resolve a a special/relative compensation ID to a concrete compensation ID that can be used to retrieve the compensation references.

        Supports the same special values as `get_spillover_table`:
        - 'current' -> the compensation currently mapped in `comp_map`
        - 'parent'  -> the parent of the current compensation
        - 'active'  -> the compensation originally declared on the sample metadata
        - 'raw'     -> identity compensation for raw/uncompensated data
        If `comp_id` is None, defaults to the current mapped compensation.
        """

        if comp_id == "current":
            comp_id = self.current_comp(sample_id)
        elif comp_id == "parent":
            current_id = self.current_comp(sample_id)
            if not current_id or current_id == "raw":
                raise KeyError(f"Sample {sample_id} has no compensation assigned")
            comp_info = self.compensations[current_id]
            comp_id = comp_info["parent"]
            if not comp_id:
                raise KeyError(f"Compensation {current_id} has no parent compensation")
        elif comp_id == "active":
            comp_id = self.samples[sample_id]["active_compensation"]
            if not comp_id:
                comp_id = "raw"

        # explicit compensation id or "raw"
        return comp_id

    def _marker_label(self, channel: str) -> str:
        if not "fluoro_markers" in self.state:
            raise KeyError("This revision session has not been properly initialized with fluorescence marker information")
        try:
            return self.state["fluoro_markers"][channel]
        except KeyError:
            raise KeyError(f"Channel {channel} not found in fluorescence marker information. Available channels: {list(self.state['fluoro_markers'].keys())}")

    def _get_identity_compensation(self) -> CompensationRef:
        """Create identity compensation matrix for raw (uncompensated) data."""

        # Get fluorescence channels
        fluoro_channels: list[str] = list(self.state["fluoro_markers"].keys())

        # Create identity matrix
        n_channels = len(fluoro_channels)
        identity_matrix = pd.DataFrame(
            np.eye(n_channels),
            index=fluoro_channels,
            columns=fluoro_channels
        )

        comp_ref = CompensationRef(
            id="raw",
            name="Raw (Identity)",
            source="identity",
            batch=[sid for sid, sinfo in self.samples.items() if sinfo["compensation"] == "raw"],
            _spill=identity_matrix,
        )

        comp_path = self.comp_dir / "identity.csv"
        comp_ref.spill.to_csv(comp_path, index=False)
        self.compensations["raw"] = {
            "id": comp_ref.id,
            "name": comp_ref.name,
            "source": comp_ref.source,
            "path": comp_path.as_posix(),
            "parent": None,  # Default compensations have no parent
            "is_new": True,
            "batch": comp_ref.batch.copy(),
        }

        self.save_session()
        return comp_ref

    # --- Table implementations ----

    def get_table(self, table_type: str, input_params: Mapping[str, Any] = {}) -> pd.DataFrame:
        if table_type == "spillover":
            return self.get_spillover_table(**input_params)
        if table_type == "channel_tests" or table_type == "pairwise_tests":
            return self.get_test_table(table_type=table_type, **input_params)
        raise ValueError(f"Unknown table type: {table_type}")

    def get_test_table(self, table_type: str, sample_ids: str | Iterable[str], comp_id: str = "current") -> pd.DataFrame:
        """Get channel QC table for a sample under a specific compensation.

        Uses cached EntityQCStatus and generates the table on demand using the
        CompensationQCEvaluator's generate_table method.

        Parameters
        ----------
        sample_ids : str | Iterable[str]
            Sample ID(s)
        comp_id : str
            Which compensation to apply:
                - "current" (mapped in workspace),
                - "parent" (parent of current),
                - "active" (compensation from sample metadata),
                - "raw" (identity matrix)

        Returns
        -------
        pd.DataFrame
            Channel QC table with columns: sample_id, compensation, channel, test_name, status, metric_name, metric_value
        """
        sample_ids = [sample_ids] if isinstance(sample_ids, str) else list(sample_ids)

        # Make sure that all samples map to the same compensation id
        comp_map = {sid: self._resolve_compensation(sid, comp_id) for sid in sample_ids}
        comp_ids = set(comp_map.values())
        if len(comp_ids) > 1:
            comp_rev_map: dict[str, list[str]] = {comp_id: [] for comp_id in comp_ids}
            for sid, cid in comp_map.items(): comp_rev_map[cid].append(sid)
            raise ValueError(f"All samples must map to the same compensation id for table generation, but got multiple:\n{comp_rev_map}")

        # Resolve compensation ID
        comp_id_resolved = comp_ids.pop()

        # Load or create QC status
        qc_status = self.load_entity_qc_cache(comp_id_resolved)

        # If a sample not in QC status, update it
        missing = [sid for sid in sample_ids if sid not in qc_status.sample_qc]
        if missing:
            n_subset: int = max(self.samples[sid]["n_events"] for sid in missing)
            sample_data = ((sid, self.load_viz_data_compensated(sid, comp_id_resolved, n_subset)) for sid in missing)
            comp_ref = self.get_comp_ref(comp_id_resolved)
            qc_status = self.qc_evaluator.update_entity_qc(entity=comp_ref, entity_qc=qc_status, sample_data=sample_data)
            self.save_entity_qc_cache(comp_id_resolved, qc_status)

        # Generate table using the evaluator
        sample_data = ((sid, ad.AnnData()) for sid in sample_ids)
        df = self.qc_evaluator.generate_table(qc_status, table_type=table_type, sample_data=sample_data)

        return df

    def get_spillover_table(self, sample_id: str, comp_id: str = "current") -> pd.DataFrame:
        """Get a specific spillover table for a sample from workspace.

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Which spillover to retrieve:
                - "current",
                - "parent" (parent of current),
                - "active" (compensation from sample metadata),
                - "raw" (identity matrix)

        Returns
        -------
        pd.DataFrame
            Spillover table with id, name, source, spill, or None if not found
        """
        comp_id_resolved = self._resolve_compensation(sample_id, comp_id)
        comp_ref = self.get_comp_ref(comp_id_resolved)
        df = comp_ref.spill
        df.index = df.columns
        return df

    # ---- Figure implementations ----

    def get_figure(self, plot_type: str, input_params: Mapping[str, Any] = {}) -> dict[str, Any]:
        if plot_type == "comp_heatmap":
            return self.comp_heatmap(**input_params)

        if plot_type == "heatmap2d":
            return self.heatmap2d(**input_params)

        if plot_type == "scatter2d":
            return self.scatter2d(**input_params)

        if plot_type == "channel_histogram":
            return self.channel_histogram(**input_params)

        if plot_type == "heatmap2d_tuner":
            return self.heatmap2d_tuner(**input_params)

        if plot_type == "qc_test_plot":
            return self.qc_test_plot(**input_params)

        if plot_type == "pairplot":
            return self.pairplot(**input_params)

        raise ValueError(f"Unknown plot type: {plot_type}")

    def comp_heatmap(
        self,
        sample_id: str | None,
        comp_id: str = "current",
        show_markers: bool = False,
        show_diagonal: bool = True,
        colorscale: Any = "RdGy",
        **kwargs
    ) -> dict[str, Any]:

        comp_id_resolved = comp_id if not sample_id else self._resolve_compensation(sample_id, comp_id)
        comp_ref = self.get_comp_ref(comp_id_resolved)
        spill_df = comp_ref.spill
        spill_df.index = spill_df.columns

        x_title, y_title = "Donor Channels", "Receiver Channels"
        if show_markers:
            markers = [self._marker_label(ch) for ch in spill_df.columns]
            spill_df.index = markers
            y_title = "Receiver Markers"

        if not show_diagonal:
            np.fill_diagonal(spill_df.values, np.nan)

        title = f"Spillover Matrix - Compensation ID: {comp_id_resolved}"
        if sample_id:
            comp_id_relative = f'{"sample_id"}' if comp_id == comp_id_resolved else f'"{sample_id}" {comp_id}'
            title += f" (Sample {comp_id_relative})"

        fig = build_matrix_heatmap(
            spill_df,
            colorscale=colorscale,
            zmid=0.0,
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            **kwargs
        )

        # TODO: If sample_id is provided: Highlight cells with test failures (WARN/SEVERE)

        return {
            "plotly": fig,
            "metadata": {
                "comp_id": comp_id,
                "n_channels": spill_df.shape[0]
            }
        }

    def heatmap2d(
        self,
        sample_id: str,
        donor: str,
        receiver: str,
        compensation_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        show_markers: bool = False,
        **kwargs,
    ):
        """2D histogram density with marginals; handler prepares data then plots.

        Loads the raw subset, applies the selected compensation and transform,
        then delegates histogram computations and Plotly figure construction to
        the shared visualization module.

        Parameters
        ----------
        sample_id : str
            Sample ID
        donor : str
            Donor channel name
        receiver : str
            Receiver channel name
        compensation_id : str, optional
            Compensation ID to apply (defaults to current from comp_map)
        n_subset : int
            Number of events to load (default: 10000)
        nbins : int
            Number of bins for histogram (default: 120)
        colorscale : Any
            Plotly colorscale (default: "Viridis")
        transformation : str
            Transformation to apply: "logicle", "linear", or "asinh" (default: "logicle")
        show_markers : bool
            Whether to display marker labels instead of channel labels (default: False)
        """

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, compensation_id, n_subset)

        donor_idx = comp_subset.var.index.get_loc(donor)
        receiver_idx = comp_subset.var.index.get_loc(receiver)

        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        x = np.asarray(comp_subset.X[:, donor_idx])
        y = np.asarray(comp_subset.X[:, receiver_idx])

        donor_label = self._marker_label(donor) if show_markers else donor
        receiver_label = self._marker_label(receiver) if show_markers else receiver

        # Delegate to visualization builder
        fig = build_histogram2d_with_marginals(
            x,
            y,
            transformation=transformation,
            title=f"{donor_label} vs {receiver_label} - {sample_id}",
            xaxis_title=donor_label,
            yaxis_title=receiver_label,
            **kwargs,
        )

        return {
            "plotly": fig,
            "metadata": {},
        }

    def scatter2d(
        self,
        sample_id: str,
        donor: str,
        receiver: str,
        compensation_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        coloraxis_log: bool = False,
        show_markers: bool = False,
        nbins: int = 50,
        marker_size: int = 5,
        **kwargs,
    ):
        """2D scatter plot with grid-based density coloring and marginals.

        Loads the compensated subset, applies the selected transformation,
        then creates a scatter plot with points colored by the bin density (count of points
        in their grid cell).

        Parameters
        ----------
        sample_id : str
            Sample ID
        donor : str
            Donor channel name
        receiver : str
            Receiver channel name
        compensation_id : str, optional
            Compensation ID to apply (defaults to current from comp_map)
        n_subset : int
            Number of events to load (default: 10000)
        transformation : str
            Transformation to apply: "logicle", "identity", "asinh" (default: "logicle")
        coloraxis_log : bool
            Whether to apply a log transform to density values (default: False)
        show_markers : bool
            Whether to display marker labels instead of channel labels (default: False)
        nbins : int
            Number of bins for 2D grid density calculation (default: 50)
        marker_size : int
            Marker size in pixels (default: 5)
        """

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, compensation_id, n_subset)

        donor_idx = comp_subset.var.index.get_loc(donor)
        receiver_idx = comp_subset.var.index.get_loc(receiver)

        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        x = np.asarray(comp_subset.X[:, donor_idx])
        y = np.asarray(comp_subset.X[:, receiver_idx])

        donor_label = self._marker_label(donor) if show_markers else donor
        receiver_label = self._marker_label(receiver) if show_markers else receiver

        # Delegate to visualization builder
        fig = build_scatter2d_density(
            x,
            y,
            nbins=nbins,
            transformation=transformation,
            coloraxis_log=coloraxis_log,
            title=f"{donor_label} vs {receiver_label} - {sample_id}",
            xaxis_title=donor_label,
            yaxis_title=receiver_label,
            marker_size=marker_size,
            **kwargs,
        )

        return {
            "plotly": fig,
            "metadata": {},
        }

    def channel_histogram(
        self,
        sample_id: str,
        channel: str,
        compensation_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        upper_bound: float | None = None,
        show_markers: bool = False,
        **kwargs,
    ):
        """Histogram from raw subset with on-the-fly compensation and transform."""

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, compensation_id, n_subset)
        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        # Extract channel values
        channel_idx = comp_subset.var.index.get_loc(channel)
        values = np.asarray(comp_subset.X[:, channel_idx])
        if upper_bound is not None:
            values = values[values <= upper_bound]

        channel_label = self._marker_label(channel) if show_markers else channel

        fig = build_histogram1d(
            values,
            title=f"{channel_label} - {sample_id}",
            xaxis_title=channel_label,
            yaxis_title="Count",
            **kwargs,
        )

        # Add vertical line at zero if in range
        x_min = np.min(values)
        if x_min < 0.0:
            fig.add_vline(
                x=0.0,
                line=dict(color="red", dash="dash"),
                annotation_text="0",
                annotation_position="top right",
            )

        return {
            "plotly": fig,
            "metadata": { }
        }

    def heatmap2d_tuner(
        self,
        sample_id: str,
        donor: str,
        receiver: str,
        compensation_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        show_markers: bool = False,
        coef_min: float = -0.5,
        coef_max: float = 0.5,
        n_steps: int = 41,
        nbins: int = 128,
        colorscale: Any = "viridis",
    ) -> dict[str, Any]:
        """Interactive 2D histogram with a slider to tune spillover coefficient.

        The slider adjusts the donor->receiver spillover entry in the workspace
        compensation and shows the resulting compensated distribution.

        Parameters
        ----------
        sample_id : str
            Sample ID.
        donor : str
            Donor channel name (column in spill matrix).
        receiver : str
            Receiver channel name (row in spill matrix).
        compensation_id : str
            Compensation to start from (default: current mapping).
        n_subset : int
            Number of events to visualize.
        transformation : str
            Transform to apply before binning (e.g., "logicle", "identity").
        show_markers : bool
            Whether to display marker labels instead of channel labels (default: False)
        coef_min, coef_max : float
            Slider coefficient range (absolute spillover value).
        n_steps : int
            Number of discrete slider steps.
        nbins : int
            Number of histogram bins for both axes.
        colorscale : Any
            Plotly colorscale for the heatmap.
        """

        # Resolve compensation and load a raw subset (creating one if needed)
        comp_id_resolved = self._resolve_compensation(sample_id, compensation_id)
        comp_ref = self.get_comp_ref(comp_id_resolved)
        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        raw_subset = self.get_or_create_viz_subset(sample_id, "raw", n_subset, self.state["seed"])

        # Determine current coefficient; default to 0 if missing
        spill_df = comp_ref.spill.copy()
        spill_df.index = spill_df.columns  # assume channels are same for index/columns

        try:
            current_coef = float(spill_df.at[receiver, donor]) # pyright: ignore[reportArgumentType]
        except Exception:
            current_coef = 0.0

        # Build slider values and choose initial index closest to current
        coef_values = np.linspace(coef_min, coef_max, int(max(2, n_steps)))
        init_idx = int(np.argmin(np.abs(coef_values - current_coef)))
        init_coef = float(coef_values[init_idx])

        # Precompute heatmaps for each coefficient; keep consistent axes and zscale
        donor_idx = raw_subset.var.index.get_loc(donor)
        receiver_idx = raw_subset.var.index.get_loc(receiver)

        donor_label = self._marker_label(donor) if show_markers else donor
        receiver_label = self._marker_label(receiver) if show_markers else receiver

        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        histograms: list[np.ndarray] = []

        # Helper to apply one modified compensation
        def _apply_with_coef(value: float) -> tuple[np.ndarray, np.ndarray]:
            mod_spill = spill_df.copy()
            if receiver in mod_spill.index and donor in mod_spill.columns:
                mod_spill.at[receiver, donor] = value
            # Create a transient compensation ref
            mod_ref = CompensationRef(
                id=f"{comp_ref.id}_tune",
                name=comp_ref.name,
                source=comp_ref.source,
                _spill=mod_spill,
            )
            comp_subset = apply_compensation(raw_subset, mod_ref, invert=False)
            if comp_subset.X is None:
                raise ValueError("Compensated subset has no data")
            xv = np.asarray(comp_subset.X[:, donor_idx])
            yv = np.asarray(comp_subset.X[:, receiver_idx])
            return xv, yv

        # First pass: compute transformed ranges and hist counts for all steps
        x_t_list: list[np.ndarray] = []
        y_t_list: list[np.ndarray] = []
        for c in coef_values:
            xv, yv = _apply_with_coef(float(c))
            # Transform values
            from cytomind.visualization.transforms import apply_transform
            x_t = apply_transform(xv, transformation=transformation)
            y_t = apply_transform(yv, transformation=transformation)
            x_t_list.append(x_t)
            y_t_list.append(y_t)
            x_min = min(x_min, float(np.min(x_t)))
            x_max = max(x_max, float(np.max(x_t)))
            y_min = min(y_min, float(np.min(y_t)))
            y_max = max(y_max, float(np.max(y_t)))

        # Robust guards for degenerate ranges
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            x_min, x_max = -1.0, 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = -1.0, 1.0

        x_edges = np.linspace(x_min, x_max, nbins + 1)
        y_edges = np.linspace(y_min, y_max, nbins + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        for x_t, y_t in zip(x_t_list, y_t_list):
            H, _, _ = np.histogram2d(y_t, x_t, bins=[y_edges, x_edges])  # pyright: ignore[reportCallIssue, reportArgumentType] # rows are y, cols are x
            # Normalize each histogram independently to [0, 1]
            H_max = float(H.max())
            if H_max > 0:
                H_normalized = H / H_max
            else:
                H_normalized = H
            histograms.append(H_normalized)

        # Build interactive figure with slider
        # Add all histograms as traces (hidden except initial)
        init_H = histograms[init_idx]
        fig = go.Figure()

        for i, (c, H) in enumerate(zip(coef_values, histograms)):
            fig.add_trace(
                go.Heatmap(
                    z=H,
                    x=x_centers,
                    y=y_centers,
                    colorscale=colorscale,
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="Density"),
                    hovertemplate="x: %{x:.3g}<br>y: %{y:.3g}<br>density: %{z:.3f}<extra></extra>",
                    visible=(i == init_idx),
                )
            )

        # Build slider steps with proper update method
        slider_steps = []
        for i, c in enumerate(coef_values):
            visibility = [j == i for j in range(len(coef_values))]
            slider_steps.append(
                {
                    "args": [
                        {"visible": visibility},  # First arg: data updates
                        {"title": f"{donor_label} vs {receiver_label} - {sample_id}<br>Spillover coefficient: {float(c):.4f}"},  # Second arg: layout updates
                    ],
                    "method": "update",  # Use update to modify both data and layout
                    "label": f"{float(c):.3f}",
                }
            )

        fig.update_layout(
            title=f"{donor_label} vs {receiver_label} - {sample_id}<br>Spillover coefficient: {init_coef:.4f}",
            xaxis_title=donor_label,
            yaxis_title=receiver_label,
            width=900,
            height=800,
            sliders=[
                {
                    "active": int(init_idx),
                    "yanchor": "top",
                    "y": -0.1,
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Spillover coefficient: ",
                        "visible": True,
                        "xanchor": "center",
                        "font": {"size": 16},
                    },
                    "pad": {"b": 10, "t": 50},
                    "len": 0.95,
                    "x": 0.05,
                    "steps": slider_steps,
                }
            ],
        )

        return {
            "plotly": fig,
            "metadata": {
                "donor": donor,
                "receiver": receiver,
                "sample_id": sample_id,
                "compensation": compensation_id,
                "coef_default": current_coef,
                "coef_min": coef_min,
                "coef_max": coef_max,
            },
        }

    def qc_test_plot(
        self,
        test_key: tuple | Mapping[str, str],
        sample_id: str | None,
        step_id: str | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Generate QC test diagnostic plot from cached EntityQCStatus.

        Parameters
        ----------
        sample_id : str
            Sample ID
        test_key : tuple or mapping
            Unique test identifier (from QCStepStatus.tests) or row dictionary from test table.
        step_id : str | None
            Optional step ID to narrow down test search
        **kwargs
            Additional arguments passed to test plotter

        Returns
        -------
        dict
            Dict with 'plotly' key containing the figure and 'metadata' with test info
        """
        # Use evaluator's helper to parse and validate test_key
        tester_class, test_key_dict = self.qc_evaluator._parse_test_key(test_key)

        # Extract sample_id and compensation_id from test_key
        if sample_id is None:
            if "sample_id" not in test_key_dict:
                raise ValueError("test_key must contain 'sample_id' when sample_id parameter is None")
            sample_id = test_key_dict["sample_id"]

        compensation_id = test_key_dict["compensation_id"]
        qc_status = self.load_entity_qc_cache(compensation_id)
        if sample_id not in qc_status.sample_qc:
            # This is also checked by the evaluator but fail early before loading data if sample is missing from QC status
            raise KeyError(f"Sample {sample_id} not found in QC status for compensation {compensation_id}")

        # Load compensated data for plotting
        n_subset = int(kwargs.get("n_subset", self.state["n_subset"]))
        adata = self.load_viz_data_compensated(sample_id, compensation_id, n_subset)

        # Generate figure using evaluator
        fig = self.qc_evaluator.generate_figure(
            entity_qc=qc_status,
            test_key=test_key_dict,
            sample_data=[(sample_id, adata)],
            step_id=step_id,
        )

        return {
            "plotly": fig,
            "metadata": {
                "sample_id": sample_id,
                "compensation": compensation_id,
                "test_key": test_key_dict,
            },
        }

    def pairplot(
        self,
        sample_id: str,
        comp_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        show_markers: bool = False,
        nbins: int = 50,
        colorscale: Any = "viridis",
        width: int = 1200,
        height: int = 1200,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a pairplot with scatter plots in lower triangle and histograms on diagonal.

        Loads compensated data and creates a grid visualization of all channel pairs.

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Compensation ID to apply (default: "current")
        n_subset : int
            Number of events to load (default: from state, typically 10000)
        transformation : str
            Transformation to apply: "logicle", "identity", "asinh" (default: "logicle")
        show_markers : bool
            Whether to display marker labels instead of channel labels (default: False)
        nbins : int
            Number of bins for histograms (default: 50)
        colorscale : Any
            Plotly colorscale for density scatter plots (default: "viridis")
        width : int
            Figure width in pixels (default: 1200)
        height : int
            Figure height in pixels (default: 1200)
        **kwargs
            Additional arguments passed to build_pairplot

        Returns
        -------
        dict
            Dict with 'plotly' key containing the figure and 'metadata'
        """

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, comp_id, n_subset)

        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        # Get only fluorescence channels for the pairplot
        fluoro_markers: dict[str, str] = self.state["fluoro_markers"]  # maps channel names to marker labels
        channel_indices = np.array([comp_subset.var.index.get_loc(ch) for ch in fluoro_markers], dtype=int)

        if show_markers:
            channel_labels = list(fluoro_markers.values())
        else:
            channel_labels = list(fluoro_markers.keys())

        # Extract data for fluorescence channels only
        data = np.asarray(comp_subset.X[:, channel_indices])

        # Build the pairplot
        fig = build_pairplot(
            data,
            channel_names=channel_labels,
            transformation=transformation,
            nbins=nbins,
            colorscale=colorscale,
            title=f"Pairplot - {sample_id}",
            width=width,
            height=height,
            **kwargs,
        )

        return {
            "plotly": fig,
            "metadata": {
                "sample_id": sample_id,
                "compensation": comp_id,
                "n_channels": len(fluoro_markers),
                "n_events": data.shape[0],
                "transformation": transformation,
            },
        }
