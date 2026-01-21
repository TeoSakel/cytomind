"""
Compensation revision handler with stateless persistence.

Handles iterative refinement of compensation results with lazy visualization
subset materialization and on-demand feedback generation.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
from shutil import rmtree
import hashlib

import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go

from cytomind.domain.flow import CompensationRef
from cytomind.domain.pipeline import  StepRun
from cytomind.steps.compensation import apply_compensation
from cytomind.revisions import RevisionHandlerRegistry
from cytomind.revisions.base import BaseRevisionHandler
from cytomind.visualization import (
    build_histogram2d_with_marginals,
    build_histogram1d,
    build_matrix_heatmap,
)
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.domain.pipeline import RevisionSession
else:
    RevisionSession = object

@RevisionHandlerRegistry.register("compensate")
class CompensationRevisionHandler(BaseRevisionHandler):
    """
    Compensation revision handler.

    Simplified implementation for iterative refinement of compensation matrices.
    """

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
                "width": "int (default: 750)",
                "height": "int (default: 750)",
                "kwargs": "additional arguments passed to heatmap2d_with_marginals",
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
                "coef_min": "float minimum allowed coefficient (default: -0.5)",
                "coef_max": "float maximum allowed coefficient (default: 0.5)",
                "n_steps": "int number of steps (default: 41)",
                "nbins": "int number of bins per dimension (default: 128)",
                "colorscale": "any colorscale (default: 'viridis')",
            }
        },
    }

    _supported_tables = {
        "spillover": {
            "description": "Spillover matrix table",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
        "channel_qc": {
            "description": "QC metrics table for individual channels",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
        "channel_pair_qc": {
            "description": "QC metrics table for channel pairs",
            "input_params": {
                "sample_id": "string",
                "comp_id": "string can be 'current', 'active', 'parent' or actual comp_id (default: 'current')",
            },
        },
    }

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

    # --- Protocol methods (revision lifecycle) ----

    def start_revision(self, input_spec: dict[str, Any]) -> RevisionSession:
        """
        Initialize revision workspace for compensation refinement.

        Sets up:
        - Copies compensation matrices from main repo
        - Tracks sample metadata (n_subset, compensation)
        - Identifies fluorescence channels from panel
        - Prepares raw data subsets for visualization
        - Initializes state with seed and n_subset

        Parameters
        ----------
        input_spec : dict
            Input specification with sample_ids, seed, n_subset

        Returns
        -------
        RevisionSession
            Initialized session with handler state
        """

        super().start_revision(input_spec)  # update self.session
        project = self.main_repo.load_project()

        # Get fluorescence channels from raw panel
        raw_panel = project.dimensions.get("raw", [])
        fluoro_channels = [dim.id for dim in raw_panel if dim.type == "fluorescence"]
        fluoro_markers = [dim.marker for dim in raw_panel if dim.type == "fluorescence"]

        # Get sample info
        samples = {}
        for sid in self.session.target_samples:
            sample = project.samples.get(sid)
            if not sample:
                raise ValueError(f"Sample {sid} not found in project")
            samples[sid] = {
                "n_events": sample.n_events,
                "active_compensation": sample.compensation,
                "compensation": sample.compensation,
            }

        # Copy compensation matrices from main repo
        self.comp_dir.mkdir(parents=True, exist_ok=True)
        compensations = {}
        for sid in self.session.target_samples:
            sample = project.samples[sid]
            comp_id = sample.compensation if sample.compensation else "raw"

            # Skip raw / identity
            if comp_id == "raw":
                continue

            # Copy compensation spillover once and track samples using it
            if comp_id not in compensations:
                comp_ref = project.compensations[comp_id]
                comp_path = self.comp_dir / f"{comp_id}.csv"
                comp_ref.spill.to_csv(comp_path, index=False)
                compensations[comp_id] = {
                    "id": comp_id,
                    "name": comp_ref.name,
                    "source": comp_ref.source,
                    "path": str(comp_path),
                    "parent": None,  # Default compensations have no parent
                    "is_default": True,
                    "batch": comp_ref.batch.copy(),
                }

            # Record that this sample uses the compensation
            if sid not in compensations[comp_id].setdefault("samples", []):
                compensations[comp_id]["samples"].append(sid)

        # Initialize handler state
        self.state.update({
            "fluoro_channels": fluoro_channels,
            "fluoro_markers": fluoro_markers,
            "samples": samples,
            "compensations": compensations,
        })

        self.save_session()
        return self.session

    def apply_revision(
        self,
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply revision modifications to update sample compensation in the workspace.

        Supported inputs (sample_id is required as str or list[str]):
          - Existing compensation: {"sample_id": "S1" | ["S1", "S2"], "comp_id": "comp_123" | "raw" | None}
          - New spillover:        {"sample_id": "S1" | ["S1", "S2"], "spillover": df, "name": "optional"}

        Notes:
        - Empty sample_id list is invalid and will raise.
        - comp_id of None or "raw" applies identity compensation.
        """

        # Normalize sample_ids
        sample_field = user_input.get("sample_id")
        if sample_field is None:
            raise ValueError("user_input must contain 'sample_id' (str or list[str])")

        if isinstance(sample_field, str):
            target_samples = [sample_field]
        elif isinstance(sample_field, list):
            if len(sample_field) == 0:
                raise ValueError("sample_id list cannot be empty")
            target_samples = sample_field
        else:
            raise TypeError("sample_id must be str or list[str]")

        # Validate samples exist
        for sid in target_samples:
            if sid not in self.samples:
                raise KeyError(f"Sample {sid} not found in workspace")

        # Case 1: Existing or trivial compensation
        if "comp_id" in user_input and "spillover" not in user_input:
            comp_id = user_input.get("comp_id")
            if comp_id is None:
                comp_id = "raw"

            if comp_id != "raw" and comp_id not in self.compensations:
                raise ValueError(f"Compensation {comp_id} not found in workspace")

            for sid in target_samples:
                self.samples[sid]["compensation"] = comp_id
                if comp_id != "raw":
                    comp_info = self.compensations[comp_id]
                    comp_info.setdefault("batch", [])
                    if sid not in comp_info["batch"]:
                        comp_info["batch"].append(sid)

            self.session.updated_at = now_iso()
            self.save_session()

            return {
                "status": "applied",
                "mode": "existing_compensation",
                "comp_id": comp_id,
                "samples_updated": target_samples,
            }

        # Case 2: Spillover provided (new or existing comp_id reuse)
        if "spillover" in user_input:
            spillover_df = user_input["spillover"]
            if not isinstance(spillover_df, pd.DataFrame):
                raise TypeError(f"spillover must be pd.DataFrame, got {type(spillover_df)}")

            comp_name = user_input.get("name")
            comp_id_override = user_input.get("comp_id")

            comp_ids_created: list[str] = []
            for sid in target_samples:
                comp_id = self.update_sample_compensation(
                    sid,
                    spill_df=spillover_df,
                    comp_name=comp_name,
                    comp_id=comp_id_override,
                )
                comp_ids_created.append(comp_id)

            self.session.updated_at = now_iso()
            self.save_session()

            return {
                "status": "applied",
                "mode": "new_spillover",
                "comp_ids": comp_ids_created,
                "samples_updated": target_samples,
                "spillover_shape": spillover_df.shape,
            }

        raise ValueError("user_input must contain either 'comp_id' or 'spillover'")

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

        comp_map = {
            sid: sinfo["compensation"]
            for sid, sinfo in self.samples.items()
            if sinfo["compensation"] != sinfo["active_compensation"]
        }

        comp_refs = {}
        for comp_id in comp_map.values():
            if comp_id in comp_refs:
                continue
            comp_refs[comp_id] = self.get_comp_ref(comp_id)

        step_run = StepRun(
            id = f"{self.session.parent_step_id}_{self.session.id}",
            step_type="compensate",
            inputs={"sample_ids": list(comp_map.keys())},
            config={"comp_id": comp_map},
            created_at=now_iso(),
        )
        self.session.state = "committed"
        self.session.updated_at = now_iso()
        self.save_session()
        return {"compensations": comp_refs.values()}, step_run

    def cleanup_workspace(self) -> None:
        super().cleanup_workspace()
        rmtree(self.comp_dir)


    # ---- Compensation accessors ----

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
        if comp_id == "raw":
            return self._get_identity_compensation()

        try:
            comp_info = self.state["compensations"][comp_id]
        except KeyError:
            raise KeyError(f"Compensation {comp_id} not found in workspace")

        spill_df = pd.read_csv(comp_info["path"], index_col=False)

        return CompensationRef(
            id=comp_info["id"],
            name=comp_info["name"],
            source=comp_info["source"],
            _spill=spill_df,
        )

    def update_sample_compensation(
        self,
        sample_id: str,
        spill_df: pd.DataFrame,
        comp_name: str | None = None,
        comp_id: str | None = None,
    ) -> str:
        """Validate spillover, create/reuse compensation, and assign to sample."""

        if sample_id not in self.samples:
            raise KeyError(f"Sample {sample_id} not found in workspace")

        # ---- Validate spillover matrix ----
        if not isinstance(spill_df, pd.DataFrame):
            raise TypeError(f"spill_df must be pd.DataFrame, got {type(spill_df)}")

        if spill_df.shape[0] != spill_df.shape[1]:
            raise ValueError(f"Spillover matrix must be square, got {spill_df.shape}")

        if not all(spill_df.index == spill_df.columns):
            raise ValueError("Spillover matrix row and column labels must match")

        fluoro_channels: list[str] = self.state["fluoro_channels"]
        if not all(ch in spill_df.columns for ch in fluoro_channels):
            raise ValueError("Spillover matrix columns do not match fluorescence channels in workspace")
        spill_df = spill_df.loc[fluoro_channels, fluoro_channels]

        if spill_df.isnull().any().any():
            raise ValueError("Spillover matrix contains NaN values")

        min_val = float(spill_df.min().min())
        max_val = float(spill_df.max().max())
        if not (0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0):
            raise ValueError(
                f"Spillover values should be in [0, 1], got range [{min_val:.4f}, {max_val:.4f}]"
            )

        # Determine parent compensation for naming lineage
        parent_comp_id = self.current_comp(sample_id)

        # If a comp_id is provided and exists, reuse and append sample
        if comp_id and comp_id in self.compensations:
            comp_info = self.compensations[comp_id]
            comp_path = Path(comp_info["path"])
            comp_info.setdefault("batch", [])
            if sample_id not in comp_info["batch"]:
                comp_info["batch"].append(sample_id)
        else:
            # Auto-generate comp_id from hash if not provided or not existing
            comp_id = comp_id or "comp_" + hashlib.md5(spill_df.to_csv().encode()).hexdigest()[:8]

            # Auto-generate name if not provided
            if comp_name is None:
                parent_info = self.compensations.get(parent_comp_id, {})
                parent_name = parent_info.get("name", f"comp_{sample_id}")
                comp_name = f"{parent_name}_revised"

            comp_path = self.spillover_path(comp_id)
            self.compensations[comp_id] = {
                "id": comp_id,
                "name": comp_name,
                "source": "user",
                "path": comp_path.as_posix(),
                "parent": parent_comp_id,
                "is_default": False,
                "batch": [sample_id],
            }

        # Persist spillover to workspace file (overwrite if exists)
        spill_df.to_csv(comp_path, index=False)

        # Assign compensation to sample
        self.samples[sample_id]["compensation"] = comp_id

        # Update session metadata
        self.session.updated_at = now_iso()
        self.save_session()

        return comp_id

    def _resolve_compensation(self, sample_id: str, comp_id: str = "current") -> tuple[str, CompensationRef]:
        """Resolve a comp_id (special or explicit) to a concrete (comp_id, CompensationRef).

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
        return comp_id, self.get_comp_ref(comp_id)

    def _get_identity_compensation(self) -> CompensationRef:
        """Create identity compensation matrix for raw (uncompensated) data."""

        # Get fluorescence channels
        fluoro_channels = self.state["fluoro_channels"]

        # Create identity matrix
        n_channels = len(fluoro_channels)
        identity_matrix = pd.DataFrame(
            np.eye(n_channels),
            index=fluoro_channels,
            columns=fluoro_channels
        )

        return CompensationRef(
            id="identity",
            name="Raw (Identity)",
            source="identity",
            _spill=identity_matrix,
        )

    # --- Table implementations ----

    def get_table(self, table_type: str, input_params: dict[str, Any]) -> pd.DataFrame:
        if table_type == "spillover":
            return self.get_spillover_table(**input_params)
        if table_type == "channel_qc":
            return self.get_channel_qc_table(**input_params)
        if table_type == "channel_pair_qc":
            return self.get_channel_pair_qc_table(**input_params)
        raise ValueError(f"Unknown table type: {table_type}")

    def get_channel_qc_table(self, sample_id: str, comp_id: str = "current") -> pd.DataFrame:
        """Get channel QC table for a sample under a specific compensation.

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Which compensation to apply:
                - "current",
                - "parent" (parent of current),
                - "active" (compensation from sample metadata),
                - "raw" (identity matrix)

        Returns
        -------
        pd.DataFrame
            Channel QC table
        """
        raise NotImplementedError("Channel QC computation not implemented yet")
        comp_id, comp_ref = self._resolve_compensation(sample_id, comp_id)

        # Load compensated visualization subset
        n_subset = int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, comp_id, n_subset)

        # Compute channel QC metrics
        qc_df = compute_channel_qc(comp_subset)

        return qc_df

    def get_channel_pair_qc_table(self, sample_id: str, comp_id: str = "current") -> pd.DataFrame:
        """Get channel pair QC table for a sample under a specific compensation.

        Parameters
        ----------
        sample_id : str
            Sample ID
        comp_id : str
            Which compensation to apply:
                - "current",
                - "parent" (parent of current),
                - "active" (compensation from sample metadata),
                - "raw" (identity matrix)

        Returns
        -------
        pd.DataFrame
            Channel pair QC table
        """
        raise NotImplementedError("Channel pair QC computation not implemented yet")
        comp_id, comp_ref = self._resolve_compensation(sample_id, comp_id)

        # Load compensated visualization subset
        n_subset = int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, comp_id, n_subset)

        # Compute channel pair QC metrics
        qc_df = compute_channel_pair_qc(comp_subset)

        return qc_df


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
        comp_id, ref = self._resolve_compensation(sample_id, comp_id)
        df = ref.spill
        df.index = df.columns
        return df

    # ---- Figure implementations ----

    def get_figure(self, plot_type: str, input_params: dict[str, Any]) -> dict[str, Any]:
        if plot_type == "comp_heatmap":
            return self.comp_heatmap(**input_params)

        if plot_type == "heatmap2d":
            return self.heatmap2d(**input_params)

        if plot_type == "channel_histogram":
            return self.channel_histogram(**input_params)

        if plot_type == "heatmap2d_tuner":
            return self.heatmap2d_tuner(**input_params)

        raise ValueError(f"Unknown plot type: {plot_type}")


    def comp_heatmap(
        self,
        sample_id: str,
        comp_id: str = "current",
        show_markers: bool = False,
        colorscale: Any = "RdBu",
        **kwargs
    ) -> dict[str, Any]:

        spill_df = self.get_spillover_table(sample_id, comp_id)
        if show_markers:
            marker_map = dict(zip(self.state["fluoro_channels"], self.state["fluoro_markers"]))
            markers = [marker_map.get(ch, ch) for ch in spill_df.columns]
            spill_df.columns = markers
            spill_df.index = markers

        fig = build_matrix_heatmap(
            spill_df,
            colorscale=colorscale,
            zmid=0.0,
            title=f"Spillover Matrix - {sample_id} \nCompensation id: {comp_id}",
            xaxis_title="Donor Channels",
            yaxis_title="Receiver Channels",
            **kwargs
        )

        return {
            "plotly": fig,
            "metadata": {
                "comp_id": comp_id,
                "n_channels": spill_df.shape[0]
            }
        }

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
        comp_id, comp_ref = self._resolve_compensation(sample_id, comp_id)

        # Load raw subset (creates if doesn't exist) and apply compensation on the fly
        raw_subset = self.get_or_create_viz_subset(sample_id, "raw", n_subset, self.state["seed"])
        comp_subset = apply_compensation(raw_subset, comp_ref, invert=False)

        return comp_subset

    def heatmap2d(
        self,
        sample_id: str,
        donor: str,
        receiver: str,
        comp_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
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
        comp_id : str, optional
            Compensation ID to apply (defaults to current from comp_map)
        n_subset : int
            Number of events to load (default: 10000)
        nbins : int
            Number of bins for histogram (default: 120)
        colorscale : Any
            Plotly colorscale (default: "Viridis")
        transformation : str
            Transformation to apply: "logicle", "linear", or "asinh" (default: "logicle")
        """

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, comp_id, n_subset)

        donor_idx = comp_subset.var.index.get_loc(donor)
        receiver_idx = comp_subset.var.index.get_loc(receiver)

        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        x = np.asarray(comp_subset.X[:, donor_idx])
        y = np.asarray(comp_subset.X[:, receiver_idx])

        # Delegate to visualization builder
        fig = build_histogram2d_with_marginals(
            x,
            y,
            transformation=transformation,
            title=f"{donor} vs {receiver} - {sample_id}",
            xaxis_title=donor,
            yaxis_title=receiver,
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
        comp_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
        **kwargs,
    ):
        """Histogram from raw subset with on-the-fly compensation and transform."""

        n_subset = n_subset if n_subset is not None else int(self.state["n_subset"])
        comp_subset = self.load_viz_data_compensated(sample_id, comp_id, n_subset)
        if comp_subset.X is None:
            raise ValueError("Compensated subset has no data")

        # Extract channel values
        channel_idx = comp_subset.var.index.get_loc(channel)
        values = np.asarray(comp_subset.X[:, channel_idx])

        fig = build_histogram1d(
            values,
            title=f"{channel} - {sample_id}",
            xaxis_title=channel,
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
        comp_id: str = "current",
        n_subset: int | None = None,
        transformation: str = "logicle",
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
        comp_id : str
            Compensation to start from (default: current mapping).
        n_subset : int
            Number of events to visualize.
        transformation : str
            Transform to apply before binning (e.g., "logicle", "identity").
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
        comp_id, comp_ref = self._resolve_compensation(sample_id, comp_id)
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
                H_normalized = H.T / H_max
            else:
                H_normalized = H.T
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
                        {"title": f"{donor} vs {receiver} - {sample_id}<br>Spillover coefficient: {float(c):.4f}"},  # Second arg: layout updates
                    ],
                    "method": "update",  # Use update to modify both data and layout
                    "label": f"{float(c):.3f}",
                }
            )

        fig.update_layout(
            title=f"{donor} vs {receiver} - {sample_id}<br>Spillover coefficient: {init_coef:.4f}",
            xaxis_title=donor,
            yaxis_title=receiver,
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
                "compensation": comp_id,
                "coef_default": current_coef,
                "coef_min": coef_min,
                "coef_max": coef_max,
            },
        }
