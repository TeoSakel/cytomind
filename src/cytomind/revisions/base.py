"""
Base revision handler with common functionality.

Provides state persistence, workspace management, and data I/O for iterative refinement.
"""
from __future__ import annotations
from typing import Any, Mapping, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
import json
import warnings

import numpy as np
import anndata as ad

from cytomind.domain.pipeline import RevisionSession, NumpyEncoder
from cytomind.domain.qc import EntityQCStatus
from cytomind.qc import EntityQCEvaluatorRegistry
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.infra.repo import ProjectRepository
    from cytomind.domain.pipeline import StepRun
    from pandas import DataFrame
else:
    ProjectRepository = object
    StepRun = object
    DataFrame = object
    PathLike = object


class BaseRevisionHandler(ABC):
    """
    Base class for revision handlers managing iterative refinement of entities.

    Revision handlers:
    - Manage a workspace separate from the main project
    - Track iterations of user modifications
    - Use a QC evaluator to assess changes
    - Produce metadata updates for commit
    - Provide data I/O with workspace-first, main-repo fallback
    - Manage visualization subset caching

    Entity types:
    - "compensation": Revise compensation matrices
    - "gating_strategy": Revise gating strategies
    - "step": Edit step configuration for rerun (use BaseStepRevisionHandler)

    Subclasses must implement:
    - apply_revision(user_input) -> None
    - _commit() -> (metadata_updates, optional_new_step)
    - get_figure(plot_type, input_params) -> dict
    - get_table(table_type, input_params) -> DataFrame
    """

    entity_type: str
    _supported_figures: dict[str, Any] = {}
    _supported_tables: dict[str, Any] = {}

    def __init__(
        self,
        main_repo: ProjectRepository,
        workspace: PathLike | None = None,
        session: RevisionSession | None = None,
        entity_id: str | None = None,
        n_subset: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize revision handler with entity context and workspace management.

        Parameters
        ----------
        entity_id : str | None
            Specific entity identifier (e.g., "comp_001", "step_0003")
        main_repo : ProjectRepository
            Main repository instance
        workspace : PathLike
            Workspace directory for this revision session
        entity_context : dict[str, Any] | None
            Entity-specific context data (e.g., compensation batch, step config)
        session : RevisionSession | str | None
            Existing session or session ID to create
        n_subset : int
            Default number of events for visualization subsets
        seed : int
            Random seed for reproducibility
        """
        self.main_repo = main_repo
        if workspace is None:
            workspace = self.main_repo.generate_revision_workspace(self.entity_type, entity_id)
        self.workspace = Path(workspace)
        if self.session_path.exists():
            self.load_session()
            if session is not None:
                warnings.warn("Session already exists on disk; ignoring provided session identifier.")
        elif isinstance(session, RevisionSession):
            if session.id != self.workspace.name:
                raise ValueError(f"Provided session ID '{session.id}' does not match workspace name '{self.workspace.name}'")
            self.session = session
            self.state["n_subset"] = self.state.get("n_subset", n_subset)
            self.state["seed"] = self.state.get("seed", seed)
            self.save_session()
        else:
            self.session = self.create_session()
            self.session.entity_id = entity_id
            self.state["n_subset"] = n_subset
            self.state["seed"] = seed

        if self.entity_type != self.session.entity_type:
            raise TypeError(f"Session entity type '{self.session.entity_type}' does not match handler entity type '{self.entity_type}'")
        if entity_id is not None and self.session.entity_id is not None and entity_id != self.session.entity_id:
            raise ValueError(f"Provided entity_id '{entity_id}' does not match session entity_id '{self.session.entity_id}'")

        qc_evaluator_class = EntityQCEvaluatorRegistry.get(self.entity_type)
        if qc_evaluator_class is None:
            raise TypeError(f"No QC evaluator registered for entity type '{self.entity_type}'")
        try:
            self.qc_evaluator = qc_evaluator_class()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QC evaluator for entity type: {e}")

        self._load_viz_metadata()

    # ---- Workspace properties ----

    @property
    def state(self) -> dict[str, Any]:
        return self.session.handler_state

    @property
    def session_path(self) -> Path:
        """Path to session metadata file"""
        return self.workspace / "session.json"

    @property
    def viz_cache_dir(self) -> Path:
        """Directory for visualization subset caching"""
        return self.workspace / "viz_cache"

    @property
    def qc_cache_dir(self) -> Path:
        """Directory for QC evaluation caching"""
        return self.workspace / "qc_cache"

    # ----- JSON I/O Helper -----

    @staticmethod
    def _write_json(path: PathLike, data: dict) -> None:
        """
        Write a JSON-serializable dictionary to a file.

        Uses NumpyEncoder to handle numpy scalar types (int64, float64, etc.)
        and numpy arrays, converting them to Python native types.

        Parameters
        ----------
        path : PathLike
            Destination file path.
        data : dict
            Data to serialize to JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    # ----- Handle QC evaluation -----

    def load_entity_qc_cache(self, entity_id: str) -> EntityQCStatus:
        """Load cached QC results for a specific entity."""
        cache_path = self.qc_cache_dir / f"{entity_id}.json"
        if cache_path.exists():
            qc_data = json.loads(cache_path.read_text())
            return EntityQCStatus.from_dict(qc_data)

        qc_path = self.main_repo.qc_entity_status_path(self.entity_type, entity_id)
        if qc_path.exists():
            qc_data = json.loads(qc_path.read_text())
            return EntityQCStatus.from_dict(qc_data)

        return EntityQCStatus(entity_id=entity_id, entity_type=self.entity_type, generated_at=now_iso())

    def save_entity_qc_cache(self, entity_id: str, qc_status: EntityQCStatus) -> None:
        """Save QC results for a specific entity to cache."""
        cache_path = self.qc_cache_dir / f"{entity_id}.json"
        self._write_json(cache_path, qc_status.to_dict())

    # ---- Viz subset management ----

    @property
    def viz_metadata_path(self) -> Path:
        """Path to viz subset metadata file."""
        return self.viz_cache_dir / "viz_metadata.json"

    def _load_viz_metadata(self):
        """Load viz subset metadata from cache."""
        try:
            self._viz_metadata = json.loads(self.viz_metadata_path.read_text())
        except FileNotFoundError:
            self._viz_metadata = {}

    def _save_viz_metadata(self):
        """Persist viz subset metadata."""
        metadata_path = self.viz_metadata_path
        self._write_json(metadata_path, self._viz_metadata)

    def get_or_create_viz_subset(
        self,
        sample_id: str,
        layer: str,
        n_subset: int | None = None,
        seed: int | None = None,
    ) -> ad.AnnData:
        """
        Get or create a visualization subset.

        Lazy: creates on first access, caches thereafter.

        Parameters
        ----------
        sample_id : str
            Sample identifier
        layer : str
            Layer name (e.g., "raw", "comp")
        n_events : int
            Number of events in subset
        seed : int
            Random seed for reproducibility

        Returns
        -------
        AnnData
            Subset loaded into memory
        """
        if n_subset is None:
            n_subset = int(self.state["n_subset"])
        if seed is None:
            seed = int(self.state["seed"])

        # Validate sample exists in main repo
        sample_ref = self.main_repo.load_sample_meta(sample_id)
        n_total = sample_ref.n_events
        if n_total <= 0:
            raise ValueError(f"Sample {sample_id} has no events to subset.")
        n_subset_cliped = min(n_subset, n_total)

        subset_key = f"{sample_id}:{layer}:{n_subset_cliped}"

        # Check if already materialized
        if subset_key in self._viz_metadata:
            subset_path = self._viz_metadata[subset_key]["path"]
            return ad.read_h5ad(subset_path)

        # Need to materialize
        print(f"Creating viz subset: {subset_key}")

        # Subsample
        if n_total > n_subset:
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_total, n_subset, replace=False)
            indices.sort()
        else:
            indices = slice(None)
            if n_total != n_subset:
                warnings.warn(f"Requested subset size {n_subset} exceeds total events {n_total} for sample {sample_id}. Using all events.")
        subset = self.main_repo.load_sample_adata(sample_id, layer=layer, mask=indices)

        # Save to disk
        self.viz_cache_dir.mkdir(parents=True, exist_ok=True)
        subset_path = self.viz_cache_dir / f"{sample_id}_{layer}_{n_subset_cliped}.h5ad"
        if n_total > n_subset:
            subset.write_h5ad(subset_path)
        else:            # If not subsetting, we can create a hard link to save space
            original_path = self.main_repo.sample_adata_path(sample_id, layer=layer)
            try:
                subset_path.hardlink_to(original_path)
            except OSError:
                # Fall back to copy if hard link fails (e.g., cross-filesystem)
                subset.write_h5ad(subset_path)

        # Track in metadata
        self._viz_metadata[subset_key] = {
            "seed": seed,
            "n_total": n_total,
            "n_subset": subset.n_obs,
            "created_at": now_iso(),
            "path": subset_path.as_posix()
        }
        self._save_viz_metadata()

        return subset

    def save_viz_object(self, key: str, adata: ad.AnnData, n_subset: int | None = None, seed: int = 42) -> ad.AnnData:
        """
        Save an anndata object to viz_cache_dir with a given key.

        Allows handlers to override default viz caching behavior.

        Parameters
        ----------
        key : str
            Cache key for the object
        adata : AnnData
            Anndata object to cache
        n_subset : int | None
            If set, subset to this many observations. Uses random sampling if needed.
        seed : int
            Random seed for reproducibility when subsetting

        Returns
        -------
        Path
            Path where the object was saved
        """

        # Subset if requested
        to_save = adata
        if n_subset is not None and adata.n_obs > n_subset:
            rng = np.random.RandomState(seed)
            indices = rng.choice(adata.n_obs, n_subset, replace=False)
            indices.sort()
            to_save = adata[indices, :].to_memory()
            key = f"{key}:{n_subset}"

        # Create a sanitized filename from the key
        filename = f"{key}.h5ad"
        obj_path = self.viz_cache_dir / filename.replace(":", "_")

        # Save to disk
        self.viz_cache_dir.mkdir(parents=True, exist_ok=True)
        to_save.write_h5ad(obj_path)

        # Track in metadata
        self._viz_metadata[key] = {
            "seed": seed,
            "n_total": adata.n_obs,
            "n_subset": to_save.n_obs,
            "created_at": now_iso(),
            "path": obj_path.as_posix()
        }
        self._save_viz_metadata()

        return to_save

    def invalidate_viz_cache(self, sample_ids: list[str], layer: str):
        """
        Remove cached viz subsets after data modification.

        Parameters
        ----------
        sample_ids : list[str]
            Sample IDs whose data was modified
        layer : str
            Layer that was modified
        """
        for sample_id in sample_ids:
            keys_to_remove = [
                k for k in self._viz_metadata.keys()
                if k.startswith(f"{sample_id}:{layer}:")
            ]

            for key in keys_to_remove:
                # Remove from disk
                subset_path = Path(self._viz_metadata[key]["path"])
                if subset_path.exists():
                    subset_path.unlink()

                # Remove from metadata
                del self._viz_metadata[key]

        self._save_viz_metadata()

    # ---- Abstract methods ----

    @abstractmethod
    def start_revision(self, input_spec: Mapping[str, Any] = {}) -> RevisionSession:
        """
        Initialize revision workspace for a new session.

        Override to set up handler-specific state (validate samples, copy resources, etc.).

        Parameters
        ----------
        input_spec : dict
            User input specification for revision.
            Must contain 'sample_ids' key with list of sample IDs to revise,
            or subclass must discover samples from entity context.

        Returns
        -------
        RevisionSession
            Initialized session
        """
        pass

    @abstractmethod
    def apply_revision(
        self,
        user_input: Mapping[str, Any] = {},
    ) -> None:
        """
        Apply revision modifications and track in history.

        Records the applied changes in the session's revision history
        for reproducibility and debugging.

        Parameters
        ----------
        user_input : dict
            User-specified modifications
        """
        self.session.updated_at = now_iso()

    @abstractmethod
    def _commit(self) -> tuple[dict[str, Any], StepRun | None]:
        """
        Commit revision changes to main project.

        Returns metadata updates to apply and optional new step to execute.

        Parameters
        ----------
        session : RevisionSession
            Revision session being committed

        Returns
        -------
        tuple
            (metadata_updates, new_step)
            - metadata_updates: dict with keys like 'compensations', 'samples', etc.
            - new_step: StepRun to execute after commit, or None
        """
        pass

    @abstractmethod
    def get_figure(self, plot_type: str, input_params: Mapping[str, Any] = {}) -> dict[str, Any]:
        """
        Generate a figure for visualization based on plot type and input parameters.

        Parameters
        ----------
        plot_type : str
            Type of plot to generate
        input_params : dict
            Parameters specific to the plot type

        Returns
        -------
        dict
            Figure data (e.g., Plotly figure dictionary)
        """
        pass

    @abstractmethod
    def get_table(self, table_type: str, input_params: Mapping[str, Any] = {}) -> DataFrame:
        """
        Generate a table based on table type and input parameters.

        Parameters
        ----------
        table_type : str
            Type of table to generate
        input_params : dict
            Parameters specific to the table type

        Returns
        -------
        pandas.DataFrame
            Table data
        """
        pass

    @classmethod
    def list_figures(cls, show_args=True) -> dict[str, Any]:
        """
        List available figure types for this revision handler.

        Returns
        -------
        dict[str, Any]
            List of figure type descriptors
        """
        figs = cls._supported_figures.copy()
        if not show_args:
            for fig in figs.values():
                fig.pop("input_specs", None)
        return figs

    @classmethod
    def list_tables(cls, show_args=True) -> dict[str, Any]:
        """
        List available table types for this revision handler.

        Returns
        -------
        dict[str, Any]
            List of table type descriptors
        """
        tabs = cls._supported_tables.copy()
        if not show_args:
            for tab in tabs.values():
                tab.pop("input_specs", None)
        return tabs

    # ---- Optional lifecycle methods ----

    def cleanup_workspace(self) -> None:
        """Clean up visualization cache but keep session info for debugging."""
        if self.viz_cache_dir.exists():
            rmtree(self.viz_cache_dir)

    def create_session(self) -> RevisionSession:
        """Create a new session with default values."""
        now = now_iso()
        return RevisionSession(
            id=self.workspace.name,
            entity_type=self.entity_type,
            entity_id=None,
            state="initialized",
            context={},
            revision_history=[],
            created_at=now,
            updated_at=now,
        )

    def save_session(self) -> None:
        """Persist current session state to disk."""
        if self.session is None:
            raise RuntimeError("Session not initialized")
        session_path = self.session_path
        self._write_json(session_path, self.session.to_dict())

    def load_session(self) -> RevisionSession:
        """Load session state from disk."""
        session_path = self.session_path
        session_data = json.loads(session_path.read_text())
        self.session = RevisionSession.from_dict(session_data)
        return self.session