"""
Base revision handler with common functionality.

Provides state persistence, workspace management, and data I/O for iterative refinement.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
import json
import warnings

import numpy as np
import anndata as ad

from cytomind.domain.pipeline import StepRun, RevisionSession
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.infra.repo import ProjectRepository
    from cytomind.qc.base import StepQCEvaluator
    from pandas import DataFrame
    PathLike = str | Path
else:
    ProjectRepository = object
    StepQCEvaluator = object
    PathLike = object

class BaseRevisionHandler(ABC):
    """
    Base class for revision handlers managing iterative refinement.

    Revision handlers:
    - Manage a workspace separate from the main project
    - Track iterations of user modifications
    - Use a QC evaluator to assess changes
    - Produce metadata updates and optional new steps for commit
    - Provide data I/O with workspace-first, main-repo fallback
    - Manage visualization subset caching

    Subclasses must implement:
    - apply_revision(session, user_input) -> dict (qc_summary)
    - commit(session) -> (metadata_updates, optional_new_step)
    """

    _supported_figures: dict[str, Any] = {}
    _supported_tables: dict[str, Any] = {}

    def __init__(
        self,
        step_run: StepRun,
        main_repo: ProjectRepository,
        workspace: PathLike,
        session: RevisionSession | str | None = None,
        qc_evaluator: StepQCEvaluator | None = None,
        n_subset: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize revision handler with workspace management.

        Parameters
        ----------
        step_run : StepRun
            The step being revised
        main_repo : ProjectRepository
            Main repository instance
        qc_evaluator : StepQCEvaluator | None
            QC evaluator for assessing changes during revision
        """
        self.step_run = step_run
        self.main_repo = main_repo
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.session_path.exists():
            self.load_session()
            if session is not None:
                warnings.warn("Session already exists on disk; ignoring provided session identifier.")
        elif isinstance(session, str):
            session = RevisionSession(
                id=session,
                parent_step_id=step_run.id,
                parent_step_type=step_run.step_type,
                state="inactive",
                created_at=now_iso(),
                updated_at=now_iso(),
                handler_state={
                    "n_subset": n_subset,
                    "seed": seed,
                }
            )
            self.session = session
        elif isinstance(session, RevisionSession):
            self.session = session
            self.state["n_subset"] = self.state.get("n_subset", n_subset)
            self.state["seed"] = self.state.get("seed", seed)
        else:
            raise ValueError("Must provide either a session identifier or a RevisionSession instance.")
        self.qc_evaluator = qc_evaluator

        # Viz subset caching
        self._viz_metadata: dict[str, dict[str, Any]] = {}
        self._load_viz_metadata()

    # ---- Workspace properties ----

    @property
    def step_id(self) -> str:
        """Step identifier"""
        return self.step_run.id

    @property
    def session_path(self) -> Path:
        """Path to session metadata file"""
        return self.workspace / "session.json"

    @property
    def viz_cache_dir(self) -> Path:
        """Directory for visualization subset caching"""
        return self.workspace / "viz_cache"

    @property
    def state(self) -> dict[str, Any]:
        return self.session.handler_state

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
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(self._viz_metadata, indent=2))

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
        subset_key = f"{sample_id}:{layer}:{n_subset}"

        # Check if already materialized
        if subset_key in self._viz_metadata:
            subset_path = self._viz_metadata[subset_key]["path"]
            return ad.read_h5ad(subset_path)

        # Need to materialize
        print(f"Creating viz subset: {subset_key}")
        sample_ref = self.main_repo.load_sample_meta(sample_id)

        # Subsample
        rng = np.random.RandomState(seed)
        n_total = sample_ref.n_events
        if n_total <= 0:
            raise ValueError(f"Sample {sample_id} has no events to subset.")
        if n_total <= n_subset:
            indices = np.arange(n_total)
        else:
            indices = rng.choice(n_total, n_subset, replace=False)
            indices.sort()
        subset = self.main_repo.load_sample_adata(sample_id, layer=layer, mask=indices)

        # Save to disk
        self.viz_cache_dir.mkdir(parents=True, exist_ok=True)
        subset_path = self.viz_cache_dir / f"{sample_id}_{layer}_{n_subset}.h5ad"
        subset.write_h5ad(subset_path)

        # Track in metadata
        self._viz_metadata[subset_key] = {
            "seed": seed,
            "n_total": n_total,
            "n_subset": len(indices),
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
    def apply_revision(
        self,
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply revision modifications and return updated QC summary.

        Parameters
        ----------
        user_input : dict
            User-specified modifications

        Returns
        -------
        dict
            Updated QC summary for the revised state
        """
        self.session.updated_at = now_iso()
        return {}

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
    def get_figure(self, plot_type: str, input_params: dict[str, Any]) -> dict[str, Any]:
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
    def get_table(self, table_type: str, input_params: dict[str, Any]) -> DataFrame:
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

    def start_revision(self, input_spec: dict[str, Any]) -> RevisionSession:
        """
        Initialize revision workspace for a new session.

        Override to set up handler-specific state (validate samples, copy resources, etc.).

        Parameters
        ----------
        input_spec : dict
            User input specification for revision

        Returns
        -------
        RevisionSession
            Initialized session
        """
        # Default: just return the session
        if self.session is None:
            raise RuntimeError("Session not initialized")
        self.session.state = "active"
        input_spec = input_spec.copy()
        self.state["n_subset"] = input_spec.pop("n_subset", self.state.get("n_subset", 10000))
        self.state["seed"] = input_spec.pop("seed", self.state.get("seed", 42))

        # Validate target samples
        original_samples: list[str] = self.step_run.inputs.get("sample_ids", [])
        self.session.target_samples = input_spec.pop("sample_ids", original_samples)
        original_samples_set = set(original_samples)
        for sid in self.session.target_samples:
            if sid not in original_samples_set:
                raise ValueError(f"Sample {sid} not in original step inputs.")

        self.session.input_spec = input_spec
        now = now_iso()
        self.session.created_at = now
        self.session.updated_at = now
        return self.session

    def cleanup_workspace(self) -> None:
        """Clean up visualization cache but keep session info for debugging."""
        if self.viz_cache_dir.exists():
            rmtree(self.viz_cache_dir)

    def save_session(self) -> None:
        """Persist current session state to disk."""
        if self.session is None:
            raise RuntimeError("Session not initialized")
        session_path = self.session_path
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps(self.session.to_dict(), indent=2))

    def load_session(self) -> RevisionSession:
        """Load session state from disk."""
        session_path = self.session_path
        session_data = json.loads(session_path.read_text())
        self.session = RevisionSession.from_dict(session_data)
        return self.session