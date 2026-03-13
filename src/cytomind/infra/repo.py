from __future__ import annotations
from typing import Any, Callable, Iterable, Sequence, Mapping, Literal, TypeVar, TYPE_CHECKING
from pathlib import Path
import warnings

from matplotlib.pyplot import flag
import pandas as pd

from cytomind.domain.flow import CompensationRef, ChannelRef, DimensionDef
from cytomind.domain.transforms import TransformDef
from cytomind.domain.pipeline import Project, SampleRef, StepRun, BatchRef, RevisionSession
from cytomind.domain.qc import EntityQCStatus
from cytomind.domain.gates import GateNode, GatingStrategyRef
from cytomind.infra.dataloader import UnifiedDataLoader

if TYPE_CHECKING:
    from anndata import AnnData
    from cytomind.domain.constants import PathLike, MaskLike, Serializable, ProjectMetadata
    import numpy as np
    from numpy.typing import NDArray
    BooleanArray = NDArray[np.bool_]
    R = TypeVar("R")  # Generic return type for parse/serialize functions
else:
    AnnData = object
    PathLike = object
    MaskLike = object
    Serializable = object
    ProjectMetadata = object
    BooleanArray = object
    R = object


class ProjectRepository:
    """A repository for loading/saving project data."""

    path_scheme: dict[str, str] = {

        # ---- Project Metadata ----
        "project": "project.json",
        # Batches
        "batches_dir": "batches",
        "batch_dir": "batches/{batch_id}",
        # Compensations
        "compensations_dir": "compensations",
        "compensation_spillover": "compensations/{comp_id}/spillover.csv",
        # Samples
        "samples_dir": "samples",
        "sample_dir": "samples/{sample_id}",
        "sample_adata": "samples/{sample_id}/{layer}.h5ad",

        # ---- Gating Strategy I/O ----
        "gating_strategy_dir": "gating_strategy",
        "gating_strategy": "gating_strategy/strategy.json",
        "gate_node": "gating_strategy/gates/{node_id}.json",
        "gating_strategy_masks_dir": "gating_strategy/masks",
        "gating_mask": "gating_strategy/masks/{mask_id}/{sample_id}.npy",

        # ---- Step/Pipeline I/O ----
        "steps_dir": "steps",
        "step_run_dir": "steps/{step_id}",
        "step_run": "steps/{step_id}/info.json",

        # ---- QC I/O ----
        "qc_dir": "qc",
        "qc_entity_dir": "qc/{entity_type}/{entity_id}",
        "qc_entity_artifacts_dir": "qc/{entity_type}/{entity_id}/artifacts",
        "qc_entity_artifact_dir": "qc/{entity_type}/{entity_id}/artifacts/{artifact_key}",
        "entity_qc": "qc/{entity_type}/{entity_id}/info.json",

        # ---- Revision Workspaces ----
        "workspaces_dir": "workspaces",
        "revision_type_dir": "workspaces/{entity_type}",
        "revision_workspace": "workspaces/{entity_type}/{session_id}",
        "revision_session": "workspaces/{entity_type}/{session_id}/session.json",
    }

    def __init__(self, root: PathLike, name: str | None = None):
        """
        Initialize repository for an existing or new project root.

        If the project does not exist on disk a minimal project.json is created.

        Parameters
        ----------
        root : PathLike
            Path to the project directory.
        name : str | None
            Optional project name used when creating a new project. Ignored when
            loading an existing project.
        """
        self.root = Path(root)
        self._dataloader = UnifiedDataLoader(
            path_scheme=self.path_scheme,
            root_dir=self.root,
            data_handlers=self._default_metadata_handlers(),
            fallback=None,  # Main repo has no fallback
            viz_cache_dir=None,  # Main repo doesn't use viz caching
        )

        try:
            self.load_project_metadata("project")
            if name is not None:
                warnings.warn("Project name is ignored when loading existing project from disk.")
        except FileNotFoundError:
            project = Project(id=name or self.root.name)
            self.save_project(project)


    def _default_metadata_handlers(self) -> dict[str, tuple[Callable[[Any], ProjectMetadata], Callable[[Any], dict|list] | None]]:
        """
        Define default metadata handlers for domain objects.

        Each entry maps a metadata type to a tuple of (parse_func, serialize_func).
        The parse_func converts raw dict/list data into domain objects, while the
        serialize_func converts domain objects back into dict/list for storage.

        Returns
        -------
        dict
            Mapping of metadata type -> (parse_func, serialize_func).
        """
        return  {
            "project":          (          Project.from_dict, None),
            "step_run":         (          StepRun.from_dict, None),
            "dimension":        (     DimensionDef.from_dict, None),
            "entity_qc":        (   EntityQCStatus.from_dict, None),
            "revision_session": (  RevisionSession.from_dict, None),
            "gating_strategy":  (GatingStrategyRef.from_dict, None),
        }

    # -------------- Project I/O -------------

    def load_project_metadata(self, entity: str, **kwargs) -> Any:
        return self._dataloader.load_data(entity, **kwargs)

    def save_project_metadata(self, entity: str, data: Any, overwrite: bool = True, **kwargs) -> Path:
        return self._dataloader.save_data(entity, data, overwrite=overwrite, **kwargs)

    def load_project(self) -> Project:
        project: Project = self.load_project_metadata("project")
        return project

    def save_project(self, project: Project) -> None:
        self.save_project_metadata("project", project)

    def update_project_metadata(
        self,
        samples: Iterable[SampleRef] = {},
        panel_catalog: Mapping[str, Sequence[ChannelRef]] = {},
        layers: Mapping[str, list[DimensionDef]] = {},
        compensations: Iterable[CompensationRef] = [],
        transforms: Iterable[TransformDef] = [],
        batches: Iterable[BatchRef] = {},
        gating_strategy: GatingStrategyRef | None = None,
        drop_samples: Iterable[str] = [],
        drop_batches: Iterable[str] = [],
    ) -> None:
        """
        Merge and persist updates to the project's registries.

        Only non-empty keyword arguments will be merged into the on-disk project metadata.
        All metadata is stored in project.json; no separate catalog files are written.

        Parameters
        ----------
        samples : Iterable[SampleRef], optional
            Iterable of SampleRef objects to add/update.
        panel_catalog : Mapping[str, Sequence[ChannelRef]], optional
            Mapping of panel_id -> channel definitions for all panels.
        layers : Mapping[str, list[DimensionDef]], optional
            Data layer dimension definitions to add/update.
        compensations : Iterable[CompensationRef], optional
            Compensation references to add/update.
        transforms : Iterable[TransformDef], optional
            Transform definitions to add/update.
        batches : Iterable[BatchRef], optional
            Iterable of BatchRef objects to add/update.
        gating_strategy : GatingStrategyRef | None, optional
            GatingStrategyRef object to set/update for the project.
        drop_samples : Iterable[str], optional
            List of sample IDs to remove from the project. This will delete sample
            directories and remove references from batches.
        drop_batches : Iterable[str], optional
            List of batch IDs to remove from the project. This will delete batch directories.
        """
        project: Project = self.load_project()
        update_needed = False

        # Handle sample deletions first
        drop_samples = list(drop_samples)
        if drop_samples:
            project = self._drop_samples(project=project, sample_ids=drop_samples)
            update_needed = True

        # Handle batch deletions
        drop_batches = list(drop_batches)
        if drop_batches:
            project = self._drop_batches(project=project, batch_ids=drop_batches)
            update_needed = True

        def iter_to_dict(iterable: Iterable[R]) -> dict[str, R]:
            """Helper to convert iterable of domain objects to dict by id."""
            return {item.id: item for item in iterable} # pyright: ignore[reportAttributeAccessIssue]

        # Handle sample additions/updates
        if samples:
            project.samples.update(iter_to_dict(samples))
            update_needed = True

        if batches:
            project.batches.update(iter_to_dict(batches))
            update_needed = True

        if panel_catalog:
            project.panel_catalog = {k: list(v) for k, v in panel_catalog.items()}
            update_needed = True

        if compensations:
            project.compensations.update(iter_to_dict(compensations))
            update_needed = True

        if layers:
            project.layers.update(layers)
            update_needed = True

        if transforms:
            for transform in transforms:
                project.register_transform(transform)
            update_needed = True

        if gating_strategy is not None:
            if not isinstance(gating_strategy, GatingStrategyRef):
                raise TypeError("gating_strategy must be a GatingStrategyRef instance.")

            # Persist gate-node JSON files via dataloader for compatibility.
            # TODO: allow partial updates to gating strategy without requiring full resave of all nodes.
            for node in gating_strategy.iter_nodes():
                self._dataloader.save_gate_node(node=node, overwrite=True)

            project.gating_strategy = gating_strategy
            update_needed = True

        if update_needed:
            self.save_project(project)


    def _drop_samples(self, project: Project, sample_ids: Sequence[str]) -> Project:
        sample_ids_set = set(sample_ids)
        sample_ids = list(sample_ids_set)  # Ensure uniqueness

        # Validate that all samples exist
        missing_samples = sample_ids_set - project.samples.keys()
        if missing_samples:
            raise ValueError(f"Sample(s) not found in project: {', '.join(sorted(missing_samples))}")

        # Remove sample data directories
        for sample_id in sample_ids:
            self._dataloader.remove_data("sample_dir", sample_id=sample_id)

        # Remove samples from project
        for sample_id in sample_ids:
            project.samples.pop(sample_id, None)

        # Remove samples from batches
        for bid, batch in project.batches.items():
            if set(batch.sample_ids).intersection(sample_ids_set):
                project.batches[bid] = batch.drop_samples(sample_ids_set)

        return project

    def _drop_batches(self, project: Project, batch_ids: Sequence[str]) -> Project:
        for batch_id in batch_ids:
            # Remove batch directory using dataloader
            self._dataloader.remove_data("batch_dir", batch_id=batch_id)

            # Remove from project batches dict
            project.batches.pop(batch_id, None)

        return project

    # ------------- Dimension I/O -----------------

    def load_layer_df(self, layer: str) -> pd.DataFrame:
        """Load dimension definitions for a data layer as a DataFrame.

        Parameters
        ----------
        layer : str
            Name of the data layer.

        Returns
        -------
        pd.DataFrame
            DataFrame with dimension definitions, indexed by dimension ID.

        Raises
        ------
        KeyError
            If the layer does not exist in the project.
        """
        project = self.load_project()
        if layer not in project.layers:
            raise KeyError(f"Data layer '{layer}' not found in project.")
        df = pd.DataFrame.from_records([dim.to_record() for dim in project.layers[layer]])
        return df.set_index("id", drop=False).sort_values("idx")

    # ------------- Sample I/O -----------------

    def sample_adata_path(self, sample: SampleRef | str, layer: str) -> Path:
        """Path to a sample's AnnData file for a given layer."""
        sample_id = sample.id if isinstance(sample, SampleRef) else sample
        pattern = self.path_scheme["sample_adata"]
        return self._dataloader._resolve_read_path(pattern=pattern,  sample_id=sample_id, layer=layer)

    def load_sample_meta(self, sample_id: str) -> SampleRef:
        project = self.load_project()
        if sample_id not in project.samples:
            raise KeyError(f"Sample '{sample_id}' not found in project.")
        return project.samples[sample_id]

    def _load_sample_layer(self, sample_id: str, layer: str, backed: bool | Literal["r", "r+"] = "r") -> AnnData:
        return self._dataloader.load_adata(
            sample_id=sample_id,
            layer=layer,
            backed=backed,
        )

    def load_sample_adata(
        self,
        sample_id: str,
        layer: str,
        mask: MaskLike = slice(None),
        select: Sequence[str] | slice = slice(None),
    ) -> AnnData:
        return self._dataloader.load_adata(
            sample_id=sample_id,
            layer=layer,
            mask=mask,
            select=select,
        )

    def save_sample_adata(self, sample_id: str, layer: str, adata: AnnData, **kwargs) -> None:
        return self._dataloader.save_adata(
            sample_id=sample_id,
            layer=layer,
            adata=adata,
            **kwargs,
        )

    # --------- Compensation I/O ----------

    def get_spill_df(self, comp: CompensationRef | str) -> pd.DataFrame:
        """
        Retrieve a compensation matrix DataFrame.

        Parameters
        ----------
        comp : CompensationRef or str
            CompensationRef instance or comp_id string.

        Returns
        -------
        pandas.DataFrame
            DataFrame representing the spillover/compensation matrix.

        Raises
        ------
        FileNotFoundError
            If the referenced spillover CSV does not exist when comp is a string id.
        """
        if isinstance(comp, CompensationRef):
            return comp.spill

        if isinstance(comp, str):
            project = self.load_project()
            try:
                return project.compensations[comp].spill
            except KeyError as e:
                 raise KeyError(f"Compensation with id '{comp}' not found in project.") from e

        raise ValueError(f"Unsupported comp type: {type(comp)}")

    def add_compensation(
        self,
        spillover: pd.DataFrame,
        name: str | None = None,
        source: str | None = None,
        batch: list[str] = [],
    ) -> str:
        """
        Write a single compensation spillover CSV and update the catalog.

        Parameters
        ----------
        spillover : pd.DataFrame
            Spillover/compensation matrix DataFrame.
        name : str | None
            Optional human-readable name for the compensation.
        source : str | None
            Optional source for the compensation.
        batch : list[str]
            Optional list of batch identifiers for the compensation.

        Raises
        ------
        ValueError
            If the spillover DataFrame is invalid.

        Returns
        -------
        str
            Identifier of the newly created CompensationRef.
        """
        if not isinstance(spillover, pd.DataFrame):
            raise TypeError("Spillover must be a pandas DataFrame.")

        ref = CompensationRef.from_dataframe(df = spillover.copy(), name=name, source=source, batch=batch)
        self.update_project_metadata(compensations=[ref])
        return ref.id

    # ---------- Pipeline I/O ----------

    @property
    def steps_dir(self) -> Path:
        """
        Path to steps directory.

        Returns
        -------
        Path
            Absolute path to the project's 'steps' directory.
        """
        return self.root / self.path_scheme["steps_dir"]

    @property
    def step_counter(self) -> int:
        """
        Number of saved step runs.

        Returns
        -------
        int
            Integer step counter used to generate new step IDs.
        """
        if not self.steps_dir.exists():
            return 0
        return sum(1 for step in self.steps_dir.iterdir() if step.is_dir())

    # ---------- QC I/O ----------

    def qc_entity_dir(self, entity_type: str, entity_id: str) -> Path:
        """Path to a specific entity QC directory."""
        pattern = self.path_scheme["qc_entity_dir"]
        return self.root / pattern.format(entity_type=entity_type, entity_id=entity_id)

    def qc_entity_tables_dir(self, entity_type: str, entity_id: str) -> Path:
        """Path to entity QC tables directory."""
        return self.qc_entity_dir(entity_type, entity_id) / "tables"

    def qc_entity_figures_dir(self, entity_type: str, entity_id: str) -> Path:
        """Path to entity QC figures directory."""
        return self.qc_entity_dir(entity_type, entity_id) / "figures"

    def qc_entity_artifacts_dir(self, entity_type: str, entity_id: str) -> Path:
        """Path to entity QC artifacts directory."""
        pattern = self.path_scheme["qc_entity_artifacts_dir"]
        return self.root / pattern.format(entity_type=entity_type, entity_id=entity_id)

    def qc_entity_artifact_dir(self, entity_type: str, entity_id: str, artifact_key: str) -> Path:
        """Path to a named artifact directory under an entity QC directory."""
        pattern = self.path_scheme["qc_entity_artifact_dir"]
        return self.root / pattern.format(entity_type=entity_type, entity_id=entity_id, artifact_key=artifact_key)

    def load_qc_entity_status(self, entity_type: str, entity_id: str) -> EntityQCStatus:
        """Load and return QC status for a specific entity."""
        return self.load_project_metadata("entity_qc", entity_type=entity_type, entity_id=entity_id)

    def save_qc_entity_status(self, qc_status: EntityQCStatus) -> None:
        self.save_project_metadata("entity_qc", qc_status, entity_type=qc_status.entity_type, entity_id=qc_status.entity_id)

    def save_step_run(self, step: StepRun) -> None:
        """
        Persist a StepRun record to disk.

        Creates a directory for the step id and writes a JSON file.

        Parameters
        ----------
        step : StepRun
            StepRun domain object to persist.
        """
        self.save_project_metadata("step_run", step, step_id=step.id)

    def load_step_run(self, step_run_id: str) -> StepRun:
        """
        Load a StepRun record from disk.

        Parameters
        ----------
        step_run_id : str
            Step run identifier (e.g., "step_0001").

        Returns
        -------
        StepRun
            Loaded StepRun domain object.

        Raises
        ------
        FileNotFoundError
            If the step run directory or JSON file does not exist.
        ValueError
            If multiple or no JSON files found in step directory.
        """
        return self.load_project_metadata("step_run", step_id=step_run_id)

    # ------------- Gating Strategy Catalog I/O -----------------

    def gate_node_path(self, node_id: str) -> Path:
        """Get the path to a gating node definition file."""
        pattern = self.path_scheme["gate_node"]
        return self.root / pattern.format(node_id=node_id)

    def load_gate_node(self, node_id: str) -> GateNode:
        """
        Load a gating node definition by ID from a gating strategy.

        REFACTORED (Phase 3): Now delegates to internal RepoDataLoader.

        Parameters
        ----------
        node_id: str
            Gating node identifier.

        Returns
        -------
        GateNode
            The gating node definition.

        Raises
        ------
        KeyError
            If the node is not found in the strategy.
        """
        project = self.load_project()
        if project.gating_strategy is None:
            raise KeyError("Gating strategy not found in project.")
        return project.gating_strategy.get_node(node_id)

    def save_gate_node(self, node: GateNode, overwrite: bool = True) -> None:
        """
        Save a gating node definition to disk.

        REFACTORED (Phase 3): Now delegates to internal RepoDataLoader.

        Parameters
        ----------
        node: GateNode
            Gating node definition to save.
        overwrite : bool
            If True, overwrite existing node.
        """
        return self._dataloader.save_gate_node(
            node=node,
            serialize_func=lambda n: n.to_dict(),
            overwrite=overwrite,
        )

    def load_gating_strategy(self) -> GatingStrategyRef:
        """
        Load a gating strategy by ID (with lazy-loaded definition).

        Parameters
        ----------
        Returns
        -------
        GatingStrategyRef
            The gating strategy reference with definition path set.

        Raises
        ------
        KeyError
            If the strategy is not found in the project.
        """
        project = self.load_project()
        return project.gating_strategy

    def save_gating_masks(
        self,
        sample: str | SampleRef,
        masks: Mapping[str, Sequence[bool] | NDArray[np.bool_]],
        overwrite: bool = True
    ) -> None:
        """
        Save gating mask for a given strategy and sample.

        Parameters
        ----------
        sample : str | SampleRef
            Sample identifier or reference.
        masks : Mapping[str, Sequence[bool] | NDArray[np.bool_]]
            Gating mask data to save.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample

        return self._dataloader.save_masks(
            sample_id=sample_id,
            masks=masks,
            overwrite=overwrite,
        )

    def load_gating_masks(
        self,
        sample: str | SampleRef,
        mask_ids: Iterable[str]
    ) -> dict[str, BooleanArray]:
        """
        Load gating mask for a given strategy, sample, and mask ID.

        REFACTORED (Phase 3): Now delegates to internal RepoDataLoader.

        Parameters
        ----------
        sample : str | SampleRef
            Sample identifier or reference.
        mask_ids : Iterable[str]
            Mask identifiers.
        Returns
        -------
        dict[str, BooleanArray]
            Decoded gating mask array.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample

        return self._dataloader.load_masks(
            sample_id=sample_id,
            gate_ids=mask_ids,  # Map 'mask_ids' to 'gate_ids' in dataloader
        )

    # ------------- Revision Session I/O -----------------

    def generate_revision_workspace(self, entity_type: str, session_id: str | None) -> Path:
        """
        Generate a new revision workspace directory for a given session.

        Parameters
        ----------
        entity_type : str
            Type of the entity for the revision session.
        session_id : str | None
            Identifier for the revision session.

        Returns
        -------
        Path
            Absolute path to the created revision workspace.
        """

        session_id = session_id or "session_001"
        pattern = self.path_scheme["revision_session"].format(
            entity_type=entity_type,
            session_id=session_id
        )
        path = (self.root / pattern).parent
        if not path.exists():
            return path

        type_dir = self.root / self.path_scheme["revision_type_dir"].format(entity_type=entity_type)
        session_id = "session" if session_id.startswith("session") else session_id
        k = sum(1 for d in type_dir.iterdir() if d.is_dir() and d.name.startswith(session_id))
        session_id = f"{session_id}_{k:03d}"
        path = self.root / self.path_scheme["revision_session"].format(entity_type=entity_type, session_id=session_id)
        return path.parent
