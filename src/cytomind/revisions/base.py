"""
Base revision handler with common functionality.

Provides state persistence, workspace management, and data I/O for iterative refinement.
"""
from __future__ import annotations
from typing import Any, Mapping, TYPE_CHECKING
from abc import ABC, abstractmethod
import inspect

from cytomind.domain.pipeline import RevisionSession
from cytomind.domain.qc import EntityQCStatus
from cytomind.infra.dataloader import UnifiedDataLoader, HandlerDictType
from cytomind.qc import EntityQCEvaluatorRegistry
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike, MaskLike
    from cytomind.infra.repo import ProjectRepository
    from cytomind.domain.pipeline import StepRun
    from pandas import DataFrame
    from anndata import AnnData
else:
    ProjectRepository = object
    StepRun = object
    DataFrame = object
    AnnData = object
    PathLike = object
    MaskLike = object


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
    """

    entity_type: str
    # Overwrite main repo paths with workspace paths
    _default_path_scheme: dict[str, str] = {
        "revision_session": "session.json",
        "viz_cache_dir":    "viz_cache",
        "viz_metadata":     "viz_cache/viz_metadata.json",
        "qc_cache_dir":     "qc_cache",
        "entity_qc":        "qc_cache/{entity_id}.json",
    }
    _default_cache_dirs: tuple[str, ...] = ("viz_cache_dir", "qc_cache_dir")

    # To be specified by the subclass if they need custom paths or metadata handlers, otherwise defaults will be used
    path_scheme: dict[str, str] = {}  # To be defined by subclasses if they need custom paths in the workspace
    metadata_handlers: HandlerDictType = {}
    cache_dirs: tuple[str, ...] = ()  # Names of cache dir required on top of viz_cache_dir and qc_cache_dir
    _default_n_subset: int = 10000
    _default_seed: int = 42


    def __init__(
        self,
        main_repo: ProjectRepository,
        workspace_root: PathLike,
        entity_id: str | None = None,
        n_subset: int | None = None,
        seed: int | None = None,
        context: dict[str, Any] = {},
        qc_evaluator_config: dict[str, Any] = {},
        **kwargs
    ):
        """
        Initialize revision handler with entity context and workspace management.

        Parameters
        ----------
        main_repo : ProjectRepository
            Main repository instance
        workspace : PathLike
            Workspace directory for this revision session
        entity_id : str | None
            Entity identifier for new sessions (e.g., "comp_001", "step_0003")
        """

        # Initialize WorkspaceRepository for workspace-first I/O
        path_scheme = {**self._default_path_scheme, **self.path_scheme}
        self.workspace = UnifiedDataLoader(
            path_scheme=path_scheme,
            root_dir=workspace_root,
            fallback=main_repo._dataloader,
            data_handlers=self.metadata_handlers,
            viz_cache_dir=path_scheme["viz_cache_dir"],
        )
        self.cache_dirs = tuple(set(self._default_cache_dirs) | set(self.cache_dirs))

        # Initialize session
        try:
            self.load_session()
        except FileNotFoundError:
            # No existing session, create a new one
            handler_state = {
                "n_subset": n_subset or self._default_n_subset,
                "seed": seed or self._default_seed,
            }
            handler_state.update(kwargs)  # Include any additional arguments in handler state
            self.session = self.create_session(entity_id=entity_id, context=context, handler_state=handler_state)
            self.save_session()

        # Sanity checks to ensure session and workspace are consistent
        if self.session.id != self.workspace.root_dir.name:
            raise ValueError(f"Session ID '{self.session.id}' does not match workspace name '{self.workspace.root_dir.name}'")

        if self.entity_type != self.session.entity_type:
            raise TypeError(f"Session entity type '{self.session.entity_type}' does not match handler entity type '{self.entity_type}'")
        if entity_id is not None and self.session.entity_id is not None and entity_id != self.session.entity_id:
            raise ValueError(f"Provided entity_id '{entity_id}' does not match session entity_id '{self.session.entity_id}'")

        # Initialize QC evaluator for this entity type
        qc_evaluator_class = EntityQCEvaluatorRegistry.get(self.entity_type)
        if qc_evaluator_class is None:
            raise TypeError(f"No QC evaluator registered for entity type '{self.entity_type}'")
        try:
            self.qc_evaluator = qc_evaluator_class(config=qc_evaluator_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QC evaluator for entity type: {e}")

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
        method = getattr(self, f"figure_{plot_type}", None)
        if method is None:
            raise ValueError(f"Unknown plot type: {plot_type}")
        return method(**input_params)

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
        method = getattr(self, f"table_{table_type}", None)
        if method is None:
            raise ValueError(f"Unknown table type: {table_type}")
        return method(**input_params)

    @classmethod
    def list_figures(cls, show_args=True) -> dict[str, Any]:
        """
        List available figure types for this revision handler.

        Returns
        -------
        dict[str, Any]
            List of figure type descriptors
        """
        figs = cls._collect_supported("figure_")
        if not show_args:
            for fig in figs.values():
                fig.pop("input_params", None)
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
        tabs = cls._collect_supported("table_")
        if not show_args:
            for tab in tabs.values():
                tab.pop("input_params", None)
        return tabs

    @classmethod
    def _collect_supported(cls, prefix: str) -> dict[str, Any]:
        supported: dict[str, Any] = {}
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):]
            summary, description, param_specs = cls._parse_spec(inspect.getdoc(func))
            input_params = cls._signature_params(func, param_specs)
            supported[key] = {
                "summary": summary,
                "description": description,
                "input_params": input_params,
            }
        return supported

    @staticmethod
    def _parse_spec(doc: str | None) -> tuple[str, str, dict[str, tuple[str, str, str]]]:
        if not doc:
            return "", "", {}
        lines = doc.splitlines()
        description = lines[0].strip()
        summary = description
        if "@spec" not in doc:
            return summary, description, {}

        param_specs: dict[str, tuple[str, str, str]] = {}
        in_params = False
        for line in lines:
            stripped = line.strip()
            if stripped == "@spec":
                in_params = False
                continue
            if stripped.startswith("summary:"):
                summary = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("description:"):
                description = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("params:"):
                in_params = True
                continue
            if in_params and ":" in stripped:
                name, value = stripped.split(":", 1)
                param_specs[name.strip()] = BaseRevisionHandler._split_param_spec(value.strip())
        return summary, description, param_specs

    @staticmethod
    def _signature_params(func: Any, param_specs: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str, str]]:
        signature = inspect.signature(func)
        params: dict[str, tuple[str, str, str]] = {}
        for param in signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue

            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # Skip kwargs in input params (used internally for extra args).
                continue

            if param.name in param_specs:
                params[param.name] = param_specs[param.name]
                continue

            annotation = BaseRevisionHandler._format_annotation(param.annotation)
            default = param.default
            default_str = "None"
            if default is not inspect._empty:
                if isinstance(default, str):
                    default_str = f'"{default}"'
                else:
                    default_str = repr(default)

            if annotation and default_str != "None":
                params[param.name] = (annotation, default_str, "")
            elif annotation:
                params[param.name] = (annotation, "None", "required")
            else:
                params[param.name] = ("", default_str, "")
        return params

    @staticmethod
    def _split_param_spec(value: str) -> tuple[str, str, str]:
        """Parse 'type | default | description' format into 3-tuple."""
        parts = value.split("|")
        if len(parts) >= 3:
            return parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            # Fallback: assume 'type | description' with no explicit default
            return parts[0].strip(), "None", parts[1].strip()
        else:
            return parts[0].strip(), "None", ""

    @staticmethod
    def _format_annotation(annotation: Any) -> str:
        if annotation is inspect._empty:
            return ""
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        text = str(annotation)
        return text.replace("typing.", "")

    # ---- Convenient methods/properties and aliases ----

    @property
    def state(self) -> dict[str, Any]:
        return self.session.handler_state

    def load_entity_qc_cache(self, entity_id: str) -> EntityQCStatus:
        """Load cached QC results for a specific entity. Returns default status if not found."""
        try:
            return self.workspace.load_data("entity_qc", entity_id=entity_id)
        except FileNotFoundError:
            return EntityQCStatus(entity_id=entity_id, entity_type=self.entity_type, generated_at=now_iso())

    def save_entity_qc_cache(self, entity_id: str, qc_status: EntityQCStatus) -> None:
        """Save QC results for a specific entity to cache."""
        self.workspace.save_data("entity_qc", qc_status, entity_id=entity_id)

    def get_or_create_viz_subset(
        self,
        sample_id: str,
        layer: str,
        mask_id: str = "root",
        mask: MaskLike = slice(None),
        select: list[str] | slice = slice(None),
        n_subset: int | None = None,
        seed: int | None = None,
    ) -> AnnData:
        n_subset = n_subset or int(self.state["n_subset"])
        seed = seed or int(self.state["seed"])
        return self.workspace.get_or_create_viz_subset(
            sample_id=sample_id,
            layer=layer,
            mask_id=mask_id,
            mask=mask,
            select=select,
            n_subset=n_subset,
            seed=seed,
        )

    # ---- Optional lifecycle methods ----

    def create_session(
        self,
        entity_id: str | None = None,
        context: dict[str, Any] = {},
        handler_state: dict[str, Any] = {}
    ) -> RevisionSession:
        now = now_iso()
        return RevisionSession(
            id=self.workspace.root_dir.name,
            entity_type=self.entity_type,
            entity_id=entity_id,
            status="initialized",
            handler_state=handler_state,
            context=context,
            revision_history=[],
            created_at=now,
            updated_at=now,
        )

    def load_session(self) -> None:
        self.session: RevisionSession = self.workspace.load_data("revision_session")

    def save_session(self) -> None:
        """Persist current session state to disk."""
        self.session.updated_at = now_iso()
        self.workspace.save_data("revision_session", self.session)

    def update_state(self, **groups: Mapping[str, dict[str, Any]]) -> None:
        """Helper to update session state and persist."""
        for group_name, updates in groups.items():
            self.state.setdefault(group_name, {}).update(updates)
        self.save_session()

    def commit(self) -> tuple[dict[str, Any], StepRun | None]:
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
        metadata, step_run = self._commit()
        self.session.status = "committed"
        self.session.updated_at = now_iso()
        self.save_session()
        self.cleanup_workspace()
        return metadata, step_run

    def cleanup_workspace(self) -> None:
        """Clean up workspace caches but keep session info for debugging."""
        for cache_dir in self.cache_dirs:
            self.workspace.remove_data(cache_dir)
