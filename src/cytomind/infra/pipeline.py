from __future__ import annotations
from datetime import datetime
from typing import Iterable, Sequence, Mapping, Any, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd

from cytomind.domain.pipeline import RevisionSession, StepRun, BatchRef, SampleRef, Project
from cytomind.domain.flow import DimensionDef
from cytomind.domain.qc import EntityQCStatus, QCFlag
from .repo import ProjectRepository
from cytomind.gates import GateRegistry
from cytomind.steps import StepRegistry
from cytomind.qc import EntityQCEvaluatorRegistry
from cytomind.revisions import RevisionHandlerRegistry
from cytomind.utils import now_iso

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.revisions.base import BaseRevisionHandler
else:
    PathLike = object
    BaseRevisionHandler = object


class InteractivePipeline:
    def __init__(self, project_root: PathLike, name: str | None = None):
        """
        Initialize an InteractivePipeline with step, QC, and revision components.

        Parameters
        ----------
        project_root : PathLike
            Path to the project root directory.
        name : str | None
            Optional project name.
        """
        self.repo = ProjectRepository(project_root, name=name)

    @property
    def project(self) -> Project:
        """Load the current project state from the repository."""
        return self.repo.load_project()

    # ---- Step lifecycle ----

    def run_step(self, step_type: str, config: dict, inputs: dict, id: str | None = None) -> StepRun:
        """
        Thin wrapper to run a step by type with config and inputs, handling the full lifecycle.
        It adds step_id and created_at metadata, then calls the `_execute_step` method to perform the computation and persist results.

        Parameters
        ----------
        step_type : str
            The registered step type name.
        config : dict
            Configuration dictionary passed to the step implementation.
        inputs : dict
            Inputs dictionary passed to the step implementation.

        id: str | None
            Optional step identifier. If None, a new sequential ID will be generated.

        Returns
        -------
        StepRun
            Completed step run with outputs and detailed QC.

        Raises
        ------
        ValueError
            If the given step_type is not registered.
        """
        # 1. Create step run
        step_run = StepRun(
            id=id or self._next_step_id(),
            step_type=step_type,
            config=config,
            inputs=inputs,
            created_at=now_iso(),
        )

        return self._execute_step(step_run)

    def _execute_step(self, step_run: StepRun) -> StepRun:
        """Execute step computation handling the full lifecycle, including validation, execution, and persistence."""
        if step_run.status != "pending":
            raise ValueError(
                f"Cannot run step {step_run.id!r} with status {step_run.status!r}, must be 'pending'."
            )

        step_class = StepRegistry.get(step_run.step_type)
        if not step_class:
            known_steps = ", ".join(repr(k) for k in StepRegistry.keys())
            raise ValueError(f"Unknown step type {step_run.step_type!r}, known steps are: {known_steps}")

        step_run = step_class(repo=self.repo).run(step_run)
        self.repo.save_step_run(step_run)
        return step_run

    def _next_step_id(self) -> str:
        """Generate the next sequential step identifier."""
        n_steps = self.repo.step_counter + 1
        return f"step_{n_steps:04d}"

    # ---- Review & Revision ----

    def start_revision(
        self,
        entity_type: str,
        entity_id: str,
        input_spec: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> BaseRevisionHandler:
        """
        Initialize revision workspace for entity refinement.

        Parameters
        ----------
        entity_type : str
            Type of entity to revise (e.g., "compensation", "gating_strategy")
        entity_id : str
            Entity identifier (e.g., "comp_001")
        input_spec : Mapping[str, Any] | None
            User input specification for the revision handler
        session_id : str | None
            Optional session identifier for the revision workspace

        Returns
        -------
        BaseRevisionHandler
            Initialized handler for managing the revision session

        Raises
        ------
        ValueError
            If no revision handler is registered for the entity type
        """
        # Get handler from registry by entity_type
        revision_handler_class = RevisionHandlerRegistry.get(entity_type)
        if not revision_handler_class:
            raise ValueError(f"No revision handler registered for entity type '{entity_type}'")

        # Generate workspace directory
        workspace = self.repo.generate_revision_workspace(
            entity_type=entity_type,
            entity_id=entity_id,
            session_id=session_id
        )

        # Instantiate handler (it will set up workspace and session)
        handler = revision_handler_class(
            main_repo=self.repo,
            workspace_root=workspace,
            entity_id=entity_id,
        )

        # Initialize session
        handler.start_revision(input_spec or {})
        return handler

    def load_revision_handler(self, entity_type: str, entity_id: str, session_id: str | None = None) -> BaseRevisionHandler:
        """
        Load an existing revision handler from the repository.

        Parameters
        ----------
        entity_type : str
            Type of entity being revised
        entity_id : str
            Identifier of the specific entity being revised
        session_id : str | None
            Session identifier. If None the latest session for the entity will be loaded.

        Returns
        -------
        BaseRevisionHandler
            Loaded revision handler

        Raises
        ------
        FileNotFoundError
            If session file does not exist
        """
        handler_class = RevisionHandlerRegistry.get(entity_type)
        if not handler_class:
            raise ValueError(f"No revision handler registered for entity type '{entity_type}'")

        if session_id is None:
            # Retrieve all sessions for the entity and find the latest non-committed one
            entity_revision_dir = self.repo.get_path(
                "revision_entity_dir",
                entity_type=entity_type,
                entity_id=entity_id,
            )
            live_sessions = []
            for session_dir in entity_revision_dir.iterdir():
                try:
                    session: RevisionSession = self.repo.load_project_metadata(
                        "revision_session",
                        entity_type=entity_type,
                        entity_id=entity_id,
                        session_id=session_dir.name
                    )
                except FileNotFoundError:
                    continue
                if session.status in {"commited", "canceled"}:
                    continue
                live_sessions.append((datetime.fromisoformat(session.updated_at), session.id))

            if not live_sessions:
                raise FileNotFoundError(
                    f"No active revision sessions found for entity '{entity_type}::{entity_id}'."
                )
            _, session_id = max(live_sessions)

        try:
            workspace_path: Path = self.repo.load_project_metadata(
                "revision_workspace",
                entity_type=entity_type,
                entity_id=entity_id,
                session_id=session_id
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "No revision session found for entity_type "
                f"'{entity_type}::{entity_id}' with session_id '{session_id}'."
            )

        # Create handler
        return handler_class(
            main_repo=self.repo,
            workspace_root=workspace_path,
        )

    def commit_revision(self, handler: BaseRevisionHandler) -> StepRun | None:
        """
        Finalize revision and optionally execute new step.

        Parameters
        ----------
        handler : BaseRevisionHandler
            The revision handler managing the session

        Returns
        -------
        StepRun | None
            New step run if handler returned a new step to execute, else None
        """
        # Get metadata updates and optional new step
        metadata_updates, new_step = handler.commit()

        # Apply metadata to main project
        if metadata_updates:
            self.repo.update_project_metadata(**metadata_updates)

        # If handler produced a new step, run it
        if new_step:
            new_step = self._execute_step(new_step)

        return new_step

    # ---- QC Analysis ----

    def get_entity_qc_table(
        self,
        *,
        entity_qc: EntityQCStatus | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        filter_by: Mapping[str, Any] | None = None,
        sample_ids: Sequence[str] | None = None,
        with_thresholds: bool = True,
        with_metadata: bool | None = None,
    ) -> pd.DataFrame:
        """
        Generate a QC table for an entity from its cached QC status.

        Parameters
        ----------
        entity_qc : EntityQCStatus | None
            Cached QC status for the entity. If None, entity_type and entity_id must be provided.
        entity_type : str | None
            Type of entity (e.g., "compensation", "gating_strategy"). Ignored if entity_qc is provided.
        entity_id : str | None
            Entity identifier (e.g., "comp_001"). Ignored if entity_qc is provided.
        filter_by : Mapping[str, Any] | None
            Optional dictionary to filter tests by their id fields (e.g., {"test_type": "compensation_sample_channel"})
            Values can be either single values or iterables of values for matching multiple options.
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to include. If None, includes all samples.
        with_thresholds : bool
            Whether to include threshold values in the output (default: True).
        with_metadata : bool | None
            Whether to include metadata fields in the output
            (default: None, which means True if filtering by a single test, False if multiple tests).

        Returns
        -------
        DataFrame
            Table with entity-specific columns matching the QC output format

        Raises
        ------
        ValueError
            If no QC evaluator is registered for the entity type
        FileNotFoundError
            If QC status file not found for the entity
        """

        if entity_qc is None:
            if entity_type is None or entity_id is None:
                raise ValueError("Either entity_qc or both entity_type and entity_id must be provided.")
            entity_qc = self.repo.load_qc_entity_status(entity_type, entity_id)

        filter_by = filter_by or {}
        if with_metadata is None:
            # default = True for a single test (otherwise they likely vary)
            test_name = filter_by.get("test_name", [])
            with_metadata = not isinstance(test_name, (list, set, tuple)) or len(test_name) == 1

        return pd.DataFrame.from_records(
            entity_qc.iter_test_records(
                filter_by=filter_by,
                sample_ids=sample_ids,
                with_thresholds=with_thresholds,
                with_metadata=with_metadata,
            )
        )

    def collect_qc_metrics(
        self,
        entity_type: str,
        entity_ids: Sequence[str] | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Aggregate QC metrics across samples by generating tables for each sample's active entity.

        - "compensation": Loops over each sample's active compensation reference, generates the specified QC table for each, and concatenates results.
        - other entity types: Loops over specified entities (or all if entity_ids is None),
        """

        # Compensation entities are sample specific so we dispatch to a dedicated method
        if entity_type == "compensation":
            return self._collect_compensation_qc_metrics(entity_ids=entity_ids, **kwargs)

        # Discover entity ids if not provided
        if entity_ids is None:
            iter_qc_entities = self.repo.iter_qc_entities(entity_type)
        else:
            iter_qc_entities = ((ent_id, self.repo.load_qc_entity_status(entity_type, ent_id)) for ent_id in entity_ids)

        records: list[dict[str, Any]] = []
        for ent_id, entity_qc in iter_qc_entities:
            for rec in entity_qc.iter_test_records(**kwargs):
                records.append(rec)

        if not records:
            return pd.DataFrame()

        return pd.DataFrame.from_records(records)

    def _collect_compensation_qc_metrics(
        self,
        entity_ids: Sequence[str] | None = None,
        sample_ids: Sequence[str] | None = None,
        **kwargs
    ) -> pd.DataFrame:

        sample_refs = self.project.samples
        sample_ids = list(sample_ids) if sample_ids is not None else list(sample_refs.keys())

        comp_map: dict[str, list[str]] = {}
        for sid in sample_ids:
            sref = sample_refs[sid]
            if sref.compensation is not None:
                comp_map.setdefault(sref.compensation, []).append(sid)

        comp_set = list(entity_ids) if entity_ids is not None else list(comp_map.keys())
        comp_dfs = []
        for comp_id in comp_set:
            df = self.get_entity_qc_table(
                entity_type="compensation",
                entity_id=comp_id,
                sample_ids=comp_map.get(comp_id),
                **kwargs
            )
            comp_dfs.append(df)
        return pd.concat(comp_dfs, ignore_index=True) if comp_dfs else pd.DataFrame()

    def aggregate_qc_metrics(
        self,
        entity_type: str,
        entity_ids: Sequence[str] | None = None,
        sample_ids: Sequence[str] | None = None,
        filter_by: Mapping[str, Any] | None = None,
        group_by: str | Sequence[str] = "target",
    ) -> pd.DataFrame:
        """
        Aggregate QC metrics across samples by collecting metrics and combining status flags.

        This method calls collect_qc_metrics internally and then aggregates the result by
        the specified grouping dimension(s). The `group_by` parameter determines which columns
        to group by - it can be a predefined shortcut or any subset of target columns.
        For each group, it combines the status values using QCFlag logic
        (FAIL > SEVERE > WARN > PASS) and counts how many tests have each flag value.

        Possible predefined shortcuts for `group_by`:

        - "test": groups by test_type and test_name
        - "target": groups by test targets (eg compensation_id, sample_id, mask). If the tests are not groupped by type it will probably create too granular groups, so "test" is usually a better option.
        - "sample": groups by sample_id
        - "entity": groups by the entity identifier for each test (specified as the 1st entry of the test target, e.g., compensation_id for compensation_sample_channel tests)
        - other groupings have to be specified by passing a list of column names from the QC metrics table (e.g., ["compensation_id", "sample_id"] to group by both compensation and sample)
        """
        df = self.collect_qc_metrics(entity_type=entity_type, entity_ids=entity_ids, sample_ids=sample_ids, filter_by=filter_by)
        if df is None or df.empty:
            return pd.DataFrame()

        # That's very hacky at least I need to check that filter_by test is applied.
        target_cols: list[str] = []
        for c in df.columns:
            if c in {"test_type", "test_name"}:
                break
            target_cols.append(c)

        # Resolve group_by shortcuts
        group_cols: list[str] = []
        if isinstance(group_by, str):
            if group_by == "test":
                group_cols = [c for c in ["test_type", "test_name"] if c in df.columns]
            elif group_by == "sample":
                group_cols = ["sample_id"] if "sample_id" in df.columns else []
            elif group_by == "entity":
                # prefer explicit entity id column
                et_col = f"{entity_type}_id"
                if et_col in df.columns:
                    group_cols = [et_col]
                elif "entity_id" in df.columns:
                    group_cols = ["entity_id"]
                elif target_cols:
                    group_cols = [target_cols[0]]
                else:
                    raise ValueError("No suitable entity identifier column found for 'entity' grouping.")
            elif group_by == "target":
                if not target_cols:
                    raise ValueError("No target columns found for 'target' grouping.")
                group_cols = target_cols
        else:
            missing = [c for c in group_by if c not in df.columns]
            if missing:
                raise ValueError(f"Invalid group_by columns: {missing} not found in QC metrics.")

        if not group_cols:
            raise ValueError("No valid grouping columns found for the provided `group_by` parameter.")

        statuses = ["PASS", "WARN", "SEVERE", "FAIL", "SKIP"]

        def combine_flags(series: pd.Series) -> str:
            vals = [s for s in series.dropna().unique()]
            flags = [QCFlag.from_str(v) for v in vals if isinstance(v, str) and v.upper() in QCFlag.__members__]
            if not flags:
                return QCFlag.PASS.value
            return QCFlag.combine(flags).value

        grp = df.groupby(group_cols, dropna=False)

        agg_rows = []
        for key, sub in grp:
            row = {}
            if isinstance(key, tuple):
                for kcol, kval in zip(group_cols, key):
                    row[kcol] = kval
            else:
                row[group_cols[0]] = key

            row["n_tests"] = len(sub)
            for s in statuses:
                row[f"n_{s.lower()}"] = int((sub.get("status") == s).sum())

            row["combined_flag"] = combine_flags(sub.get("status", pd.Series(dtype=object)))
            # percentages
            for s in statuses:
                row[f"pct_{s.lower()}"] = (row.get(f"n_{s.lower()}", 0) / row["n_tests"]) if row["n_tests"] else 0.0

            agg_rows.append(row)

        res = pd.DataFrame.from_records(agg_rows)
        # ensure group columns first
        res = res[[c for c in group_cols if c in res.columns] + [c for c in res.columns if c not in group_cols]]
        return res

    def get_entity_qc_figure(
        self,
        entity_type: str,
        entity_id: str,
        test_key: Any,
        sample_ids: Sequence[str] | None = None,
        step_id: str | None = None,
        **kwargs
    ) -> Any:
        """
        Generate a QC diagnostic figure for an entity from its cached QC status.

        Parameters
        ----------
        entity_type : str
            Type of entity (e.g., "compensation", "gating_strategy")
        entity_id : str
            Entity identifier (e.g., "comp_001")
        test_key : Any
            Unique test identifier (from QCStepStatus.tests)
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to include. If None, includes all samples.
        step_id : str | None
            Optional step ID for contextual information

        Returns
        -------
        Figure
            Diagnostic figure for the specified test

        Raises
        ------
        ValueError
            If no QC evaluator is registered for the entity type
        FileNotFoundError
            If QC status file not found for the entity
        """
        # Get evaluator
        qc_evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
        if qc_evaluator_class is None:
            raise ValueError(f"No QC evaluator registered for entity type '{entity_type}'")

        qc_evaluator = qc_evaluator_class()

        # Load QC status from repository
        qc_status = self.repo.load_qc_entity_status(entity_type, entity_id)

        # Build context for dataloader
        dataloader_context = {}
        if sample_ids:
            # For figures, we typically need a specific sample_id
            # If multiple sample_ids provided, use the first one
            dataloader_context["sample_id"] = sample_ids[0] if isinstance(sample_ids, (list, tuple)) else next(iter(sample_ids))

        # Generate figure
        return qc_evaluator.generate_figure(
            qc_status,
            test_key,
            dataloader=self.repo._dataloader,
            dataloader_context=dataloader_context if dataloader_context else None,
            step_id=step_id,
            **kwargs
        )

    # --- Convenience Methods for Common Operations ----

    def add_samples(
        self,
        samples: Mapping[str, PathLike],
        config: Mapping[str, Any] = {},
        channel_mapping: Mapping[str, Mapping] | None = None
    ) -> StepRun:
        """
        Initialize a new project by parsing FCS files and building registries.

        Parameters
        ----------
        samples : Mapping[str, PathLike]
            Mapping from sample_id to FCS file path.
        config : dict[str, Any]
            Additional configuration for the add_samples step.
        channel_mapping : dict[str, dict] | None
            Optional channel mapping data (from parse_channel_json_filtered).
            Maps FCS filenames to drop/rename information.

        Returns
        -------
        StepRun
            The completed add_samples step run.
        """

        # The step will create SampleRef instances internally, but we need placeholders for BaseStep.run
        # Temporarily register minimal samples in project so run can iterate
        sample_refs = [SampleRef(id=sid, fcs=Path(fcs).as_posix()) for sid, fcs in samples.items()]
        batch_ref = BatchRef(
            id="__all__",
            sample_ids={sref.id for sref in sample_refs},
            tags={"all_samples"},
            meta={}
        )
        self.repo.update_project_metadata(samples=sample_refs, batches=[batch_ref])

        # Add channel_mapping to config if provided
        step_config = dict(config)
        if channel_mapping is not None:
            step_config["channel_mapping"] = channel_mapping

        return self.run_step(
            step_type="add_samples",
            config=step_config,
            inputs={"batch_ids": ["__all__"]},
        )

    def load_fcs(self, sample_ids: Sequence[str] | None = None) -> StepRun:
        """
        Load FCS files for the specified samples.

        Parameters
        ----------
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to load. If None, loads all samples.

        Returns
        -------
        StepRun
            The completed load_fcs step run.
        """
        if sample_ids is None:
            sample_ids = list(self.repo.load_project().samples.keys())
        return self.run_step(
            step_type="load_fcs",
            config={},
            inputs={"sample_ids": list(sample_ids)},
        )

    def compensate_samples(
        self,
        comp_id: str | Mapping[str, str],
        sample_ids: Sequence[str] | None = None,
        store_raw: bool = False
    ) -> StepRun:
        """
        Compensates the specified samples using the given compensation ID(s).

        Parameters
        ----------
        comp_id : str | Mapping[str, str]
            The compensation ID or a mapping from sample ID to compensation ID.
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to compensate. If None, compensates all samples.
        store_raw : bool
            Whether to store raw data (default: False).

        Returns
        -------
        StepRun
            The completed compensate step run.
        """
        if sample_ids is None:
            if isinstance(comp_id, Mapping):
                sample_ids = list(comp_id.keys())
            else:
                raise ValueError("sample_ids must be provided when comp_id is a single string.")
        sample_ids = list(sample_ids)
        step_comp = self.run_step(
            step_type="compensate",
            config={"comp_id": comp_id, "store_raw": store_raw},
            inputs={"sample_ids": sample_ids}
        )
        return step_comp

    def add_gate(
        self,
        gate_id: str,
        gate_type: str,
        dimensions: Sequence[str] = [],
        parent_ids: str | Iterable[str] = [],
        layer: str = "xf",
        name: str | None = None,
        use_as_complement: bool = False,
        fit_on_batch: bool = False,
        custom_fit: Iterable[str] = [],
        **gate_params
    ) -> StepRun:
        """
        Add a gate to a gating strategy and compute masks for samples.

        Parameters
        ----------
        gate_id : str
            Unique identifier for the new gate.
        gate_type : str
            Gate class name (e.g., "RectangleGate", "PolygonGate").
        dimensions : Sequence[str]
            Dimension/channel IDs the gate operates on.
        parent_id : str
            Parent gate ID or "root" for ungated (default: "root").
            Can be a single ID or comma-separated list for multiple parents.
        layer : str
            Data layer to use (default: "xf").
        name : str | None
            Human-readable name (default: gate_id).
        use_as_complement : bool
            Whether to use gate complement (default: False).
        fit_on_batch : bool
            If True, fit gate on pooled batch data (default: False).
        save_masks : bool
            Whether to save masks to sample .obs (default: True).
        **gate_params
            Gate-specific parameters (e.g., min_vals, max_vals for RectangleGate).

        Returns
        -------
        StepRun
            The completed add_gate step run.
        """

        # Convert parent_id to list
        if isinstance(parent_ids, str):
            parent_ids = [parent_ids]
        else:
            parent_ids = list(parent_ids)

        if gate_type == "Boolean":
            dim_set: set[str] = set()
            for parent in parent_ids:
                parent_node = self.repo.load_gate_node(node_id=parent)
                dim_set.update(parent_node.dimensions)
            dimensions = sorted(dim_set)

        gate_node = {
            "id": gate_id,
            "gate_type": gate_type,
            "dimensions": list(dimensions),
            "parent_ids": parent_ids,
            "layer": layer,
            "name": name or gate_id,
            "use_as_complement": use_as_complement,
            "hyperparams": gate_params,
        }

        return self.run_step(
            step_type="add_gate",
            config={
                "gate_node": gate_node,
                "fit_on_batch": fit_on_batch,
                "custom_fit": list(custom_fit)
            },
            inputs={
                "batch_ids": ['__all__']
            },
        )

    def plot_gate(
        self,
        gate_id: str,
        sample_id: str,
        plot_all_events: bool = False,
        layer: str | None = None,
        select: Sequence[str] | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot a gate for a specific sample context.

        Parameters
        ----------
        gate_id : str
            Gate node identifier within the strategy.
        sample_id : str
            Sample identifier used for per-sample gate parameters and masks.
        plot_all_events : bool
            If True, load all sample events. If False, parent masks are loaded,
            collapsed with OR across parents, and used to filter loaded events.
        layer : str | None
            Optional data layer override. If None, uses the gate node layer.
        select : Sequence[str] | None
            Optional dimensions to load for plotting. If None, loads gate
            dimensions; for gates without dimensions (e.g., Boolean), loads all.
        **plot_kwargs : Any
            Additional keyword arguments passed directly to ``Gate.plot``.

        Returns
        -------
        Any
            Figure returned by the underlying gate implementation.

        Raises
        ------
        ValueError
            If the gate type is unknown or required gate dimensions are missing
            from ``select``.
        """
        gate_node = self.repo.load_gate_node(node_id=gate_id)

        try:
            gate_class = GateRegistry.get(gate_node.gate_type)
        except KeyError as e:
            available = sorted(GateRegistry.list_gates().keys())
            raise ValueError(
                f"Unknown gate type {gate_node.gate_type!r} for gate {gate_id!r}. "
                f"Known gate types are: {available}"
            ) from e

        gate = gate_class.from_node(gate_node, sample_id=sample_id)

        if select is None:
            selected_dims: Sequence[str] | slice = list(gate.dimensions) if gate.dimensions else slice(None)
        else:
            selected_dims = list(select)
            if gate.dimensions:
                missing = sorted(set(gate.dimensions) - set(selected_dims))
                if missing:
                    raise ValueError(
                        f"select is missing required gate dimensions for {gate_id!r}: {missing}"
                    )

        parent_ids = [parent_id for parent_id in gate_node.parent_ids if parent_id != "root"]
        parent_masks: dict[str, Any] = {}

        if plot_all_events:
            event_mask = slice(None)
        elif parent_ids:
            parent_masks = self.repo.load_gating_masks(
                sample=sample_id,
                mask_ids=parent_ids,
            )
            parent_arrays = [np.asarray(mask, dtype=bool) for mask in parent_masks.values()]
            event_mask = np.logical_or.reduce(parent_arrays)
        else:
            event_mask = slice(None)

        plot_layer = layer or gate_node.layer
        events = self.repo.load_sample_adata(
            sample_id=sample_id,
            layer=plot_layer,
            mask=event_mask,
            select=selected_dims,
        )

        if parent_masks:
            if isinstance(event_mask, slice):
                plot_masks = parent_masks
            else:
                plot_masks = {
                    key: np.asarray(mask, dtype=bool)[event_mask]
                    for key, mask in parent_masks.items()
                }
        elif gate.glm_type == "BooleanGate":
            parent_masks = self.repo.load_gating_masks(
                sample=sample_id,
                mask_ids=parent_ids,
            )
            if isinstance(event_mask, slice):
                plot_masks = parent_masks
            else:
                plot_masks = {
                    key: np.asarray(mask, dtype=bool)[event_mask]
                    for key, mask in parent_masks.items()
                }
        else:
            plot_masks = {"root": np.ones(events.n_obs, dtype=bool)}

        return gate.plot(events=events, mask=plot_masks, **plot_kwargs)

    def add_layer(
        self,
        layer: str,
        dimensions: Iterable[DimensionDef] | None = None,
        batch_id: str = "panel",
        default: bool = False,
    ) -> StepRun:
        """
        Adds a new data layer to the specified samples.

        Parameters
        ----------
        layer : str
            The name of the new data layer to add.
        dimensions : Iterable[DimensionDef] | None
            A list of dimension definitions to create the new layer using DimensionDef constructor.
            If None, the layer will be created empty.
        batch_id: str
            Batch ID to which the new layer will be applied.
        default : bool
            If True, sets the new layer as the default data layer for the samples. Defaults to False.

        Returns
        -------
        StepRun
            The completed add_layer step run.
        """

        project = self.repo.load_project()
        if layer in project.layers and dimensions is not None:
            raise ValueError(f"Data layer {layer!r} already exists use add_dimensions instead.")

        dimensions_dicts = [dim.to_dict() for dim in dimensions] if dimensions is not None else []

        return self.run_step(
            step_type="add_layer",
            config={"layer": layer, "dimensions": dimensions_dicts, "default": default},
            inputs={"batch_ids": [batch_id]},
        )

    def add_dimensions(
        self,
        layer: str,
        dimensions: Iterable[DimensionDef],
        batch_id: str = "panel",
    ) -> StepRun:
        """
        Adds dimensions to an existing data layer.

        Parameters
        ----------
        layer : str
            The data layer to which dimensions will be added.
        dimensions : Iterable[DimensionDef]
            A list of dimension definitions to add, created via DimensionDef constructor.
        batch_id: str
            Batch ID to which the new dimensions will be applied.

        Returns
        -------
        StepRun
            The completed add_dimensions step run.
        """
        dimensions_dicts = [dim.to_dict() for dim in dimensions]

        return self.run_step(
            step_type="add_dimensions",
            config={"layer": layer, "dimensions": dimensions_dicts},
            inputs={"batch_ids": [batch_id]},
        )

    def add_transformation(
        self,
        transform_id: str,
        transform_type: str,
        params: Mapping[str, Any] | None = None,
        tags: Iterable[str] = (),
    ) -> None:
        """
        Register a custom transformation in the project.

        Parameters
        ----------
        transform_id : str
            Unique identifier for the transformation.
        transform_type : str
            Human-readable type (e.g., "logicle", "asinh", "custom_transform").
        params : Mapping[str, Any] | None
            Parameters required by the transformation (e.g., {"param_t": 262144.0}).
            If None, defaults to an empty dict.
        tags : Iterable[str]
            Optional semantic tags for downstream consumers (e.g., "boundary", "score").

        Returns
        -------
        None

        Examples
        --------
        Register a custom transformation with parameters:

        >>> pipeline = InteractivePipeline("/path/to/project")
        >>> pipeline.add_transformation(
        ...     transform_id="my_logicle",
        ...     transform_type="logicle",
        ...     params={"param_t": 262144.0, "param_m": 4.5, "param_a": 0.0, "param_w": 0.5}
        ... )

        Then use it in dimension definitions:

        >>> dims = [
        ...     DimensionDef(
        ...         id="CD3",
        ...         source_dims=["CD3"],
        ...         marker="CD3",
        ...         type="fluorescence",
        ...         source_layer="comp",
        ...         transform_id="my_logicle"
        ...     )
        ... ]
        >>> pipeline.add_layer("xf", dimensions=dims)
        """
        from cytomind.domain.transforms import TransformDef, SCALE_TRANSFORM_TYPES

        transform_ref = TransformDef(
            id=transform_id,
            type=transform_type,
            params=dict(params or {}),
            tags=tuple(set(tags) | ({"scale"} if transform_type in SCALE_TRANSFORM_TYPES else set())),
        )
        self.repo.update_project_metadata(transforms=[transform_ref])

    def add_batch(
        self,
        batch_id: str,
        sample_ids: Iterable[str],
        tags: Iterable[str] | None = None,
        **meta,
    ) -> BatchRef:
        """
        Group existing samples into a batch for subsequent analysis.

        Parameters
        ----------
        batch_id : str
            Optional batch identifier. If None, auto-generated.
        tags : Iterable[str] | None
            Optional tags to label the batch.
        **meta
            Additional metadata to store with the batch.

        Returns
        -------
        BatchRef
            The created batch reference.
        """

        project = self.repo.load_project()
        if not batch_id or batch_id in project.batches:
            raise ValueError(f"Batch ID {batch_id!r} is invalid or already exists.")

        sample_list = list(sample_ids)
        sample_set = set(sample_list)
        if len(sample_list) != len(sample_set):
            raise ValueError("Duplicate sample IDs in batch.")
        if len(sample_set) < 2:
            raise ValueError("At least two samples are required to create a batch.")

        batch = BatchRef(
            id=batch_id,
            sample_ids=sample_set,
            tags=set(tags or []),
            meta=dict(meta),
        )

        # Add batch to project and persist
        self.repo.update_project_metadata(batches=[batch])
        return batch

    # TODO: move this to Project?
    def get_steps_history(self) -> list[dict[str, str]] :
        summary: list[dict[str, str]] = []
        for step_dir in self.repo.steps_dir.iterdir():
            try:
                step_run = self.repo.load_step_run(step_run_id=step_dir.name)
            except (FileNotFoundError, ValueError):
                continue
            try:
                flag = step_run.qc.overall_flag.value
            except:
                flag = "None"
            summary.append({
                "created_at": step_run.created_at,
                "type": step_run.step_type,
                "id": step_run.id,
                "status": step_run.status,
                "flag": flag
            })
        return summary

