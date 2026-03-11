from __future__ import annotations
from typing import Iterable, Sequence, Mapping, Any, TYPE_CHECKING
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cytomind.domain.pipeline import StepRun, BatchRef, SampleRef, Project
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

    # ---- Step lifecycle ----

    def run_step(self, step_type: str, config: dict, inputs: dict) -> StepRun:
        """
        Execute complete step lifecycle: compute → evaluate QC → persist.

        Parameters
        ----------
        step_type : str
            The registered step type name.
        config : dict
            Configuration dictionary passed to the step implementation.
        inputs : dict
            Inputs dictionary passed to the step implementation.

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
            id=self._next_step_id(),
            step_type=step_type,
            config=config,
            inputs=inputs,
            created_at=now_iso(),
        )

        step_run = self._execute_step(step_run)
        return step_run

    def _execute_step(self, step_run: StepRun) -> StepRun:
        """Execute step computation (computation phase only)."""
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

    # ---- Review & Revision ----

    def start_revision(
        self,
        entity_type: str,
        session_id: str | None = None,
        entity_id: str | None = None,
        input_spec: Mapping[str, Any] = {},
    ) -> BaseRevisionHandler:
        """
        Initialize revision workspace for entity refinement.

        Parameters
        ----------
        entity_type : str
            Type of entity to revise (e.g., "compensation", "gating_strategy")
        session_id : str | None
            Optional specific session identifier to load or create (e.g., "comp_001", "step_0003").
        entity_id : str | None
            Entity identifier (e.g., "comp_001")
        input_spec : Mapping[str, Any]
            User input specification for the revision handler

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
        workspace = self.repo.generate_revision_workspace(entity_type=entity_type, session_id=entity_id)

        # Instantiate handler (it will set up workspace and session)
        handler = revision_handler_class(
            main_repo=self.repo,
            workspace_root=workspace,
            entity_id=entity_id,
        )

        # Initialize session
        session = handler.start_revision(input_spec)

        return handler

    def load_revision_handler(self, entity_type: str, session_id: str) -> BaseRevisionHandler:
        """
        Load an existing revision handler from the repository.

        Parameters
        ----------
        entity_type : str
            Type of entity being revised
        session_id : str
            Session identifier

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

        # Calculate workspace path
        try:
            workspace_path: Path = self.repo.load_project_metadata(
                "revision_workspace",
                entity_type=entity_type,
                session_id=session_id
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "No revision session found for entity_type "
                f"'{entity_type}' with session_id '{session_id}'."
            )

        # Create handler
        handler = handler_class(
            main_repo=self.repo,
            workspace_root=workspace_path,
        )

        return handler

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
        entity_type: str,
        entity_id: str,
        table_type: str | None = None,
        sample_ids: Sequence[str] | None = None,
        test_name: str | None = None,
    ) -> Any:
        """
        Generate a QC table for an entity from its cached QC status.

        Parameters
        ----------
        entity_type : str
            Type of entity (e.g., "compensation", "gating_strategy")
        entity_id : str
            Entity identifier (e.g., "comp_001")
        table_type : str | None
            Type of table to generate (entity-specific)
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to include. If None, includes all samples.
        test_name : str | None
            Optional specific test name to filter by. If provided, takes precedence over table_type
            and only tests matching this name will be included in the table.

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
        if table_type is None and test_name is None:
            raise ValueError("Must specify either table_type or test_name to generate QC table.")

        # Get evaluator
        qc_evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
        if qc_evaluator_class is None:
            raise ValueError(f"No QC evaluator registered for entity type '{entity_type}'")

        qc_evaluator = qc_evaluator_class()

        # Load QC status from repository
        qc_status = self.repo.load_qc_entity_status(entity_type, entity_id)

        # Generate table
        return qc_evaluator.generate_table(
            entity_qc=qc_status,
            table_type=table_type,
            sample_ids=sample_ids,
            test_name=test_name
        )

    def get_entity_qc_figure(
        self,
        entity_type: str,
        entity_id: str,
        test_key: Any,
        sample_ids: Sequence[str] | None = None,
        step_id: str | None = None,
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
        )

    def collect_qc_metrics(
        self,
        entity_type: str,
        table_type: str,
        sample_ids: Sequence[str] | None = None,
        entity_ids: Sequence[str] | None = None,
        test_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate QC metrics across samples by generating tables for each sample's active entity.

        This method generalizes EntityQCEvaluator.generate_table() across samples. For sample-based
        entities like "compensation", it loops over each sample's active entity reference (specified
        in the sample metadata), generates the QC table, and concatenates results.

        For entity types that don't have sample-specific references (like "step" or "gate_node"), provide
        explicit entity_ids to aggregate.

        Parameters
        ----------
        entity_type : str
            Type of entity to aggregate metrics for. Common values:
            - "compensation": Aggregates compensation QC across samples' active compensations
            - "gating_strategy": Aggregates gating strategy QC across samples' active strategies
            - "gate_node": Aggregates gate node QC (requires entity_ids; strategy auto-discovered)
            - "step": Aggregates step QC (requires entity_ids to be specified)
        table_type : str
            Type of table to generate for each entity (evaluator-specific).
            For compensation: "compensation_channel", "pairwise_tests", etc.
            For gating_strategy: "gate_event_counts", "gate_ratios"
            For gate_node: "event_metrics", "fitting_quality", etc.
            For step: "per_sample_step", "heatmap", "per_sample", etc.
        sample_ids : Sequence[str] | None, default None
            Sample IDs to include in aggregation. If None, includes all samples in the project.
        entity_ids : Sequence[str] | None, default None
            For sample-based entities (compensation, gating_strategy): ignored.
            For non-sample-based entities (gate_node, step): required. List of entity IDs to aggregate across.
        test_name : str | None, default None
            Optional specific test name to filter by. If provided, only rows for this test are included
            (overrides table_type filtering). Returns table with test-specific columns.

        Returns
        -------
        pd.DataFrame
            Concatenated table with all rows from individual entity tables.
            For compensation example: Aggregates compensation_channel tables across all samples'
            active compensations, with sample_id preserved in output.

        Raises
        ------
        ValueError
            If entity_type is not registered
        ValueError
            If any sample_ids are not found in project
        ValueError
            If entity_ids is required for entity_type but not provided

        Examples
        --------
        Aggregate compensation QC metrics across all samples:

        >>> pipeline = InteractivePipeline("/path/to/project")
        >>> df = pipeline.aggregate_qc_metrics(
        ...     entity_type="compensation",
        ...     table_type="compensation_channel",
        ... )
        >>> # Returns table with compensation channel test results from all samples' active compensations
        >>> print(df.columns)
        # sample_id, compensation, channel, test_name, status, metric_name, metric_value, ...

        Aggregate for specific samples:

        >>> df = pipeline.aggregate_qc_metrics(
        ...     entity_type="compensation",
        ...     table_type="pairwise_tests",
        ...     sample_ids=["sample_001", "sample_002"],
        ... )

        Aggregate step QC across specific step entities:

        >>> df = pipeline.aggregate_qc_metrics(
        ...     entity_type="step",
        ...     table_type="per_sample_step",
        ...     entity_ids=["step_0001", "step_0002"],
        ... )
        """
        # Validate entity type
        qc_evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
        if qc_evaluator_class is None:
            raise ValueError(f"No QC evaluator registered for entity type '{entity_type}'")

        qc_evaluator = qc_evaluator_class()

        # Load project metadata
        project = self.repo.load_project()

        # Determine sample IDs
        if sample_ids is None:
            sample_ids = list(project.samples.keys())
        else:
            sample_ids = list(sample_ids)
            # Validate that all provided sample_ids exist in the project
            missing_samples = set(sample_ids) - set(project.samples.keys())
            if missing_samples:
                raise ValueError(
                    f"The following sample IDs were not found in the project: {sorted(missing_samples)}"
                )

        if entity_type == "compensation":
            tables = self._aggregate_qc_metrics_compensation(
                qc_evaluator=qc_evaluator,
                table_type=table_type,
                project=project,
                sample_ids=sample_ids,
                test_name=test_name,
            )
        elif entity_type == "gating_strategy":
            tables = self._aggregate_qc_metrics_gating_strategy(
                qc_evaluator=qc_evaluator,
                table_type=table_type,
                project=project,
                sample_ids=sample_ids,
                test_name=test_name,
            )
        else:
            tables = self._aggregate_qc_metrics_by_entity_ids(
                qc_evaluator=qc_evaluator,
                entity_type=entity_type,
                table_type=table_type,
                sample_ids=sample_ids,
                entity_ids=entity_ids,
                test_name=test_name,
            )

        # Concatenate all tables
        if not tables:
            return pd.DataFrame()

        return pd.concat(tables, ignore_index=True)

    def aggregate_qc_metrics(
        self,
        entity_type: str,
        table_type: str | Sequence[str] | None,
        sample_ids: Sequence[str] | None = None,
        entity_ids: Sequence[str] | None = None,
        by: str | Sequence[str] = "target",
        test_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate QC metrics across samples by collecting metrics and combining status flags.

        This method calls collect_qc_metrics internally and then aggregates the result by
        the specified grouping dimension(s). The `by` parameter determines which columns
        to group by - it can be a predefined shortcut or any subset of target columns.
        For each group, it combines the status values using QCFlag logic
        (FAIL > SEVERE > WARN > PASS) and counts how many tests have each flag value.

        If table_type is "all" (or contains "all" when a sequence is provided), all test
        table types are loaded using the evaluator class registered in
        EntityQCEvaluatorRegistry for the given entity_type via evaluator_class.get_test_types().

        Parameters
        ----------
        entity_type : str
            Type of entity to aggregate metrics for. Common values:
            - "compensation": Aggregates compensation QC across samples' active compensations
            - "gating_strategy": Aggregates gating strategy QC across samples' active strategies
            - "step": Aggregates step QC (requires entity_ids to be specified)
        table_type : str | Sequence[str]
            Type(s) of table(s) to generate for each entity (evaluator-specific).
            For compensation: "compensation_channel", "pairwise_tests", etc.
            For gating_strategy: "gate_event_counts", "gate_ratios"
            For step: "per_sample_step", "heatmap", "per_sample", etc.
            If set to "all" (or if "all" is included in a list), aggregates metrics
            across all evaluator test table types.
        sample_ids : Sequence[str] | None, default None
            Sample IDs to include in aggregation. If None, includes all samples in the project.
        entity_ids : Sequence[str] | None, default None
            For sample-based entities (compensation, gating_strategy): ignored.
            For non-sample-based entities (step): required. List of entity IDs to aggregate across.
        by : str | Sequence[str], default "target"
            Grouping dimension(s) for aggregation. Can be:
            - "target": Groups by all target columns (entity_id, sample_id, etc.)
            - "sample": Groups by sample_id only (if present in targets)
            - "entity": Groups by entity identifier (first target column, or first 2 for gates)
            - A sequence of column names: Groups by any subset of target columns

            Target columns vary by entity type:
        test_name : str | None, default None
            Optional specific test name to filter by. If provided, only data from this test are aggregated
            (takes precedence over table_type).
            - compensation: ["compensation_id", "sample_id", "mask"]
            - gating_strategy: ["sample_id", "gate_id"]
            - step: ["step_id", "sample_id"]

        Returns
        -------
        pd.DataFrame
            Aggregated table with columns:
            - All target columns (those before test_type, e.g., compensation_id, sample_id, mask)
            - status: Combined QCFlag across all tests in the group
            - n_tests: Total number of tests in the group
            - n_pass, n_warn, n_severe, n_fail, n_skip: Count of tests with each flag status

        Raises
        ------
        ValueError
            If entity_type is not registered
        ValueError
            If any sample_ids are not found in project
            If entity_ids is required for entity_type but not provided
            If `by` is a string not in {"target", "sample", "entity"}
            If `by` is a sequence containing columns not in target columns

        Examples
        --------
        Aggregate compensation QC metrics across all samples (by all targets):

        >>> pipeline = InteractivePipeline("/path/to/project")
        >>> df_agg = pipeline.aggregate_qc_metrics(
        ...     entity_type="compensation",
        ...     table_type="compensation_channel",
        ...     by="target",  # Groups by compensation_id, sample_id, mask
        ... )

        Aggregate by sample only:

        >>> df_agg = pipeline.aggregate_qc_metrics(
        ...     entity_type="compensation",
        ...     table_type="compensation_channel",
        ...     by="sample",  # Groups by sample_id only
        ... )

        Aggregate by custom columns:

        >>> df_agg = pipeline.aggregate_qc_metrics(
        ...     entity_type="compensation",
        ...     table_type="compensation_channel",
        ...     by=["compensation_id", "sample_id"],  # Custom grouping
        ... )
        """
        # Validate entity type
        qc_evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
        if qc_evaluator_class is None:
            raise ValueError(f"No QC evaluator registered for entity type '{entity_type}'")

        qc_evaluator = qc_evaluator_class()

        # Get target columns from evaluator
        targets =  list(qc_evaluator_class.targets) + ["test_type", "test_name"]
        all_types = sorted(qc_evaluator_class.get_test_types())

        # Determine id_vars based on 'by' parameter
        if isinstance(by, str):
            if by == "target":
                id_vars = targets[:-2] # All target columns except test_type and test_name
            elif by == "test":
                id_vars = ["test_type", "test_name"]
            elif by == "sample":
                if "sample_id" not in targets:
                    raise ValueError(
                        f"Cannot group by 'sample': 'sample_id' not in target columns {targets}"
                    )
                id_vars = ["sample_id"]
            elif by == "entity":
                # First target is the entity identifier for each evaluator.
                id_vars = [targets[0]] if targets else []
            else:
                # Treat as a single column name
                if by not in targets:
                    raise ValueError(
                        f"Column '{by}' not found in target columns {targets}. "
                        f"Use one of: 'target', 'sample', 'entity', or a valid column name."
                    )
                id_vars = [by]
        elif isinstance(by, (list, tuple)):
            by_list = list(by)
            invalid = set(by_list) - set(targets)
            if invalid:
                raise ValueError(
                    f"Invalid columns in 'by': {sorted(invalid)}. "
                    f"Must be a subset of target columns: {targets}"
                )
            id_vars = by_list
        else:
            raise ValueError(
                f"'by' must be a string or sequence of strings, got {type(by).__name__}"
            )

        value_vars = ["TOTAL", "PASS", "WARN", "SEVERE", "FAIL", "SKIP"]
        subset_vars = targets + ["status", "metric"]

        cols = id_vars + ["status"] + value_vars

        if table_type is None:
            table_types = all_types
        elif isinstance(table_type, str):
            table_types = all_types if table_type == "all" else [table_type]
        else:
            table_types = list(table_type)
        unknown_types = set(table_types) - set(all_types) - {"all"}
        if unknown_types:
            raise ValueError(
                f"Unknown table_type(s) {sorted(unknown_types)} for entity_type '{entity_type}'. "
                f"Known types are: {sorted(all_types)} or 'all'."
            )

        if not table_types:
            return pd.DataFrame(columns=cols)

        normalized_tables: list[pd.DataFrame] = []
        for current_table_type in table_types:
            current_df = self.collect_qc_metrics(
                entity_type=entity_type,
                table_type=current_table_type,
                sample_ids=sample_ids,
                entity_ids=entity_ids,
                test_name=test_name,
            )
            if not current_df.empty:
                normalized_tables.append(current_df[subset_vars])

        if not normalized_tables:
            return pd.DataFrame(columns=cols)

        # Add Counts
        combined = (pd.concat(normalized_tables, ignore_index=True, sort=False)
                      .groupby(id_vars + ["status"], dropna=True)["metric"]
                      .count()
                      .reset_index()
                      .pivot(index=id_vars, columns="status", values="metric")
                      .fillna(0)
                      .reset_index())
        for var in value_vars:
            if var not in combined.columns:
                combined[var] = 0
        combined = combined.astype({var: int for var in value_vars}, errors="ignore")
        combined["TOTAL"] = combined[value_vars[1:]].sum(axis=1)

        # Add Combined Status
        def decide_status(row):
            if row["FAIL"] > 0:
                return "FAIL"
            elif row["SEVERE"] > 0:
                return "SEVERE"
            elif row["WARN"] > 0:
                return "WARN"
            elif row["PASS"] > 0:
                return "PASS"
            else:
                return "SKIP"
        combined["status"] = combined.apply(decide_status, axis=1)
        return combined[cols]

    def _aggregate_qc_metrics_compensation(
        self,
        qc_evaluator: Any,
        table_type: str,
        project: Project,
        sample_ids: Sequence[str],
        test_name: str | None = None,
    ) -> list[pd.DataFrame]:
        """Aggregate QC tables for compensation entities, batched by active compensation."""

        tables: list[pd.DataFrame] = []
        sample_ids_by_comp: dict[str, list[str]] = {}
        for sample_id in sample_ids:
            comp_id = project.samples[sample_id].compensation
            if comp_id is None:
                continue
            if comp_id not in sample_ids_by_comp:
                sample_ids_by_comp[comp_id] = []
            sample_ids_by_comp[comp_id].append(sample_id)

        for comp_id, comp_sample_ids in sample_ids_by_comp.items():
            try:
                qc_status = self.repo.load_qc_entity_status("compensation", comp_id)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"QC status for compensation {comp_id!r} not found. "
                    f"Cannot aggregate QC metrics for samples {comp_sample_ids!r}."
                )

            try:
                df = qc_evaluator.generate_table(
                    entity_qc=qc_status,
                    table_type=table_type,
                    sample_ids=comp_sample_ids,
                    test_name=test_name,
                )
                if not df.empty:
                    tables.append(df)
            except (ValueError, KeyError) as e:
                warnings.warn(
                    f"Failed to generate {table_type} table for compensation {comp_id!r} "
                    f"(samples={comp_sample_ids!r}): {e}"
                )
        return tables

    def _aggregate_qc_metrics_gating_strategy(
        self,
        qc_evaluator: Any,
        project: Project,
        sample_ids: Sequence[str],
        table_type: str | None = None,
        test_name: str | None = None,
    ) -> list[pd.DataFrame]:
        """Aggregate QC tables for the project gating strategy across selected samples."""

        tables: list[pd.DataFrame] = []
        gs = project.gating_strategy

        try:
            qc_status = self.repo.load_qc_entity_status("gating_strategy", gs.id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"QC status for gating_strategy {gs.id!r} not found."
            )

        gs_samples = list(sample_ids)
        try:
            df = qc_evaluator.generate_table(
                entity_qc=qc_status,
                sample_ids=gs_samples,
                table_type=table_type,
                test_name=test_name,
            )
            if not df.empty:
                tables.append(df)
        except (ValueError, KeyError) as e:
            warnings.warn(
                f"Failed to generate table for samples {gs_samples!r}, "
                f"gating_strategy {gs.id!r}: {e}"
            )
        return tables

    def _aggregate_qc_metrics_by_entity_ids(
        self,
        qc_evaluator: Any,
        entity_type: str,
        table_type: str,
        sample_ids: Sequence[str],
        entity_ids: Sequence[str] | None,
        test_name: str | None = None,
    ) -> list[pd.DataFrame]:
        """Aggregate QC tables for entity types that require explicit entity IDs.

        For gate_node: validates gate IDs against the project gating strategy.
        """

        if entity_ids is None:
            raise ValueError(
                f"entity_ids must be specified for entity_type '{entity_type}' "
                "(only 'compensation' and 'gating_strategy' support sample-based aggregation)"
            )

        tables: list[pd.DataFrame] = []

        # For gate_node, validate gate IDs against the project strategy
        if entity_type == "gate_node":
            project = self.repo.load_project()
            strategy = project.gating_strategy
            if strategy is None:
                warnings.warn("No gating strategy found in project, skipping gate_node QC aggregation.")
                return tables
            for gate_id in entity_ids:
                try:
                    strategy.get_node(gate_id)
                except (KeyError, AttributeError):
                    warnings.warn(f"Gate {gate_id!r} not found in project gating strategy, skipping.")

        for entity_id in entity_ids:
            try:
                qc_status = self.repo.load_qc_entity_status(entity_type, entity_id)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"QC status for entity {entity_id!r} not found. "
                    f"Cannot aggregate QC metrics for samples {sample_ids!r}."
                )

            dataloader_context = {"sample_ids": [sid for sid in sample_ids if sid in qc_status.sample_qc]}

            try:
                df = qc_evaluator.generate_table(
                    entity_qc=qc_status,
                    table_type=table_type,
                    sample_ids=dataloader_context["sample_ids"] if dataloader_context["sample_ids"] else None,
                    test_name=test_name,
                )
                if not df.empty:
                    tables.append(df)
            except (ValueError, KeyError) as e:
                warnings.warn(f"Failed to generate {table_type} table for entity {entity_id!r}: {e}")
        return tables

    # ---- Helpers ----

    def _next_step_id(self) -> str:
        """Generate the next sequential step identifier."""
        n_steps = self.repo.step_counter + 1
        return f"step_{n_steps:04d}"

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
        project = self.repo.load_project()
        sample_ids = set(project.samples.keys())
        if not sample_ids:
            raise ValueError("Cannot add gate: project has no samples.")

        batch_id = next(
            (bid for bid, batch in project.batches.items() if batch.sample_ids == sample_ids),
            "",
        )
        if not batch_id:
            batch_id = "__all__"
            if batch_id in project.batches and project.batches[batch_id].sample_ids != sample_ids:
                suffix = 1
                while f"{batch_id}_{suffix}" in project.batches:
                    suffix += 1
                batch_id = f"{batch_id}_{suffix}"

            self.repo.update_project_metadata(
                batches=[BatchRef(id=batch_id, sample_ids=sample_ids, tags={"all_samples"}, meta={})]
            )

        # Convert parent_id to list
        if isinstance(parent_ids, str):
            parent_ids = [parent_ids]
        else:
            parent_ids = list(parent_ids)

        if gate_type == "Boolean":
            dime_set: set[str] = set()
            for parent in parent_ids:
                parent_node = self.repo.load_gate_node(node_id=parent)
                dime_set.update(parent_node.dimensions)
            dimensions = sorted(dime_set)

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
                "batch_ids": [batch_id],
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
        dimensions: Iterable[Mapping[str, Any]] | None = None,
        batch_id: str = "panel",
        default: bool = False
    ) -> StepRun:
        """
        Adds a new data layer to the specified samples.

        Parameters
        ----------
        layer : str
            The name of the new data layer to add.
        dimensions : Iterable[Mapping[str, Any]] | None
            A list of dimension definitions to create the new layer. If None, the layer will be created empty.
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

        if layer not in project.layers:
            if dimensions is None:
                raise ValueError(f"Data layer {layer!r} does not exist. Provide dimensions to create it.")
            self.repo.add_data_layer(layer, dimensions=dimensions)

        return self.run_step(
            step_type="add_layer",
            config={"layer": layer, "default": default},
            inputs={"batch_ids": [batch_id]},
        )

    def add_dimensions(
        self,
        layer: str,
        dimensions: Sequence[Mapping[str, Any]],
        batch_id: str = "panel",
    ) -> StepRun:
        """
        Adds dimensions to an existing data layer.

        Parameters
        ----------
        layer : str
            The data layer to which dimensions will be added.
        dimensions : Sequence[Mapping[str, Any]]
            A list of dimension definitions to add.
        batch_id: str
            Batch ID to which the new dimensions will be applied.

        Returns
        -------
        StepRun
            The completed add_dimensions step run.
        """
        return self.run_step(
            step_type="add_dimensions",
            config={"layer": layer, "dimensions": dimensions},
            inputs={"batch_ids": [batch_id]},
        )

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

