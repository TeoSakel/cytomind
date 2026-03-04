"""
Entity QC evaluators and registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Mapping, Iterator, Iterable, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from cytomind.infra.repo import ProjectRepository
from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.utils import now_iso

from . import EntityQCEvaluatorRegistry

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.domain.pipeline import StepRun
    from cytomind.infra.dataloader import UnifiedDataLoader
    from pandas import DataFrame
    from plotly.graph_objects import Figure
else:
    StepRun = object
    PathLike = object
    UnifiedDataLoader = object
    Figure = object
    DataFrame = object


class QCTester(ABC):
    """
    Base class for QC test with fit-classify-plot pipeline.

    Workflow:
    1. fit() - compute metrics from adata, optionally return plot data
    2. classify() - apply thresholds, update status to PASS/WARN/FAIL
    3. plot() - generate diagnostic visualization (can reuse plot_data from fit)

    This design supports the hybrid QC pattern:
    - Steps emit test records during execution (status="PENDING")
    - Evaluators classify records post-execution with configurable thresholds
    - Plots are generated on-demand for flagged tests

    Plotting Metadata:
    - plot_type: str - Category of plot (e.g., "histogram", "scatter", "heatmap"). Empty string if plot() not implemented.
    - plot_description: str - Human-readable description for frontend UI

    Concrete implementations should:
    - Set test_type and test_name class attributes
    - Define default_config and default_thresholds
    - Implement fit/classify/plot/make_key for their specific entity type
    - If plot() is implemented, populate plot_type and plot_description
    - Use **kwargs for entity-specific dimensions (donors, parents, receivers, etc.)
    """

    test_type: str                           # Evaluator type
    test_name: str                           # name of the test
    target_keys: tuple[str, ...] = ()        # Fields from targets that identify tested entity instance(s)
    meta_keys: tuple[str, ...] = ()          # Fields from metadata that identify tested dimensions
    default_config: dict[str, Any] = {}      # Default config parameters for the tester
    default_thresholds: dict[str, Any] = {}  # Default thresholds for classifying test results
    plot_type: str = ""                      # Category of plot (e.g., "histogram", "scatter", "heatmap"). Empty if no plot.
    plot_description: str = ""               # Human-readable description for frontend UI

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        # Keep a concrete, instance-level tuple for downstream code paths.
        cfg = dict(self.default_config)
        for key in cfg:
            if key in config:
                cfg[key] = config[key]
        self.metadata = cfg
        thres = dict(self.default_thresholds)
        for key in thres:
            if key in thresholds:
                thres[key] = thresholds[key]
        self.thresholds = thres

    @property
    def key_fields(self) -> tuple[str, ...]:
        return self.target_keys + ("test_type", "test_name") + self.meta_keys

    def fit(self, *args, **kwargs) -> Iterable[QCTestRecord]:
        raise NotImplementedError("fit() method not implemented for this tester. This tester cannot be used for testing or plotting.")

    @abstractmethod
    def classify(self, test: QCTestRecord, **kwargs) -> QCTestRecord:
        """
        Populate test_record.status based on metrics and thresholds.

        Parameters
        ----------
        test : QCTestRecord
            Test record from fit() with status="PENDING"
        **kwargs : dict
            Optional threshold overrides (supersede self.thresholds)

        Returns
        -------
        QCTestRecord
            Updated test with:
            - status: "PASS", "WARN", "SEVERE", "FAIL", or "SKIP"
            - thresholds: applied threshold values
            - message: human-readable summary
        """
        pass

    def fit_classify( self, **kwargs) -> Iterable[QCTestRecord]:
        """
        Convenience method to run fit and classify sequentially.

        Iterates over records from fit(), classifies each, and yields classified records.
        """
        for test in self.fit(**kwargs):
            classified_test = self.classify(test, **kwargs)
            yield classified_test

    @abstractmethod
    def plot(self, adata: AnnData, test: QCTestRecord, output_path: PathLike | None = None, **kwargs) -> Figure:
        """
        Generate diagnostic plot for test.

        Implementation can call fit(adata, plot_data=True) internally to avoid
        duplication if plot-specific data is needed.

        Parameters
        ----------
        adata : AnnData
            Data to visualize (same as used in fit())
        test : QCTestRecord
            Classified test record with metrics and status
        output_path : PathLike | None
            Optional path to save figure
        **kwargs : dict
            Entity-specific context and plotting options
            (channel names, colors, bins, transformations, etc.)

        Returns
        -------
        Figure
            Plotly figure object
        """
        pass

    def key_dict(
        self,
        targets: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        targets = targets or {}
        metadata = metadata or {}

        missing = [key for key in self.target_keys if key not in targets]
        if missing:
            raise KeyError(f"Missing target keys: {missing}")
        missing = [key for key in self.meta_keys if key not in metadata]
        if missing:
            raise KeyError(f"Missing metadata keys: {missing}")

        d: dict[str, Any] = {
            "test_type": self.test_type,
            "test_name": self.test_name
        }
        for key in self.target_keys: d[key] = targets[key]
        for key in self.meta_keys:   d[key] = metadata[key]

        return d

    def make_key(
        self,
        targets: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple:
        keys =  self.target_keys + ("test_type", "test_name") + self.meta_keys
        key_dict = self.key_dict(targets=targets, metadata=metadata)
        return tuple(key_dict[key] for key in keys)

    @classmethod
    def from_dict(cls, test: QCTestRecord | Mapping[str, Any]) -> QCTester:
        """Factory method to create tester from dict config."""
        test = test if isinstance(test, QCTestRecord) else QCTestRecord.from_dict(test)  # Validate required fields and types
        cls._check_test_record(test)  # Validate required fields and types
        return cls(config=test.metadata, thresholds=test.thresholds)

    @classmethod
    def _check_test_record(cls, test: QCTestRecord | Mapping[str, Any]):
        if not isinstance(test, QCTestRecord):
            test = QCTestRecord.from_dict(test)  # Validate required fields and types

        if test.test_type != cls.test_type:
            raise ValueError(f"Test record has type '{test.test_type}' but expected '{cls.test_type}'")
        if test.test_name != cls.test_name:
            raise ValueError(f"Test record has name '{test.test_name}' but expected '{cls.test_name}'")
        for field in cls.target_keys:
            if field not in test.targets:
                raise ValueError(f"Test record is missing target key field '{field}' in targets")
        for field in cls.meta_keys:
            if field not in test.metadata:
                raise ValueError(f"Test record is missing metadata key field '{field}' in metadata")


class EntityQCEvaluator(ABC):
    """
    Unified QC evaluation for any entity type (compensation, gating_strategy, step, etc.).

    Philosophy:
    - All entities (including steps) are evaluated using the same interface
    - Steps are just entities of type "step" with special product evaluation hooks
    - Entity QC operates on persistent data and can be re-run without re-execution

    Best Hybrid QC Pattern:
    - Steps emit test records with metrics and status="PENDING" during execution
    - Evaluators classify pending records, apply thresholds, assign final status
    - Thresholds are configurable and can be adjusted post-execution

    Artifact Declaration:
    - _supported_tables: dict[str, dict] - Mapping of table_type → artifact spec
    - _supported_figures: dict[str, dict] - Mapping of figure_type → artifact spec
    - Each spec dict should include:
        - "description": str - Human-readable description
        - "input_params": dict - Required/optional parameters for generation
        - Optionally other metadata

    Example:
        _supported_tables = {
            "my_table": {
                "description": "Summary table of test results",
                "input_params": {
                    "sample_data": "optional"
                }
            }
        }

    Test Registration:
    - Subclasses implement get_tests() → dict[str, type[QCTester]]
    - Composition via super().get_tests() and dict.update() for inheritance
    - Tests route to per-sample or batch evaluation based on test_type

    Artifact Listing:
    - Call list_artifacts(entity_ref=None) to get available artifacts
    - This combines class-level declarations with test-derived plots
    """

    entity_type: str
    targets: tuple[str, ...] = ()  # Fields that identify the entity instance (e.g., compensation_id, sample_id, mask)
    default_config: dict[str, Any] = {}
    _supported_tables: dict[str, dict[str, Any]] = {}    # Table type → artifact spec
    _supported_figures: dict[str, dict[str, Any]] = {}   # Figure type → artifact spec

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        self.config = cfg

    @property
    def test_types(self) -> set[str]:
        """Return the set of test types for this evaluator."""
        return self.get_test_types()

    @classmethod
    def get_test_types(cls, entity: Any = None) -> set[str]:
        """Return the set of test types for this evaluator.

        Parameters
        ----------
        entity : Any, optional
            Entity for which to get test types (used by subclasses for entity-specific tests).

        Returns
        -------
        set[str]
            Set of test type identifiers
        """
        tests = cls.get_tests(entity=entity)
        return set(tester.test_type for tester in tests.values())

    @classmethod
    def get_tests(cls, entity: Any = None) -> dict[str, type[QCTester]]:
        """
        Return dictionary of test classes for this evaluator.

        Subclasses compose tests via:
            tests = super().get_tests(entity=entity)  # Get parent tests
            tests.update({"new_test": NewTestClass})  # Add own tests
            return tests

        Parameters
        ----------
        entity : Any, optional
            Entity for which to get tests (used by subclasses for entity-specific tests).
            Compensation and step evaluators will ignore this parameter.

        Returns
        -------
        dict[str, type[QCTester]]
            Mapping of test_name → QCTester subclass
        """
        return {}

    def list_artifacts(self, entity_ref: Any = None) -> dict[str, list[dict[str, Any]]]:
        """
        List available artifacts (tables and figures) for this evaluator.

        Combines:
        - Class-level artifact declarations (_supported_tables, _supported_figures)
        - Test-derived plots from registered tests (tests with supports_plot=True)

        The entity_ref parameter allows evaluators to determine entity-dependent artifacts.
        For example, GatingStrategyQCEvaluator can use StrategyRef to decide which
        gate-specific visualizations to include.

        Parameters
        ----------
        entity_ref : Any, optional
            Entity reference (e.g., CompensationRef, StrategyRef) used by subclasses
            to determine entity-dependent artifacts. Default None.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Dictionary with "tables" and "figures" keys, each mapping to a list of artifact specs.
            Each artifact spec is a dict with at minimum:
            - "type": str - identifier for the artifact (e.g., "compensation_channel")
            - "description": str - human-readable description
            - For test plots, also includes: "test_name", "test_type", "plot_type"

        Examples
        --------
        >>> evaluator = CompensationQCEvaluator()
        >>> artifacts = evaluator.list_artifacts()
        >>> artifacts["tables"]
        [{"type": "compensation_channel", "description": "..."},
         {"type": "compensation_pair", "description": "..."}]
        >>> artifacts["figures"]  # Includes test plots
        [{"type": "qc_test_plot", "test_name": "NegativeFluorescence", ...},
         ...]
        """
        # Start with class-level artifact declarations
        tables = []
        for table_type, spec in self._supported_tables.items():
            table_spec = dict(spec)  # Copy to avoid mutating class attribute
            table_spec["type"] = table_type
            tables.append(table_spec)

        figures = []
        for figure_type, spec in self._supported_figures.items():
            figure_spec = dict(spec)  # Copy to avoid mutating class attribute
            figure_spec["type"] = figure_type
            figures.append(figure_spec)

        # Add test-derived plots from registered tests
        tests = self.get_tests(entity=entity_ref)
        for test_name, tester_class in tests.items():
            plot_type = getattr(tester_class, "plot_type", "")
            if plot_type:  # Non-empty plot_type means plot is supported
                test_plot_spec = {
                    "type": "qc_test_plot",
                    "test_name": tester_class.test_name,
                    "test_type": tester_class.test_type,
                    "plot_type": plot_type,
                    "description": getattr(tester_class, "plot_description", ""),
                }
                figures.append(test_plot_spec)

        return {
            "tables": tables,
            "figures": figures,
        }

    def update_entity_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus | None = None,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        entity_qc = entity_qc or EntityQCStatus(entity_id=entity.id, entity_type=self.entity_type, generated_at=now_iso())
        entity_qc = self.update_sample_qc(entity, entity_qc, dataloader, dataloader_context, context=context)
        entity_qc = self.update_batch_qc(entity, entity_qc, dataloader, dataloader_context, context=context)
        entity_qc.summary.update(self.basic_summary(entity_qc))
        summary_dict = self.summarize_entity_qc(entity_qc)
        if "status" in summary_dict:
            raise ValueError("Summary dict cannot contain reserved key 'status'")
        if "aggregated_flag_counts" in summary_dict:
            raise ValueError("Summary dict cannot contain reserved key 'aggregated_flag_counts'")
        entity_qc.summary.update(summary_dict)
        return entity_qc

    @abstractmethod
    def update_sample_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Update the QC for a specific entity instance.

        This method is stateless - it takes the entity, QC status, and dataloader
        and updates the per_sample_qc and batch_qc based on the tests defined for this entity type.

        Parameters
        ----------
        entity : Any
            The entity to evaluate (type depends on entity_type).
        entity_qc : EntityQCStatus
            QC status object to update.
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading sample data (AnnData, masks, etc.)
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader (e.g., layer, sample_ids)
        context : dict[str, Any]
            Optional metadata to attach to the QC status.

        Returns
        -------
        EntityQCStatus
            Updated QC status with test results and summary.
        """
        pass

    @abstractmethod
    def update_batch_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """
        Update batch-level QC tests (run once across all samples).

        Store results in entity_qc.batch_qc. Subclasses that have no batch tests
        can implement as no-op.

        Parameters
        ----------
        entity : Any
            The entity being evaluated
        entity_qc : EntityQCStatus
            QC status to update with batch test results
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading sample data (AnnData, masks, etc.)
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader (e.g., layer, sample_ids)
        context : dict[str, Any]
            Optional evaluation context

        Returns
        -------
        EntityQCStatus
            Updated entity_qc with batch test results in batch_qc
        """
        pass

    @abstractmethod
    def summarize_entity_qc(
        self,
        entity_qc: EntityQCStatus,
    ) -> dict[str, Any]:
        """Generate user-facing summary for review UI.

        Transforms detailed QC data into formatted tables, metrics,
        and recommendations suitable for user review.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            QC status with detailed test records

        Returns
        -------
        dict
            User-facing summary with tables, metrics, recommendations
        """
        pass


    def parse_step(
        self,
        step_run: StepRun,
        entity_id: str | None = None,
    ) -> EntityQCStatus:
        """
        Parse QC information from step execution and create initial EntityQCStatus.

        This method is called by the step evaluator to extract QC data generated
        during step execution (e.g., gate fitting metrics, compensation application results)
        and populate the EntityQCStatus for an entity created by the step.

        The step_run.qc contains fine-grained execution data that should be captured
        in the entity's QCStatus before full entity-level QC evaluation runs.

        Parameters
        ----------
        step_run : StepRun
            Step execution context with qc data from computation phases
        entity_id : str | None
            Optional entity identifier when parsing QC for step products.

        Returns
        -------
        EntityQCStatus
            Initial EntityQCStatus with data parsed from step execution.

        Subclasses should override to extract entity-specific test records and metrics
        from step_run.qc and populate an EntityQCStatus.
        """
        target_id = entity_id or step_run.id
        return EntityQCStatus(entity_id=target_id, entity_type=self.entity_type, generated_at=now_iso())

    def evaluate_step_products(
        self,
        repo: ProjectRepository,
        step_run: StepRun,
    ) -> Iterator[EntityQCStatus]:
        """
        Optional hook: Evaluate products created by this entity.

        For step entities: parse step-level QC data and optionally run full entity evaluation.
        For other entities: no-op (return empty dict).

        Workflow:
        1. Extract entities from step_run.evaluable_products (only includes products actually ready for QC)
        2. Call parse_step() for each entity to create initial EntityQCStatus with
           computation-stage data (fitting results, metrics, etc.)
        3. Optionally run additional run_entity_qc() for full evaluation

        Note: Products in project_updates but not in evaluable_products are skipped.
        This distinguishes "entities registered in project" from "entities actually ready to evaluate".
        Example: Compensations created by AddSamplesStep are in project_updates but not evaluable_products
        (they haven't been applied to sample data yet).

        Parameters
        ----------
        step_run : StepRun
            Step run associated with the entity (only for step entities)

        Returns
        -------
        Iterator[EntityQCStatus]
            Iterator over EntityQCStatus objects for each product entity
        """
        # Loop through evaluable products only (products explicitly marked as ready for QC)
        for entity_type, entities in step_run.evaluable_products.items():

            # Check if evaluator exists for this entity type
            evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
            if not evaluator_class:
                warnings.warn(f"No EntityQCEvaluator registered for entity type '{entity_type}'. Skipping QC evaluation for these products.")
                continue

            evaluator = evaluator_class()  # TODO: consider passing entity-specific config if needed
            for entity_id, context in entities.items():
                if "sample_ids" in context:
                    sample_ids: list[str] = context.pop("sample_ids")
                else:
                    sample_ids = list(step_run.sample_outputs.keys())  # Default to all samples in step outputs if not specified in context
                qc_status = evaluator.parse_step(step_run, entity_id)
                entity = evaluator.load_entity(repo._dataloader, entity_id)  # Load full entity for evaluation

                # Build dataloader context with sample_ids only
                dataloader_context = {}
                if sample_ids:
                    dataloader_context["sample_ids"] = sample_ids

                qc_status = evaluator.update_entity_qc(
                    entity=entity,
                    entity_qc=qc_status,
                    dataloader=repo._dataloader,
                    dataloader_context=dataloader_context if dataloader_context else None,
                    context=context,
                )
                yield qc_status

    def basic_summary(
        self,
        entity_qc: EntityQCStatus,
    ) -> dict[str, Any]:
        """
        Generate user-facing summary for review UI.

        Transforms detailed QC data into formatted tables, metrics,
        and recommendations suitable for user review.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            QC status with detailed test records

        Returns
        -------
        dict
            User-facing summary with tables, metrics, recommendations
        """
        if entity_qc.entity_type != self.entity_type:
            raise TypeError(f"EntityQCEvaluator for '{self.entity_type}' cannot summarize QC for entity type '{entity_qc.entity_type}'")

        per_sample_flags = {sid: flag.value for sid, flag in entity_qc.sample_flags.items()}
        sample_counts = Counter(qc.overall_flag.value for qc in entity_qc.sample_qc.values())

        # Build test summary by counting status for each test_name
        test_summary: dict[str, dict[tuple, dict[str, int]]] = {}
        for sample_id in entity_qc.sample_qc:
            for step_name, test_key, test in entity_qc.iter_sample_tests(sample_id):
                if step_name not in test_summary:
                    test_summary[step_name] = {test_key: {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}}
                if test_key not in test_summary[step_name]:
                    test_summary[step_name][test_key] = {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}
                test_summary[step_name][test_key][test.status] += 1

        return {
            "status": {
                "overall": entity_qc.overall_flag.value,
                "batch": entity_qc.batch_qc.overall_flag.value if entity_qc.batch_qc else None,
                "per_sample": per_sample_flags,
            },
            "aggregated_flag_counts": {
                "overall": dict(sample_counts),
                "by_test": {step_name: list(test_dict.items()) for step_name, test_dict in test_summary.items()},
            }
        }

    def generate_table(
        self,
        entity_qc: EntityQCStatus,
        table_type: str,
        sample_ids: Iterable[str] | None = None,
        table_path: PathLike | None = None,
    ) -> DataFrame:
        """Generate a table from EntityQCStatus in long (melted) format.

        This is a generic implementation that extracts all tests for a given type,
        converts them to a DataFrame, and melts metrics to name/value columns.

        Subclasses can override this method to provide entity-specific table formats
        by calling super().generate_table() and then reshaping the result as needed.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records.
        table_type : str
            Type of tests to extract/table to generate (entity-specific).
        sample_ids : Iterable[str] | None
            Optional list of sample IDs to filter the table.
        table_path : PathLike | None
            Optional output path to save the table CSV.

        Returns
        -------
        DataFrame
            Table in long (melted) format with columns:
            - sample_id, mask, status, [metadata keys from tester.key_dict()], metric, value

        Examples
        --------
        >>> # In a subclass override:
        >>> df_long = super().generate_table(entity_qc, table_type, ...)
        >>> df_wide = df_long.pivot_table(...)  # Reshape as needed
        >>> return df_wide
        """
        # Identify key columns from the first test of this type's target/meta keys
        tests = self.get_tests(entity=None)
        id_vars = []

        # Get key fields from any test of the requested type
        for tester_class in tests.values():
            if tester_class.test_type == table_type:
                id_vars = list(tester_class().key_fields)
                break
        if not id_vars:
            raise ValueError(f"No tests found for table_type '{table_type}'. Cannot determine id_vars for table generation.")
        id_vars += ["status"]  # Always include status as an id_var for melting

        # Extract records using helper method
        records = self._extract_table_records(entity_qc, table_type, sample_ids=sample_ids)

        if not records:
            # Return empty DataFrame with correct column structure
            final_columns = id_vars + ["metric", "value"]
            return pd.DataFrame(columns=final_columns)

        df = pd.DataFrame.from_records(records)
        df = df.melt(id_vars=id_vars, var_name="metric", value_name="value").dropna(subset=["value"])

        # Save to file if output path provided
        if table_path is not None:
            output_path = Path(table_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                warnings.warn(f"Output path {output_path} already exists and will be updated with new samples.")
                old = pd.read_csv(output_path, index_col=False)
                # Use base class method to update the table
                updated = self._update_qc_table(old, df)
                updated.to_csv(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)

        return df

    def _extract_table_records(
        self,
        entity_qc: EntityQCStatus,
        table_type: str,
        sample_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract test records across all samples and steps into a flattened list.

        This is a generalized helper method that iterates through all tests in the
        EntityQCStatus and extracts their data into records containing sample_id,
        status, test metadata (via key_dict), and all metrics.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records.
        table_type : str
            Type of tests to extract (matches test_type field).
        sample_ids : Iterable[str] | None
            Optional list of sample IDs to filter which samples to include.

        Returns
        -------
        list[dict[str, Any]]
            List of records, one per test, where each record contains:
            - "sample_id": str
            - "mask": str (from metadata.mask_key, defaults to "root")
            - "status": str
            - Test metadata from tester.key_dict() (e.g., "compensation_id", "channel")
            - All test metrics flattened as individual columns
        """
        # Use sample_ids from argument if provided, otherwise use all samples
        sample_filter = set(sample_ids or entity_qc.sample_qc.keys())

        # Validate table_type
        test_types = self.get_test_types()
        if table_type not in test_types:
            raise ValueError(f"Invalid table_type '{table_type}'. Must be one of {test_types}.")

        records = []
        tests = self.get_tests(entity=None)

        # Iterate through samples, steps, and tests
        for sample_id, sample_run in entity_qc.sample_qc.items():
            if sample_id not in sample_filter:
                continue

            for step in sample_run.steps.values():
                for test in step.tests.values():
                    # Skip tests that don't match the requested table_type
                    if test.test_type != table_type:
                        continue

                    # Get tester class and validate test_name
                    try:
                        tester_class = tests[test.test_name]
                    except KeyError:
                        valid_names = [key for key, val in tests.items() if val.test_type == table_type]
                        raise KeyError(f"Unknown test_name '{test.test_name}' for table_type '{table_type}'. Valid names are: {valid_names}")

                    # Create tester instance and extract metadata keys
                    tester = tester_class.from_dict(test)
                    key_dict = tester.key_dict(test.targets, test.metadata)

                    # Build record with status, metadata, and all metrics
                    records.append({
                        "status": test.status,
                        **key_dict,
                        **test.metrics,
                    })

        return records

    @staticmethod
    def _update_qc_table(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
        """Helper to update an existing QC table with new results based on sample_id and mask.

        Merges old and new tables, prioritizing new data for samples that appear in both.
        Validates that both tables have the same columns and test_type (if present).

        Parameters
        ----------
        df_old : pd.DataFrame
            Existing QC table
        df_new : pd.DataFrame
            New QC results to merge

        Returns
        -------
        pd.DataFrame
            Combined table with new results for updated samples and old results preserved for others
        """
        if df_old.empty:
            return df_new
        if df_new.empty:
            return df_old

        # Validate test_type compatibility if column exists
        test_type_old = df_old["test_type"].iloc[0]
        test_type_new = df_new["test_type"].iloc[0]
        if test_type_old != test_type_new:
            raise ValueError(f"Cannot merge tables with different test types: '{test_type_old}' vs '{test_type_new}'")

        if not df_old.columns.equals(df_new.columns):
            raise ValueError("Old and new DataFrames must have the same columns to merge.")

        # Remove old rows for (sample_id, mask) keys being updated, then concatenate with new.
        value_cols = {"metric", "value", "status"}  # Columns that contain test results
        key_cols = [col for col in df_old.columns if col not in value_cols]  # All other columns are keys
        update_keys = df_new[key_cols].drop_duplicates()
        old_with_marker = df_old.merge(update_keys.assign(_to_replace=True), on=key_cols, how="left")
        df_old_filtered = old_with_marker[old_with_marker["_to_replace"].isna()].drop(columns=["_to_replace"])
        df_combined = pd.concat([df_old_filtered, df_new], ignore_index=True)
        return df_combined

    @abstractmethod
    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Mapping[str, Any],
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
        figure_dir: PathLike | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Generate a diagnostic figure on demand.

        Creates a visualization identified by test_key. The interpretation of test_key
        is entity-specific:
        - For compensation: test_key is a test identifier (channel name, donor/receiver pair)
        - For gates: test_key identifies which gate/test to visualize
        - For step: test_key can be a specific test identifier or a visualization type
          (e.g., "heatmap" for a step-level overview)

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records and data.
        test_key : Mapping[str, Any]
            Entity-specific identifier for which figure to generate.
            Could be a test record key, visualization type, or other lookup value.
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading additional data (AnnData, masks, etc.)
            needed to generate the figure.
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader (e.g., strategy_id,
            layer, etc.). Used when loading data for figure generation.
        step_id : str | None
            Optional step ID to narrow scope of search/visualization.
            Meaning depends on entity type.
        figure_dir : PathLike | None
            Optional directory to save the figure.
        **kwargs : Any
            Additional entity-specific plotting options.

        Returns
        -------
        Figure
            Plotly figure object ready to be serialized or displayed.
        """
        pass

    @abstractmethod
    def required_layer(self, entity: Any = None) -> str | None:
        """Return the name of the AnnData layer required for this evaluator's tests.

        This is used to determine which layer to load for each sample when running
        entity QC. If the required layer is not present in a sample's AnnData, the
        evaluator should skip tests for that sample and mark it as "SKIP" with an
        appropriate message.

        Returns
        -------
        str | None
            Name of the required AnnData layer (e.g., "raw_counts", "compensated", "gate_mask").
            If None, no layer is required.
        """
        pass

    @abstractmethod
    def load_entity(self, dataloader: UnifiedDataLoader, entity_id: Hashable) -> Any:
        """Load the entity object from the dataloader given its ID.

        This method is used to retrieve the full entity (e.g., GateNode, CompensationRef)
        for a given entity_id when running QC. The implementation should handle loading
        the appropriate data structure based on the entity type.

        Parameters
        ----------
        dataloader : UnifiedDataLoader
            Dataloader instance to load data from.
        entity_id : str
            Unique identifier of the entity to load.

        Returns
        -------
        Any
            The loaded entity object (type depends on entity_type).
        """
        pass

    def _parse_test_key(
        self,
        test_key: tuple | Mapping[str, str],
    ) -> tuple[type[QCTester], dict[str, Any]]:
        """Parse and validate a test_key, extracting tester_class and normalizing the key.

        Parameters
        ----------
        test_key : tuple | Mapping[str, str]
            Test key as either (test_type, test_name, ...) tuple or mapping with 'test_type' and 'test_name'.

        Returns
        -------
        tuple[type[QCTester], dict[str, Any]]
            (tester_class, normalized_test_key_dict)
            The test_type and test_name can be retrieved from test_key_dict or tester_class attributes.

        Raises
        ------
        ValueError
            If test_key format is invalid or test_type is unsupported.
        KeyError
            If test_name is not in registry.
        """
        # Parse test_key to extract test_type and test_name
        if isinstance(test_key, Mapping):
            test_key_dict = dict(test_key)  # Make a copy to avoid mutating input
            try:
                test_type = test_key_dict["test_type"]
                test_name = test_key_dict["test_name"]
            except KeyError as e:
                raise ValueError(f"Invalid test_key mapping. Missing required key: {e.args[0]}")
        elif isinstance(test_key, tuple):
            test_type = str(test_key[0])
            test_name = str(test_key[1])
            test_key_dict = None
        else:
            raise ValueError("test_key must be either a tuple or a mapping with 'test_type' and 'test_name' keys.")

        # Validate test_type
        if test_type not in self.get_test_types():
            raise ValueError(f"Unsupported test_type '{test_type}'. Expected one of: {self.get_test_types()}")

        # Look up tester_class from get_tests()
        tests = self.get_tests(entity=None)
        try:
            tester_class = tests[test_name]
        except KeyError:
            raise ValueError(f"Unknown test name '{test_name}'. Available: {list(tests.keys())}")

        # Normalize test_key to dict if it was a tuple
        if test_key_dict is None:
            key_fields = tester_class.target_keys + tester_class.meta_keys
            test_key_dict = dict(zip(("test_type", "test_name") + key_fields, test_key))

        return tester_class, test_key_dict
