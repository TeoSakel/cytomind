"""
Entity QC evaluators and registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Hashable, Iterable, Mapping, Iterator, TYPE_CHECKING
import warnings

from anndata import AnnData

from cytomind.infra.repo import ProjectRepository
from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.utils import now_iso

from . import EntityQCEvaluatorRegistry

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.domain.pipeline import StepRun
    from pandas import DataFrame
    from plotly.graph_objects import Figure
else:
    StepRun = object
    PathLike = object
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
    key_fields: tuple[str, ...]              # Fields from metadata that uniquely identify this test instance (used for make_key)
    default_config: dict[str, Any] = {}      # Default config parameters for the tester
    default_thresholds: dict[str, Any] = {}  # Default thresholds for classifying test results
    plot_type: str = ""                       # Category of plot (e.g., "histogram", "scatter", "heatmap"). Empty if no plot.
    plot_description: str = ""                # Human-readable description for frontend UI

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
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

    @abstractmethod
    def fit(self, entity: Any, adata: AnnData,  **kwargs) -> QCTestRecord:
        """
        Compute test metrics from AnnData.

        Parameters
        ----------
        adata : AnnData
            Annotated data object (may be subset or full sample)
        plot_data : bool
            If True, return additional data needed for plotting
        **kwargs : dict
            Entity-specific context (sample_id, gate_id, channel names, donors, parents, etc.)

        Returns
        -------
        test : QCTestRecord
            Test record with:
            - id: unique identifier for this test instance (used for storing results)
            - test_type, test_name: from class attributes
            - metadata: context from kwargs
            - metrics: computed values
            - status: "PENDING"
        """
        pass

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

    def fit_classify(
        self,
        entity: Any,
        adata: AnnData,
        *,
        plot_data: bool = False,
        classify_kwargs: dict[str, Any] = {},
        **kwargs
    ) -> QCTestRecord:
        """Convenience method to run fit and classify sequentially."""
        test = self.fit(entity, adata, plot_data=plot_data, **kwargs)
        classified_test = self.classify(test, **classify_kwargs)
        return classified_test

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

    def key_dict(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        d: dict[str, Any] = {"test_type": self.test_type, "test_name": self.test_name}
        d.update({field: metadata[field] for field in self.key_fields})
        return d

    def make_key(self, metadata: Mapping[str, Any]) -> tuple:
        return tuple((self.test_type, self.test_name) + \
            tuple(metadata[field] for field in self.key_fields))

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
        for field in cls.key_fields:
            if field not in test.metadata:
                raise ValueError(f"Test record is missing key field '{field}' in metadata")


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

    def get_test_types(self, entity: Any = None) -> set[str]:
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
        tests = self.get_tests(entity=entity)
        return set(tester.test_type for tester in tests.values())

    def get_tests(self, entity: Any = None) -> dict[str, type[QCTester]]:
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


    @abstractmethod
    def update_batch_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        all_samples: Iterable[tuple[str, AnnData]] | None = None,
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
        all_samples : Iterable[tuple[str, AnnData]] | None
            Iterable of (sample_id, adata) tuples for all samples
        context : dict[str, Any]
            Optional evaluation context

        Returns
        -------
        EntityQCStatus
            Updated entity_qc with batch test results in batch_qc
        """
        pass

    def update_entity_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus | None = None,
        sample_data: Iterable[tuple[str, AnnData]] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        entity_qc = entity_qc or EntityQCStatus(entity_id=entity.id, entity_type=self.entity_type, generated_at=now_iso())
        entity_qc = self.update_sample_qc(entity, entity_qc, sample_data, context=context)
        entity_qc = self.update_batch_qc(entity, entity_qc, sample_data, context=context)  # TODO: problem if sample_data is iterator and is consumed by sample_qc?
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
        sample_data: Iterable[tuple[str, AnnData]] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Update the QC for a specific entity instance.

        This method is stateless - it takes the entity, QC status, and sample data
        and updates the per_sample_qc and batch_qc based on the tests defined for this entity type.

        Parameters
        ----------
        entity : Any
            The entity to evaluate (type depends on entity_type).
        entity_qc : EntityQCStatus | None
            Optional existing QC status to update. If None, creates a new one.
        sample_data : Iterable[tuple[str, AnnData]] | None
            Iterable of tuples mapping sample_id to AnnData for evaluation.
            If None, no samples will be evaluated.
        context : dict[str, Any] | None
            Optional metadata to attach to the QC status.

        Returns
        -------
        EntityQCStatus
            Updated QC status with test results and summary.
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
                entity = evaluator.load_entity(repo, entity_id)  # Load full entity for evaluation
                layer = evaluator.required_layer(entity)
                if layer:
                    sample_data = ((sid, repo.load_sample_adata(sid, layer=layer)) for sid in sample_ids)
                else:
                    sample_data = ((sid, AnnData()) for sid in sample_ids)
                qc_status = evaluator.update_entity_qc(entity=entity, entity_qc=qc_status, sample_data=sample_data, context=context)
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

    @abstractmethod
    def generate_table(
        self,
        entity_qc: EntityQCStatus,
        table_type: str,
        sample_data: Iterable[tuple[str, AnnData]]| None = None,
        table_dir: PathLike | None = None,
    ) -> DataFrame:
        """Generate a table from cached EntityQCStatus on demand.

        Reconstructs the specified table type from the stored test records
        in the QC status. This allows generating different views of the QC
        data without re-running tests.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records.
        table_type : str
            Type of table to generate (entity-specific).
        sample_data : Mapping[str, AnnData] | None
            Optional mapping of sample_id to AnnData. If provided, implementations
            may filter results to only include samples in this mapping.
            If None, returns all available data.

        Returns
        -------
        DataFrame
            Table with entity-specific columns matching the QC output format.
        """
        pass

    @abstractmethod
    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Any,
        sample_data: Iterable[tuple[str, AnnData]]| None = None,
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
        test_key : Any
            Entity-specific identifier for which figure to generate.
            Could be a test record key, visualization type, or other lookup value.
        sample_data : Mapping[str, AnnData] | None
            Optional mapping of sample_id to AnnData for plotting.
            Meaning and requirement depends on entity type.
        step_id : str | None
            Optional step ID to narrow scope of search/visualization.
            Meaning depends on entity type.
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
    def load_entity(self, repo: ProjectRepository, entity_id: Hashable) -> Any:
        """Load the entity object from the repository given its ID.

        This method is used to retrieve the full entity (e.g., GateNode, CompensationRef)
        for a given entity_id when running QC. The implementation should handle loading
        the appropriate data structure based on the entity type.

        Parameters
        ----------
        repo : ProjectRepository
            Repository instance to load data from.
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
            test_key_dict = dict(zip(("test_type", "test_name") + tester_class.key_fields, test_key))

        return tester_class, test_key_dict
