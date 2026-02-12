"""
Entity QC evaluators and registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Hashable, Mapping, TYPE_CHECKING

from matplotlib.pylab import f

from . import EntityQCEvaluatorRegistry

if TYPE_CHECKING:
    from cytomind.infra.repo import ProjectRepository
    from cytomind.domain.pipeline import StepRun
    from cytomind.domain.qc import EntityQCStatus, QCTestRecord
    from anndata import AnnData
    from pathlib import Path
    PathLike = Path | str
    from plotly.graph_objects import Figure
else:
    ProjectRepository = object
    StepRun = object
    EntityQCStatus = object
    QCTestRecord = object
    AnnData = object
    PathLike = object
    Figure = object


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

    Concrete implementations should:
    - Set test_type and test_name class attributes
    - Define default_config and default_thresholds
    - Implement fit/classify/plot/make_key for their specific entity type
    - Use **kwargs for entity-specific dimensions (donors, parents, receivers, etc.)
    """

    test_type: str  # type of test (e.g. "compensation", "gate_fit", etc.)
    test_name: str  # name of the test (e.g. "donor_high_correlation", "gate_event_count", etc.)
    default_config: dict[str, Any] = {}      # Default config parameters for the tester
    default_thresholds: dict[str, Any] = {}  # Default thresholds for classifying test results

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        self.metadata = cfg
        thres = dict(self.default_thresholds)
        if thresholds:
            thres.update(thresholds)
        self.thresholds = thres

    def _check_test_record(self, test: QCTestRecord):
        if test.test_type != self.test_type:
            raise ValueError(f"Test record has type '{test.test_type}' but expected '{self.test_type}'")
        if test.test_name != self.test_name:
            raise ValueError(f"Test record has name '{test.test_name}' but expected '{self.test_name}'")

    @abstractmethod
    def fit(self, entity: Any, adata: AnnData,  **kwargs) -> tuple[Hashable, QCTestRecord]:
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
        key : Hashable
            Unique identifier for this test instance (from make_key())
        test : QCTestRecord
            Test record with:
            - test_type, test_name: from class attributes
            - metadata: context from kwargs
            - metrics: computed values
            - status: "PENDING"
        plot_data_dict : dict[str, Any]
            Additional data for plotting (empty dict if plot_data=False)
            Can include masks, histograms, fitted curves, etc.
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
    ) -> tuple[Hashable, QCTestRecord]:
        """Convenience method to run fit and classify sequentially."""
        key, test = self.fit(entity, adata, plot_data=plot_data, **kwargs)
        classified_test = self.classify(test, **classify_kwargs)
        return key, classified_test

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

    @abstractmethod
    def make_key(self, test: QCTestRecord) -> Hashable:
        """
        Generate unique key for this test instance.

        Parameters
        ----------
        test : QCTestRecord
            Test record from fit() with status="PENDING"

        Returns
        -------
        Hashable
            Unique key for storing test in QCStepStatus.tests
            Examples: channel_name, gate_id, (donor_channel, receiver_channel)
        """
        pass

    @classmethod
    def from_dict(cls, test: QCTestRecord | Mapping[str, Any]) -> QCTester:
        """Factory method to create tester from dict config."""
        if isinstance(test, QCTestRecord):
            test_dict = {
                "test_type": test.test_type,
                "test_name": test.test_name,
                **test.metadata,
                **test.thresholds,
            }
        else:
            test_dict = dict(test)
        if test_dict["test_type"] != cls.test_type:
            raise ValueError(f"Cannot create tester of type '{cls.test_type}' from test record with type '{test_dict['test_type']}'")
        if test_dict["test_name"] != cls.test_name:
            raise ValueError(f"Cannot create tester of type '{cls.test_type}' with name '{cls.test_name}' from test record with name '{test_dict['test_name']}'")

        thres = test_dict.pop("thresholds", {})
        return cls(config=test_dict, thresholds=thres)

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
    """

    entity_type: str = ""
    default_config: dict[str, Any] = {}

    def __init__(self, repo: ProjectRepository, config: Mapping[str, Any] | None = None):
        self.repo = repo
        self.project = repo.load_project()
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        self.config = cfg

    @abstractmethod
    def run_entity_qc(
        self,
        entity_id: str,
        *,
        sample_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EntityQCStatus:
        """Run QC for a specific entity instance."""
        pass

    def run_product_qc(
        self,
        step_run: StepRun,
    ) -> dict[str, EntityQCStatus]:
        """
        Optional hook: Evaluate products created by this entity.

        For step entities: evaluate compensations, gating strategies, etc. created by step.
        For other entities: no-op (return empty dict).

        Parameters
        ----------
        step_run : StepRun
            Step run associated with the entity (only for step entities)

        Returns
        -------
        dict[str, EntityQCStatus]
            Mapping product_entity_id -> EntityQCStatus
        """
        results: dict[str, EntityQCStatus] = {}
        sample_ids = step_run.inputs.get("sample_ids")

        # Loop through all project updates
        for update in step_run.project_updates:
            # Loop through each key-value pair in the update
            for entity_type, payload in update.items():

                # Check if evaluator exists for this entity type
                evaluator_class = EntityQCEvaluatorRegistry.get(entity_type)
                if not evaluator_class:
                    continue

                # Extract entity IDs from payload (handle both list and dict formats)
                if isinstance(payload, dict):
                    entity_ids = list(payload.keys())
                else:
                    entity_ids = [
                        e.id if hasattr(e, 'id') else e
                        for e in payload
                    ]

                # Run QC for each entity
                evaluator = evaluator_class(self.repo, config=self.config)
                for entity_id in entity_ids:
                    qc_status = evaluator.run_entity_qc(
                        entity_id,
                        sample_ids=sample_ids,
                        context={"trigger": "step", "step_id": step_run.id},
                    )
                    if qc_status:
                        results[entity_id] = qc_status

        return results


    def summarize(
        self,
        entity_qc: EntityQCStatus,
    ) -> dict[str, Any]:
        """
        Generate user-facing summary for review UI.

        Transforms detailed QC data into formatted tables, metrics,
        and recommendations suitable for user review.

        Parameters
        ----------
        repo : ProjectRepository
            Repository for reading data
        entity_id : str
            ID of the entity
        entity_qc : EntityQCStatus
            QC status with detailed test records

        Returns
        -------
        dict
            User-facing summary with tables, metrics, recommendations
        """
        if entity_qc.entity_type != self.entity_type:
            raise TypeError(f"EntityQCEvaluator for '{self.entity_type}' cannot summarize QC for entity type '{entity_qc.entity_type}'")

        sample_qc = entity_qc.sample_qc
        per_sample_flags = {sid: flag.value for sid, flag in entity_qc.sample_flags().items()}
        sample_counts = Counter(qc.overall_flag.value for qc in sample_qc.values())

        # Build test summary by counting status for each test_name
        test_summary: dict[str, dict[Hashable, dict[str, int]]] = {}
        for sample_id in sample_qc:
            for step_name, test_key, test in entity_qc.iter_sample_tests(sample_id):
                if step_name not in test_summary:
                    test_summary[step_name] = {test_key: {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}}
                if test_key not in test_summary[step_name]:
                    test_summary[step_name][test_key] = {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}
                test_summary[step_name][test_key][test.status] += 1

        return {
            "overall": entity_qc.overall_flag.value,
            "batch": entity_qc.batch_qc.overall_flag.value if entity_qc.batch_qc else None,
            "n_samples": sample_counts.total(),
            "n_pass": sample_counts["PASS"],
            "n_warn": sample_counts["WARN"],
            "n_severe": sample_counts["SEVERE"],
            "n_fail": sample_counts["FAIL"],
            "per_sample_flags": per_sample_flags,
            "test_summary": test_summary,
        }
