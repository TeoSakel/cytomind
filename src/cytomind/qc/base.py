"""
Entity QC evaluators and registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Mapping, Iterator, Iterable, TYPE_CHECKING
import math
import warnings

import pandas as pd

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.utils import now_iso

from . import EntityQCEvaluatorRegistry

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.domain.pipeline import StepRun
    from cytomind.infra.repo import ProjectRepository
    from cytomind.infra.dataloader import UnifiedDataLoader
    from pandas import DataFrame
    from plotly.graph_objects import Figure
else:
    StepRun = object
    PathLike = object
    UnifiedDataLoader = object
    ProjectRepository = object
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
    - Define meta_fields: list[tuple[name, description]] for metadata created in QCTestRecord.metadata
    - Define metric_fields: list[tuple[name, description]] for metrics created in QCTestRecord.metrics
    - Define default_config and default_thresholds
    - Use systematic threshold format: {metric_name: {"warn": (low, high), "severe": (low, high)}}
      where low/high can be None if no threshold in that direction
    - Implement fit/classify/plot/make_key for their specific entity type
    - If plot() is implemented, populate plot_type and plot_description
    - Use **kwargs for entity-specific dimensions (donors, parents, receivers, etc.)
    """

    test_type: str                           # Evaluator type
    test_name: str                           # name of the test
    target_keys: tuple[str, ...] = ()        # Fields from targets that identify tested entity instance(s)
    meta_keys: tuple[str, ...] = ()          # Fields from metadata that identify tested dimensions
    meta_fields: list[tuple[str, str]] = []   # [(name, description)] for metadata keys in QCTestRecord.metadata
    metric_fields: list[tuple[str, str]] = [] # [(name, description)] for metric keys in QCTestRecord.metrics
    default_config: dict[str, Any] = {}      # Default config parameters for the tester
    default_thresholds: dict[str, Any] = {}  # Default thresholds for classifying test results
                                             # Format: {metric_name: {"warn": (low, high), "severe": (low, high)}}
                                             # where low/high can be None if no threshold in that direction
    plot_type: str = ""                      # Category of plot (e.g., "histogram", "scatter", "heatmap"). Empty if no plot.
    plot_description: str = ""               # Human-readable description for frontend UI

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        # Validate threshold format
        self._validate_thresholds_format(thresholds if thresholds else self.default_thresholds)

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

    @classmethod
    def _validate_thresholds_format(cls, thresholds: Mapping[str, Any]) -> None:
        """
        Validate that thresholds follow the systematic format:
        {metric_name: {"warn": (low, high), "severe": (low, high)}}

        Parameters
        ----------
        thresholds : Mapping[str, Any]
            Threshold dictionary to validate

        Raises
        ------
        ValueError
            If thresholds don't follow the expected format
        """
        for metric_name, threshold_spec in thresholds.items():
            if not isinstance(threshold_spec, dict):
                raise ValueError(
                    f"Threshold for metric '{metric_name}' must be a dict with 'warn' and 'severe' keys, "
                    f"got {type(threshold_spec).__name__}"
                )

            required_keys = {"warn", "severe"}
            if not required_keys.issubset(threshold_spec.keys()):
                missing = required_keys - threshold_spec.keys()
                raise ValueError(
                    f"Threshold for metric '{metric_name}' missing required keys: {missing}. "
                    f"Expected format: {{'warn': (low, high), 'severe': (low, high)}}"
                )

            for level in ["warn", "severe"]:
                value = threshold_spec[level]
                if not isinstance(value, (tuple, list)):
                    raise ValueError(
                        f"Threshold '{metric_name}.{level}' must be a tuple/list (low, high), "
                        f"got {type(value).__name__}"
                    )
                if len(value) != 2:
                    raise ValueError(
                        f"Threshold '{metric_name}.{level}' must have exactly 2 elements (low, high), "
                        f"got {len(value)} elements"
                    )
                low, high = value
                if low is not None and not isinstance(low, (int, float)):
                    raise ValueError(
                        f"Threshold '{metric_name}.{level}' low bound must be numeric or None, "
                        f"got {type(low).__name__}"
                    )
                if high is not None and not isinstance(high, (int, float)):
                    raise ValueError(
                        f"Threshold '{metric_name}.{level}' high bound must be numeric or None, "
                        f"got {type(high).__name__}"
                    )
                if low is not None and high is not None and low > high:
                    raise ValueError(
                        f"Threshold '{metric_name}.{level}' low bound ({low}) cannot be greater than "
                        f"high bound ({high})"
                    )

    @property
    def key_fields(self) -> tuple[str, ...]:
        return self.target_keys + ("test_type", "test_name") + self.meta_keys

    @classmethod
    def _meta_field_names(cls) -> tuple[str, ...]:
        return tuple(name for name, _ in cls.meta_fields)

    @classmethod
    def _metric_field_names(cls) -> tuple[str, ...]:
        return tuple(name for name, _ in cls.metric_fields)

    def get_threshold(self, metric_name: str, level: str = "warn") -> tuple[float | None, float | None]:
        """
        Get threshold bounds for a specific metric and level.

        Parameters
        ----------
        metric_name : str
            Name of the metric
        level : str
            Threshold level: "warn" or "severe"

        Returns
        -------
        tuple[float | None, float | None]
            (low, high) bounds where None means no threshold in that direction

        Raises
        ------
        KeyError
            If metric_name or level not found in thresholds
        """
        if metric_name not in self.thresholds:
            raise KeyError(f"Metric '{metric_name}' not found in thresholds")
        if level not in self.thresholds[metric_name]:
            raise KeyError(f"Level '{level}' not found in thresholds for metric '{metric_name}'")
        return tuple(self.thresholds[metric_name][level])

    def check_threshold(
        self,
        value: float,
        metric_name: str,
        level: str = "warn"
    ) -> bool:
        """
        Check if a value violates the threshold for a metric at a given level.

        Parameters
        ----------
        value : float
            The metric value to check
        metric_name : str
            Name of the metric
        level : str
            Threshold level: "warn" or "severe"

        Returns
        -------
        bool
            True if value violates threshold (outside bounds), False otherwise
        """
        low, high = self.get_threshold(metric_name, level)

        if low is not None and value < low:
            return True
        if high is not None and value > high:
            return True
        return False

    def fit(self, *args, **kwargs) -> Iterable[QCTestRecord]:
        raise NotImplementedError("fit() method not implemented for this tester. This tester cannot be used for testing or plotting.")

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
        if test.status == "SKIP":
            return test

        thresholds = kwargs.get("thresholds", self.thresholds)
        self._validate_thresholds_format(thresholds)
        test.thresholds = dict(thresholds)

        severe_hits: list[str] = []
        warn_hits: list[str] = []
        skipped_metrics: list[str] = []

        for metric_name, threshold_spec in thresholds.items():
            if metric_name not in test.metrics:
                continue

            raw_value = test.metrics[metric_name]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                skipped_metrics.append(metric_name)
                continue

            if math.isnan(value):
                skipped_metrics.append(metric_name)
                continue

            severe_low, severe_high = threshold_spec["severe"]
            warn_low, warn_high = threshold_spec["warn"]

            severe = ((severe_low is not None and value < severe_low) or
                      (severe_high is not None and value > severe_high))
            warn = ((warn_low is not None and value < warn_low) or
                    (warn_high is not None and value > warn_high))

            if severe:
                severe_hits.append(metric_name)
            elif warn:
                warn_hits.append(metric_name)

        if severe_hits:
            test.status = "SEVERE"
            test.message = f"Severe threshold violation for: {', '.join(sorted(severe_hits))}"
        elif warn_hits:
            test.status = "WARN"
            test.message = f"Warning threshold violation for: {', '.join(sorted(warn_hits))}"
        elif skipped_metrics and len(skipped_metrics) == len(thresholds):
            test.status = "SKIP"
            test.message = "No valid metric values available for threshold classification."
        else:
            test.status = "PASS"
            test.message = "All metric values are within configured thresholds."

        return test

    def fit_classify(self, **kwargs) -> Iterable[QCTestRecord]:
        """
        Convenience method to run fit and classify sequentially.

        Iterates over records from fit(), classifies each, and yields classified records.
        """
        for test in self.fit(**kwargs):
            classified_test = self.classify(test, **kwargs)
            yield classified_test

    def plot(self, test: QCTestRecord, **kwargs) -> Figure:
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
        raise NotImplementedError("This tester does not support plotting.")

    def key_dict(
        self,
        targets: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        targets = targets or {}
        metadata = metadata or {}
        common = {**targets, **metadata}


        missing = [key for key in self.target_keys if key not in common]
        if missing:
            raise KeyError(f"Missing target keys: {missing}")
        missing = [key for key in self.meta_keys if key not in common]
        if missing:
            raise KeyError(f"Missing metadata keys: {missing}")

        d: dict[str, Any] = {
            "test_type": self.test_type,
            "test_name": self.test_name
        }
        keys = self.target_keys + self.meta_keys
        for key in keys: d[key] = common[key]

        return d

    def make_key(
        self,
        targets: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple:
        keys =  self.target_keys + ("test_type", "test_name") + self.meta_keys
        key_dict = self.key_dict(targets=targets, metadata=metadata)
        return tuple(key_dict[key] for key in keys)

    def to_record(self, test: QCTestRecord) -> dict[str, Any]:
        """
        Create a row record from a QCTestRecord, including key fields, status, and metrics.

        The record combines:
        - Key fields from key_dict (targets + test_type + test_name + metadata keys)
        - Status from the test record
        - All metrics from the test record

        Parameters
        ----------
        test : QCTestRecord
            The test record to convert to a table row

        Returns
        -------
        dict[str, Any]
            Record dict with keys from key_dict, 'status', and all metrics
        """
        # Build key_dict with targets and test metadata
        key_dict = self.key_dict(test.targets, test.metadata)

        # Build record starting with key_dict and status
        record = {
            **key_dict,
            "status": test.status,
        }

        # Add all metrics
        record.update(test.metrics)

        return record

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

        # meta_fields can include conditional metadata (method-specific values, etc.).
        # Only meta_keys are strictly required for record identity.

        # Validate metric_fields if defined
        if cls.metric_fields:
            for field in cls._metric_field_names():
                if field not in test.metrics:
                    raise ValueError(f"Test record is missing expected metric field '{field}' in metrics")

        # Validate thresholds format
        cls._validate_thresholds_format(test.thresholds)


# ---------------------------------------------------------------------------
# Generic scalar outlier tester
# ---------------------------------------------------------------------------

class _ScalarOutlierTester(QCTester):
    """
    Concrete QCTester produced by `make_scalar_outlier_tester`.

    Scores per-sample scalar values for outliers using either IQR or Z-score.
    Keys: (entity_id, metric_type, sample_id, metric_name)
    Metadata additionally carries `metric_value` (not a key).
    """
    # Class attributes overwritten per-instance by make_scalar_outlier_tester.
    test_type: str = "outlier_scalar"
    test_name: str = "scalar_outlier"
    target_keys: tuple = ("entity_id", "metric_type")
    meta_keys: tuple = ("sample_id", "metric_name")
    metric_fields = [("outlier_score", "Outlier score (IQR or Z-score)")]
    meta_fields = [
        ("metric_name",   "Metric being tested"),
        ("metric_value",  "Raw metric value for this sample"),
        ("sample_id",     "Sample ID"),
        ("outlier_method","Method used (iqr or zscore)"),
    ]
    default_config = {
        "min_samples":    6,
        "outlier_method": "iqr",
        "use_mad":        True,
    }
    default_thresholds = {
        "outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)},
    }
    plot_type = "box"
    plot_description = "Box plot of scalar metric outlier scores per sample"

    # Set at factory time
    _extra_static_meta: dict

    def fit(
        self,
        targets: dict,
        sample_values: dict[str, float],
        metric_name: str,
        *,
        sample_meta: dict[str, dict] | None = None,
        **kwargs,
    ) -> Iterable[QCTestRecord]:
        """
        Parameters
        ----------
        targets       : must contain the target_keys (entity_id, metric_type)
        sample_values : sample_id → scalar value
        metric_name   : name of the metric being scored
        sample_meta   : optional per-sample extra metadata to embed in the record
        """
        from cytomind.qc.utils import dict_iqr_score, dict_zscore  # local import avoids circular

        use_mad = bool(self.metadata["use_mad"])
        method  = str(self.metadata["outlier_method"])
        min_n   = int(self.metadata["min_samples"])
        thresholds = {"outlier_score": self.thresholds["outlier_score"]}
        targets = dict(targets)
        sample_meta = sample_meta or {}

        base_meta = {**self._extra_static_meta, "metric_name": metric_name}

        if len(sample_values) < min_n:
            for sid, val in sample_values.items():
                try:
                    safe_val = float(val)
                except (TypeError, ValueError):
                    safe_val = float("nan")
                meta = {**base_meta, "sample_id": sid,
                        "metric_value": safe_val, "outlier_method": method,
                        **sample_meta.get(sid, {})}
                yield QCTestRecord(
                    id=self.make_key(targets, meta),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=meta,
                    metrics={"outlier_score": float("nan")},
                    thresholds=thresholds,
                    status="SKIP",
                    message="Not enough samples for outlier detection",
                )
            return

        score_fn = dict_iqr_score if method == "iqr" else dict_zscore
        # Filter to only scalar-convertible values (some metric dicts may hold
        # nested structures like bounding-box dicts that cannot be scored).
        clean_values: dict[str, float] = {}
        for k, v in sample_values.items():
            try:
                clean_values[k] = float(v)
            except (TypeError, ValueError):
                pass
        try:
            scores, score_stats = score_fn(clean_values, use_mad=use_mad)
        except ValueError as exc:
            for sid, val in sample_values.items():
                try:
                    mv = float(val)
                except (TypeError, ValueError):
                    mv = float("nan")
                meta = {**base_meta, "sample_id": sid,
                        "metric_value": mv,
                        "outlier_method": method,
                        **sample_meta.get(sid, {})}
                yield QCTestRecord(
                    id=self.make_key(targets, meta),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=meta,
                    metrics={"outlier_score": float("nan")},
                    thresholds=thresholds,
                    status="SKIP",
                    message=str(exc),
                )
            return

        for sid, score in scores.items():
            raw_val = sample_values.get(sid)
            if raw_val is None:
                metric_value = float("nan")
            else:
                try:
                    metric_value = float(raw_val)
                except (TypeError, ValueError):
                    metric_value = float("nan")
            meta = {
                **base_meta,
                "sample_id": sid,
                "metric_value": metric_value,
                "outlier_method": method,
                **score_stats,
                **sample_meta.get(sid, {}),
            }
            yield QCTestRecord(
                id=self.make_key(targets, meta),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=meta,
                metrics={"outlier_score": score},
                thresholds=thresholds,
                status="PENDING",
            )


def make_scalar_outlier_tester(
    entity: Any,
    *,
    test_name: str,
    test_type: str | None = None,
    config: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
    extra_meta_fields: list[tuple[str, str]] | None = None,
    extra_static_meta: dict[str, Any] | None = None,
    plot_description: str = "",
) -> "_ScalarOutlierTester":
    """
    Factory that returns a configured `_ScalarOutlierTester`.

    Parameters
    ----------
    entity           : domain entity being scored (e.g. GateNode).  Used to
                       derive `test_type` and `entity_id`.
    test_name        : short snake_case identifier for this test.
        test_type        : explicit test_type string; defaults to
                           ``f"outlier_scalar_{type(entity).__name__}"``.
    config           : overrides for default_config (min_samples, outlier_method, use_mad).
    thresholds       : overrides for default_thresholds (outlier_score warn/severe).
    extra_meta_fields: additional (name, description) metadata field declarations.
    extra_static_meta: values embedded verbatim into every record's metadata.
    plot_description : human-readable description for the frontend.
    """
    entity_type = type(entity).__name__
    entity_id   = getattr(entity, "id", str(entity))

    # --- build config -------------------------------------------------
    base_cfg = dict(_ScalarOutlierTester.default_config)
    if config:
        base_cfg.update(config)

    # --- build thresholds ---------------------------------------------
    method = base_cfg.get("outlier_method", "iqr")
    if method == "iqr":
        base_thresholds = {"outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)}}
    elif method == "zscore":
        base_thresholds = {"outlier_score": {"warn": (-3.0, 3.0), "severe": (-5.0, 5.0)}}
    else:
        raise ValueError(f"outlier_method must be 'iqr' or 'zscore', got '{method}'")
    if thresholds:
        for k, v in thresholds.items():
            existing = base_thresholds.get(k, {})
            merged = {**existing, **{lv: tv for lv, tv in v.items()}} if isinstance(v, dict) else v
            base_thresholds[k] = merged

    # --- instantiate --------------------------------------------------
    # Set instance-level overrides BEFORE __init__ so validation and
    # self.metadata / self.thresholds are built from the factory-provided values.
    tester = _ScalarOutlierTester.__new__(_ScalarOutlierTester)

    tester.test_type       = test_type or f"outlier_scalar_{entity_type}"
    tester.test_name       = test_name
    tester.default_config     = base_cfg
    tester.default_thresholds = base_thresholds
    tester._extra_static_meta = dict(extra_static_meta) if extra_static_meta else {}
    tester.meta_fields = list(_ScalarOutlierTester.meta_fields) + (extra_meta_fields or [])
    tester.plot_description = (
        plot_description or f"Scalar outlier scores for {test_name} on {entity_type}"
    )
    # Run through __init__ so _validate_thresholds_format is exercised and
    # self.metadata / self.thresholds are set via the standard path.
    _ScalarOutlierTester.__init__(tester)
    return tester


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

    def prepare_artifacts(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
        """Optional hook for artifact invalidation or precomputation before QC updates."""
        return

    def artifact_root_dir(
        self,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
    ) -> Path:
        """Return the artifact root directory for an entity QC status."""
        pattern = dataloader.path_scheme.get("qc_entity_artifacts_dir", "qc/{entity_type}/{entity_id}/artifacts")
        return dataloader.root_dir / pattern.format(
            entity_type=entity_qc.entity_type,
            entity_id=entity_qc.entity_id,
        )

    def artifact_dir(
        self,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        artifact_key: str,
    ) -> Path:
        """Return a specific artifact directory for an entity QC status."""
        return self.artifact_root_dir(entity_qc, dataloader) / Path(artifact_key)

    def get_artifact_metadata(
        self,
        entity_qc: EntityQCStatus,
        artifact_key: str,
    ) -> dict[str, Any] | None:
        """Return artifact metadata stored on the QC status, if present."""
        metadata = entity_qc.artifacts.get(artifact_key)
        if isinstance(metadata, dict):
            return metadata
        return None

    def set_artifact_metadata(
        self,
        entity_qc: EntityQCStatus,
        artifact_key: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Persist lightweight artifact metadata on the QC status."""
        entity_qc.artifacts[artifact_key] = dict(metadata)

    def invalidate_artifact(
        self,
        entity_qc: EntityQCStatus,
        artifact_key: str,
    ) -> None:
        """Remove lightweight artifact metadata from the QC status."""
        entity_qc.artifacts.pop(artifact_key, None)

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
        if entity_qc is None:
            entity_qc = EntityQCStatus(
                entity_id=entity.id,
                entity_type=self.entity_type,
                generated_at=now_iso()
            )
        self.prepare_artifacts(entity, entity_qc, dataloader, dataloader_context, context=context)
        self.update_sample_qc(entity, entity_qc, dataloader, dataloader_context, context=context)
        self.update_batch_qc(entity, entity_qc, dataloader, dataloader_context, context=context)
        entity_qc.summary.update(self.basic_summary(entity_qc))
        summary_dict = self.summarize_entity_qc(entity_qc)
        if "status" in summary_dict:
            raise ValueError("Summary dict cannot contain reserved key 'status'")
        if "aggregated_flag_counts" in summary_dict:
            raise ValueError("Summary dict cannot contain reserved key 'aggregated_flag_counts'")
        entity_qc.summary.update(summary_dict)
        return entity_qc

    def update_sample_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
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
        return

    def update_batch_qc(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
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
        None
        """
        return

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
        return {}


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
        evaluable_products = step_run.evaluable_products

        # Process entity types in evaluator registry order (priority, then registration order).
        for entity_type, evaluator_class in EntityQCEvaluatorRegistry.iter_evaluators():
            entities = evaluable_products.get(entity_type)
            if not entities:
                continue

            # Preserve caller-provided order within entity type.
            for entity_id, raw_context in entities.items():
                context = dict(raw_context)
                sample_ids = context.pop("sample_ids", list(step_run.sample_outputs.keys()))

                evaluator = evaluator_class()  # TODO: consider passing entity-specific config if needed
                qc_status = evaluator.parse_step(step_run, entity_id)
                entity = evaluator.load_entity(repo._dataloader, entity_id, context=context)  # Load full entity for evaluation

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

        # Warn for unknown types encountered in step output.
        for entity_type in evaluable_products:
            if EntityQCEvaluatorRegistry.get(entity_type) is None:
                warnings.warn(
                    f"No EntityQCEvaluator registered for entity type '{entity_type}'. "
                    "Skipping QC evaluation for these products."
                )

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
            raise TypeError(
                f"EntityQCEvaluator for '{self.entity_type}' cannot summarize "
                f"QC for entity type '{entity_qc.entity_type}'"
            )

        per_sample_flags = {sid: flag.value for sid, flag in entity_qc.sample_flags.items()}
        sample_counts = Counter(qc.overall_flag.value for qc in entity_qc.sample_qc.values())

        # Build test summary by counting status for each test_name
        test_summary: dict[str, dict[tuple, dict[str, int]]] = {}
        for sample_id in entity_qc.sample_qc:
            for step_name, test_key, test in entity_qc.iter_sample_tests(sample_id):
                if step_name not in test_summary:
                    test_summary[step_name] = {
                        test_key: {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}
                    }
                if test_key not in test_summary[step_name]:
                    test_summary[step_name][test_key] = {
                        "PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0
                    }
                test_summary[step_name][test_key][test.status] += 1

        return {
            "status": {
                "overall": entity_qc.overall_flag.value,
                "batch": entity_qc.batch_qc.overall_flag.value if entity_qc.batch_qc else None,
                "per_sample": per_sample_flags,
            },
            "aggregated_flag_counts": {
                "overall": dict(sample_counts),
                "by_test": {
                    step_name: list(test_dict.items())
                    for step_name, test_dict in test_summary.items()
                },
            }
        }

    def generate_table(
        self,
        entity_qc: EntityQCStatus,
        table_type: str | None = None,
        test_name: str | None = None,
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
        test_name : str | None
            Optional specific test name to filter by. If provided, takes precedence over table_type.

        Returns
        -------
        DataFrame
            Table in long (melted) format with columns:
            - Key fields from tester.key_fields (targets, test_type, test_name, metadata keys)
            - status
            - Tester configuration metadata (if any)
            - Melted metrics as 'metric' and 'value' columns

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
        tester_instance = None

        # Get key fields from the appropriate test
        if test_name is not None:
            if test_name not in tests:
                raise ValueError(
                    f"Invalid test_name '{test_name}'. Must be one of {sorted(tests.keys())}."
                )
            tester_instance = tests[test_name]()
            id_vars = list(tester_instance.key_fields)
            table_type = table_type or tester_instance.test_type
            if table_type != tester_instance.test_type:
                raise ValueError(
                    f"test_name '{test_name}' does not match table_type '{table_type}'. "
                    f"Test '{test_name}' has type '{tester_instance.test_type}'."
                )
        elif table_type is not None:
            # Otherwise use the first test of the requested table_type
            for tester_class in tests.values():
                if tester_class.test_type == table_type:
                    id_vars = list(tester_class().key_fields)
                    break
        else:
            raise ValueError("Must specify either table_type or test_name to determine which tests to include in the table.")

        if not id_vars:
            raise ValueError(
                f"No tests found for table_type '{table_type}'"
                + (f" with test_name '{test_name}'" if test_name else "")
                + ". Cannot determine id_vars for table generation."
            )
        id_vars += ["status"]  # Always include status as an id_var for melting

        # Extract records using helper method
        records = self._extract_table_records(entity_qc, table_type, sample_ids=sample_ids, test_name=test_name)

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
                warnings.warn(
                    f"Output path {output_path} already exists "
                    "and will be updated with new samples."
                )
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
        test_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract test records across all samples and steps into a flattened list.

        This is a generalized helper method that iterates through all tests in the
        EntityQCStatus and extracts their data into records containing sample_id,
        status, test metadata, and metrics.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records.
        table_type : str
            Type of tests to extract (matches test_type field).
        sample_ids : Iterable[str] | None
            Optional list of sample IDs to filter which samples to include.
        test_name : str | None
            Optional specific test name to filter by. If provided, takes precedence over table_type
            and only tests matching this name will be included. Must be a valid test name from get_tests().

        Returns
        -------
        list[dict[str, Any]]
            List of records, one per test, where each record contains:
            - Key fields from tester.key_dict() (test_type, test_name, targets, metadata keys)
            - "status": str
            - All test metrics flattened as individual columns
        """
        # Use sample_ids from argument if provided, otherwise use all samples
        sample_filter = set(sample_ids or entity_qc.sample_qc.keys())

        # Get all available tests and validate parameters
        tests = self.get_tests(entity=None)

        # Validate test_name against available test keys if specified
        if test_name is not None:
            if test_name not in tests:
                valid_names = sorted(tests.keys())
                raise ValueError(
                    f"Invalid test_name '{test_name}'. Must be one of {valid_names}."
                )
        else:
            # Validate table_type
            test_types = self.get_test_types()
            if table_type not in test_types:
                raise ValueError(f"Invalid table_type '{table_type}'. Must be one of {test_types}.")

        records = []

        def append_test_record(test: QCTestRecord) -> None:
            # If test_name is specified, filter by that; otherwise filter by table_type
            if test_name is not None:
                if test.test_name != test_name:
                    return
            else:
                if test.test_type != table_type:
                    return

            # Get tester class and validate test_name
            try:
                tester_class = tests[test.test_name]
            except KeyError:
                valid_names = [key for key, val in tests.items() if val.test_type == table_type]
                raise KeyError(
                    f"Unknown test_name '{test.test_name}' for table_type '{table_type}'."
                    f" Valid names are: {valid_names}"
                )

            # Create tester instance and use to_record to create the row record
            tester = tester_class.from_dict(test)
            record = tester.to_record(test)
            records.append(record)

        # Iterate through sample-level tests
        for sample_id, sample_run in entity_qc.sample_qc.items():
            if sample_id not in sample_filter:
                continue

            for step in sample_run.steps.values():
                for test in step.tests.values():
                    append_test_record(test)

        # Iterate through batch-level tests as well
        for step in entity_qc.batch_qc.steps.values():
            for test in step.tests.values():
                append_test_record(test)

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
            raise ValueError(
                "Cannot merge tables with different test types: "
                f"'{test_type_old}' vs '{test_type_new}'"
            )

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

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Mapping[str, Any],
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
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
            Optional context parameters for the dataloader (e.g., sample IDs,
            layer, etc.). Used when loading data for figure generation.
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
        raise NotImplementedError("This evaluator does not support figure generation.")

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
        return None

    @abstractmethod
    def load_entity(
        self,
        dataloader: UnifiedDataLoader,
        entity_id: Hashable,
        context: dict[str, Any] | None = None
    ) -> Any:
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
        context : dict[str, Any] | None
            Optional context dict with additional metadata for entity-specific loading.

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
            raise ValueError(
                "test_key must be either a tuple or a mapping with "
                "'test_type' and 'test_name' keys."
            )

        # Validate test_type
        if test_type not in self.get_test_types():
            raise ValueError(
                f"Unsupported test_type '{test_type}'. "
                f"Expected one of: {self.get_test_types()}"
            )

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
