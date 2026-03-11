"""
Gating Strategy QC Evaluator.

Performs QC analysis on gating strategies by evaluating event counts and ratios
for each gate, including ratio to total events and ratio to parent gate events.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, TYPE_CHECKING

import numpy as np

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.domain.gates import GatingStrategyRef
from cytomind.domain.pipeline import Project

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester

if TYPE_CHECKING:
    from cytomind.infra.dataloader import UnifiedDataLoader
else:
    UnifiedDataLoader = object


# ============================================================================
# QC Test Classes
# ============================================================================

class MultiMetricOutlierTester(QCTester):
    """Outlier test for cross-gate feature vectors using Mahalanobis distance.

    Detects samples with outlier patterns across multiple gates by computing
    Mahalanobis distance on feature vectors composed of gate-agnostic metrics
    (e.g., ratio_parent, centrality_score).

    Each metric type (ratio_parent, centrality_score) is treated as a separate
    multidimensional feature where each dimension corresponds to a gate.
    """

    test_type = "strategy_batch"
    test_name = "cross_gate_outlier"
    target_keys = ("metric_type",)
    meta_keys = ("sample_id", "metric_name")
    default_config = {
        "min_samples": 3,
        "min_gates": 2,
        "use_robust_cov": True,
    }
    meta_fields = [
           ("sample_id", "Sample ID"),
           ("metric_name", "Name of cross-gate metric"),
           ("n_features", "Number of gates used as features"),
           ("n_gates", "Total number of gates"),
           ("n_valid_gates", "Number of gates with valid data"),
           ("feature_names", "Names of valid gate features"),
       ]
    metric_fields = [("mahalanobis_distance", "Mahalanobis distance in cross-gate feature space")]
    default_thresholds = {
            "mahalanobis_distance": {"warn": (None, 3.0), "severe": (None, 4.0)},
       }
    plot_type = "scatter"
    plot_description = "Scatter plot of Mahalanobis distances for cross-gate outlier detection"

    def fit(
        self,
        targets: dict[str, Any],
        feature_data: dict[str, dict[str, dict[str, float]]],
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute Mahalanobis distance for each sample based on cross-gate features.

        Each metric type (ratio_parent, centrality_score) is processed separately,
        treating gates as dimensions in a multidimensional feature space.

        Parameters
        ----------
        targets : dict[str, Any]
            Target identifiers (metric_type)
        feature_data : dict[str, dict[str, dict[str, float]]]
            Nested dict: sample_id → gate_id → {ratio_parent, centrality_score}
        **kwargs
            Additional parameters (ignored)

        Yields
        ------
        QCTestRecord
            Test record for each sample and metric with mahalanobis_distance metric
        """
        metadata = self.metadata.copy()
        thresholds = self.thresholds.copy()

        metric_type = targets.get("metric_type", "cross_gate")

        min_samples = int(metadata["min_samples"])
        min_gates = int(metadata["min_gates"])
        use_robust_cov = bool(metadata["use_robust_cov"])

        # Process each metric separately as a multidimensional feature
        for metric_name, metric_data in feature_data.items():
            # Collect all unique sample IDs and gate IDs for this metric
            sample_ids = sorted(metric_data.keys())
            if not sample_ids:
                continue

            all_gates = set()
            for sample_gates in metric_data.values():
                all_gates.update(sample_gates.keys())
            gate_ids = sorted(all_gates)

            # Build feature matrix for this metric: samples × gates
            # Each gate is a dimension in the feature space
            feature_names = [f"{gate_id}_{metric_name}" for gate_id in gate_ids]

            X = np.full((len(sample_ids), len(gate_ids)), np.nan, dtype=float)
            for i, sample_id in enumerate(sample_ids):
                sample_gates = metric_data[sample_id]
                for j, gate_id in enumerate(gate_ids):
                    X[i, j] = sample_gates.get(gate_id, np.nan)

            # Check minimum samples
            if len(sample_ids) < min_samples:
                for sample_id in sample_ids:
                    meta = {**metadata, "sample_id": sample_id, "metric_name": metric_name}
                    tgt = {**targets, "metric_type": metric_type}
                    yield QCTestRecord(
                        id=self.make_key(tgt, meta),
                        test_type=self.test_type,
                        test_name=self.test_name,
                        targets=tgt,
                        metadata=meta,
                        metrics={"mahalanobis_distance": float('nan')},
                        thresholds=thresholds,
                        status="SKIP",
                        message=f"Not enough samples for cross-gate outlier detection (n={len(sample_ids)}, min={min_samples})",
                    )
                continue

            # Remove features (gates) with too many missing values (>50% missing)
            valid_gates = np.sum(~np.isnan(X), axis=0) > (len(sample_ids) / 2)
            n_valid_gates = np.sum(valid_gates)

            if n_valid_gates < min_gates:
                for sample_id in sample_ids:
                    meta = {**metadata, "sample_id": sample_id, "metric_name": metric_name}
                    tgt = {**targets, "metric_type": metric_type}
                    yield QCTestRecord(
                        id=self.make_key(tgt, meta),
                        test_type=self.test_type,
                        test_name=self.test_name,
                        targets=tgt,
                        metadata=meta,
                        metrics={"mahalanobis_distance": float('nan')},
                        thresholds=thresholds,
                        status="SKIP",
                        message=f"Not enough gates with valid {metric_name} for cross-gate outlier detection (n={n_valid_gates}, min={min_gates})",
                    )
                continue

            X_valid = X[:, valid_gates]
            valid_feature_names = [name for name, valid in zip(feature_names, valid_gates) if valid]

            # Impute remaining missing values with column median
            for j in range(X_valid.shape[1]):
                col = X_valid[:, j]
                valid_mask = ~np.isnan(col)
                if np.any(valid_mask):
                    col[~valid_mask] = np.median(col[valid_mask])

            # Check if we still have variance
            if X_valid.shape[1] < 2 or np.any(np.std(X_valid, axis=0) == 0):
                for sample_id in sample_ids:
                    meta = {**metadata, "sample_id": sample_id, "metric_name": metric_name}
                    tgt = {**targets, "metric_type": metric_type}
                    yield QCTestRecord(
                        id=self.make_key(tgt, meta),
                        test_type=self.test_type,
                        test_name=self.test_name,
                        targets=tgt,
                        metadata=meta,
                        metrics={"mahalanobis_distance": float('nan')},
                        thresholds=thresholds,
                        status="SKIP",
                        message=f"Insufficient variance in {metric_name} for cross-gate outlier detection",
                    )
                continue

            # Compute Mahalanobis distance for this metric
            try:
                if use_robust_cov:
                    from sklearn.covariance import MinCovDet
                    mcd = MinCovDet(store_precision=True, random_state=42)
                    mcd.fit(X_valid)
                    center = mcd.location_
                    precision = mcd.get_precision()
                else:
                    center = np.mean(X_valid, axis=0)
                    cov = np.cov(X_valid, rowvar=False)
                    precision = np.linalg.inv(cov)

                # Compute Mahalanobis distance for each sample
                diff = X_valid - center
                mahal_dist = np.sqrt(np.sum(diff @ precision * diff, axis=1))

            except (np.linalg.LinAlgError, ValueError) as e:
                for sample_id in sample_ids:
                    meta = {**metadata, "sample_id": sample_id, "metric_name": metric_name}
                    tgt = {**targets, "metric_type": metric_type}
                    yield QCTestRecord(
                        id=self.make_key(tgt, meta),
                        test_type=self.test_type,
                        test_name=self.test_name,
                        targets=tgt,
                        metadata=meta,
                        metrics={"mahalanobis_distance": float('nan')},
                        thresholds=thresholds,
                        status="SKIP",
                        message=f"Numerical error in Mahalanobis distance computation for {metric_name}: {e}",
                    )
                continue

            # Yield test records for this metric
            for i, sample_id in enumerate(sample_ids):
                distance = float(mahal_dist[i])
                meta = {
                    **metadata,
                    "sample_id": sample_id,
                    "metric_name": metric_name,
                    "n_features": int(X_valid.shape[1]),
                    "n_gates": len(gate_ids),
                    "n_valid_gates": n_valid_gates,
                    "feature_names": valid_feature_names,
                }
                tgt = {**targets, "metric_type": metric_type}

                yield QCTestRecord(
                    id=self.make_key(tgt, meta),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=tgt,
                    metadata=meta,
                    metrics={"mahalanobis_distance": distance},
                    thresholds=thresholds,
                    status="PENDING",
                )


@EntityQCEvaluatorRegistry.register("gating_strategy")
class GatingStrategyQCEvaluator(EntityQCEvaluator):
    """QC evaluator for gating strategy entities."""

    entity_type = "gating_strategy"
    _supported_tables = {
        "gate_event_counts": {
            "description": "Event counts per gate across samples",
            "input_params": {}
        },
        "gate_ratios": {
            "description": "Ratios (to total, to parent) per gate across samples",
            "input_params": {}
        },
        "gate_fitting_diagnostics": {
            "description": "Gate fitting quality diagnostics (r_squared, residual_std, n_outliers) per gate and sample",
            "input_params": {}
        },
        "sample_outliers": {
            "description": "Samples flagged as outliers for gate event metrics",
            "input_params": {}
        },
    }
    _supported_figures = {}  # Test plots are auto-discovered from registered tests

    default_config = {
        "ratio_total_min": 0.0,
        "ratio_total_max": 1.0,
        "ratio_parent_min": 0.0,
        "ratio_parent_max": 1.0,
        # Gate fitting quality thresholds
        "r_squared_min": 0.7,
        "residual_std_max": 1.0,
        "n_outliers_max": 100,
        # Cross-gate outlier detection (Mahalanobis)
        "min_samples": 3,
        "min_gates": 2,
        "mahalanobis_thresholds": {
            "warn": (None, 3.0),
            "severe": (None, 4.0),
        },
        "use_robust_cov": True,
    }

    @classmethod
    def get_tests(cls, entity: GatingStrategyRef | None = None) -> dict[str, type[QCTester]]:
        """Return dictionary of test classes for gating strategy QC.

        Parameters
        ----------
        entity : GatingStrategyRef | None
            Gating strategy entity (optional, can be used for entity-specific tests)

        Returns
        -------
        dict[str, type[QCTester]]
            Mapping of test_name → QCTester subclass
        """
        return {
            MultiMetricOutlierTester.test_name: MultiMetricOutlierTester,
        }

    def load_entity(self, dataloader: UnifiedDataLoader, entity_id: Hashable, context: dict[str, Any] | None = None) -> GatingStrategyRef:
        """Load a gating strategy from the dataloader."""
        return dataloader.load_data("project", parse_func=Project.from_dict).gating_strategy

    def update_batch_qc(
        self,
        entity: GatingStrategyRef,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
        """Update batch-level QC tests for gating strategy.

        Performs cross-gate outlier detection using Mahalanobis distance on
        feature vectors composed of ratio_parent and centrality_score for each gate.
        """
        config = self.config.copy()
        config.update(context)

        # Get batch QC step
        batch_step = entity_qc.batch_qc.get_step("GATING_STRATEGY_BATCH_QC")

        # Extract cross-gate feature vectors for each sample
        feature_data = self._extract_cross_gate_features(entity_qc)

        if not feature_data: return

        # Run Mahalanobis distance-based outlier detection
        maha_thresholds = {
            "mahalanobis_distance": dict(
                config.get("mahalanobis_thresholds", self.default_config["mahalanobis_thresholds"])
            )
        }
        outlier_tester = MultiMetricOutlierTester(
            config={
                "min_samples": config.get("min_samples", self.default_config["min_samples"]),
                "min_gates": config.get("min_gates", self.default_config["min_gates"]),
                "use_robust_cov": config.get("use_robust_cov", self.default_config["use_robust_cov"]),
            },
            thresholds=maha_thresholds,
        )

        for test in outlier_tester.fit_classify(
            targets={"metric_type": "cross_gate"},
            feature_data=feature_data,
        ):
            if test.status in {"WARN", "SEVERE", "FAIL"}:
                batch_step.add_reason(
                    code=f"CROSS_GATE_OUTLIER_{test.status}",
                    message=test.message,
                    tests=[test],
                )
            else:
                batch_step.add_test(test)

    def _extract_cross_gate_features(self, entity_qc: EntityQCStatus) -> dict[str, dict[str, dict[str, float]]]:
        """Extract cross-gate feature vectors for each sample.

        Returns nested dict: metric_name → sample_id → gate_id → value
        """
        feature_data: dict[str, dict[str, dict[str, float]]] = {
            "ratio_parent": {},
            "centrality_score": {},
        }

        # Iterate through all sample QC records
        for sample_id, sample_qc_run in entity_qc.sample_qc.items():
            for step in sample_qc_run.steps.values():
                for test_record in step.tests.values():
                    # Extract ratio_parent from gate event count tests
                    if test_record.test_type == "gate" and test_record.test_name == "gate_event_count":
                        gate_id = test_record.targets.get("gate_id")
                        ratio_parent = test_record.metrics.get("ratio_parent")
                        if gate_id and ratio_parent is not None:
                            if sample_id not in feature_data["ratio_parent"]:
                                feature_data["ratio_parent"][sample_id] = {}
                            feature_data["ratio_parent"][sample_id][gate_id] = float(ratio_parent)

                    # Extract centrality_score from GLM gate outlier tests
                    elif test_record.test_type == "gate_batch" and test_record.test_name == "gate_coverage_outlier":
                        gate_id = test_record.targets.get("gate_id")
                        test_sample_id = test_record.metadata.get("sample_id")
                        centrality_score = test_record.metadata.get("centrality_score")
                        if gate_id and test_sample_id == sample_id and centrality_score is not None:
                            if sample_id not in feature_data["centrality_score"]:
                                feature_data["centrality_score"][sample_id] = {}
                            feature_data["centrality_score"][sample_id][gate_id] = float(centrality_score)

        return feature_data
