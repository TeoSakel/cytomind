"""
Gating Strategy QC Evaluator.

Performs QC analysis on gating strategies by evaluating event counts and ratios
for each gate, including ratio to total events and ratio to parent gate events.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, TYPE_CHECKING
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.domain.gates import GatingStrategyRef

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester

if TYPE_CHECKING:
    from anndata import AnnData
    from cytomind.infra.repo import ProjectRepository
    from cytomind.domain.constants import PathLike
else:
    AnnData = object
    ProjectRepository = object
    PathLike = object


# ============================================================================
# Validator helpers for threshold checks
# ============================================================================

def _validate_percentage_range(name: str, value: tuple[float, float]) -> None:
    """Raise ValueError if values are not valid percentages."""
    if value is None or len(value) != 2:
        return
    if not (0.0 <= value[0] <= 1.0 and 0.0 <= value[1] <= 1.0):
        raise ValueError(f"{name} must be a tuple of values in [0, 1] range.")
    if value[0] > value[1]:
        raise ValueError(f"{name}: min threshold must be <= max threshold.")


# ============================================================================
# QC Test Classes
# ============================================================================

class GateMaskEventCountTest(QCTester):
    """Test for event counts and ratios in a gated region."""

    test_type = "gating_mask"
    test_name = "gate_mask_event_count"
    key_fields = ("gating_strategy_id", "gate_id", "mask_key")
    default_config = {}
    default_thresholds = {
        "ratio_total": (0.0, 1.0),      # min, max ratio relative to total events
        "ratio_parent": (0.0, 1.0),      # min, max ratio relative to parent gate events
    }
    plot_type = "bar"
    plot_description = "Gate event counts and ratios relative to total and parent gate"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config=config, thresholds=thresholds)
        _validate_percentage_range("ratio_total", self.thresholds.get("ratio_total", (0.0, 1.0)))
        _validate_percentage_range("ratio_parent", self.thresholds.get("ratio_parent", (0.0, 1.0)))

    def fit(
        self,
        entity: GatingStrategyRef,
        adata: AnnData,
        gate_id: str | None = None,
        mask_key: str | None = None,
        parent_mask: dict[str, Any] | None = None,
        **kwargs
    ) -> QCTestRecord:
        """Compute test metrics for a gate mask.

        Parameters
        ----------
        entity : GatingStrategyRef
            The gating strategy entity
        adata : AnnData
            Event data with gate masks in adata.obs
        gate_id : str
            Gate identifier
        mask_key : str
            Key in adata.obs containing the boolean mask for this gate
        parent_mask : dict[str, Any]
            Optional parent mask to compute ratio relative to parent
        """
        if gate_id is None or mask_key is None:
            raise ValueError("Gate ID and mask key must be provided for GateMaskEventCountTest.")

        metadata = {
            "gating_strategy_id": entity.id,
            "gate_id": gate_id,
            "mask_key": mask_key,
        }
        test = QCTestRecord(
            id=self.make_key(metadata),
            test_type=self.test_type,
            test_name=self.test_name,
            metadata=metadata,
            metrics={
                "n_events_passing": 0,
                "n_events_total": adata.n_obs,
                "ratio_total": 0.0,
                "ratio_parent": np.nan,  # Will be NaN if no parent mask
            },
            status="PENDING",
        )

        if adata.X is None or adata.n_obs == 0:
            test.status = "SKIP"
            test.message = "No events to test."
            return test

        # Extract mask if it exists in adata.obs
        if mask_key not in adata.obs.columns:
            test.status = "SKIP"
            test.message = f"Mask '{mask_key}' not found in adata.obs."
            return test

        # Get the mask
        gate_mask = np.array(adata.obs[mask_key].astype(bool))

        # Calculate basic metrics
        n_passing = np.sum(gate_mask)
        test.metrics["n_events_passing"] = int(n_passing)
        test.metrics["n_events_total"] = int(adata.n_obs)

        # Ratio relative to total
        test.metrics["ratio_total"] = float(n_passing / adata.n_obs) if adata.n_obs > 0 else 0.0

        # Ratio relative to parent (if parent_mask provided)
        if parent_mask is not None and len(parent_mask) > 0:
            parent_mask_array = np.array(next(iter(parent_mask.values())).astype(bool))
            n_parent = np.sum(parent_mask_array)
            if n_parent > 0:
                # Count events that pass both parent and gate masks
                n_passing_in_parent = np.sum(gate_mask[parent_mask_array])
                test.metrics["ratio_parent"] = float(n_passing_in_parent / n_parent)
                test.metadata["n_parent_events"] = int(n_parent)
        else:
            # If no parent, ratio_parent is same as ratio_total
            test.metrics["ratio_parent"] = test.metrics["ratio_total"]
            test.metadata["n_parent_events"] = adata.n_obs

        return test

    def classify(
        self,
        test: QCTestRecord,
        ratio_total_min: float | None = None,
        ratio_total_max: float | None = None,
        ratio_parent_min: float | None = None,
        ratio_parent_max: float | None = None,
        **kwargs
    ) -> QCTestRecord:
        """Classify test results based on thresholds.

        Parameters
        ----------
        test : QCTestRecord
            Test record from fit()
        ratio_total_min : float | None
            Min threshold for ratio_total (default: 0.0)
        ratio_total_max : float | None
            Max threshold for ratio_total (default: 1.0)
        ratio_parent_min : float | None
            Min threshold for ratio_parent (default: 0.0)
        ratio_parent_max : float | None
            Max threshold for ratio_parent (default: 1.0)
        """
        _ratio_total_min = ratio_total_min if ratio_total_min is not None else self.thresholds["ratio_total"][0]
        _ratio_total_max = ratio_total_max if ratio_total_max is not None else self.thresholds["ratio_total"][1]
        _ratio_parent_min = ratio_parent_min if ratio_parent_min is not None else self.thresholds["ratio_parent"][0]
        _ratio_parent_max = ratio_parent_max if ratio_parent_max is not None else self.thresholds["ratio_parent"][1]

        _validate_percentage_range("ratio_total", (_ratio_total_min, _ratio_total_max))
        _validate_percentage_range("ratio_parent", (_ratio_parent_min, _ratio_parent_max))

        test.thresholds["ratio_total"] = (_ratio_total_min, _ratio_total_max)
        test.thresholds["ratio_parent"] = (_ratio_parent_min, _ratio_parent_max)

        if test.status == "SKIP":
            return test

        ratio_total = test.metrics["ratio_total"]
        ratio_parent = test.metrics["ratio_parent"]

        # Check against thresholds
        issues = []
        if ratio_total < _ratio_total_min:
            issues.append(
                f"Event ratio to total below minimum: {ratio_total:.2%} < {_ratio_total_min:.2%}"
            )
        if ratio_total > _ratio_total_max:
            issues.append(
                f"Event ratio to total above maximum: {ratio_total:.2%} > {_ratio_total_max:.2%}"
            )

        if not np.isnan(ratio_parent):
            if ratio_parent < _ratio_parent_min:
                issues.append(
                    f"Event ratio to parent below minimum: {ratio_parent:.2%} < {_ratio_parent_min:.2%}"
                )
            if ratio_parent > _ratio_parent_max:
                issues.append(
                    f"Event ratio to parent above maximum: {ratio_parent:.2%} > {_ratio_parent_max:.2%}"
                )

        if issues:
            test.status = "WARN"
            test.message = "; ".join(issues)
        else:
            test.status = "PASS"

        return test

    def plot(
        self,
        adata: AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Generate diagnostic plot for gate mask event counts.

        Creates a simple figure showing event counts and ratios.
        """
        gate_id = test.metadata.get("gate_id", "Unknown")
        mask_key = test.metadata.get("mask_key", "Unknown")

        n_passing = test.metrics.get("n_events_passing", 0)
        n_total = test.metrics.get("n_events_total", 0)
        n_not_passing = n_total - n_passing

        ratio_total = test.metrics.get("ratio_total", 0.0)
        ratio_parent = test.metrics.get("ratio_parent", np.nan)

        # Create bar chart with event counts
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Passing Gate",
            x=[gate_id],
            y=[n_passing],
            text=[f"{n_passing:,}<br>({ratio_total:.1%})"],
            textposition="outside",
            marker=dict(color="rgba(0, 128, 0, 0.7)"),
        ))

        fig.add_trace(go.Bar(
            name="Not Passing Gate",
            x=[gate_id],
            y=[n_not_passing],
            text=[f"{n_not_passing:,}"],
            textposition="outside",
            marker=dict(color="rgba(192, 192, 192, 0.7)"),
        ))

        fig.update_layout(
            title=f"Gate: {gate_id} (Mask: {mask_key})",
            xaxis_title="Gate",
            yaxis_title="Event Count",
            barmode="stack",
            hovermode="closest",
            height=400,
        )

        # Add threshold annotation if available
        if not np.isnan(ratio_parent):
            fig.add_annotation(
                text=f"Ratio to parent: {ratio_parent:.2%}",
                xref="paper", yref="paper",
                x=0.5, y=-0.2,
                showarrow=False,
                font=dict(size=12),
            )

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


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
    }
    _supported_figures = {}  # Test plots are auto-discovered from registered tests

    default_config = {
        "ratio_total_min": 0.0,
        "ratio_total_max": 1.0,
        "ratio_parent_min": 0.0,
        "ratio_parent_max": 1.0,
    }

    def get_tests(self, entity: GatingStrategyRef | None = None) -> dict[str, type[QCTester]]:
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
            "gate_mask_event_count": GateMaskEventCountTest,
        }

    def required_layer(self, entity: GatingStrategyRef | None = None) -> str | None:
        """Return the required AnnData layer for gating strategy QC.

        Gating strategies typically require the gated layer with mask information.
        """
        return None  # Gates work with any layer; layer is determined by the gate definition

    def load_entity(self, repo: ProjectRepository, entity_id: Hashable) -> GatingStrategyRef:
        """Load a gating strategy from the repository."""
        return repo.load_gating_strategy(str(entity_id))

    def update_sample_qc(
        self,
        entity: GatingStrategyRef,
        entity_qc: EntityQCStatus,
        sample_data: Iterable[tuple[str, AnnData]] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Evaluate gating strategy QC against sample data.

        Parameters
        ----------
        entity : GatingStrategyRef
            The gating strategy entity to evaluate
        entity_qc : EntityQCStatus
            QC status to update
        sample_data : Iterable[tuple[str, AnnData]]
            Iterable of (sample_id, adata) tuples with gate masks in adata.obs
        context : dict[str, Any]
            Optional evaluation context

        Returns
        -------
        EntityQCStatus
            Updated QC status with test results
        """
        config = self.config.copy()
        config.update(context)
        entity_qc.context = config

        if sample_data is None:
            return entity_qc

        for sample_id, adata in sample_data:
            self._evaluate_gating_strategy(
                entity=entity,
                adata=adata,
                entity_qc=entity_qc,
                sample_id=sample_id,
                config=config,
            )

        return entity_qc

    def update_batch_qc(
        self,
        entity: GatingStrategyRef,
        entity_qc: EntityQCStatus,
        all_samples: Iterable[tuple[str, AnnData]] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Update batch-level QC tests for gating strategy.

        Gating strategies have no batch-level tests, so this is a no-op.
        """
        return entity_qc

    def summarize_entity_qc(self, entity_qc: EntityQCStatus) -> dict[str, Any]:
        """Generate summary for gating strategy QC."""
        return {}

    def _evaluate_gating_strategy(
        self,
        entity: GatingStrategyRef,
        adata: AnnData,
        entity_qc: EntityQCStatus,
        sample_id: str,
        config: Mapping[str, Any],
    ) -> None:
        """Evaluate all gates in the gating strategy for a sample.

        Parameters
        ----------
        entity : GatingStrategyRef
            The gating strategy
        adata : AnnData
            Sample data with gate masks in adata.obs
        entity_qc : EntityQCStatus
            QC status to update
        sample_id : str
            Sample identifier
        config : Mapping[str, Any]
            Evaluation config with threshold settings
        """
        # Create tester with config thresholds
        tester = GateMaskEventCountTest(
            thresholds={
                "ratio_total": (
                    config.get("ratio_total_min", self.default_config["ratio_total_min"]),
                    config.get("ratio_total_max", self.default_config["ratio_total_max"]),
                ),
                "ratio_parent": (
                    config.get("ratio_parent_min", self.default_config["ratio_parent_min"]),
                    config.get("ratio_parent_max", self.default_config["ratio_parent_max"]),
                ),
            }
        )

        # Get QC step for this sample
        step = entity_qc.get_sample_steps(sample_id).get_step("GATING_STRATEGY_QC")

        # Iterate through gates in the gating strategy
        # For each gate, look for its mask in adata.obs
        gate_id_to_parents: dict[str, list[str]] = {}

        # Build parent mapping
        try:
            for gate_node in entity.iter_nodes():
                gate_id_to_parents[gate_node.id] = gate_node.parent_ids.copy()
        except (ValueError, FileNotFoundError) as e:
            # Graph not available, skip batch evaluation
            step.add_reason(
                code="GATING_STRATEGY_LOAD_ERROR",
                message=f"Could not load gating strategy graph: {e}",
            )
            return

        # Evaluate each gate
        for gate_id, parent_ids in gate_id_to_parents.items():
            # Look for mask column in adata.obs
            mask_key = f"{gate_id}.pos"
            if mask_key not in adata.obs.columns:
                # Try without suffix
                if gate_id not in adata.obs.columns:
                    continue
                mask_key = gate_id

            # Get parent mask if available
            parent_mask = {}
            if parent_ids:
                for parent_id in parent_ids:
                    parent_mask_key = f"{parent_id}.pos"
                    if parent_mask_key in adata.obs.columns:
                        parent_mask[parent_mask_key] = adata.obs[parent_mask_key]
                        break  # Use first available parent mask

            # Fit and classify test
            test = tester.fit(
                entity=entity,
                adata=adata,
                gate_id=gate_id,
                mask_key=mask_key,
                parent_mask=parent_mask if parent_mask else None,
            )
            classified_test = tester.classify(test)

            # Add to step
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                step.add_reason(
                    code=f"GATING_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                step.add_test(classified_test)

    def generate_table(
        self,
        entity_qc: EntityQCStatus,
        table_type: str = "gate_event_counts",
        sample_data: Iterable[tuple[str, AnnData]] | None = None,
        table_dir: PathLike | None = None,
    ) -> pd.DataFrame:
        """Generate a table from gating strategy QC results.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object
        table_type : str
            Type of table: "gate_event_counts" or "gate_ratios"
        sample_data : Iterable[tuple[str, AnnData]] | None
            Optional sample data for filtering results
        table_dir : PathLike | None
            Optional directory to save the table

        Returns
        -------
        pd.DataFrame
            Table with gate QC results
        """
        if sample_data is None:
            sample_filter = set(entity_qc.sample_qc.keys())
        else:
            sample_filter = set(sid for sid, _ in sample_data)

        records = []
        for sample_id, sample_run in entity_qc.sample_qc.items():
            if sample_id not in sample_filter:
                continue

            for step in sample_run.steps.values():
                for test in step.tests.values():
                    if test.test_type != "gating_mask":
                        continue

                    record = {
                        "sample_id": sample_id,
                        "gate_id": test.metadata.get("gate_id"),
                        "status": test.status,
                        "n_events_passing": test.metrics.get("n_events_passing", 0),
                        "n_events_total": test.metrics.get("n_events_total", 0),
                        "ratio_total": test.metrics.get("ratio_total", 0.0),
                        "ratio_parent": test.metrics.get("ratio_parent", np.nan),
                    }
                    records.append(record)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)

        if table_type == "gate_event_counts":
            return df[["sample_id", "gate_id", "n_events_passing", "n_events_total", "status"]]
        elif table_type == "gate_ratios":
            return df[["sample_id", "gate_id", "ratio_total", "ratio_parent", "status"]]
        else:
            raise ValueError(
                f"Unknown table_type '{table_type}'. Must be one of: "
                "'gate_event_counts', 'gate_ratios'"
            )

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Any,
        sample_data: Iterable[tuple[str, AnnData]] | None = None,
        step_id: str | None = None,
        figure_dir: PathLike | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Generate a diagnostic figure for gating strategy QC.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object
        test_key : Any
            Test key identifying which gate/test to visualize
        sample_data : Iterable[tuple[str, AnnData]] | None
            Sample data for visualization
        step_id : str | None
            Optional step ID to narrow search
        figure_dir : PathLike | None
            Optional directory to save figure

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if sample_data is None:
            raise ValueError("generate_figure requires sample_data to be provided")

        sample_data = iter(sample_data)
        sample_id, adata = next(sample_data)

        # Retrieve test from QC status
        if sample_id not in entity_qc.sample_qc:
            raise KeyError(f"Sample {sample_id} not found in QC status")

        sample_run = entity_qc.sample_qc[sample_id]
        tester_class = GateMaskEventCountTest
        tester = tester_class()
        test_key_tuple = tester.make_key(dict(test_key) if isinstance(test_key, Mapping) else test_key)

        # Find test record
        test = None
        if step_id is not None:
            step = sample_run.steps.get(step_id)
            if step and test_key_tuple in step.tests:
                test = step.tests[test_key_tuple]
        else:
            for step in sample_run.steps.values():
                if test_key_tuple in step.tests:
                    test = step.tests[test_key_tuple]
                    break

        if test is None:
            raise KeyError(f"Test {test_key_tuple} not found for sample {sample_id}")

        # Generate figure using tester
        tester = tester_class.from_dict(test)
        return tester.plot(adata=adata, test=test)
