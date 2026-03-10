"""
Gate Node QC Evaluator.

Performs QC analysis on individual gates within their gating strategy context.
Evaluates event counts, ratios, fitting quality, and outlier detection for a single gate.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, Sequence, TYPE_CHECKING
from pathlib import Path

import numpy as np
import anndata as ad
import plotly.graph_objects as go

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.gates import GateRegistry

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester
from .utils import (
    validate_percentage_range,
    dict_iqr_score,
    dict_zscore,
)

if TYPE_CHECKING:
    from cytomind.domain.constants import BooleanArray, FloatArray, PathLike
    from cytomind.domain.gates import GateNode, GatingStrategyRef
    from cytomind.infra.dataloader import UnifiedDataLoader
else:
    BooleanArray = object
    FloatArray = object
    PathLike = object
    GateNode = object
    GatingStrategyRef = object
    UnifiedDataLoader = object


# ============================================================================
# QC Test Classes
# ============================================================================

class GateMaskRatioTest(QCTester):
    """Test for event counts and ratios in a gated region."""

    test_type = "gate"
    test_name = "gate_event_count"
    target_keys = ("strategy_id", "gate_id", "sample_id")
    meta_keys = ("parent_id", )
    default_config = {}
    meta_fields = [
        ("parent_id", "Parent gate ID(s)"),
        ("n_events_total", "Total number of events"),
        ("n_events_parent", "Number of events in parent gate"),
        ("n_events_passing", "Number of events passing this gate"),
    ]
    metric_fields = [
        ("ratio_total", "Proportion of events passing gate relative to total"),
        ("ratio_parent", "Proportion of events passing gate relative to parent"),
    ]
    default_thresholds = {
        "ratio_total": {"warn": (0.0, 1.0), "severe": (None, None)},
        "ratio_parent": {"warn": (0.0, 1.0), "severe": (None, None)},
    }
    plot_type = "bar"
    plot_description = "Gate event counts and ratios relative to total and parent gate"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config=config, thresholds=thresholds)

    def fit(
        self,
        targets: dict[str, Any],
        entity: GateNode,
        masks: dict[str, BooleanArray],
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all gates in the mask dict.

        Yields one QCTestRecord per gate.

        Parameters
        ----------
        targets : dict[str, Any]
            Base target identifiers (strategy_id, sample_id)
        entity : GateNode
            Gate node entity for the test.
        masks : dict[str, BooleanArray]
            Mapping of gate IDs to boolean masks for ratio calculation.
        **kwargs
            Additional test-specific parameters.
        """

        metadata = self.metadata.copy()  # Start with default metadata
        # Calculate basic metrics
        gate_id = entity.id
        mask = masks[gate_id]
        n_passing = int(np.sum(mask))
        n_total = int(mask.shape[0])

        # Gate Parent Mask. If multiple (Boolean Gate) combine with OR and concatenate parent IDs:
        if entity.parent_ids:
            parent_id = "|".join(sorted(entity.parent_ids))
            parent_mask = np.logical_or.reduce([masks[parent] for parent in entity.parent_ids])
            if parent_mask.shape[0] != n_total:
                raise ValueError(
                    f"Mask length mismatch for gate {targets['gate_id']}: "
                    f"gate mask length={n_total}, parent mask length={parent_mask.shape[0]}"
                )
        else:
            parent_id = "root"
            parent_mask = np.ones_like(mask)

        metadata["parent_id"] = parent_id
        metadata["n_events_total"] = n_total
        metadata["n_events_parent"] = n_parent = int(np.sum(parent_mask))
        metadata["n_events_passing"] = n_passing

        if n_total == 0:
            yield QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={
                    "ratio_total": float("nan"),
                    "ratio_parent": float("nan"),
                },
                thresholds=self.thresholds.copy(),
                status="SKIP",
                message="No events to test.",
            )
            return

        ratio_total = float(n_passing / n_total)
        ratio_parent = float("nan")
        n_passing_in_parent = float(np.sum(mask & parent_mask))
        ratio_parent = n_passing_in_parent / n_parent if n_parent else float("nan")

        test = QCTestRecord(
            id=self.make_key(targets, metadata),
            test_type=self.test_type,
            test_name=self.test_name,
            targets=targets,
            metadata=metadata,
            metrics={
                "ratio_total": ratio_total,
                "ratio_parent": ratio_parent,
            },
            thresholds=self.thresholds.copy(),
            status="PENDING",
        )

        yield test

    def plot(
        self,
        test: QCTestRecord,
        *,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Generate diagnostic plot for gate mask event counts.

        Creates two normalized stacked bars (height 1.0):
        1. Total view: shows gate/parent/other proportions relative to total events
        2. Parent view: shows gate/non-gate proportions relative to parent events

        Both bars overlay their corresponding ratio thresholds.
        """
        gate_id = test.targets.get("gate_id", "Unknown")
        parent_id = test.metadata.get("parent_id", "Unknown")

        n_total = test.metadata.get("n_events_total", 0)
        n_parent = test.metadata.get("n_events_parent", 0)
        n_passing = test.metadata.get("n_events_passing", 0)

        ratio_total = test.metrics.get("ratio_total", 0.0)
        ratio_parent = test.metrics.get("ratio_parent", 0.0)

        # Calculate event counts for stacked bars
        # For parent view: need events in gate AND parent
        if not np.isnan(ratio_parent) and n_parent > 0:
            n_passing_in_parent = int(ratio_parent * n_parent)
        else:
            n_passing_in_parent = 0

        # Normalized proportions for first bar (Total view)
        if n_total > 0:
            prop_gate_total = n_passing_in_parent / n_total
            prop_parent_not_gate_total = (n_parent - n_passing_in_parent) / n_total
            prop_not_parent_total = (n_total - n_parent) / n_total
        else:
            prop_gate_total = prop_parent_not_gate_total = prop_not_parent_total = 0

        # Normalized proportions for second bar (Parent view)
        if n_parent > 0:
            prop_gate_parent = ratio_parent
            prop_parent_not_gate_parent = 1.0 - ratio_parent
        else:
            prop_gate_parent = prop_parent_not_gate_parent = 0

        # Create figure
        fig = go.Figure()

        # First bar: Total view (3 parts)
        fig.add_trace(go.Bar(
            name="Not in parent",
            x=["Total View"],
            y=[prop_not_parent_total],
            marker=dict(color="rgba(211, 211, 211, 0.7)"),
            text=f"Other<br>{prop_not_parent_total:.1%}",
            textposition="inside",
            hovertemplate="<b>Not in parent</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Parent (not gate)",
            x=["Total View"],
            y=[prop_parent_not_gate_total],
            marker=dict(color="rgba(100, 149, 237, 0.7)"),
            text=f"Parent<br>{prop_parent_not_gate_total:.1%}",
            textposition="inside",
            hovertemplate="<b>In parent, not in gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Gate",
            x=["Total View"],
            y=[prop_gate_total],
            marker=dict(color="rgba(60, 179, 113, 0.7)"),
            text=f"Gate<br>{prop_gate_total:.1%}",
            textposition="inside",
            hovertemplate="<b>In gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        # Second bar: Parent view (2 parts)
        fig.add_trace(go.Bar(
            name="Parent (not gate)",
            x=["Parent View"],
            y=[prop_parent_not_gate_parent],
            marker=dict(color="rgba(100, 149, 237, 0.7)"),
            text=f"Parent<br>{prop_parent_not_gate_parent:.1%}",
            textposition="inside",
            showlegend=False,
            hovertemplate="<b>In parent, not in gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Gate",
            x=["Parent View"],
            y=[prop_gate_parent],
            marker=dict(color="rgba(60, 179, 113, 0.7)"),
            text=f"Gate<br>{prop_gate_parent:.1%}",
            textposition="inside",
            showlegend=False,
            hovertemplate="<b>In gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        # Add threshold lines
        thresholds = test.thresholds or {}

        # Total ratio threshold (on first bar)
        if "ratio_total" in thresholds and "warn" in thresholds["ratio_total"]:
            warn_range = thresholds["ratio_total"]["warn"]
            if warn_range and len(warn_range) == 2 and warn_range[0] is not None:
                fig.add_shape(
                    type="line",
                    x0=-0.4, x1=0.4,  # First bar position
                    y0=warn_range[0], y1=warn_range[0],
                    line=dict(color="orange", width=3, dash="dash"),
                    xref="x", yref="y",
                )
            if warn_range and len(warn_range) == 2 and warn_range[1] is not None and warn_range[1] < 1.0:
                fig.add_shape(
                    type="line",
                    x0=-0.4, x1=0.4,  # First bar position
                    y0=warn_range[1], y1=warn_range[1],
                    line=dict(color="orange", width=3, dash="dash"),
                    xref="x", yref="y",
                )

        # Parent ratio threshold (on second bar)
        if "ratio_parent" in thresholds and "warn" in thresholds["ratio_parent"]:
            warn_range = thresholds["ratio_parent"]["warn"]
            if warn_range and len(warn_range) == 2 and warn_range[0] is not None:
                fig.add_shape(
                    type="line",
                    x0=0.6, x1=1.4,  # Second bar position
                    y0=warn_range[0], y1=warn_range[0],
                    line=dict(color="red", width=3, dash="dash"),
                    xref="x", yref="y",
                )
            if warn_range and len(warn_range) == 2 and warn_range[1] is not None and warn_range[1] < 1.0:
                fig.add_shape(
                    type="line",
                    x0=0.6, x1=1.4,  # Second bar position
                    y0=warn_range[1], y1=warn_range[1],
                    line=dict(color="red", width=3, dash="dash"),
                    xref="x", yref="y",
                )

        # Update layout
        title = f"Gate: {gate_id} (parent: {parent_id})<br>" \
                f"<sub>Total events: {n_total:,} | Parent events: {n_parent:,} | " \
                f"Gate events: {n_passing_in_parent:,}</sub>"

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="Proportion",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            barmode="stack",
            hovermode="closest",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


class GateFitDiagnosticTest(QCTester):
    """Test for gate fitting quality based on diagnostics.

    Evaluates gate fitting diagnostics (e.g., r_squared, residual_std, n_outliers)
    for gates that have been fitted via automated methods. Skips gates with manual
    parameter overrides (which lack diagnostics).

    Gate types can define their own expected diagnostics and thresholds via
    _get_gate_type_config().
    """

    test_type = "gate"
    test_name = "gate_fit_quality"
    target_keys = ("strategy_id", "gate_id", "sample_id")
    meta_keys = ("parent_id", )
    default_config = {}
    meta_fields = [
           ("glm_type", "Type of GLM model used"),
           ("parent_id", "Parent gate ID(s)"),
       ]
    metric_fields = [
           ("r_squared", "R-squared value of the fit"),
           ("residual_std", "Standard deviation of residuals"),
           ("n_outliers", "Number of outlier points"),
       ]
    default_thresholds = {
           "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
           "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
           "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
       }
    plot_type = ""  # No plot for now
    plot_description = "Gate fitting quality diagnostics"

    # Gate type configuration registry
    _gate_type_configs = {
        "ols_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
                   "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
                   "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
            },
        },
        "wls_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
                   "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
                   "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
            },
        },
        "logistic_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.6, None), "severe": (0.4, None)},
                   "residual_std": {"warn": (None, 1.5), "severe": (None, 2.0)},
                   "n_outliers": {"warn": (None, 150), "severe": (None, 200)},
            },
        },
        "min_density": {
            "expected_diagnostics": [],  # Per-dimension diagnostics only
            "thresholds": {},
        },
    }

    def fit(
        self,
        targets: dict[str, Any],
        entity: GateNode,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for gate fitting diagnostics.

        Yields one QCTestRecord per diagnostic key found in the gate node.
        Yields SKIP records if diagnostics are missing (manual gate override).

        Parameters
        ----------
        targets : dict[str, Any]
            Target identifiers (strategy_id, sample_id, gate_id)
        **kwargs
            Additional test-specific parameters (gate_node, sample_id, entity, etc.)
        """
        sample_id: str = targets["sample_id"]
        thresholds = self.thresholds.copy()
        metadata = self.metadata.copy()
        metadata["glm_type"] = entity.glm_type
        # Add parent_id for comopatibility with older test records
        metadata["parent_id"] = "|".join(sorted(entity.parent_ids)) if entity.parent_ids else "root"

        # Extract diagnostics from gate node
        node_params = entity.get_params_for_sample(sample_id)
        diagnostics: dict[str, Any] = node_params.get("diagnostics", {})

        gate_config = self._gate_type_configs.get(entity.gate_type, {})
        thresholds.update(gate_config.pop("thresholds", {}))  # Override thresholds with gate-type-specific values if available
        metadata.update(gate_config)  # Add expected diagnostics and thresholds to metadata
        for diag_key, diag_value in diagnostics.items():
            try:
                metric_value = float(diag_value)
            except (TypeError, ValueError):
                # Skip non-scalar diagnostics
                continue

            yield QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={diag_key: metric_value},
                thresholds=thresholds,
                status="PENDING",
            )

class MetricOutlierTest(QCTester):
    """Outlier test for scalar metrics computed per sample.

    This tester is intended for simple scalar metrics, such as masked event
    ratios from GateMaskRatioTest, scalar diagnostics, or other numeric metrics.
    """

    test_type = "gate_batch"
    test_name = "gate_metric_outlier"
    target_keys = ("strategy_id", "gate_id", "metric_type")
    meta_keys = ("sample_id", "metric_name")
    default_config = {
        "min_samples": 6,        # Minimum samples required for outlier detection
        "outlier_method": "iqr", # "iqr" or "zscore"
        "use_mad": True,         # Use MAD for robust z-score
    }
    meta_fields = [
           ("metric_name", "Name of the gate metric being tested"),
           ("sample_id", "Sample ID"),
           ("outlier_method", "Method used for outlier detection (iqr or zscore)"),
           ("q1", "First quartile (IQR method only)"),
           ("q3", "Third quartile (IQR method only)"),
           ("iqr", "Interquartile range (IQR method only)"),
           ("center", "Center value: median (MAD) or mean (zscore method only)"),
           ("scale", "Scale value: MAD or standard deviation (zscore method only)"),
       ]
    metric_fields = [("outlier_score", "Outlier score computed by IQR or Z-score method")]
    default_thresholds = {
           "outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)},
       }
    plot_type = "box"
    plot_description = "Box plot showing outlier samples for gate event metrics"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        # Determine the outlier method to set appropriate default thresholds
        method = config.get("outlier_method", self.default_config["outlier_method"])

        # Set method-specific default thresholds
        if method == "iqr":
            # IQR method: 1.5 for warn, 3.0 for severe
            method_defaults = {
                "outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)},
            }
        elif method == "zscore":
            # Z-score method: 3 for warn, 5 for severe (more conservative)
            method_defaults = {
                "outlier_score": {"warn": (-3.0, 3.0), "severe": (-5.0, 5.0)},
            }
        else:
            raise ValueError(f"Invalid outlier method: {method}. Must be 'iqr' or 'zscore'.")

        # Update class-level defaults with method-specific defaults
        self.default_thresholds = method_defaults

        # Perform smarter merging of thresholds at warn/severe level
        merged_thresholds = {}
        for metric_name, metric_defaults in method_defaults.items():
            if metric_name in thresholds:
                # Merge warn/severe levels individually
                merged_thresholds[metric_name] = {}
                for level in ["warn", "severe"]:
                    if level in thresholds[metric_name]:
                        merged_thresholds[metric_name][level] = thresholds[metric_name][level]
                    elif level in metric_defaults:
                        merged_thresholds[metric_name][level] = metric_defaults[level]
            else:
                merged_thresholds[metric_name] = metric_defaults

        # Add any user-provided metrics not in defaults
        for metric_name in thresholds:
            if metric_name not in merged_thresholds:
                merged_thresholds[metric_name] = thresholds[metric_name]

        # Now validate and initialize with fully merged thresholds
        super().__init__(config=config, thresholds=merged_thresholds)
        min_samples = int(self.metadata["min_samples"])
        if min_samples <= 2:
            raise ValueError("MetricOutlierTest requires min_samples > 2.")

    def fit(
        self,
        targets: dict[str, Any],
        sample_metrics: dict[str, dict[str, Any]],
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute per-sample outlier scores for gate metrics.

        Parameters
        ----------
        targets : dict[str, Any]
            Base target identifiers (strategy_id, gate_id, parent_id)
        **kwargs
            Additional test-specific parameters:
            - gate_metrics: dict[str, dict[str, float]] - Nested dict: sample_id → {metric_name: value}
            - metric_names: list[str] | None - List of metric names to test
            - entity: GatingStrategyRef - The gating strategy entity
        """

        use_mad = bool(self.metadata["use_mad"])
        method = self.metadata.get("outlier_method", "iqr")
        if method not in ("iqr", "zscore"):
            raise ValueError(
                f"Invalid outlier method: {method}. Must be 'iqr' or 'zscore'."
            )

        metric_names = sorted(set().union(*(set(metrics.keys()) for metrics in sample_metrics.values())))
        thresholds = {"outlier_score": self.thresholds.get("outlier_score", self.default_thresholds["outlier_score"])}
        score_fn = dict_iqr_score if method == "iqr" else dict_zscore

        if len(sample_metrics) < self.metadata["min_samples"]:
            for sample_id in sample_metrics.keys():
                for metric_name in metric_names:
                    meta = {**self.metadata, "metric_name": metric_name, "sample_id": sample_id}
                    yield QCTestRecord(
                        id=self.make_key(targets, meta),
                        test_type=self.test_type,
                        test_name=self.test_name,
                        targets=targets,
                        metadata=meta,
                        metrics={"outlier_score": 0.},
                        thresholds=thresholds,
                        status="SKIP",
                        message=f"Not enough samples for outlier detection",
                    )
            return

        # Evaluate each metric across all samples in the current batch.
        for metric_name in metric_names:
            values_by_sample: dict[str, float] = {}
            for sample_id, metrics in sample_metrics.items():
                value = metrics.get(metric_name, np.nan)
                try:
                    values_by_sample[sample_id] = float(value)
                except (TypeError, ValueError):
                    continue

            try:
                scores, meta = score_fn(values_by_sample, use_mad=use_mad)
            except ValueError:
                # Skip metrics that do not have enough valid values for scoring.
                continue

            base_metadata = self.metadata.copy()
            base_metadata["metric_name"] = metric_name
            base_metadata["outlier_method"] = method
            base_metadata.update(meta)  # Add scoring metadata (e.g., center, scale for z-score)

            for sample_id, score in scores.items():
                base_metadata["sample_id"] = sample_id
                yield QCTestRecord(
                    id=self.make_key(targets, base_metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=base_metadata.copy(),
                    metrics={"outlier_score": score},
                    thresholds=thresholds.copy(),
                    status="PENDING",
                )

    def plot(
        self,
        test: QCTestRecord,
        **kwargs
    ) -> go.Figure:
        """Generate diagnostic plot for metric outlier test.

        Creates a combined histogram and marginal boxplot showing the distribution
        of metric values across samples with threshold lines for warn/severe levels.

        Parameters
        ----------
        test : QCTestRecord
            A test record to extract metadata (metric_name, thresholds)
        **kwargs
            Must include 'sample_metrics': dict[str, dict[str, Any]]
                Nested dict: sample_id → {metric_name: value}
            Optional 'output_path': PathLike | None
                Path to save the figure as HTML

        Returns
        -------
        go.Figure
            Plotly figure with histogram and marginal boxplot
        """
        sample_metrics = kwargs.get("sample_metrics", {})
        output_path = kwargs.get("output_path")

        metric_name = test.metadata.get("metric_name", "Unknown")
        gate_id = test.targets.get("gate_id", "Unknown")
        strategy_id = test.targets.get("strategy_id", "Unknown")

        # Extract metric values from sample_metrics
        sample_ids = []
        values = []
        for sample_id, metrics in sample_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                try:
                    values.append(float(value))
                    sample_ids.append(sample_id)
                except (TypeError, ValueError):
                    continue

        if not values:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No valid values for metric: {metric_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            return fig

        values_array = np.array(values)

        # Create figure with secondary y-axis for boxplot
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.15, 0.85],
            vertical_spacing=0.05,
            shared_xaxes=True,
            subplot_titles=("", f"Distribution of {metric_name}")
        )

        # Add boxplot on top
        fig.add_trace(
            go.Box(
                x=values_array,
                name="",
                orientation='h',
                marker=dict(color='rgba(100, 149, 237, 0.6)'),
                boxmean='sd',
                showlegend=False,
                hovertemplate="<b>Boxplot</b><br>Value: %{x:.4f}<extra></extra>",
            ),
            row=1, col=1
        )

        # Add histogram on bottom
        fig.add_trace(
            go.Histogram(
                x=values_array,
                name=metric_name,
                marker=dict(
                    color='rgba(100, 149, 237, 0.7)',
                    line=dict(color='rgba(100, 149, 237, 1)', width=1)
                ),
                showlegend=False,
                hovertemplate="<b>Bin</b><br>Range: %{x}<br>Count: %{y}<extra></extra>",
            ),
            row=2, col=1
        )

        # Extract thresholds
        thresholds = test.thresholds or {}
        outlier_thresholds = thresholds.get("outlier_score", {})

        # Get the method to understand if we're looking at scores or raw values
        method = test.metadata.get("outlier_method", "iqr")

        # Calculate statistics for reference
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        median = np.median(values_array)
        mean = np.mean(values_array)
        std = np.std(values_array)

        # Calculate MAD if using robust z-score
        use_mad = test.metadata.get("use_mad", True)
        if use_mad:
            mad = np.median(np.abs(values_array - median))
            scale = mad if mad > 0 else std
        else:
            scale = std

        # Convert outlier score thresholds to raw metric values
        warn_range = outlier_thresholds.get("warn", (None, None))
        severe_range = outlier_thresholds.get("severe", (None, None))

        # Compute threshold boundaries in raw metric space
        threshold_lines = []
        if method == "iqr":
            # IQR method: score = (value - Q2) / IQR
            # So: value = Q2 + score * IQR
            if warn_range[0] is not None and iqr > 0:
                warn_lower = median + warn_range[0] * iqr
                threshold_lines.append(("warn_lower", warn_lower, "orange", "dash"))
            if warn_range[1] is not None and iqr > 0:
                warn_upper = median + warn_range[1] * iqr
                threshold_lines.append(("warn_upper", warn_upper, "orange", "dash"))
            if severe_range[0] is not None and iqr > 0:
                severe_lower = median + severe_range[0] * iqr
                threshold_lines.append(("severe_lower", severe_lower, "red", "dashdot"))
            if severe_range[1] is not None and iqr > 0:
                severe_upper = median + severe_range[1] * iqr
                threshold_lines.append(("severe_upper", severe_upper, "red", "dashdot"))

        elif method == "zscore":
            # Z-score method: score = (value - center) / scale
            # So: value = center + score * scale
            center = median if use_mad else mean
            if warn_range[0] is not None and scale > 0:
                warn_lower = center + warn_range[0] * scale
                threshold_lines.append(("warn_lower", warn_lower, "orange", "dash"))
            if warn_range[1] is not None and scale > 0:
                warn_upper = center + warn_range[1] * scale
                threshold_lines.append(("warn_upper", warn_upper, "orange", "dash"))
            if severe_range[0] is not None and scale > 0:
                severe_lower = center + severe_range[0] * scale
                threshold_lines.append(("severe_lower", severe_lower, "red", "dashdot"))
            if severe_range[1] is not None and scale > 0:
                severe_upper = center + severe_range[1] * scale
                threshold_lines.append(("severe_upper", severe_upper, "red", "dashdot"))

        # Add statistical reference lines
        colors = {
            'median': 'green',
            'mean': 'blue',
            'q1': 'orange',
            'q3': 'orange',
        }

        # Add vertical lines using shapes for subplot compatibility
        # Boxplot (row 1)
        fig.add_shape(
            type="line",
            x0=median, x1=median,
            y0=0, y1=1,
            line=dict(color=colors['median'], width=2, dash='dash'),
            xref="x", yref="y domain",
            row=1, col=1
        )

        fig.add_shape(
            type="line",
            x0=mean, x1=mean,
            y0=0, y1=1,
            line=dict(color=colors['mean'], width=2, dash='dot'),
            xref="x", yref="y domain",
            row=1, col=1
        )

        # Add threshold lines to boxplot
        for label, value, color, dash in threshold_lines:
            fig.add_shape(
                type="line",
                x0=value, x1=value,
                y0=0, y1=1,
                line=dict(color=color, width=2, dash=dash),
                xref="x", yref="y domain",
                row=1, col=1
            )

        # Histogram (row 2) - with annotations
        fig.add_shape(
            type="line",
            x0=median, x1=median,
            y0=0, y1=1,
            line=dict(color=colors['median'], width=2, dash='dash'),
            xref="x2", yref="y2 domain",
            row=2, col=1
        )

        fig.add_annotation(
            x=median, y=1.05,
            text=f"Median: {median:.4f}",
            showarrow=False,
            xref="x2", yref="y2 domain",
            font=dict(size=10, color=colors['median']),
            row=2, col=1
        )

        fig.add_shape(
            type="line",
            x0=mean, x1=mean,
            y0=0, y1=1,
            line=dict(color=colors['mean'], width=2, dash='dot'),
            xref="x2", yref="y2 domain",
            row=2, col=1
        )

        fig.add_annotation(
            x=mean, y=0.95,
            text=f"Mean: {mean:.4f}",
            showarrow=False,
            xref="x2", yref="y2 domain",
            font=dict(size=10, color=colors['mean']),
            row=2, col=1
        )

        # Add threshold lines to histogram
        for label, value, color, dash in threshold_lines:
            fig.add_shape(
                type="line",
                x0=value, x1=value,
                y0=0, y1=1,
                line=dict(color=color, width=2, dash=dash),
                xref="x2", yref="y2 domain",
                row=2, col=1
            )
            # Add annotation for threshold
            if "lower" in label:
                y_pos = 0.85 if "severe" in label else 0.75
            else:
                y_pos = 0.65 if "severe" in label else 0.55

            fig.add_annotation(
                x=value, y=y_pos,
                text=f"{label.replace('_', ' ').title()}: {value:.4f}",
                showarrow=False,
                xref="x2", yref="y2 domain",
                font=dict(size=9, color=color),
                textangle=-90,
                row=2, col=1
            )

        # Update layout
        title = (
            f"<b>Gate: {gate_id}</b> | Metric: {metric_name}<br>"
            f"<sub>Strategy: {strategy_id} | Samples: {len(values)} | "
            f"Method: {method.upper()} | "
            f"μ={mean:.4f}, σ={std:.4f}, Q1={q1:.4f}, Q3={q3:.4f}</sub>"
        )

        fig.update_layout(
            title=title,
            height=600,
            hovermode="closest",
            showlegend=False,
        )

        # Update axes
        fig.update_xaxes(title_text=metric_name, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # Add threshold information as annotation
        warn_range = outlier_thresholds.get("warn", (None, None))
        severe_range = outlier_thresholds.get("severe", (None, None))
        threshold_text = (
            f"<b>Outlier Thresholds ({method.upper()})</b><br>"
            f"Warn score: [{warn_range[0]}, {warn_range[1]}]<br>"
            f"Severe score: [{severe_range[0]}, {severe_range[1]}]<br>"
        )

        # Add converted thresholds if applicable
        if threshold_lines:
            threshold_text += "<br><b>In metric space:</b><br>"
            for label, value, color, dash in threshold_lines:
                threshold_text += f"{label.replace('_', ' ').title()}: {value:.4f}<br>"

        fig.add_annotation(
            text=threshold_text,
            xref="paper", yref="paper",
            x=1.02, y=0.5,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font=dict(size=9),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            align="left",
        )

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig

class GLMGateOutlierTest(QCTester):
    """Outlier test for full gate geometry grouped by `glm_type`.

    Computes pairwise Jaccard distance between sample gate masks, derives a
    per-sample centrality score from mean pairwise similarity, and then computes
    an outlier z-score over centrality values.
    """

    test_type = "gate_batch"
    test_name = "gate_coverage_outlier"
    target_keys = ("strategy_id", "gate_id", "metric_type")
    meta_keys = ("sample_id", "metric_name")
    default_config = {
        "min_samples": 6,
        "outlier_method": "zscore",
        "use_mad": True,
        "n_points": 2048,
        "seed": 42,
    }
    meta_fields = [
        ("glm_type", "Type of GLM gate"),
        ("metric_name", "Name of centrality metric"),
        ("sample_id", "Sample ID"),
        ("centrality_score", "Computed centrality score"),
        ("n_eval_points", "Number of evaluation points"),
    ]
    metric_fields = [("outlier_score", "Outlier score computed by IQR or Z-score method")]
    default_thresholds = {
        "outlier_score": {"warn": (-3.0, 3.0), "severe": (-5.0, 5.0)},
    }
    plot_type = "scatter"
    plot_description = "Sample gate-space centrality outlier score"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        method = config.get("outlier_method", self.default_config["outlier_method"])

        if method == "iqr":
            method_defaults = {
                "outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)},
            }
        elif method == "zscore":
            method_defaults = {
                "outlier_score": {"warn": (-3.0, 3.0), "severe": (-5.0, 5.0)},
            }
        else:
            raise ValueError(
                f"Invalid outlier method: {method}. Must be 'iqr' or 'zscore'."
            )

        self.default_thresholds = method_defaults

        # Merge user thresholds at warn/severe level so partial overrides are valid.
        merged_thresholds: dict[str, Any] = {}
        for metric_name, metric_defaults in method_defaults.items():
            if metric_name in thresholds:
                merged_thresholds[metric_name] = {}
                user_spec = thresholds[metric_name]
                for level in ("warn", "severe"):
                    if isinstance(user_spec, Mapping) and level in user_spec:
                        merged_thresholds[metric_name][level] = user_spec[level]
                    else:
                        merged_thresholds[metric_name][level] = metric_defaults[level]
            else:
                merged_thresholds[metric_name] = metric_defaults

        for metric_name, user_spec in thresholds.items():
            if metric_name not in merged_thresholds:
                merged_thresholds[metric_name] = user_spec

        super().__init__(config=config, thresholds=merged_thresholds)
        min_samples = int(self.metadata["min_samples"])
        if min_samples <= 2:
            raise ValueError("GLMGateOutlierTest requires min_samples > 2.")

    def fit(
        self,
        targets: dict[str, Any],
        entity: GateNode,
        sample_ids: Iterable[str],
        **kwargs,
    ) -> Iterable[QCTestRecord]:
        sample_ids = list(sample_ids)

        metadata = self.metadata.copy()
        metadata["glm_type"] = entity.glm_type
        metadata["metric_name"] = "centrality_score"
        method = self.metadata["outlier_method"]
        thresholds = {"outlier_score": self.thresholds.get("outlier_score", self.default_thresholds["outlier_score"])}

        # Check if enough samples are available.
        min_samples = int(self.metadata["min_samples"])
        targets = targets.copy()
        if entity.gate_type in {"Boolean", "Quadrant"} or len(sample_ids) < min_samples:
            if entity.gate_type in {"Boolean", "Quadrant"}:
                msg = f"GLM gate outlier test is not applicable for gate_type '{entity.gate_type}'"
            else:
                msg = "GLM gate outlier test is not applicable due to insufficient samples"
            for sample_id in sample_ids:
                meta = {**metadata, "sample_id": sample_id}
                yield QCTestRecord(
                    id=self.make_key(targets, meta),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=meta,
                    metrics={"outlier_score": float('nan')},
                    thresholds=thresholds.copy(),
                    status="SKIP",
                    message=msg,
                )
            return

        masks = self._compute_gate_coverage(entity=entity, sample_ids=sample_ids, metadata=metadata)
        if not masks:
            msg = "No evaluation points could be generated for GLM gate outlier test, cannot compute gate-space similarity."
            for sample_id in sample_ids:
                meta = {
                    **metadata,
                    "sample_id": sample_id,
                    metadata["metric_name"]: float('nan')
                }
                yield QCTestRecord(
                    id=self.make_key(targets, meta),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=meta,
                    metrics={"outlier_score": float('nan')},
                    thresholds=thresholds.copy(),
                    status="SKIP",
                    message=msg,
                )
            return

        centrality = self._compute_gate_centralities(masks)

        use_mad = bool(self.metadata["use_mad"])
        if method == "iqr":
            scores, score_meta = dict_iqr_score(centrality, use_mad=use_mad)
        else:
            scores, score_meta = dict_zscore(centrality, use_mad=use_mad)
        metadata.update(score_meta)

        for sample_id, z_score in scores.items():
            meta = {
                **metadata,
                "sample_id": sample_id,
                "centrality_score": centrality[sample_id],
            }
            yield QCTestRecord(
                id=self.make_key(targets, meta),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=meta,
                metrics={"outlier_score": z_score},
                thresholds=thresholds.copy(),
                status="PENDING",
            )

    def _make_eval_points(
        self,
        *,
        gates: Mapping[str, Any],
        dimensions: Sequence[str],
        glm_type: str | None,
    ) -> ad.AnnData:
        dims = len(dimensions)
        n_points = int(self.metadata.get("n_points", 2048))
        seed = int(self.metadata.get("seed", 0))

        low = np.full(dims, np.inf, dtype=float)
        high = np.full(dims, -np.inf, dtype=float)

        if glm_type == "RectangleGate":
            for gate in gates.values():
                mins = np.array([float(gate.min_vals.get(d, -np.inf)) for d in dimensions], dtype=float)
                maxs = np.array([float(gate.max_vals.get(d, np.inf)) for d in dimensions], dtype=float)
                low = np.minimum(low, np.where(np.isfinite(mins), mins, np.inf))
                high = np.maximum(high, np.where(np.isfinite(maxs), maxs, -np.inf))
        elif glm_type == "PolygonGate":
            for gate in gates.values():
                verts = np.asarray(gate.vertices, dtype=float)
                low = np.minimum(low, np.min(verts, axis=0))
                high = np.maximum(high, np.max(verts, axis=0))
        elif glm_type == "EllipsoidGate":
            for gate in gates.values():
                center = np.asarray(gate.center, dtype=float)
                cov = np.asarray(gate.covariance_matrix, dtype=float)
                radius = np.sqrt(max(float(gate.distance_square), 0.0))
                extent = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=np.inf)) * radius
                low = np.minimum(low, center - extent)
                high = np.maximum(high, center + extent)
        else:
            # For other gate types, use a stable reference domain.
            low = np.full(dims, 0., dtype=float)
            high = np.full(dims, 1.0, dtype=float)

        finite = np.isfinite(low) & np.isfinite(high)
        if not np.any(finite):
            return ad.AnnData()  # No valid bounds to sample from

        low: FloatArray = np.where(np.isfinite(low), low, 0.0)
        high: FloatArray = np.where(np.isfinite(high), high, 1.0)
        span: FloatArray = np.maximum(high - low, 1e-9)
        low = low - 0.05 * span
        high = high + 0.05 * span

        if dims == 1:
            X = np.linspace(low[0], high[0], max(n_points, 64)).reshape(-1, 1)
        elif dims == 2:
            side = int(max(8, np.ceil(np.sqrt(n_points))))
            x = np.linspace(low[0], high[0], side)
            y = np.linspace(low[1], high[1], side)
            gx, gy = np.meshgrid(x, y, indexing="xy")
            X = np.column_stack((gx.ravel(), gy.ravel()))
        else:
            rng = np.random.default_rng(seed)
            X = rng.uniform(low=low, high=high, size=(n_points, dims))

        adata = ad.AnnData(X=X)
        adata.var_names = dimensions
        return adata

    def _compute_gate_coverage(
        self,
        entity: GateNode,
        sample_ids: Iterable[str],
        metadata: dict[str, Any]
    ) -> dict[str, BooleanArray]:

        gate_cls = GateRegistry.get(entity.gate_type)
        gates = {sample_id: gate_cls.from_node(entity, sample_id=sample_id) for sample_id in sample_ids}
        adata = self._make_eval_points(gates=gates, dimensions=entity.dimensions, glm_type=entity.glm_type)
        metadata["n_eval_points"] = adata.n_obs
        if adata.X is None or adata.n_obs == 0:
            return {}

        # Compute space-coverage for each sample
        root_mask = np.ones(adata.n_obs, dtype=bool)
        masks: dict[str, BooleanArray] = {}
        for sample_id, gate in gates.items():
            gate_masks = gate.apply(adata, {"root": root_mask})
            if len(gate_masks) > 1:
                raise ValueError(
                    f"Expected a single mask for gate {entity.id} in sample {sample_id}, "
                    f"but got {len(gate_masks)} masks from apply()"
                )
            masks[sample_id] = next(iter(gate_masks.values()))
        return masks

    def _compute_gate_centralities(self, masks: dict[str, BooleanArray]) -> dict[str, float]:
        n_samples = len(masks)
        mask_matrix = np.vstack(list(masks.values()))
        mask_matrix_u16 = mask_matrix.astype(np.uint16)

        # Pairwise Jaccard distance matrix using mask vectors over evaluation points.
        intersections = mask_matrix_u16 @ mask_matrix_u16.T
        support = np.sum(mask_matrix, axis=1, dtype=float)
        unions = support[:, None] + support[None, :] - intersections
        jaccard_distance = 1. - np.divide(
            intersections,
            unions,
            out=np.ones_like(intersections, dtype=float),
            where=unions > 0,
        )
        np.fill_diagonal(jaccard_distance, 0.0)

        centrality: FloatArray = 1.0 - np.sum(jaccard_distance, axis=1) / (n_samples - 1)
        return dict(zip(masks.keys(), centrality.tolist()))


@EntityQCEvaluatorRegistry.register("gate_node")
class GateNodeQCEvaluator(EntityQCEvaluator):
    """QC evaluator for individual gate nodes within a gating strategy.

    Evaluates gate performance within its strategy context, including:
    - Event counts and ratios (total and parent-relative)
    - Gate fitting quality diagnostics
    - Sample-level outlier detection

    Requires strategy_id in entity_qc.context to load the gating strategy.
    """

    entity_type = "gate_node"
    _supported_tables = {
        "event_metrics": {
            "description": "Event counts and ratios per sample for this gate",
            "input_params": {}
        },
        "fitting_quality": {
            "description": "Gate fitting diagnostics across samples",
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
        "r_squared": 0.7,
        "residual_std": 1.0,
        "n_outliers": 100,
        # Outlier detection config/thresholds
        "min_samples": 6,
        "outlier_method": "iqr",
        "outlier_thresholds": {
            "warn": (-1.5, 1.5),
            "severe": (-3.0, 3.0),
        },
        "use_mad": True,
        # Full-gate geometry outlier config
        "gate_space_min_samples": 6,
        "gate_space_n_points": 2048,
        "gate_space_thresholds": {
            "warn": (-3.0, 3.0),
            "severe": (-5.0, 5.0),
        },
        "gate_space_seed": 0,
    }

    @staticmethod
    def _normalize_outlier_thresholds(raw: Any) -> dict[str, Any]:
        """Normalize evaluator threshold config to QCTester threshold schema.

        Accepted inputs:
        - None: use tester defaults
        - {"warn": (...), "severe": (...)}
        - {"outlier_score": {"warn": (...), "severe": (...)}}
        - (low, high): interpreted as warn only; severe is filled by tester defaults
        """
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            if "outlier_score" in raw:
                return {"outlier_score": raw["outlier_score"]}
            if "warn" in raw or "severe" in raw:
                return {"outlier_score": dict(raw)}
            return dict(raw)
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return {"outlier_score": {"warn": tuple(raw)}}
        raise ValueError(
            "Invalid thresholds config. Expected None, (low, high), "
            "{'warn': (...)}, or {'outlier_score': {'warn': (...), 'severe': (...)}}."
        )

    @classmethod
    def get_tests(cls, entity: GateNode | None = None) -> dict[str, type[QCTester]]:
        """Return dictionary of test classes for gate node QC.

        Reuses the same test classes as GatingStrategyQCEvaluator.

        Parameters
        ----------
        entity : GateNode | None
            Gate node entity (optional, can be used for entity-specific tests)

        Returns
        -------
        dict[str, type[QCTester]]
            Mapping of test_name → QCTester subclass
        """
        qc_testers: list[type[QCTester]] = [
            GateMaskRatioTest,
            GateFitDiagnosticTest,
            MetricOutlierTest,
            GLMGateOutlierTest,
        ]
        return {tester.test_name: tester for tester in qc_testers}


    def required_layer(self, entity: GateNode | None = None) -> str | None:
        """Return the required AnnData layer for gate node QC."""
        if entity is None:
            return None
        return entity.layer

    def load_entity(
        self,
        dataloader: UnifiedDataLoader,
        entity_id: Hashable,
        context: dict[str, Any] | None = None
    ) -> GateNode:
        """Load a gate node from the dataloader.

        Requires strategy_id in context to load the gate from its strategy.
        """
        # Gate nodes are loaded from their parent gating strategy
        gate_id = str(entity_id)

        # Get strategy_id from context
        context = context or {}
        strategy_id = context.get("strategy_id")

        if strategy_id is None:
            raise ValueError(
                f"Cannot load gate node {gate_id}: strategy_id must be provided in context"
            )

        # Load the gating strategy and extract the gate node
        strategy = dataloader.load_gating_strategy(strategy_id=strategy_id)
        gate_node = strategy.get_node(gate_id)

        return gate_node

    def update_sample_qc(
        self,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
        """Evaluate gate node QC against sample data.

        Requires strategy_id in entity_qc.context to evaluate the gate
        within its strategy context.

        Parameters
        ----------
        entity : GateNode
            The gate node entity to evaluate
        entity_qc : EntityQCStatus
            QC status to update
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading sample data
        dataloader_context : dict[str, Any] | None
            Optional context with sample_ids, layer, etc.
        context : dict[str, Any]
            Optional evaluation context (must include strategy_id)

        Returns
        -------
        None
        """
        # Default context to empty dict to avoid None checks
        dataloader_context = dataloader_context or {}

        config = self.config.copy()
        config.update(context)
        entity_qc.context = config

        # Get strategy_id from context
        strategy_id = entity_qc.context.get("strategy_id")
        if strategy_id is None:
            raise ValueError(
                "Cannot evaluate gate node QC: strategy_id must be provided in entity_qc.context"
            )

        # Load the gating strategy
        if dataloader is None:
            raise ValueError("dataloader must be provided to load gating strategy")

        strategy = dataloader.load_gating_strategy(strategy_id=strategy_id)

        # Gate event count QC only needs masks; no need to load AnnData.
        sample_ids = dataloader_context.get("sample_ids") or list(entity_qc.sample_qc.keys())

        # Process all samples
        for sample_id in sample_ids:
            self._evaluate_gate_node(
                entity=entity,
                strategy=strategy,
                entity_qc=entity_qc,
                sample_id=sample_id,
                config=config,
                dataloader=dataloader,
            )

    def update_batch_qc(
        self,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> None:
        """Update batch-level QC tests for gate node.

        Runs outlier detection across samples for this gate's event metrics.
        """
        config = self.config.copy()
        config.update(context)

        # Get strategy_id from context
        strategy_id = entity_qc.context.get("strategy_id")
        if strategy_id is None:
            raise ValueError(
                "Cannot evaluate gate node batch QC: strategy_id must be provided in entity_qc.context"
            )

        # Get batch QC step
        batch_step = entity_qc.batch_qc.get_step("GATE_NODE_BATCH_QC")
        sample_metrics = self._collect_sample_params(entity_qc, gate_node=entity)
        sample_metrics["mask"] = self._collect_sample_metrics(entity_qc, gate_id=entity.id)

        # Run outlier detection for this gate
        method = config["outlier_method"]
        outlier_thresholds = self._normalize_outlier_thresholds(
            config.get("outlier_thresholds", config.get("thresholds"))
        )
        outlier_tester = MetricOutlierTest(
            config={
                "min_samples": config["min_samples"],
                "outlier_method": method,
                "use_mad": config["use_mad"],
            },
            thresholds=outlier_thresholds,
        )
        for metric_type, metrics in sample_metrics.items():
            outlier_targets = {
                "strategy_id": strategy_id,
                "gate_id": entity.id,
                "metric_type": metric_type,
            }

            # Run outlier detection (tests ratio_total and ratio_parent)
            for classified_test in outlier_tester.fit_classify(
                targets=outlier_targets,
                sample_metrics=metrics,
            ):
                # Add to batch step
                if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                    batch_step.add_reason(
                        code=f"GATE_OUTLIER_{classified_test.status}",
                        message=classified_test.message,
                        tests=[classified_test],
                    )
                else:
                    batch_step.add_test(classified_test)

        # Compare full gate spaces across samples using glm_type-specific geometry.
        gate_space_sample_ids = sorted(entity_qc.sample_qc.keys())
        gate_space_thresholds = self._normalize_outlier_thresholds(
            config.get("gate_space_thresholds", config.get("outlier_thresholds", config.get("thresholds")))
        )

        space_tester = GLMGateOutlierTest(
            config={
                "min_samples": config["gate_space_min_samples"],
                "outlier_method": method,
                "use_mad": config["use_mad"],
                "n_points": config["gate_space_n_points"],
                "seed": config["gate_space_seed"],
            },
            thresholds=gate_space_thresholds,
        )
        for classified_test in space_tester.fit_classify(
            targets={
                "strategy_id": strategy_id,
                "gate_id": entity.id,
                "metric_type": "params",
            },
            entity=entity,
            sample_ids=gate_space_sample_ids,
        ):
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                batch_step.add_reason(
                    code=f"GATE_GLM_SPACE_OUTLIER_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                batch_step.add_test(classified_test)

    def _collect_sample_metrics(
        self,
        entity_qc: EntityQCStatus,
        gate_id: str
    ) -> dict[str, dict[str, float]]:
        """Collect event metrics for a specific gate across all samples.

        Returns a nested dict: sample_id → {metric_name: value}
        """
        sample_metrics: dict[str, dict[str, float]] = {}

        for sample_id, sample_qc_run in entity_qc.sample_qc.items():
            for step in sample_qc_run.steps.values():
                for test_record in step.tests.values():
                    if (test_record.test_type == "gate" and
                        test_record.test_name == GateMaskRatioTest.test_name and
                        test_record.targets.get("gate_id") == gate_id):

                        if sample_id not in sample_metrics:
                            sample_metrics[sample_id] = {}

                        # Extract metrics (only float values)
                        for metric_key, metric_value in test_record.metrics.items():
                            try:
                                sample_metrics[sample_id][metric_key] = float(metric_value)
                            except (TypeError, ValueError):
                                continue

        return sample_metrics

    def _collect_sample_params(
        self,
        entity_qc: EntityQCStatus,
        gate_node: GateNode
    ) -> dict[str, dict[str, Any]]:
        """Collect gate parameters for a specific gate across all samples.

        Returns a nested dict: sample_id → {param_name: value}
        """
        sample_params: dict[str, dict[str, float]] = {}
        sample_diagnostics: dict[str, dict[str, float]] = {}
        for sample_id in entity_qc.sample_qc:
            node_params = gate_node.get_params_for_sample(sample_id)
            sample_params[sample_id] = node_params.get("params", {})
            sample_diagnostics[sample_id] = node_params.get("diagnostics", {})

        return {
            "params": sample_params,
            "diagnostics": sample_diagnostics,
        }

    def _evaluate_gate_node(
        self,
        entity: GateNode,
        strategy: GatingStrategyRef,
        entity_qc: EntityQCStatus,
        sample_id: str,
        config: Mapping[str, Any],
        dataloader: UnifiedDataLoader,
    ) -> None:
        """Evaluate the gate node for a single sample.

        Parameters
        ----------
        entity : GateNode
            The gate node to evaluate
        strategy : GatingStrategyRef
            The parent gating strategy
        entity_qc : EntityQCStatus
            QC status to update
        sample_id : str
            Sample identifier
        config : Mapping[str, Any]
            Evaluation config with threshold settings
        dataloader : UnifiedDataLoader
            Dataloader for loading gate masks
        """
        # Get QC step for this sample
        step = entity_qc.get_sample_steps(sample_id).get_step("GATE_NODE_QC")

        # Load gate masks from dataloader
        gate_id = entity.id
        masks_to_load = [gate_id] + entity.parent_ids
        # try:
        #     masks_to_load.remove("root")
        # except ValueError:
        #     pass  # No root mask to pop, continue with available masks

        try:
            masks = dataloader.load_masks(
                sample_id=sample_id,
                strategy_id=strategy.id,
                gate_ids=masks_to_load,
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"Missing required masks for sample {sample_id}, strategy {strategy.id}: {e}"
            )


        gate_targets = {
            "strategy_id": strategy.id,
            "sample_id": sample_id,
            "gate_id": gate_id,
        }

        # Run event count test
        event_tester = GateMaskRatioTest(
            config={},
            thresholds={
                "ratio_total": {
                    "warn": (config["ratio_total_min"], config["ratio_total_max"]),
                    "severe": (None, None),
                },
                "ratio_parent": {
                    "warn": (config["ratio_parent_min"], config["ratio_parent_max"]),
                    "severe": (None, None),
                },
            }
        )

        for classified_test in event_tester.fit_classify(
            targets=gate_targets,
            entity=entity,
            masks=masks,
        ):
            # Add to step
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                step.add_reason(
                    code=f"GATING_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                step.add_test(classified_test)

        # Run gate fitting quality test
        fitting_tester = GateFitDiagnosticTest(
            thresholds={
                "r_squared": {"warn": (config["r_squared"], None), "severe": (None, None)},
                "residual_std": {"warn": (None, config["residual_std"]), "severe": (None, None)},
                "n_outliers": {"warn": (None, config["n_outliers"]), "severe": (None, None)},
            }
        )

        for classified_test in fitting_tester.fit_classify(
            targets=gate_targets,
            entity=entity,
        ):
            # Add to step
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                step.add_reason(
                    code=f"GATE_FITTING_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                step.add_test(classified_test)