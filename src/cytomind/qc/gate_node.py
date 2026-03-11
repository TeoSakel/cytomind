"""
Gate Node QC Evaluator.

Performs QC analysis on individual gates within their gating strategy context.
Evaluates event counts, ratios, fitting quality, and outlier detection for a single gate.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, Sequence, TYPE_CHECKING
from pathlib import Path
import hashlib
import json

import numpy as np
import anndata as ad
import plotly.graph_objects as go

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.domain.pipeline import NumpyEncoder
from cytomind.gates import GateRegistry
from cytomind.utils import now_iso

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester, make_scalar_outlier_tester
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


_GLM_GATE_SPACE_ARTIFACT_VERSION = 1


def _compute_gate_space_bounds(
    *,
    gates: Mapping[str, Any],
    dimensions: Sequence[str],
    glm_type: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    dims = len(dimensions)
    low = np.full(dims, np.inf, dtype=float)
    high = np.full(dims, -np.inf, dtype=float)

    for gid, gate in gates.items():
        if list(gate.dimensions) != list(dimensions):
            raise ValueError(
                f"Gate {gid} dimensions must match space dimensions: "
                f"expected {list(dimensions)}, got {list(gate.dimensions)}"
            )
    if glm_type == "RectangleGate":
        for gate in gates.values():
            mins, maxs = np.asarray(gate.params["boundaries"], dtype=float)
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
        low = np.full(dims, 0.0, dtype=float)
        high = np.full(dims, 1.0, dtype=float)

    finite = np.isfinite(low) & np.isfinite(high)
    if not np.any(finite):
        return None

    low = np.where(np.isfinite(low), low, 0.0)
    high = np.where(np.isfinite(high), high, 1.0)
    span = np.maximum(high - low, 1e-9)
    return low - 0.05 * span, high + 0.05 * span


def _make_gate_space_eval_points(
    *,
    dimensions: Sequence[str],
    low: np.ndarray,
    high: np.ndarray,
    n_points: int,
    seed: int,
) -> ad.AnnData:
    dims = len(dimensions)
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
    adata.var_names = list(dimensions)
    return adata


def _compute_gate_space_masks(
    *,
    gates: Mapping[str, Any],
    eval_points: ad.AnnData,
    gate_id: str,
) -> dict[str, np.ndarray]:
    root_mask = np.ones(eval_points.n_obs, dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for sample_id, gate in gates.items():
        gate_masks = gate.apply(eval_points, {"root": root_mask})
        if len(gate_masks) > 1:
            raise ValueError(
                f"Expected a single mask for gate {gate_id} in sample {sample_id}, "
                f"but got {len(gate_masks)} masks from apply()"
            )
        masks[sample_id] = np.asarray(next(iter(gate_masks.values())), dtype=bool)
    return masks


def _compute_jaccard_distance_matrix(mask_matrix: np.ndarray) -> np.ndarray:
    if mask_matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    mask_matrix_u16 = mask_matrix.astype(np.uint16)
    intersections = mask_matrix_u16 @ mask_matrix_u16.T
    support = np.sum(mask_matrix, axis=1, dtype=float)
    unions = support[:, None] + support[None, :] - intersections
    distance = 1.0 - np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=float),
        where=unions > 0,
    )
    np.fill_diagonal(distance, 0.0)
    return distance


def _compute_jaccard_distance_between(mask_matrix_a: np.ndarray, mask_matrix_b: np.ndarray) -> np.ndarray:
    if mask_matrix_a.shape[0] == 0 or mask_matrix_b.shape[0] == 0:
        return np.zeros((mask_matrix_a.shape[0], mask_matrix_b.shape[0]), dtype=float)
    mask_matrix_a_u16 = mask_matrix_a.astype(np.uint16)
    mask_matrix_b_u16 = mask_matrix_b.astype(np.uint16)
    intersections = mask_matrix_a_u16 @ mask_matrix_b_u16.T
    support_a = np.sum(mask_matrix_a, axis=1, dtype=float)
    support_b = np.sum(mask_matrix_b, axis=1, dtype=float)
    unions = support_a[:, None] + support_b[None, :] - intersections
    return 1.0 - np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=float),
        where=unions > 0,
    )


def _compute_gate_space_centrality(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-sample centrality as mean deviation from the group.

    Centrality = mean pairwise Jaccard distance to all other samples.
    - Low centrality (→ 0): Sample has gate coverage similar to the group → typical
    - High centrality (→ 1): Sample has gate coverage dissimilar from group → outlier
    """
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=float)
    return np.sum(distance_matrix, axis=1) / (n_samples - 1)


def _pack_mask_matrix(mask_matrix: np.ndarray) -> np.ndarray:
    return np.packbits(mask_matrix.astype(np.uint8), axis=1)


def _unpack_mask_matrix(packed_masks: np.ndarray, n_eval_points: int) -> np.ndarray:
    return np.unpackbits(packed_masks, axis=1, count=n_eval_points).astype(bool)


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, cls=NumpyEncoder)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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


# ---------------------------------------------------------------------------
# Read-only proxy stubs for gate outlier tests (used by generate_table)
# ---------------------------------------------------------------------------

class _GateMetricOutlierProxy(QCTester):
    """Table-generation proxy for gate_metric_outlier tests (no fit())."""
    test_type = "gate_batch"
    test_name = "gate_metric_outlier"
    target_keys = ("entity_id", "metric_type")
    meta_keys = ("sample_id", "metric_name")
    metric_fields = [("outlier_score", "Outlier score")]
    meta_fields = [
        ("metric_name",    "Name of the metric"),
        ("metric_value",   "Raw metric value"),
        ("sample_id",      "Sample ID"),
        ("outlier_method", "Outlier detection method"),
    ]
    default_config = {"min_samples": 6, "outlier_method": "iqr", "use_mad": True}
    default_thresholds = {"outlier_score": {"warn": (-1.5, 1.5), "severe": (-3.0, 3.0)}}


class _GateCoverageOutlierProxy(QCTester):
    """Table-generation proxy for gate_coverage_outlier tests (no fit())."""
    test_type = "gate_batch"
    test_name = "gate_coverage_outlier"
    target_keys = ("entity_id", "metric_type")
    meta_keys = ("sample_id", "metric_name")
    metric_fields = [("outlier_score", "Outlier score")]
    meta_fields = [
        ("metric_name",    "Name of the metric"),
        ("metric_value",   "Raw centrality value"),
        ("sample_id",      "Sample ID"),
        ("glm_type",       "GLM gate type"),
        ("n_eval_points",  "Evaluation points used"),
    ]
    default_config = {"min_samples": 6, "outlier_method": "zscore", "use_mad": True}
    default_thresholds = {"outlier_score": {"warn": (-3.0, 3.0), "severe": (-5.0, 5.0)}}


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
    _glm_gate_space_artifact_name = "glm_gate_space"

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
        "gate_space_update_policy": "incremental",
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
            _GateMetricOutlierProxy,
            _GateCoverageOutlierProxy,
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
        param_metrics = self._collect_sample_params(entity_qc, gate_node=entity)
        sample_ids = param_metrics["sample_ids"]

        has_custom_gates = bool(entity.custom_gates)
        # Without per-sample custom gates, params and gate-space centrality are
        # deterministic across samples and not informative for outlier detection.
        skip_param_and_centrality = not has_custom_gates

        mask_metrics = self._collect_sample_metrics(
            entity_qc,
            gate_id=entity.id,
            sample_ids=sample_ids,
        )

        sample_metrics: dict[str, dict[str, list[float]]] = {
            "diagnostics": param_metrics["diagnostics"],
            "mask": mask_metrics,
        }
        if not skip_param_and_centrality:
            sample_metrics["params"] = param_metrics["params"]

        # Run outlier detection for this gate
        method = config["outlier_method"]
        outlier_thresholds = self._normalize_outlier_thresholds(
            config.get("outlier_thresholds", config.get("thresholds"))
        )
        outlier_config = {
            "min_samples": config["min_samples"],
            "outlier_method": method,
            "use_mad": config["use_mad"],
        }
        for metric_type, metrics_by_name in sample_metrics.items():
            outlier_targets = {
                "entity_id": entity.id,
                "metric_type": metric_type,
            }
            for metric_name, metric_series in metrics_by_name.items():
                outlier_tester = make_scalar_outlier_tester(
                    entity,
                    test_name="gate_metric_outlier",
                    test_type="gate_batch",
                    config=outlier_config,
                    thresholds=outlier_thresholds,
                )
                sample_values = {sid: metric_series[idx] for idx, sid in enumerate(sample_ids)}
                # Run outlier detection per metric
                for classified_test in outlier_tester.fit_classify(
                    targets=outlier_targets,
                    sample_values=sample_values,
                    metric_name=metric_name,
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
        # Skip this when gate parameters are deterministic across samples.
        if not skip_param_and_centrality:
            gate_space_sample_ids = list(entity_qc.sample_qc.keys())
            gate_space_thresholds = self._normalize_outlier_thresholds(
                config.get("gate_space_thresholds", config.get("outlier_thresholds", config.get("thresholds")))
            )
            gate_space_state = None
            if dataloader is not None:
                gate_space_state = self._get_or_build_glm_gate_space_state(
                    entity=entity,
                    entity_qc=entity_qc,
                    dataloader=dataloader,
                    strategy_id=strategy_id,
                    sample_ids=gate_space_sample_ids,
                    config=config,
                )

            gate_space_targets = {"entity_id": entity.id, "metric_type": "params"}
            gate_space_min_samples = int(config["gate_space_min_samples"])
            gate_space_config = {
                "min_samples": gate_space_min_samples,
                "outlier_method": "zscore",
                "use_mad": config["use_mad"],
            }

            # Determine if scoring is possible before building the tester.
            skip_msg: str | None = None
            if entity.gate_type in {"Boolean", "Quadrant"}:
                skip_msg = f"GLM gate outlier test is not applicable for gate_type '{entity.gate_type}'"
            elif len(gate_space_sample_ids) < gate_space_min_samples:
                skip_msg = "GLM gate outlier test is not applicable due to insufficient samples"
            elif gate_space_state is None:
                skip_msg = (
                    "gate_space_state must be provided by GateNodeQCEvaluator"
                    " via _get_or_build_glm_gate_space_state()"
                )

            n_eval_points = 0
            centrality: dict[str, float] = {}
            if skip_msg is None:
                assert gate_space_state is not None
                n_eval_points = int(gate_space_state.get("n_eval_points", 0))
                centrality = {
                    sid: float(gate_space_state.get("centrality", {}).get(sid, float("nan")))
                    for sid in gate_space_sample_ids
                }
                if not n_eval_points or not any(np.isfinite(v) for v in centrality.values()):
                    skip_msg = (
                        "No evaluation points could be generated for GLM gate outlier test,"
                        " cannot compute gate-space similarity."
                    )

            space_tester = make_scalar_outlier_tester(
                entity,
                test_name="gate_coverage_outlier",
                test_type="gate_batch",
                config=gate_space_config,
                thresholds=gate_space_thresholds,
                extra_meta_fields=[
                    ("glm_type",      "Type of GLM gate"),
                    ("n_eval_points", "Number of evaluation points used"),
                ],
                extra_static_meta={
                    "glm_type":      entity.glm_type or "",
                    "n_eval_points": n_eval_points,
                },
                plot_description="Gate-space centrality outlier score per sample",
            )

            if skip_msg is not None:
                thresholds_rec = {"outlier_score": space_tester.thresholds["outlier_score"]}
                for sid in gate_space_sample_ids:
                    meta = {
                        "metric_name":  "centrality_score",
                        "metric_value": float("nan"),
                        "sample_id":    sid,
                        "glm_type":     entity.glm_type or "",
                        "n_eval_points": n_eval_points,
                    }
                    classified_test = QCTestRecord(
                        id=space_tester.make_key(gate_space_targets, meta),
                        test_type=space_tester.test_type,
                        test_name=space_tester.test_name,
                        targets=gate_space_targets,
                        metadata=meta,
                        metrics={"outlier_score": float("nan")},
                        thresholds=thresholds_rec,
                        status="SKIP",
                        message=skip_msg,
                    )
                    batch_step.add_test(classified_test)
            else:
                for classified_test in space_tester.fit_classify(
                    targets=gate_space_targets,
                    sample_values=centrality,
                    metric_name="centrality_score",
                ):
                    if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                        batch_step.add_reason(
                            code=f"GATE_GLM_SPACE_OUTLIER_{classified_test.status}",
                            message=classified_test.message,
                            tests=[classified_test],
                        )
                    else:
                        batch_step.add_test(classified_test)

    def _gate_space_artifact_key(self, strategy_id: str) -> str:
        return f"{self._glm_gate_space_artifact_name}:{strategy_id}"

    def _gate_space_artifact_dir(
        self,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
    ) -> Path:
        return self.artifact_dir(
            entity_qc,
            dataloader,
            f"{self._glm_gate_space_artifact_name}/{strategy_id}",
        )

    def _gate_space_artifact_paths(
        self,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
    ) -> dict[str, Path]:
        artifact_dir = self._gate_space_artifact_dir(entity_qc, dataloader, strategy_id)
        return {
            "dir": artifact_dir,
            "basis": artifact_dir / "basis.json",
            "sample_index": artifact_dir / "sample_index.json",
            "eval_points": artifact_dir / "eval_points.npz",
            "coverage_masks": artifact_dir / "coverage_masks.npz",
            "distance_state": artifact_dir / "distance_state.npz",
        }

    def _gate_space_config_payload(
        self,
        entity: GateNode,
        strategy_id: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": _GLM_GATE_SPACE_ARTIFACT_VERSION,
            "strategy_id": strategy_id,
            "gate_id": entity.id,
            "gate_type": entity.gate_type,
            "glm_type": entity.glm_type,
            "dimensions": list(entity.dimensions),
            "n_points": int(config["gate_space_n_points"]),
            "seed": int(config["gate_space_seed"]),
        }

    def _sample_gate_signature(self, entity: GateNode, sample_id: str) -> str:
        return _hash_payload(entity.get_params_for_sample(sample_id))

    def _load_glm_gate_space_state(
        self,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
    ) -> dict[str, Any] | None:
        artifact_key = self._gate_space_artifact_key(strategy_id)
        metadata = self.get_artifact_metadata(entity_qc, artifact_key)
        if not metadata:
            return None

        paths = self._gate_space_artifact_paths(entity_qc, dataloader, strategy_id)
        if not all(path.exists() for name, path in paths.items() if name != "dir"):
            self.invalidate_artifact(entity_qc, artifact_key)
            return None

        basis = json.loads(paths["basis"].read_text())
        sample_index = json.loads(paths["sample_index"].read_text())
        eval_points = np.load(paths["eval_points"], allow_pickle=False)
        coverage_masks = np.load(paths["coverage_masks"], allow_pickle=False)
        distance_state = np.load(paths["distance_state"], allow_pickle=False)

        n_eval_points = int(basis.get("n_eval_points", 0))
        sample_ids = list(sample_index.get("sample_ids", []))
        packed_masks = np.asarray(coverage_masks["packed_masks"])
        mask_matrix = _unpack_mask_matrix(packed_masks, n_eval_points) if n_eval_points else np.zeros((0, 0), dtype=bool)
        centrality_values = np.asarray(distance_state["centrality"], dtype=float)

        return {
            "metadata": metadata,
            "basis": basis,
            "sample_index": sample_index,
            "eval_points": np.asarray(eval_points["X"], dtype=float),
            "mask_matrix": mask_matrix,
            "distance_matrix": np.asarray(distance_state["distance_matrix"], dtype=float),
            "centrality_vector": centrality_values,
            "sample_ids": sample_ids,
            "sample_hashes": dict(sample_index.get("sample_hashes", {})),
            "n_eval_points": n_eval_points,
            "centrality": dict(zip(sample_ids, centrality_values.tolist())),
        }

    def _save_glm_gate_space_state(
        self,
        *,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
        config: Mapping[str, Any],
        sample_ids: list[str],
        sample_hashes: dict[str, str],
        low: np.ndarray,
        high: np.ndarray,
        eval_points: np.ndarray,
        mask_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        centrality_vector: np.ndarray,
    ) -> dict[str, Any]:
        paths = self._gate_space_artifact_paths(entity_qc, dataloader, strategy_id)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        basis = {
            **self._gate_space_config_payload(entity, strategy_id, config),
            "low": low.tolist(),
            "high": high.tolist(),
            "n_eval_points": int(eval_points.shape[0]),
        }
        sample_index = {
            "sample_ids": sample_ids,
            "sample_hashes": sample_hashes,
        }

        paths["basis"].write_text(json.dumps(basis, indent=2, cls=NumpyEncoder))
        paths["sample_index"].write_text(json.dumps(sample_index, indent=2, cls=NumpyEncoder))
        np.savez_compressed(paths["eval_points"], X=eval_points)
        np.savez_compressed(paths["coverage_masks"], packed_masks=_pack_mask_matrix(mask_matrix))
        np.savez_compressed(
            paths["distance_state"],
            distance_matrix=distance_matrix.astype(np.float32),
            centrality=centrality_vector.astype(np.float32),
        )

        relative_dir = paths["dir"].relative_to(dataloader.root_dir).as_posix()
        artifact_key = self._gate_space_artifact_key(strategy_id)
        artifact_metadata = {
            "artifact_type": self._glm_gate_space_artifact_name,
            "strategy_id": strategy_id,
            "schema_version": _GLM_GATE_SPACE_ARTIFACT_VERSION,
            "relative_dir": relative_dir,
            "config_fingerprint": _hash_payload(self._gate_space_config_payload(entity, strategy_id, config)),
            "sample_count": len(sample_ids),
            "n_eval_points": int(eval_points.shape[0]),
            "updated_at": now_iso(),
        }
        self.set_artifact_metadata(entity_qc, artifact_key, artifact_metadata)

        return {
            "metadata": artifact_metadata,
            "basis": basis,
            "sample_index": sample_index,
            "eval_points": eval_points,
            "mask_matrix": mask_matrix,
            "distance_matrix": distance_matrix,
            "centrality_vector": centrality_vector,
            "sample_ids": sample_ids,
            "sample_hashes": sample_hashes,
            "n_eval_points": int(eval_points.shape[0]),
            "centrality": dict(zip(sample_ids, centrality_vector.tolist())),
        }

    def _rebuild_glm_gate_space_state(
        self,
        *,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
        sample_ids: list[str],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        gate_cls = GateRegistry.get(entity.gate_type)
        gates = {sample_id: gate_cls.from_node(entity, sample_id=sample_id) for sample_id in sample_ids}
        bounds = _compute_gate_space_bounds(
            gates=gates,
            dimensions=entity.dimensions,
            glm_type=entity.glm_type,
        )
        if bounds is None:
            self.invalidate_artifact(entity_qc, self._gate_space_artifact_key(strategy_id))
            return {"n_eval_points": 0, "centrality": {}, "sample_ids": sample_ids}

        low, high = bounds
        eval_points_adata = _make_gate_space_eval_points(
            dimensions=entity.dimensions,
            low=low,
            high=high,
            n_points=int(config["gate_space_n_points"]),
            seed=int(config["gate_space_seed"]),
        )
        eval_points = np.asarray(eval_points_adata.X, dtype=float)
        masks = _compute_gate_space_masks(
            gates=gates,
            eval_points=eval_points_adata,
            gate_id=entity.id,
        )
        mask_matrix = np.vstack([masks[sample_id] for sample_id in sample_ids]) if sample_ids else np.zeros((0, eval_points.shape[0]), dtype=bool)
        distance_matrix = _compute_jaccard_distance_matrix(mask_matrix)
        centrality_vector = _compute_gate_space_centrality(distance_matrix)
        sample_hashes = {sample_id: self._sample_gate_signature(entity, sample_id) for sample_id in sample_ids}
        return self._save_glm_gate_space_state(
            entity=entity,
            entity_qc=entity_qc,
            dataloader=dataloader,
            strategy_id=strategy_id,
            config=config,
            sample_ids=sample_ids,
            sample_hashes=sample_hashes,
            low=low,
            high=high,
            eval_points=eval_points,
            mask_matrix=mask_matrix,
            distance_matrix=distance_matrix,
            centrality_vector=centrality_vector,
        )

    def _maybe_incremental_glm_gate_space_state(
        self,
        *,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
        sample_ids: list[str],
        config: Mapping[str, Any],
        existing_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        existing_sample_ids = list(existing_state["sample_ids"])
        if sample_ids[:len(existing_sample_ids)] != existing_sample_ids:
            return None

        sample_hashes = {sample_id: self._sample_gate_signature(entity, sample_id) for sample_id in sample_ids}
        for sample_id in existing_sample_ids:
            if sample_hashes[sample_id] != existing_state["sample_hashes"].get(sample_id):
                return None

        new_sample_ids = sample_ids[len(existing_sample_ids):]
        if not new_sample_ids:
            centrality_vector = _compute_gate_space_centrality(existing_state["distance_matrix"])
            existing_state["centrality_vector"] = centrality_vector
            existing_state["centrality"] = dict(zip(existing_sample_ids, centrality_vector.tolist()))
            return existing_state

        basis = existing_state["basis"]
        gate_cls = GateRegistry.get(entity.gate_type)
        new_gates = {sample_id: gate_cls.from_node(entity, sample_id=sample_id) for sample_id in new_sample_ids}
        new_bounds = _compute_gate_space_bounds(
            gates=new_gates,
            dimensions=entity.dimensions,
            glm_type=entity.glm_type,
        )
        if new_bounds is None:
            return None

        low = np.asarray(basis["low"], dtype=float)
        high = np.asarray(basis["high"], dtype=float)
        new_low, new_high = new_bounds
        if np.any(new_low < low) or np.any(new_high > high):
            return None

        eval_points_adata = ad.AnnData(X=existing_state["eval_points"])
        eval_points_adata.var_names = entity.dimensions
        new_masks = _compute_gate_space_masks(
            gates=new_gates,
            eval_points=eval_points_adata,
            gate_id=entity.id,
        )
        new_mask_matrix = np.vstack([new_masks[sample_id] for sample_id in new_sample_ids])

        old_mask_matrix = existing_state["mask_matrix"]
        new_to_old = _compute_jaccard_distance_between(new_mask_matrix, old_mask_matrix)
        new_to_new = _compute_jaccard_distance_matrix(new_mask_matrix)
        upper = np.hstack([existing_state["distance_matrix"], new_to_old.T])
        lower = np.hstack([new_to_old, new_to_new])
        distance_matrix = np.vstack([upper, lower])
        mask_matrix = np.vstack([old_mask_matrix, new_mask_matrix])
        centrality_vector = _compute_gate_space_centrality(distance_matrix)

        return self._save_glm_gate_space_state(
            entity=entity,
            entity_qc=entity_qc,
            dataloader=dataloader,
            strategy_id=strategy_id,
            config=config,
            sample_ids=sample_ids,
            sample_hashes=sample_hashes,
            low=low,
            high=high,
            eval_points=existing_state["eval_points"],
            mask_matrix=mask_matrix,
            distance_matrix=distance_matrix,
            centrality_vector=centrality_vector,
        )

    def _get_or_build_glm_gate_space_state(
        self,
        *,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        strategy_id: str,
        sample_ids: list[str],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        if entity.gate_type in {"Boolean", "Quadrant"}:
            return {"n_eval_points": 0, "centrality": {}, "sample_ids": sample_ids}

        config_payload = self._gate_space_config_payload(entity, strategy_id, config)
        config_fingerprint = _hash_payload(config_payload)
        artifact_key = self._gate_space_artifact_key(strategy_id)
        artifact_metadata = self.get_artifact_metadata(entity_qc, artifact_key)
        existing_state = self._load_glm_gate_space_state(entity_qc, dataloader, strategy_id)
        if artifact_metadata is None or existing_state is None:
            return self._rebuild_glm_gate_space_state(
                entity=entity,
                entity_qc=entity_qc,
                dataloader=dataloader,
                strategy_id=strategy_id,
                sample_ids=sample_ids,
                config=config,
            )

        if artifact_metadata.get("schema_version") != _GLM_GATE_SPACE_ARTIFACT_VERSION:
            return self._rebuild_glm_gate_space_state(
                entity=entity,
                entity_qc=entity_qc,
                dataloader=dataloader,
                strategy_id=strategy_id,
                sample_ids=sample_ids,
                config=config,
            )
        if artifact_metadata.get("config_fingerprint") != config_fingerprint:
            return self._rebuild_glm_gate_space_state(
                entity=entity,
                entity_qc=entity_qc,
                dataloader=dataloader,
                strategy_id=strategy_id,
                sample_ids=sample_ids,
                config=config,
            )

        update_policy = config.get("gate_space_update_policy", "incremental")
        if update_policy == "incremental":
            updated_state = self._maybe_incremental_glm_gate_space_state(
                entity=entity,
                entity_qc=entity_qc,
                dataloader=dataloader,
                strategy_id=strategy_id,
                sample_ids=sample_ids,
                config=config,
                existing_state=existing_state,
            )
            if updated_state is not None:
                return updated_state

        return self._rebuild_glm_gate_space_state(
            entity=entity,
            entity_qc=entity_qc,
            dataloader=dataloader,
            strategy_id=strategy_id,
            sample_ids=sample_ids,
            config=config,
        )

    def _collect_sample_metrics(
        self,
        entity_qc: EntityQCStatus,
        gate_id: str,
        sample_ids: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Collect scalar mask metrics in metric-centric format.

        Returns
        -------
        dict[str, list[float]]
            Mapping ``metric_name -> [value_per_sample, ...]`` aligned with
            ``sample_ids``. Missing values are filled with ``NaN``.
        """
        aligned_sample_ids = sample_ids or list(entity_qc.sample_qc.keys())
        n_samples = len(aligned_sample_ids)
        sample_index = {sid: idx for idx, sid in enumerate(aligned_sample_ids)}
        metrics_by_name: dict[str, list[float]] = {}

        for sample_id, sample_qc_run in entity_qc.sample_qc.items():
            if sample_id not in sample_index:
                continue

            for step in sample_qc_run.steps.values():
                for test_record in step.tests.values():
                    if (test_record.test_type == "gate" and
                        test_record.test_name == GateMaskRatioTest.test_name and
                        test_record.targets.get("gate_id") == gate_id):

                        idx = sample_index[sample_id]
                        # Extract metrics and align them by sample index.
                        for metric_key, metric_value in test_record.metrics.items():
                            metric_series = metrics_by_name.setdefault(
                                metric_key,
                                [float("nan")] * n_samples,
                            )
                            try:
                                metric_series[idx] = float(metric_value)
                            except (TypeError, ValueError):
                                continue

        return metrics_by_name

    def _collect_sample_params(
        self,
        entity_qc: EntityQCStatus,
        gate_node: GateNode
    ) -> dict[str, Any]:
        """Collect scalar gate metrics across samples in metric-centric format.

        Expected shape (by convention, not enforced by all gates):
        - node_params["params"]:      {metric_name: scalar | array}
        - node_params["diagnostics"]: {metric_name: scalar | array}

        Arrays/non-scalars are ignored for now to keep this path strictly 1D.

        Returns
        -------
        dict[str, Any]
            {
                "sample_ids": [sample_id, ...],
                "params": {metric_name: [value_per_sample, ...]},
                "diagnostics": {metric_name: [value_per_sample, ...]},
            }
            Metric series are aligned with "sample_ids".
        """
        sample_ids = list(entity_qc.sample_qc.keys())
        n_samples = len(sample_ids)

        sample_params: dict[str, list[float]] = {}
        sample_diagnostics: dict[str, list[float]] = {}

        for idx, sample_id in enumerate(sample_ids):
            node_params = gate_node.get_params_for_sample(sample_id)
            for section_name, sink in (("params", sample_params), ("diagnostics", sample_diagnostics)):
                section = node_params.get(section_name, {})
                if not isinstance(section, Mapping):
                    continue

                for metric_name, metric_value in section.items():
                    # Skip array-like values until multi-dimensional tests are implemented.
                    if isinstance(metric_value, (list, tuple, dict, np.ndarray)):
                        continue

                    metric_series = sink.setdefault(metric_name, [float("nan")] * n_samples)
                    try:
                        metric_series[idx] = float(metric_value)
                    except (TypeError, ValueError):
                        # Keep NaN for non-numeric values.
                        continue

        return {
            "sample_ids": sample_ids,
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