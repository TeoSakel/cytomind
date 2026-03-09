"""
Step-level QC summarization.
"""
from __future__ import annotations
from typing import Any, Hashable, TYPE_CHECKING, Iterable

import pandas as pd
import plotly.graph_objects as go
import networkx as nx

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator
from cytomind.domain.qc import EntityQCStatus, QCFlag

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike
    from cytomind.domain.pipeline import StepRun
    from cytomind.infra.dataloader import UnifiedDataLoader
else:
    PathLike = object
    StepRun = object
    UnifiedDataLoader = object


@EntityQCEvaluatorRegistry.register("step")
class StepQCEvaluator(EntityQCEvaluator):
    """Default step QC evaluator (entity_type='step')."""

    entity_type = "step"
    _supported_tables = {
        "per_sample_step": {
            "description": "One row per sample-step combination",
            "input_params": {}
        },
        "heatmap": {
            "description": "Heatmap format with samples as rows and steps as columns",
            "input_params": {}
        },
        "per_sample": {
            "description": "Summary per sample with overall flag and test counts",
            "input_params": {}
        },
        "per_step": {
            "description": "Summary per step with sample-level flag counts and rates",
            "input_params": {}
        },
        "per_test": {
            "description": "Summary per test (test_type/test_name) with flag counts and rates",
            "input_params": {}
        },
        "all": {
            "description": "All tests with flag for every combination of sample_id, step_name, test_type, and test_name",
            "input_params": {}
        },
    }
    _supported_figures = {
        "heatmap": {
            "description": "Sample × Step QC heatmap with flags and hover messages",
            "input_params": {}
        },
    }

    def load_entity(self, dataloader: UnifiedDataLoader, entity_id: Hashable, context: dict[str, Any] | None = None) -> StepRun:
        return dataloader.load_data("step_run", step_id=str(entity_id))

    def parse_step(self, step_run: StepRun, entity_id: str | None = None) -> EntityQCStatus:
        """Return the step_run's execution QC, preserving all data from run phases."""
        return step_run.qc

    def _get_step_order(self, entity_qc: EntityQCStatus) -> list[str]:
        """Get steps in execution order using topological sort.

        Builds a directed graph from observed step sequences across all samples,
        then performs topological sort to get a consistent ordering that respects
        all partial orderings observed in the data.

        Falls back to using the sample with the most steps if a cycle is detected
        (indicating conflicting orderings across samples).

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing sample data.

        Returns
        -------
        list[str]
            Steps ordered by execution order.
        """
        # Build a directed graph of step dependencies based on execution order
        graph = nx.DiGraph()
        for qc_run in entity_qc.sample_qc.values():
            step_names = list(qc_run.steps.keys())
            graph.add_nodes_from(step_names)
            graph.add_edges_from(zip(step_names, step_names[1:]))

        # Try topological sort; fall back if there's a cycle
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Cycle detected - conflicting orderings across samples
            # Fall back to the sample with the most steps
            sample_with_most = max(entity_qc.sample_qc.values(), key=len, default=None)

            if not sample_with_most:
                # Fallback: return any consistent ordering
                return sorted(graph.nodes)

            # Start with the longest sample's step order
            step_order = list(sample_with_most.steps.keys())
            seen = set(step_order)
            step_order.extend(step for step in graph.nodes - seen)
            return step_order

    def generate_table(
        self,
        entity_qc: EntityQCStatus,
        table_type: str = "per_sample_step",
        test_name: str | None = None,
        sample_ids: Iterable[str] | None = None,
        table_path: PathLike | None = None,
    ) -> pd.DataFrame:
        """Generate tables from step-level QC status.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing step and sample data.
        table_type : str, default "per_sample_step"
            Type of table to generate:
            - "per_sample_step": One row per sample-step combination
            - "heatmap": Heatmap format with samples as rows and steps as columns
            - "per_sample": Summary per sample with overall flag and counts
            - "per_step": Summary per step with sample-level flag counts and rates
            - "per_test": Summary per test (test_type/test_name) with flag counts and rates
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading additional data if needed.
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader.
        table_path : PathLike | None
            Optional output path to save the table.

        Returns
        -------
        pd.DataFrame
            Table in the requested format.
        """
        # Use sample_ids from context if provided, otherwise use all samples
        sample_filter = set(sample_ids or entity_qc.sample_qc.keys())

        df_all = self._generate_all_tests(entity_qc, sample_filter)
        df_sample_step = self._aggregate_per_sample_step(df_all)
        if "_is_placeholder" in df_all.columns:
            df_all_public = df_all.drop(columns=["_is_placeholder"])
        else:
            df_all_public = df_all

        # Derive other formats from all-tests table
        # NOTE: Should this be done in the front-end?
        if table_type == "per_sample_step":
            return df_sample_step
        elif table_type == "heatmap":
            return self._pivot_to_heatmap(df_sample_step, entity_qc)
        elif table_type == "per_sample":
            return self._aggregate_per_sample(df_sample_step)
        elif table_type == "per_step":
            return self._aggregate_per_step(df_sample_step)
        elif table_type == "per_test":
            return self._aggregate_per_test(df_all)
        elif table_type == "all":
            return df_all_public
        else:
            raise ValueError(
                f"Unknown table_type '{table_type}'. Must be one of: "
                "'per_sample_step', 'heatmap', 'per_sample', 'per_step', 'per_test', 'all'"
            )

    def _generate_long_format(
        self,
        entity_qc: EntityQCStatus,
        sample_filter: set[str],
    ) -> pd.DataFrame:
        """Generate long format table with one row per sample-step combination.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object.
        sample_filter : set[str]
            Sample IDs to include.

        Returns
        -------
        pd.DataFrame
            Columns: sample_id, step_name, flag, n_tests, n_pass, n_warn, n_severe, n_fail, n_skip, message
        """
        records = []

        for sample_id, qc_run in entity_qc.sample_qc.items():
            if sample_id not in sample_filter:
                continue

            for step_name, step in qc_run.steps.items():
                # Count test statuses
                test_counts = {"PASS": 0, "WARN": 0, "SEVERE": 0, "FAIL": 0, "SKIP": 0}
                for test in step.tests.values():
                    status = test.status
                    if status in test_counts:
                        test_counts[status] += 1

                # Build reason codes
                reason_codes = list(step.reasons.keys())

                records.append({
                    "sample_id": sample_id,
                    "step_name": step_name,
                    "flag": step.flag.value,
                    "n_tests": len(step.tests),
                    "n_pass": test_counts["PASS"],
                    "n_warn": test_counts["WARN"],
                    "n_severe": test_counts["SEVERE"],
                    "n_fail": test_counts["FAIL"],
                    "n_skip": test_counts["SKIP"],
                    "message": ", ".join(reason_codes) if reason_codes else "",
                })

        if not records:
            return pd.DataFrame(columns=[
                "sample_id", "step_name", "flag", "n_tests", "n_pass",
                "n_warn", "n_severe", "n_fail", "n_skip", "message"
            ])

        return pd.DataFrame(records)

    def _aggregate_per_sample_step(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Aggregate all-tests table into one row per sample-step.

        Parameters
        ----------
        df_all : pd.DataFrame
            All-tests table from _generate_all_tests.

        Returns
        -------
        pd.DataFrame
            Columns: sample_id, step_name, flag, n_tests, n_pass, n_warn, n_severe, n_fail, n_skip, message
        """
        if df_all.empty:
            return pd.DataFrame(columns=[
                "sample_id", "step_name", "flag", "n_tests", "n_pass",
                "n_warn", "n_severe", "n_fail", "n_skip", "message"
            ])

        flag_order = ["PASS", "WARN", "SEVERE", "FAIL", "SKIP"]
        if "_is_placeholder" in df_all.columns:
            df_non_placeholder = df_all[~df_all["_is_placeholder"]]
            placeholder_flags = (
                df_all[df_all["_is_placeholder"]]
                .set_index(["sample_id", "step_name"])["flag"]
                .to_dict()
            )
        else:
            df_non_placeholder = df_all
            placeholder_flags = {}

        base_index = df_all.groupby(["sample_id", "step_name"]).size().index
        flag_counts = (
            df_non_placeholder.groupby(["sample_id", "step_name"])
            .flag.value_counts()
            .unstack(fill_value=0)
            .reindex(index=base_index, columns=flag_order, fill_value=0)
        )

        df_sample_step = flag_counts.rename(columns={
            "PASS": "n_pass",
            "WARN": "n_warn",
            "SEVERE": "n_severe",
            "FAIL": "n_fail",
            "SKIP": "n_skip",
        }).reset_index()

        df_sample_step["n_tests"] = df_sample_step[[
            "n_pass", "n_warn", "n_severe", "n_fail", "n_skip"
        ]].sum(axis=1)

        def _combine_flags(row: pd.Series) -> str:
            key = row.name
            if row["n_tests"] == 0 and key in placeholder_flags:
                return placeholder_flags[key]
            flags = []
            if row["n_pass"]:
                flags.append(QCFlag.PASS)
            if row["n_warn"]:
                flags.append(QCFlag.WARN)
            if row["n_severe"]:
                flags.append(QCFlag.SEVERE)
            if row["n_fail"]:
                flags.append(QCFlag.FAIL)
            return QCFlag.combine(flags).value

        df_sample_step["flag"] = df_sample_step.apply(_combine_flags, axis=1)

        df_sample_step["message"] = ""

        df_sample_step = df_sample_step[[
            "sample_id", "step_name", "flag", "n_tests", "n_pass",
            "n_warn", "n_severe", "n_fail", "n_skip", "message"
        ]]

        return df_sample_step

    def _pivot_to_heatmap(self, df_long: pd.DataFrame, entity_qc: EntityQCStatus) -> pd.DataFrame:
        """Convert long format to heatmap (pivot on samples × steps).

        Parameters
        ----------
        df_long : pd.DataFrame
            Long format table from _generate_long_format.
        entity_qc : EntityQCStatus
            The QC status object (used to determine step order).

        Returns
        -------
        pd.DataFrame
            Index: sample_id, Columns: step names, Values: flag strings
        """
        df_heatmap = df_long.pivot_table(
            index="sample_id",
            columns="step_name",
            values="flag",
            aggfunc="first"  # Should be one row per sample-step, but use first just in case
        )
        # Fill missing steps with "SKIP"
        df_heatmap = df_heatmap.fillna("SKIP")
        # Reorder columns by step execution order and sort index
        step_order = self._get_step_order(entity_qc)
        df_heatmap = df_heatmap.reindex(step_order, axis=1)
        df_heatmap = df_heatmap.sort_index()
        return df_heatmap

    def _aggregate_per_sample(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Aggregate long format table by sample.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long format table from _generate_long_format.

        Returns
        -------
        pd.DataFrame
            Columns: sample_id, overall_flag, n_steps, n_tests, n_pass, n_warn, n_severe, n_fail, n_skip
        """
        agg_dict = {
            "flag": lambda flags: QCFlag.combine([QCFlag(f) for f in flags]).value,
            "n_tests": "sum",
            "n_pass": "sum",
            "n_warn": "sum",
            "n_severe": "sum",
            "n_fail": "sum",
            "n_skip": "sum",
            "step_name": "count",  # Count of steps per sample
        }

        df_per_sample = df_long.groupby("sample_id", as_index=False).agg(agg_dict)
        df_per_sample = df_per_sample.rename(columns={
            "flag": "overall_flag",
            "step_name": "n_steps"
        })

        # Reorder columns
        df_per_sample = df_per_sample[[
            "sample_id", "overall_flag", "n_steps", "n_tests",
            "n_pass", "n_warn", "n_severe", "n_fail", "n_skip"
        ]]

        return df_per_sample

    def _aggregate_per_step(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Aggregate long format table by step.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long format table from _generate_long_format.

        Returns
        -------
        pd.DataFrame
            Columns: step_name, n_samples, n_pass, n_warn, n_severe, n_fail, n_skip, pass_rate, warn_rate, fail_rate
        """
        flag_order = ["PASS", "WARN", "SEVERE", "FAIL", "SKIP"]
        flag_counts = (
            df_long.groupby("step_name")
            .flag.value_counts()
            .unstack(fill_value=0)
            .reindex(columns=flag_order, fill_value=0)
        )

        df_per_step = flag_counts.rename(columns={
            "PASS": "n_pass",
            "WARN": "n_warn",
            "SEVERE": "n_severe",
            "FAIL": "n_fail",
            "SKIP": "n_skip",
        }).reset_index()

        df_per_step["n_samples"] = df_per_step[[
            "n_pass", "n_warn", "n_severe", "n_fail", "n_skip"
        ]].sum(axis=1)

        # Calculate rates based on sample-level flags
        df_per_step["pass_rate"] = df_per_step["n_pass"] / df_per_step["n_samples"]
        df_per_step["warn_rate"] = df_per_step["n_warn"] / df_per_step["n_samples"]
        df_per_step["fail_rate"] = df_per_step["n_fail"] / df_per_step["n_samples"]

        # Reorder columns
        df_per_step = df_per_step[[
            "step_name", "n_samples", "n_pass", "n_warn", "n_severe",
            "n_fail", "n_skip", "pass_rate", "warn_rate", "fail_rate"
        ]]

        return df_per_step

    def _aggregate_per_test(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Aggregate test-level table by test_type and test_name.

        Parameters
        ----------
        df_all : pd.DataFrame
            All-tests table from _generate_all_tests.

        Returns
        -------
        pd.DataFrame
            Columns: test_type, test_name, n_tests, n_pass, n_warn, n_severe, n_fail, n_skip,
            pass_rate, warn_rate, fail_rate
        """
        if df_all.empty:
            return pd.DataFrame(columns=[
                "test_type", "test_name", "n_tests", "n_pass", "n_warn",
                "n_severe", "n_fail", "n_skip", "pass_rate", "warn_rate", "fail_rate"
            ])

        if "_is_placeholder" in df_all.columns:
            df_all = df_all[~df_all["_is_placeholder"]]

        flag_order = ["PASS", "WARN", "SEVERE", "FAIL", "SKIP"]
        flag_counts = (
            df_all.groupby(["test_type", "test_name"])
            .flag.value_counts()
            .unstack(fill_value=0)
            .reindex(columns=flag_order, fill_value=0)
        )

        df_per_test = flag_counts.rename(columns={
            "PASS": "n_pass",
            "WARN": "n_warn",
            "SEVERE": "n_severe",
            "FAIL": "n_fail",
            "SKIP": "n_skip",
        }).reset_index()

        df_per_test["n_tests"] = df_per_test[[
            "n_pass", "n_warn", "n_severe", "n_fail", "n_skip"
        ]].sum(axis=1)

        df_per_test["pass_rate"] = df_per_test["n_pass"] / df_per_test["n_tests"]
        df_per_test["warn_rate"] = df_per_test["n_warn"] / df_per_test["n_tests"]
        df_per_test["fail_rate"] = df_per_test["n_fail"] / df_per_test["n_tests"]

        df_per_test = df_per_test[[
            "test_type", "test_name", "n_tests", "n_pass", "n_warn", "n_severe",
            "n_fail", "n_skip", "pass_rate", "warn_rate", "fail_rate"
        ]]

        return df_per_test

    def _generate_all_tests(self, entity_qc: EntityQCStatus, sample_filter: set[str]) -> pd.DataFrame:
        """Generate table with all test combinations for every sample-step pair.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object.
        sample_filter : set[str]
            Sample IDs to include.

        Returns
        -------
        pd.DataFrame
            Columns: sample_id, step_name, test_type, test_name, flag
            Includes placeholder rows with empty test_type/test_name for steps without tests.
        """
        records = []

        for sample_id, qc_run in entity_qc.sample_qc.items():
            if sample_id not in sample_filter:
                continue

            for step_name, step in qc_run.steps.items():
                # Iterate through all tests in this step
                for test_id, test in step.tests.items():
                    records.append({
                        "sample_id": sample_id,
                        "step_name": step_name,
                        "test_type": test.test_type,
                        "test_name": test.test_name,
                        "flag": test.flag.value,
                        "_is_placeholder": False,
                    })
                if not step.tests:
                    records.append({
                        "sample_id": sample_id,
                        "step_name": step_name,
                        "test_type": "",
                        "test_name": "",
                        "flag": step.flag.value,
                        "_is_placeholder": True,
                    })

        if not records:
            return pd.DataFrame(columns=[
                "sample_id", "step_name", "test_type", "test_name", "flag", "_is_placeholder"
            ])

        return pd.DataFrame(records)

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Any,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Generate a diagnostic figure identified by test_key.

        For step-level QC, test_key specifies the visualization type:
        - "heatmap": Sample × Step QC heatmap with flags and hover messages

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing step and sample data.
        test_key : Any
            Visualization type identifier:
            - "heatmap": Samples as rows, steps as columns, colored by flag
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading additional data if needed.
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader.
        step_id : str | None
            Unused for step-level QC (placeholder for interface compatibility).
        **kwargs : Any
            Additional plotting options (passed to visualization functions).

        Returns
        -------
        go.Figure
            Plotly figure object.

        Raises
        ------
        ValueError
            If test_key is not a recognized visualization type.
        """
        # Dispatch to appropriate visualization function
        if test_key == "heatmap":
            return self._generate_heatmap_figure(entity_qc, **kwargs)
        else:
            available = ["heatmap"]
            raise ValueError(
                f"Unknown figure type '{test_key}' for step-level QC. "
                f"Available types: {', '.join(available)}"
            )

    def _generate_heatmap_figure(
        self,
        entity_qc: EntityQCStatus,
        **kwargs: Any,
    ) -> go.Figure:
        """Generate a scatter plot of samples vs steps with flags.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object.
        **kwargs : Any
            Additional plotting options (e.g., height, width adjustments).

        Returns
        -------
        go.Figure
            Plotly scatter plot figure with square markers colored by flag.
        """
        # Determine which samples to include
        sample_filter = set(entity_qc.sample_qc.keys())

        # Generate long format
        df_long = self._generate_long_format(entity_qc, sample_filter)

        if df_long.empty:
            raise ValueError("No QC data available to generate figure")

        # Get steps in execution order
        step_order = self._get_step_order(entity_qc)

        # Create step to x-coordinate mapping
        step_to_x = {step: i for i, step in enumerate(step_order)}

        # Get unique samples and sort them
        samples = sorted(df_long['sample_id'].unique())
        sample_to_y = {sample: i for i, sample in enumerate(samples)}

        # Map flags to colors
        flag_to_color = {
            "PASS": "#2ecc71",    # green
            "WARN": "#f39c12",    # orange
            "SEVERE": "#e74c3c",  # red
            "FAIL": "#c0392b",    # dark red
            "SKIP": "#95a5a6",    # grey
        }

        # Create figure
        fig = go.Figure()

        # Add a trace for each flag type (for legend)
        flag_order = ["PASS", "WARN", "SEVERE", "FAIL", "SKIP"]

        for flag in flag_order:
            flag_data = df_long[df_long['flag'] == flag]

            x_coords = [step_to_x[row['step_name']] for _, row in flag_data.iterrows()]
            y_coords = [sample_to_y[row['sample_id']] for _, row in flag_data.iterrows()]

            hover_texts = []
            for _, row in flag_data.iterrows():
                hover_text = f"<b>{row['sample_id']} / {row['step_name']}</b><br>"
                hover_text += f"Flag: <b>{row['flag']}</b>"
                if row['message']:
                    hover_text += f"<br>Issues: {row['message']}"
                hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                name=flag,
                marker=dict(
                    size=20,
                    color=flag_to_color[flag],
                    symbol='square',
                    line=dict(width=1, color='black'),
                ),
                text=hover_texts,
                hoverinfo='text',
            ))

        # Apply custom sizing
        height = kwargs.get("height")
        width = kwargs.get("width")
        if height is None:
            height = max(400, len(samples) * 20 + 200)
        if width is None:
            width = max(600, len(step_order) * 100 + 300)

        fig.update_layout(
            title="QC Status: Samples × Steps",
            xaxis_title="Step",
            yaxis_title="Sample",
            xaxis=dict(
                tickvals=list(range(len(step_order))),
                ticktext=step_order,
            ),
            yaxis=dict(
                tickvals=list(range(len(samples))),
                ticktext=samples,
            ),
            height=height,
            width=width,
            hovermode='closest',
            legend=dict(
                title="QC Status",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            ),
        )

        return fig