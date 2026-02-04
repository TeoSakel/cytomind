from __future__ import annotations
from typing import Any, Mapping, Callable, TypeVar, Sequence, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from cytomind.infra.repo import ProjectRepository
else:
    ProjectRepository = object

R = TypeVar("R") # Type variable for the result returned by the provided callable

import numpy as np
from numpy.typing import NDArray
from anndata import AnnData
from cytomind.domain.pipeline import SampleRef, StepRun, QCRunStatus, QCStepStatus, QCFlag


class BaseStep:
    """
    Default: run per sample; subclasses implement run_for_sample.

    Steps own `per_sample_qc` - computation-time QC that captures:
    - Transient data only available during processing
    - Detailed metrics from processing algorithms
    - Structured test records with thresholds and measurements

    Steps should NOT compute user-facing summary_qc for review purposes.
    That responsibility belongs to RevisionHandlers via generate_review_summary().

    The base summarize_qc() provides minimal aggregation (counts of PASS/WARN/FAIL)
    for simple step completion tracking. For detailed review summaries,
    use RevisionHandler.generate_review_summary().
    """
    default_config: dict[str, Any] = {}

    def __init__(self, repo: ProjectRepository) -> None:
        self.repo = repo
        self.config = dict(self.default_config)
        self.project = self.repo.load_project()

    # --- default run: three-phase execution per batch -----

    def run(self, step_run: StepRun) -> StepRun:
        """Main orchestrator using three-phase execution model.

        For each batch:
        1. prepare_batch() - set up batch-level context (priors, thresholds, etc.)
        2. run_sample() - process each sample using batch context
        3. finalize_batch() - aggregate per-sample results and populate project_updates

        Then:
        4. Summarize QC from all phases
        5. Apply accumulated project_updates to project via update_project()
        6. Track project updates

        Sample filtering:
        - If step_run.inputs["sample_ids"] is provided, only those samples are processed
        - This allows partial reruns within a batch or sample-only processing
        - If no batch_ids but sample_ids are specified, samples are processed directly
          without prepare_batch/finalize_batch phases

        This enables patterns like:
        - prepare_batch computes global thresholds, run_sample applies them
        - finalize_batch aggregates per-sample results for batch-level summaries
        - project_updates defers all persistence until full batch aggregation is visible

        Args:
            step_run (StepRun): execution context (mutable, shared across phases)

        Returns:
            StepRun: updated with outputs, QC, project_updates, and status
        """
        step_run.config = self.merge_config(step_run)
        batch_ids = step_run.inputs.get("batch_ids", [])
        input_sample_ids = step_run.inputs.get("sample_ids", [])  # Optional filter

        # Execute batches with three-phase model
        for batch_id in batch_ids:

            # Phase 1: Prepare batch context
            try:
                output_info, batch_qc = self.prepare_batch(batch_id, step_run)
            except Exception as exc:
                output_info = {}
                batch_qc = QCRunStatus(sample_id=f"batch_{batch_id}", step_run_id=step_run.id)
                step = batch_qc.get_step(self.__class__.__name__)
                step.flag = QCFlag.FAIL
                step.add_reason("PREPARE_BATCH_RUNTIME_ERROR", str(exc))
                step_run.batch_outputs[batch_id] = output_info
                step_run.qc_summary[batch_id] = batch_qc

            step_run.batch_outputs[batch_id] = output_info
            step_run.qc_summary[batch_id] = batch_qc
            if batch_qc.overall_flag == QCFlag.FAIL:
                # Skip processing samples in this batch if preparation failed
                continue

            # Phase 2: Process each sample in batch
            batch = self.project.batches[batch_id]  # prepare_batch ensures batch exists
            # Determine which samples to process: use input filter if provided
            if input_sample_ids:
                samples_to_process = [sid for sid in input_sample_ids if sid in set(batch.sample_ids)]
            else:
                samples_to_process = batch.sample_ids

            for sample_id in samples_to_process:
                try:
                    output_info, qc = self.run_sample(sample_id, step_run)
                except Exception as exc:
                    output_info = {}
                    qc = QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)
                    step = qc.get_step(self.__class__.__name__)
                    step.flag = QCFlag.FAIL
                    step.add_reason("RUN_SAMPLE_RUNTIME_ERROR", str(exc))
                step_run.sample_outputs[sample_id] = output_info
                step_run.per_sample_qc[sample_id] = qc

            # Phase 3: Finalize batch (aggregate per-sample results)
            try:
                # Reuse the same QCRunStatus from prepare_batch to persist steps
                output_info, batch_qc = self.finalize_batch(batch_id, step_run, batch_qc)
            except Exception as exc:
                output_info = {}
                step = batch_qc.get_step(self.__class__.__name__)
                step.flag = QCFlag.FAIL
                step.add_reason("FINALIZE_BATCH_RUNTIME_ERROR", str(exc))

            step_run.batch_outputs[batch_id] = output_info
            step_run.qc_summary[batch_id] = batch_qc

        # If no batches specified but sample_ids provided, process samples directly
        if not batch_ids and input_sample_ids:
            for sample_id in input_sample_ids:
                try:
                    output_info, qc = self.run_sample(sample_id, step_run)
                except Exception as exc:
                    output_info = {}
                    qc = QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)
                    step = qc.get_step(self.__class__.__name__)
                    step.flag = QCFlag.FAIL
                    step.add_reason("RUN_SAMPLE_RUNTIME_ERROR", str(exc))

                step_run.sample_outputs[sample_id] = output_info
                step_run.per_sample_qc[sample_id] = qc

        # Aggregate overall QC summary from per-sample QC (dict format)
        step_run.qc_summary["_overall_"] = self.summarize_qc(step_run.per_sample_qc)
        step_run = self.update_project(step_run)
        step_run = self.cleanup_step_run(step_run)
        step_run.status = "completed"

        return step_run

    def prepare_batch(self, batch_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        """
        Prepare batch-level context before processing samples.

        This phase runs first for each batch. Use it to:
        - Compute batch-level statistics or priors (global thresholds, fitted parameters)
        - Load pooled batch data for fitting gates, compensation matrices, etc.
        - Set up shared state in step_run.batch_outputs[batch_id]

        Per-sample methods will read batch_outputs[batch_id] to access these priors.

        Default implementation does nothing (no-op).

        Args:
            batch_id: ID of the batch to prepare
            step_run: shared execution context (write to step_run.batch_outputs[batch_id])

        Returns:
            (output_info, batch_qc) where:
                output_info: dict with batch context/outputs
                batch_qc: QCRunStatus for batch-level QC (stored in step_run.qc_summary[batch_id])
        """
        qc = QCRunStatus(sample_id=f"batch_{batch_id}", step_run_id=step_run.id)
        try:
            batch = self.project.batches[batch_id]
        except KeyError:
            step = qc.get_step("LoadBatch")
            step.flag = QCFlag.FAIL
            step.add_reason("BATCH_NOT_FOUND", f"Batch {batch_id} not found in project.")
        return {}, qc

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        """
        Process one sample, optionally using batch context from prepare_batch.

        Can read batch context from step_run.batch_outputs[batch_id] that was
        set up by prepare_batch(). Write per-sample results to
        step_run.sample_outputs[sample_id] for finalize_batch() to aggregate.

        Returns:
          (output_info, qc_info) for this sample
          where:
            output_info: dict to store under step_run.sample_outputs[sample.id]
            qc_info: QCRunStatus to store under step_run.per_sample_qc[sample.id]
        """
        qc = QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason("SAMPLE_NOT_FOUND", f"Sample {sample_id} not found in project.")
        return {}, qc

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        """
        Finalize batch after all samples are processed.

        This phase runs last for each batch. Use it to:
        - Read per-sample results from step_run.sample_outputs
        - Aggregate results (pooled statistics, batch-level summaries)
        - Populate step_run.project_updates with changes to persist
        - Update step_run.batch_outputs[batch_id] with aggregated results
        - Add finalize-specific QC steps to the provided qc

        Default implementation does nothing (no-op).

        Args:
            batch_id: ID of the batch to finalize
            step_run: shared execution context (read from sample_outputs, write to project_updates)
            qc: QCRunStatus from prepare_batch (add new steps to this)

        Returns:
            (output_info, batch_qc) where:
                output_info: dict with aggregated batch results
                qc: Updated QCRunStatus with finalize steps added
        """
        # Do nothing by default
        output_info = step_run.batch_outputs.get(batch_id, {})
        return output_info, qc

    def summarize_qc(self, per_sample_qc: Mapping[str, QCRunStatus]) -> dict:
        sample_qc = [qc for sid, qc in per_sample_qc.items() if sid in self.project.samples]
        overall = QCFlag.combine(qc.overall_flag for qc in sample_qc)
        sample_counts = Counter(qc.overall_flag.value for qc in sample_qc)

        # Include per-sample flags for detailed review
        per_sample_flags = {
            qc.sample_id: qc.overall_flag.value
            for qc in sample_qc
        }

        return {
            "overall_flag": overall.value,
            "n_samples": sample_counts.total(),
            "n_pass": sample_counts["PASS"],
            "n_warn": sample_counts["WARN"],
            "n_fail": sample_counts["FAIL"],
            "per_sample_flags": per_sample_flags,
        }

    def update_project(self, step_run: StepRun) -> StepRun:
        """Apply project_updates accumulated by phases to the project.

        The step implementation (prepare_batch/finalize_batch) appends update dicts
        to project_updates list. This method applies them sequentially, then cleans
        up the updates to only keep IDs of updated entries.

        Subclasses can override to add step-specific cleanup or validation before
        or after calling super().update_project().

        Args:
            step_run: execution context with accumulated project_updates list

        Returns:
            step_run: with project updates applied and cleaned up
        """
        # Apply updates and collect cleaned versions
        cleaned_updates = []
        for updates in step_run.project_updates:
            self.repo.update_project_metadata(**updates)

            # Clean up: keep only IDs of updated entries
            cleaned = updates.copy()
            if "samples" in updates:
                cleaned["samples"] = list(updates["samples"].keys())
            if "batches" in updates:
                cleaned["batches"] = list(updates["batches"].keys())
            if "compensations" in updates:
                cleaned["compensations"] = [c.id if hasattr(c, 'id') else c for c in updates["compensations"]]
            if "panel" in updates:
                cleaned["panel"] = [ch.pnn if hasattr(ch, 'pnn') else ch for ch in updates["panel"]]
            if "dimensions" in updates:
                cleaned["dimensions"] = {
                    layer: [dim.id if hasattr(dim, 'id') else dim for dim in dims]
                    for layer, dims in updates["dimensions"].items()
                }
            if "transformations" in updates:
                cleaned["transformations"] = list(updates["transformations"].keys())
            if "gating_strategies" in updates:
                cleaned["gating_strategies"] = [gs.id if hasattr(gs, 'id') else gs for gs in updates["gating_strategies"]]

            cleaned_updates.append(cleaned)

        # Replace full updates with cleaned versions
        step_run.project_updates = cleaned_updates
        return step_run

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        """
        Cleanup any temporary data in step_run before persisting.

        Default implementation does nothing. Subclasses can override
        to remove large intermediate data structures if needed.
        """
        return step_run

    # --- utility methods for steps -----

    def merge_config(self, step_run: StepRun) -> dict:
        cfg = self.config.copy()
        cfg.update(step_run.config)
        return cfg

    def run_step(
        self,
        qc: QCRunStatus,
        step_name: str,
        func: Callable[..., R],
        *,
        reason_code_fail: str | None = None,
        reraise: bool = False,
        **kwargs,
    ) -> tuple[R | None, QCStepStatus]:
        """
        Run `func(**kwargs)` and capture QC for this sub-step.
        Returns (result, success: bool).
        """
        step_status = qc.get_step(step_name)
        if reason_code_fail is None:
            reason_code_fail = f"{step_name.upper()}_RUN_ERROR"
        try:
            result = func(**kwargs)
            step_status.flag = QCFlag.PASS
            return result, step_status
        except Exception as e:
            step_status.flag = QCFlag.FAIL
            # prefer add_reason to capture both reason code and message together
            step_status.add_reason(reason_code_fail or "STEP_ERROR", str(e))
            if reraise:
                raise
            return None, step_status

    def load_layer(
        self,
        sample: SampleRef,
        qc: QCRunStatus,
        layer: str | None = None,
    ) -> tuple[AnnData | None, QCStepStatus]:
        """
        Load AnnData for the sample's layer and update QC.
        Returns (AnnData, updated qc).
        """
        return self.run_step(
            qc,
            f"load_anndata_{layer}",
            self.repo._load_sample_layer,
            reason_code_fail="LOAD_ANNDATA_ERROR",
            sample_id=sample.id,
            layer=layer or sample.default_layer,
        )

    def load_adata(
        self,
        sample: SampleRef,
        qc: QCRunStatus,
        layer: str | None = None,
        mask: NDArray[np.bool_] | slice | None = None,
        select: Sequence[str] | slice |None = None,
    ) -> tuple[AnnData | None, QCStepStatus]:
        """
        Load AnnData for the sample and update QC.
        Returns (AnnData, updated qc).
        """
        layer = layer or sample.default_layer
        mask = mask or slice(None)
        select = select or slice(None)

        return self.run_step(
            qc,
            f"load_anndata_{layer}",
            self.repo.load_sample_adata,
            reason_code_fail="LOAD_ANNDATA_ERROR",
            sample_id=sample.id,
            layer=layer,
            mask=mask,
            select=select,
        )

    def save_adata(
        self,
        sample: SampleRef,
        adata: AnnData,
        qc: QCRunStatus,
        layer: str
    ) -> tuple[None, QCStepStatus]:
        """
        Save AnnData to the sample's path and update QC.
        Returns True if successful, False otherwise.
        """

        return self.run_step(
            qc,
            f"save_anndata_{layer}",
            self.repo.save_sample_adata,
            reason_code_fail="SAVE_ANNDATA_ERROR",
            sample_id=sample.id,
            adata=adata,
            layer=layer
        )