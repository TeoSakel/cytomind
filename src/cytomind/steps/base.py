from __future__ import annotations
from re import A
from typing import Any, Callable, TypeVar, Sequence, TYPE_CHECKING

import numpy as np
from cytomind.domain.qc import QCRunStatus, QCStepStatus, QCFlag
from cytomind.qc.step import StepQCEvaluator

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray
    from cytomind.domain.pipeline import SampleRef, StepRun
    from cytomind.infra.repo import ProjectRepository
    R = TypeVar("R") # Type variable for the result returned by the provided callable
else:
    AnnData = object
    NDArray = object
    SampleRef = object
    StepRun = object
    ProjectRepository = object
    R = object


class BaseStep:
    """
    Default: run per sample; subclasses implement run_for_sample.

    Best Hybrid QC Pattern:
    Steps emit test records during execution with:
    - test_type, test_name: categorize the test
    - metadata: context (gate_id, channel, cutpoint, R², bounds, etc.)
    - metrics: measured values (proportion_passing, sigma, etc.)
    - status: "PENDING" (no threshold decisions during execution)

    Example:
        test = QCTestRecord(
            test_type="gate_fit",
            test_name="mindensity_1d",
            metadata={"gate_id": "cd3", "channel": "CD3", "cutpoint": 2500.3},
            metrics={"proportion_passing": 0.42, "r_squared": 0.91},
            status="PENDING",  # Evaluator will classify
        )
        step = qc.get_step("FitGate")
        step.add_reason("GATE_FIT_RESULT", test=test)

    QC Evaluators (run after execution) will:
    - Apply thresholds based on config
    - Assign final status (PASS/WARN/FAIL)
    - Add human-readable messages

    Steps should NOT compute user-facing summary_qc for review purposes.
    That responsibility belongs to RevisionHandlers via generate_review_summary().
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
                batch_qc = step_run.qc.batch_qc
                step = batch_qc.get_step(self.__class__.__name__)
                step.flag = QCFlag.FAIL
                step.add_reason("PREPARE_BATCH_RUNTIME_ERROR", str(exc))

            step_run.batch_outputs[batch_id] = output_info
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
                    qc = step_run.qc.get_sample_steps(sample_id)
                    step = qc.get_step(self.__class__.__name__)
                    step.flag = QCFlag.FAIL
                    step.add_reason("RUN_SAMPLE_RUNTIME_ERROR", str(exc))
                step_run.sample_outputs[sample_id] = output_info

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

        # If no batches specified but sample_ids provided, process samples directly
        if not batch_ids and input_sample_ids:
            for sample_id in input_sample_ids:
                try:
                    output_info, qc = self.run_sample(sample_id, step_run)
                except Exception as exc:
                    output_info = {}
                    qc = step_run.qc.get_sample_steps(sample_id)
                    step = qc.get_step(self.__class__.__name__)
                    step.flag = QCFlag.FAIL
                    step.add_reason("RUN_SAMPLE_RUNTIME_ERROR", str(exc))

                step_run.sample_outputs[sample_id] = output_info

        # Apply project updates, then evaluate and persist QC
        step_run = self.update_project(step_run)
        try:
            self.evaluate_step_run(step_run)
        except Exception as exc:
            step_qc = step_run.qc.batch_qc.get_step(self.__class__.__name__)
            step_qc.flag = QCFlag.FAIL
            step_qc.add_reason("STEP_QC_EVALUATION_ERROR", str(exc))

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
                batch_qc: QCRunStatus for batch-level QC (accessible via step_run.qc.batch_qc)
        """
        qc = step_run.qc.batch_qc
        try:
            batch = self.project.batches[batch_id]
            out = {"batch": batch}
        except KeyError:
            step = qc.get_step("LoadBatch")
            step.flag = QCFlag.FAIL
            step.add_reason("BATCH_NOT_FOUND", f"Batch {batch_id} not found in project.")
            out = {}
        return out, qc

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        """
        Process one sample, optionally using batch context from prepare_batch.

        Can read batch context from step_run.batch_outputs that was
        set up by prepare_batch(). Write per-sample results to
        step_run.sample_outputs[sample_id] for finalize_batch() to aggregate.

        Returns:
          (output_info, qc_info) for this sample
          where:
            output_info: dict to store under step_run.sample_outputs[sample.id]
            qc_info: QCRunStatus accessible via step_run.qc.per_sample_steps[sample.id]
        """
        qc = step_run.qc.get_sample_steps(sample_id)
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
        - Populate step_run.evaluable_products with products ready for QC evaluation
        - Add finalize-specific QC steps to the provided qc

        Note: Only add to evaluable_products if the product is fully processed and safe to evaluate.
        Example: AddSamplesStep adds samples, panel, dimensions, batches but NOT compensations
        (they are registered but not yet applied to sample data).

        Default implementation does nothing (no-op).

        Args:
            batch_id: ID of the batch to finalize
            step_run: shared execution context (read from sample_outputs, write to project_updates & evaluable_products)
            qc: QCRunStatus from prepare_batch (add new steps to this)

        Returns:
            (output_info, batch_qc) where:
                output_info: dict with aggregated batch results
                qc: Updated QCRunStatus with finalize steps added
        """
        # Do nothing by default
        output_info = step_run.batch_outputs.get(batch_id, {})
        return output_info, qc

    def evaluate_step_run(self, step_run: StepRun) -> None:
        """
        Evaluate and persist QC for the step run and its products.

        Uses entity-oriented evaluation:
        - Step's own QC always uses StepQCEvaluator (entity_type="step")
        - Product QC delegates to entity-specific evaluators from registry
        """
        step_evaluator = StepQCEvaluator(config=self.config)

        # Evaluate and persist product QC
        for qc_status in step_evaluator.evaluate_step_products(self.repo, step_run):
            step_run.qc.update_batch_steps(qc_status.batch_qc, merge=True)
            for sid, qc in qc_status.sample_qc.items():
                step_run.qc.update_sample_steps(sid, qc, merge=True)
            self.repo.save_qc_entity_status(qc_status)

        # Evaluate step's own QC
        step_qc = step_evaluator.parse_step(step_run)
        step_qc = step_evaluator.update_entity_qc(
            entity=step_run,
            entity_qc=step_qc,
            dataloader=self.repo._dataloader,  # Not needed for step QC
            dataloader_context=None,
        )
        self.repo.save_qc_entity_status(step_qc)


    def update_project(self, step_run: StepRun) -> StepRun:
        """Apply project_updates accumulated by phases to the project.

        The step implementation (prepare_batch/finalize_batch) appends update dicts
        to project_updates list. This method applies them sequentially.

        Subclasses can override to add step-specific validation before
        or after calling super().update_project().

        Args:
            step_run: execution context with accumulated project_updates list

        Returns:
            step_run: with project updates applied
        """
        # Apply updates to project
        for updates in step_run.project_updates:
            self.repo.update_project_metadata(**updates)

        return step_run

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        """
        Cleanup any temporary data in step_run before persisting.

        Removes custom objects from project_updates and keeps only IDs
        to avoid JSON serialization issues. Subclasses can override
        to remove additional large intermediate data structures if needed.

        Args:
            step_run: execution context to cleanup

        Returns:
            step_run: with custom objects replaced by IDs
        """
        # Clean up project_updates: keep only IDs of updated entries
        for updates in step_run.project_updates:
            if "samples" in updates:
                updates["samples"] = [sref.id for sref in updates["samples"]]
            if "batches" in updates:
                updates["batches"] = [bref.id for bref in updates["batches"]]
            if "compensations" in updates:
                updates["compensations"] = [cref.id for cref in updates["compensations"]]
            if "panel_catalog" in updates:
                updates["panel_catalog"] = list(updates["panel_catalog"].keys())
            if "layers" in updates:
                updates["layers"] = {
                    layer: [dim.id for dim in dims]
                    for layer, dims in updates["layers"].items()
                }
            if "transforms" in updates:
                updates["transforms"] = list(updates["transforms"].keys())
            if "gating_strategy" in updates and updates["gating_strategy"] is not None:
                updates["gating_strategy"] = updates["gating_strategy"].id

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
        mask: NDArray[np.bool_] | dict[str, NDArray[np.bool_]] | slice | None = None,
        select: Sequence[str] | slice | None = None,
    ) -> tuple[AnnData | None, QCStepStatus]:
        """
        Load AnnData for the sample and update QC.
        Returns (AnnData, updated qc).
        """
        layer = layer or sample.default_layer
        layer_dims = [dim.id for dim in self.project.layers.get(layer, [])]
        select = select or slice(None)
        if isinstance(select, Sequence) and list(select) == layer_dims:
            select = slice(None) # optimize for common case of selecting all dimensions
        mask = mask or slice(None)
        if isinstance(mask, dict):
            mask = next(iter(mask.values()))
        if isinstance(mask, np.ndarray) and mask.all():
            mask = slice(None) # optimize for common case of no masking

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
        layer: str,
        overwrite: bool = False,
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
            layer=layer,
            overwrite=overwrite
        )