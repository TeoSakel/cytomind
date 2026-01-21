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

    # --- default run: loops over samples (optionally parallel) -----

    def run(self, step_run: StepRun) -> StepRun:
        """Main Function that must be called from Pipeline orchestrator

        First loops over samples and calls `run_sample` for each sample.
        Then calls `run_all_samples` to produce overall outputs.
        Finally summarizes QC information generated from all previous steps
        using `summarize_qc` and and updates project state using `update_project`.

        Args:
            step_run (StepRun): contains the information necessary to
            run the step.

        Returns:
            StepRun: updated step run with outputs and QC information
        """

        sample_ids = step_run.inputs.get("sample_ids", [])
        per_sample_qc: dict[str, QCRunStatus] = {}

        step_run.config = self.merge_config(step_run)
        # TODO: here you can plug in multiprocessing/threading if you want
        for sample_id in sample_ids:
            try:
                output_info, qc = self.run_sample(sample_id, step_run)
            except Exception as exc:
                # on error, produce empty output and a failing QC with message
                output_info = {}
                qc = QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)
                step = qc.get_step(self.__class__.__name__)
                step.flag = QCFlag.FAIL
                step.add_reason("RUN_TIME_ERROR", str(exc))

            step_run.outputs[sample_id] = output_info
            step_run.per_sample_qc[sample_id] = qc

        # Run step at batch level if needed
        batch_ids = step_run.inputs.get("batch_ids", [])
        for batch_id in batch_ids:
            try:
                output_info, qc = self.run_batch(batch_id, step_run)
            except Exception as exc:
                output_info = {}
                qc = QCRunStatus(sample_id=f"batch_{batch_id}", step_run_id=step_run.id)
                step = qc.get_step(self.__class__.__name__)
                step.flag = QCFlag.FAIL
                step.add_reason("RUN_TIME_ERROR", str(exc))

            step_run.outputs[qc.sample_id] = output_info
            step_run.per_sample_qc[qc.sample_id] = qc

        step_run.qc_summary = self.summarize_qc(per_sample_qc)
        step_run = self.update_project(step_run)
        step_run.status = "completed"

        return step_run

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        """
        Do the actual work for *one* sample.

        Returns:
          (output_info, qc_info) for this sample
          where:
            output_info: dict to store under step_run.outputs[sample.id]
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

    def run_batch(self, batch_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        """Summarize overall output from this step run. Method to run after run()."""
        qc = QCRunStatus(sample_id=f"batch_{batch_id}", step_run_id=step_run.id)
        return {}, qc

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
        """Update project metadata based on step run results."""
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