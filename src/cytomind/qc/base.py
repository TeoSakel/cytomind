"""
Base class for pluggable QC evaluation.

QC evaluators provide detailed analysis on completed step runs,
computing metrics, test results, and generating user-facing summaries.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from collections import Counter
from pathlib import Path

from cytomind.domain.pipeline import QCRunStatus, QCFlag

if TYPE_CHECKING:
    from cytomind.domain.pipeline import StepRun
    from cytomind.infra.repo import ProjectRepository
else:
    StepRun = object
    ProjectRepository = object


class StepQCEvaluator(ABC):
    """
    Pluggable QC evaluation for a specific step type.

    Evaluators are stateless and can be called multiple times:
    - On original step execution
    - During revision to re-evaluate
    - For generating review summaries

    Key responsibility: Transform raw step execution into actionable QC insights.
    """

    @abstractmethod
    def run_step_qc(self, repo: ProjectRepository, step_run: StepRun) -> StepRun:
        """
        Perform detailed QC analysis on a completed step run.

        Updates step_run.per_sample_qc with detailed metrics and test results.
        Updates step_run.qc_summary with aggregated statistics.
        Returns the modified step_run.

        Parameters
        ----------
        repo : ProjectRepository
            Main repository for reading data
        step_run : StepRun
            Completed step run to analyze

        Returns
        -------
        StepRun
            Modified step_run with detailed QC data populated
        """
        pass

    @abstractmethod
    def generate_review_summary(
        self, repo: ProjectRepository, step_run: StepRun
    ) -> dict[str, Any]:
        """
        Generate user-facing summary for review UI.

        Transforms detailed QC data into formatted tables, metrics,
        and recommendations suitable for user review.

        Parameters
        ----------
        repo : ProjectRepository
            Main repository for reading data
        step_run : StepRun
            Completed step run with QC data

        Returns
        -------
        dict
            User-facing summary with tables, metrics, recommendations
        """
        pass

    def get_sample_qc(self, sample_id: str, step_run: StepRun) -> QCRunStatus:
        qc = step_run.per_sample_qc.get(sample_id)
        if not qc:
            return QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)

        return QCRunStatus.from_dict(qc) if isinstance(qc, dict) else qc

    def _summarize_qc(self, step_run: StepRun, *args, **kwargs) -> dict[str, Any]:
        """Aggregate per-sample QC flags into a step-level summary."""
        sample_qc = list(step_run.per_sample_qc.values())
        overall = QCFlag.combine(qc.overall_flag for qc in sample_qc)
        counts = Counter(qc.overall_flag.value for qc in sample_qc)
        per_sample_flags = {qc.sample_id: qc.overall_flag.value for qc in sample_qc}
        return {
            "overall_flag": overall.value,
            "n_samples": len(sample_qc),
            "n_pass": counts.get("PASS", 0),
            "n_warn": counts.get("WARN", 0),
            "n_fail": counts.get("FAIL", 0),
            "per_sample_flags": per_sample_flags,
        }