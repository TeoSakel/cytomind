"""
Step-level QC summarization.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

from .base import EntityQCEvaluator

if TYPE_CHECKING:
    from cytomind.domain.qc import EntityQCStatus
else:
    EntityQCStatus = object


class StepQCEvaluator(EntityQCEvaluator):
    """Default step QC summarizer."""

    def run_entity_qc(
        self,
        entity_id: str,  # step_run_id
        *,
        sample_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EntityQCStatus:

        step_run = self.repo.load_step_run(entity_id)
        return step_run.qc