from __future__ import annotations

from cytomind.domain.constants import GML_VERSION
from cytomind.domain.pipeline import StepRun
from cytomind.domain.qc import QCRunStatus, QCFlag
from cytomind.domain.gates import GatingStrategyRef
from cytomind.utils import string_to_filename
from .base import BaseStep
from . import register_step

@register_step("add_gating_strategy")
class AddGatingStrategyStep(BaseStep):
    """
    Add a gating strategy to the project.

    This step creates a new gating strategy reference and initializes it in the project.
    The gating strategy can be populated with gates and transformations in subsequent steps.

    Input config should include:
    - strategy_id: str - unique identifier for the gating strategy
    - strategy_name: str - human-readable name for the gating strategy
    - batch_id: str - batch this strategy applies to
    - description: str (optional) - human-readable description
    """

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        step_config = qc.get_step("validate_config")
        try:
            given_strategy_id: str = step_run.config["strategy_id"]
        except KeyError:
            step_config.flag = QCFlag.FAIL
            step_config.add_reason(
                code="INVALID_CONFIG",
                message="Missing required config: strategy_id is required."
            )
            return {}, qc

        # Check and sanitize strategy_id
        strategy_id = string_to_filename(given_strategy_id)
        if strategy_id != given_strategy_id:
            step_config.flag = QCFlag.WARN
            step_config.add_reason(
                code="STRATEGY_ID_SANITIZED",
                message=f"strategy_id '{given_strategy_id}' was sanitized to '{strategy_id}'."
            )

        strategy_name: str = step_run.config.get("strategy_name", given_strategy_id)
        description: str = step_run.config.get("description", "")
        glm_version: str = step_run.config.get("glm_version", GML_VERSION)

        # Check if strategy ID already exists
        step_check = qc.get_step("check_strategy_uniqueness")
        if strategy_id in self.project.gating_strategies:
            step_check.flag = QCFlag.FAIL
            step_check.add_reason(
                code="STRATEGY_EXISTS",
                message=f"Gating strategy with ID '{strategy_id}' already exists in project."
            )
            return {}, qc

        # collect compensations and transformations from samples in the batch
        # samples = self.project.batches[batch_id].sample_ids
        # comps = set(self.project.samples[sid].compensation or "raw" for sid in samples)
        # comps.discard("raw")
        # layers = set(self.project.samples[sid].default_layer for sid in samples)
        # transf = set(dim.transform_id for layer in layers for dim in self.project.dimensions[layer])

        # Create new gating strategy to add to project
        strategy = GatingStrategyRef(
            id=strategy_id,
            name=strategy_name,
            batch_id=batch_id,
            path=None,
            description=description,
            glm_version=glm_version,
        )
        strategy.init_graph()
        step_run.project_updates.append({"gating_strategies": [strategy]})

        return {}, qc

    def merge_config(self, step_run: StepRun) -> dict:
        batch_ids = step_run.inputs.get("batch_ids", [])
        if not batch_ids:
            raise ValueError("AddGatingStrategyStep requires one batch_id in inputs.")
        if len(step_run.inputs.get("batch_ids", [])) != 1:
            raise ValueError("AddGatingStrategyStep only supports single batch_id per run.")
        if step_run.inputs.get("sample_ids"):
            raise ValueError("AddGatingStrategyStep does not support sample_ids in inputs.")

        return super().merge_config(step_run)