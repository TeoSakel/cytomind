from __future__ import annotations
from typing import Any, TYPE_CHECKING, Sequence
from numpy.typing import NDArray

import numpy as np
import anndata as ad

from cytomind.domain.gates import GateNode
from cytomind.domain.qc import QCRunStatus, QCFlag
from cytomind.gates import GateRegistry

if TYPE_CHECKING:
    from cytomind.domain.pipeline import BatchRef, SampleRef, StepRun
    from cytomind.gates.base import Gate
    from cytomind.gates.glm_gates import QuadrantGate
else:
    Gate = object
    QuadrantGate = object

from .base import BaseStep
from . import register_step


@register_step("add_gate")
class AddGateStep(BaseStep):
    """
    Add a gate to a gating strategy and compute masks for samples.

    This step:
    1. Loads the gating strategy from the project
    2. For each sample in the batch, fits and applies the gate
    3. Saves resulting masks to sample .obs
    4. Updates the gating strategy graph with the new gate node
    5. Persists the updated strategy

    Handles special cases:
    - QuadrantGate: returns multiple region masks
    - BooleanGate: requires all expression variable masks
    """

    default_config = {
        "n_events_per_sample": 10_000,
        "seed": 42,
        "fit_on_batch": True,
        "custom_fit": set(),
    }

    def merge_config(self, step_run: StepRun) -> dict:
        """Validate config before execution."""
        # TODO: should I move checks to run so that I can capture them in QC?
        batch_ids = step_run.inputs.get("batch_ids", [])
        if not batch_ids:
            raise ValueError("AddGateStep requires one batch_id in inputs.")
        if len(batch_ids) != 1:
            raise ValueError("AddGateStep only supports single batch_id per run.")

        if "strategy_id" not in step_run.config:
            raise ValueError("AddGateStep requires 'strategy_id' in config.")
        if "gate_node" not in step_run.config:
            raise ValueError("AddGateStep requires 'gate_node' in config.")

        step_run.config["custom_fit"] = set(step_run.config.get("custom_fit", []))
        return super().merge_config(step_run)

    def prepare_batch(self, batch_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        output, qc = super().prepare_batch(batch_id, step_run)

        try:
            batch_ref = self.project.batches[batch_id]
        except KeyError:
            step = qc.get_step("LoadBatch")
            step.flag = QCFlag.FAIL
            step.add_reason("BATCH_NOT_FOUND", f"Batch {batch_id} not found in project.")
            return {}, qc

        strategy_id = step_run.config["strategy_id"]
        if not strategy_id in self.project.gating_strategies:
            step = qc.get_step("LoadGatingStrategy")
            step.flag = QCFlag.FAIL
            step.add_reason("STRATEGY_NOT_FOUND", f"Gating strategy {strategy_id} not found in project.")
            return {}, qc

        # Load gate
        gate_node, _ = self.run_step(
            qc,
            "ParseGateNode",
            GateNode.from_dict,
            data=step_run.config["gate_node"],
            reason_code_fail="GATE_NODE_PARSE",
        )
        if gate_node is None:
            return {}, qc

        # Check if gate already exists
        if self.repo.gate_node_path(strategy_id, gate_node.id).exists():
            step = qc.get_step("CheckGateExists")
            step.flag = QCFlag.FAIL
            step.add_reason(
                "GATE_ALREADY_EXISTS",
                f"Gate '{gate_node.id}' already exists in strategy. Use `RerunGateStep` to update existing gates."
            )
            return {}, qc

        # Initialize gate instance
        gate, step = self.run_step(
            qc,
            "InitializeGate",
            self._initialize_gate,
            gate_node=gate_node,
            reason_code_fail="GATE_INIT_ERROR"
        )
        if gate is None:
            return {}, qc

        output["gate_node"] = gate_node
        output["gate"] = gate

        if not (step_run.config.get("fit_on_batch", False) and gate.tunable):
            return output, qc

        # Build pooled batch AnnData for batch-level fitting
        batch_adata, step = self.run_step(
            qc,
            "BuildBatchAnnData",
            self._build_batch_adata,
            reason_code_fail="BUILD_BATCH_ADATA_ERROR",
            batch=batch_ref,
            strategy_id=strategy_id,
            gate_node=gate_node,
            step_run=step_run,
        )
        if batch_adata is None:
            return {}, qc

        # Fit gate on pooled batch data
        # For batch fitting, pass empty mask dict since masks already applied
        gate, step = self.run_step(
            qc,
            "FitGateOnBatch",
            gate.fit,
            events=batch_adata,
            mask={},
            reason_code_fail="GATE_FIT_ERROR"
        )
        if gate is None:
            return {}, qc

        # Store fitted gate for reuse in run_sample
        output["gate"] = gate
        output["fit_on_batch"] = True

        return output, qc

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        """
        Fit and apply gate to one sample, saving masks to .obs.

        Args:
            sample_id: Sample identifier
            step_run: Execution context with config

        Returns:
            (output_info, qc) tuple
        """
        qc = QCRunStatus(sample_id=sample_id, step_run_id=step_run.id)

        # Get sample using run_step
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason("SAMPLE_NOT_FOUND", f"Sample {sample_id} not found in project.")
            return {}, qc

        # Load Gate
        strategy_id = step_run.config["strategy_id"]
        batch_id = step_run.inputs["batch_ids"][0]
        gate_node: GateNode = step_run.batch_outputs[batch_id]["gate_node"]
        gate: Gate = step_run.batch_outputs[batch_id]["gate"]
        should_fit = gate.tunable and (not step_run.config["fit_on_batch"] or sample_id in step_run.config["custom_fit"])

        # Load sample data (backed on disk for now)
        adata, _ = self.load_layer(sample, qc, layer=gate_node.layer)
        if adata is None:
            return {}, qc

        # Load Parent Masks
        parent_dict, _ = self.run_step(
            qc,
            "LoadParentMasks",
            self.repo.load_gating_masks,
            reason_code_fail="PARENT_MASK_ERROR",
            strategy=strategy_id,
            sample=sample,
            mask_ids=gate_node.parent_ids
        )
        if parent_dict is None:
            return {}, qc

        # Fit gate (skip if already fitted on batch)
        if should_fit:
            gate, step = self.run_step(
                qc,
                "FitGate",
                gate.copy().fit,
                events=adata,
                mask=parent_dict,
                reason_code_fail="GATE_FIT_ERROR"
            ) # pyright: ignore[reportAssignmentType]
            if gate is None:
                return {}, qc

        # Apply gate
        mask, _ = self.run_step(
            qc,
            "ApplyGate",
            gate.apply,
            events=adata,
            mask=parent_dict,
            reason_code_fail="GATE_APPLY_ERROR"
        )
        if mask is None:
            return {}, qc

        # Save masks
        _, step = self.run_step(
            qc,
            "SaveGatingMasks",
            self.repo.save_gating_masks,
            reason_code_fail="SAVE_MASKS_ERROR",
            strategy=strategy_id,
            sample=sample,
            masks=mask
        )
        if step.flag == QCFlag.FAIL:
            return {}, qc

        output_info = {
            "n_events": len(adata),
            "n_pass": {k: int(np.sum(v)) for k, v in mask.items()},
            "params": gate.to_dict().get("params", {}) if should_fit else None,
        }

        return output_info, qc

    def finalize_batch(
        self,
        batch_id: str,
        step_run: StepRun,
        qc: QCRunStatus
    ) -> tuple[dict, QCRunStatus]:
        """
        Update gating strategy graph with new gate node.

        Args:
            batch_id: Batch identifier
            step_run: Execution context
            qc: QC status from prepare_batch

        Returns:
            (output_info, qc) tuple
        """

        output_info: dict[str, Any] = step_run.batch_outputs[batch_id]

        # Load the gating strategy
        strategy_id: str = step_run.config["strategy_id"]
        strategy = self.project.gating_strategies[strategy_id]

        # Create GateNode from config
        gate_node: GateNode = output_info["gate_node"]
        gate_data: dict[str, Any] = output_info["gate"].to_dict()
        gate_node.params = gate_data.get("params", {})
        for sample_id, sample_info in step_run.sample_outputs.items():
            params = sample_info.pop("params", None)
            if params:
                gate_node.custom_gates[sample_id] = params

        # Add the gate node with its attributes
        strategy.add_node(gate_node)

        # For QuadrantGate, create individual GateNodes for each quadrant
        if gate_node.gate_type == "QuadrantGate":
            gate: QuadrantGate = output_info["gate"]
            for qid, loc in gate.locations.items():  # type: ignore[attr-defined]
                # Create a GateNode for this quadrant
                quadrant_node = GateNode(
                    id=qid,
                    gate_type="Quadrant",
                    glm_type="Quadrant",
                    dimensions=list(loc.keys()),
                    layer=gate_node.layer,
                    name=qid,
                    parent_ids=[gate_node.id],
                    use_as_complement=False,
                    params=gate_node.params.get(qid, {}),
                    custom_gates={sid: params.get(qid, {}) for sid, params in gate_node.custom_gates.items()}
                )
                # Add node with attributes to graph
                strategy.add_node(quadrant_node)

        step_run.project_updates.append({"gating_strategies": [strategy]})
        output_info["gate_node"] = gate_node.to_dict()
        output_info["strategy_id"] = strategy_id
        del output_info["gate"]

        return output_info, qc

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        step_run.config["custom_fit"] = list(step_run.config.get("custom_fit", []))
        return step_run

    def _initialize_gate(self, gate_node: GateNode) -> Gate:
        try:
            gate_class = GateRegistry.get(gate_node.gate_type)
        except KeyError:
            raise ValueError(
                f"UNKNOWN_GATE_TYPE:"
                f"Gate type '{gate_node.gate_type}' not recognized. "
                f"Available: {list(GateRegistry.list_gates().keys())}"
            )
        return gate_class(
            gate_name=gate_node.name or gate_node.id,
            dimensions=gate_node.dimensions,
            use_as_complement=gate_node.use_as_complement,
            **gate_node.hyperparams
        )

    def _build_batch_adata(
        self,
        batch: BatchRef,
        strategy_id: str,
        gate_node: GateNode,
        seed: int,
        n_events: int,
    ) -> ad.AnnData:
        """
        Build an AnnData pooling events from all samples in batch,
        with up to n_events_per_sample events per sample.

        Parameters
        ----------
        batch : BatchRef
            Batch whose sample_ids will be pooled.
        strategy_id : str
            Gating strategy identifier for loading parent masks.
        gate_node : GateNode
            Gate configuration with layer, dimensions, and parent_ids.
        step_run : StepRun
            Step execution context with config (n_events_per_sample, seed).

        Returns
        -------
        batch_adata : AnnData
            Concatenated AnnData with events from all samples.
            Has a column `sample_id` in .obs.
        """
        rng = np.random.default_rng(seed)
        per_sample_adatas: list[ad.AnnData] = []

        for sample_id in batch.sample_ids:
            try:
                sample = self.project.samples[sample_id]
            except KeyError:
                raise KeyError(f"SAMPLE_NOT_FOUND: Sample {sample_id} not found in project.")

            # Get parent mask if gate has parent
            try:
                parent_mask = self._get_parent_mask(strategy_id, gate_node.parent_ids, sample)
            except Exception as e:
                raise RuntimeError(f"PARENT_MASK_ERROR: Failed to load parent masks for sample {sample_id}: {e}")

            # Create subsampled mask
            try:
                mask = self._create_subsample_mask(
                    parent_mask=parent_mask,
                    n_events=n_events,
                    rng=rng
                )
            except Exception as e:
                raise RuntimeError(f"SUBSAMPLE_MASK_ERROR: Failed to create subsample mask for sample {sample_id}: {e}")

            # Load sample data
            try:
                sample_sub = self.repo.load_sample_adata(
                    sample_id=sample_id,
                    layer=gate_node.layer,
                    mask=mask,
                    select=gate_node.dimensions
                )
            except Exception as e:
                raise RuntimeError(f"SAMPLE_DATA_LOAD_ERROR: Failed to load data for sample {sample_id}: {e}")

            per_sample_adatas.append(sample_sub)

        if not per_sample_adatas:
            raise ValueError("No events collected for this batch (all empty after filtering?)")

        batch_adata = ad.concat(
            per_sample_adatas,
            axis=0,
            join="inner",
            label="sample_id",
            keys=batch.sample_ids,
            index_unique="-",
        )

        return batch_adata

    def _get_parent_mask(self, strategy_id: str, parent_ids: Sequence[str], sample: SampleRef) -> NDArray[np.bool_]:
        """Load parent mask for a sample. Returns None if no parent."""
        if not parent_ids:
            return np.ones(sample.n_events, dtype=bool)

        if len(parent_ids) > 1:
            raise ValueError(f"Gate has {len(parent_ids)} parents. Only single-parent gates can use batch fitting.")

        parent_dict = self.repo.load_gating_masks(
            strategy=strategy_id,
            sample=sample,
            mask_ids=parent_ids
        )
        return parent_dict[parent_ids[0]]

    def _create_subsample_mask(
        self,
        parent_mask: NDArray[np.bool_],
        n_events: int,
        rng: np.random.Generator
    ) -> NDArray[np.bool_] | slice:

        if parent_mask.sum() <= n_events:
            return parent_mask

        # Subsample from parent events
        parent_indices = np.where(parent_mask)[0]
        selected_indices = rng.choice(parent_indices, size=n_events, replace=False)
        mask = np.zeros_like(parent_mask)
        mask[selected_indices] = True
        return mask
