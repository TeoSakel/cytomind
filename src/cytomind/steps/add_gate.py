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
        batch_ids = step_run.inputs.get("batch_ids", [])
        if not batch_ids:
            raise ValueError("AddGateStep requires one batch_id in inputs.")
        if len(batch_ids) != 1:
            raise ValueError("AddGateStep only supports single batch_id per run.")

        if "gate_node" not in step_run.config:
            raise ValueError("AddGateStep requires 'gate_node' in config.")

        # Keep custom_fit as list; convert to set locally when needed for membership tests
        if "custom_fit" not in step_run.config:
            step_run.config["custom_fit"] = []
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

        strategy = self.project.gating_strategy
        if strategy is None:
            step = qc.get_step("LoadGatingStrategy")
            step.flag = QCFlag.FAIL
            step.add_reason("STRATEGY_NOT_FOUND", "Gating strategy not found in project.")
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
        if gate_node.id in strategy.graph:
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

        gate_node.dimensions = gate.dimensions  # Ensure dimensions are populated in node for downstream use
        gate_node.use_as_complement = gate.use_as_complement  # Ensure complement flag is in node for downstream use
        gate_node.glm_type = gate.glm_type  # Ensure glm_type is in node for downstream use
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
        qc = step_run.qc.get_sample_steps(sample_id)

        # Get sample using run_step
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason("SAMPLE_NOT_FOUND", f"Sample {sample_id} not found in project.")
            return {}, qc

        # Load Gate
        batch_id = step_run.inputs["batch_ids"][0]
        gate_node: GateNode = step_run.batch_outputs[batch_id]["gate_node"]
        gate_batch: Gate = step_run.batch_outputs[batch_id]["gate"]

        # Check if this sample should be refitted
        custom_fit_set = set(step_run.config["custom_fit"])
        # Refit if: gate is tunable AND (batch fitting disabled OR sample explicitly in custom_fit list)
        should_fit = not step_run.config.get("fit_on_batch", False) or sample_id in custom_fit_set
        should_fit = gate_batch.tunable and should_fit

        # Load Parent Masks
        parent_ids = [pid for pid in gate_node.parent_ids if pid != "root"]
        parent_dict, _ = self.run_step(
            qc,
            "LoadParentMasks",
            self.repo.load_gating_masks,
            reason_code_fail="PARENT_MASK_ERROR",
            sample=sample,
            mask_ids=parent_ids
        )
        if parent_dict is None:
            return {}, qc

        if gate_node.gate_type == "Boolean":
            adata = ad.AnnData(np.empty((0, 0), dtype=np.float32))
        else:
            if len(parent_dict) == 0:
                parent_dict = {"root": np.ones(sample.n_events, dtype=bool)}
            elif len(parent_dict) > 1:
                step = qc.get_step("ValidateParentMasks")
                step.flag = QCFlag.FAIL
                step.add_reason(
                    "MULTIPLE_PARENTS_NOT_SUPPORTED",
                    f"Gate of type '{gate_node.gate_type}' expects exactly one parent/root mask, got {len(parent_dict)}."
                )
                return {}, qc

            # Load only parent-selected events; Gate.apply expands back to full length.
            adata, _ = self.load_adata(
                sample,
                qc,
                layer=gate_node.layer,
                mask=parent_dict,
                select=gate_node.dimensions
            )
            if adata is None:
                return {}, qc

        # Fit gate (skip if already fitted on batch)
        if should_fit:
            gate, step = self.run_step(
                qc,
                "FitGate",
                gate_batch.copy().fit,
                events=adata,
                mask=parent_dict,
                reason_code_fail="GATE_FIT_ERROR"
            ) # pyright: ignore[reportAssignmentType]
            if gate is None:
                return {}, qc
        else:
            if sample_id in gate_node.custom_gates:
                params_to_use = gate_node.custom_gates[sample_id]
            else:
                params_to_use = gate_batch.to_node_params()
            gate = gate_batch.update_params(params_to_use)

        # Apply gate with required mask parameter
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
            sample=sample,
            masks=mask,
            overwrite=True  # Overwrite any existing masks for this gate (e.g., from previous batch fit or previous runs)
        )
        if step.flag == QCFlag.FAIL:
            return {}, qc

        # Store counts in nested dict structure for easier access
        summary = {
            "root": sample.n_events,
            "parents": {pid: int(parent_mask.sum()) for pid, parent_mask in parent_dict.items()},
            "gates": {mask_id: int(gate_mask.sum()) for mask_id, gate_mask in mask.items()}
        }
        output_info = {
            "summary": summary,
            "params": gate.to_node_params() if should_fit else None,
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
        strategy = self.project.gating_strategy
        if strategy is None:
            step = qc.get_step("LoadGatingStrategy")
            step.flag = QCFlag.FAIL
            step.add_reason("STRATEGY_NOT_FOUND", "Gating strategy not found in project.")
            return {}, qc

        # Create GateNode from config
        gate_node: GateNode = output_info["gate_node"]
        gate_node.params = output_info["gate"].to_node_params()

        # Store sample-specific parameter overrides (only if sample was actually fitted)
        for sample_id, sample_info in step_run.sample_outputs.items():
            params = sample_info.get("params")  # None if sample wasn't fitted
            if params:
                gate_node.custom_gates[sample_id] = params

        # Add the gate node with its attributes
        try:
            strategy.add_node(gate_node)
        except ValueError as e:
            # Has been checked in prepare_batch but re-checking here for safety since graph is being modified
            step = qc.get_step("AddGateNode")
            step.flag = QCFlag.FAIL
            step.add_reason("CYCLE_DETECTED", str(e))
            return {}, qc

        gate_ids = [gate_node.id]

        # For QuadrantGate, create individual GateNodes for each quadrant
        if gate_node.glm_type == "QuadrantGate":
            gate = output_info["gate"]
            # Extract nested "params" dict which contains computed quadrants
            quadrants = gate_node.params["params"]["quadrants"]
            for qid, loc in gate.locations.items():  # type: ignore[attr-defined]
                # Create a GateNode for this quadrant
                quadrant_params = {"boundaries": quadrants[qid]}
                quadrant_node = GateNode(
                    id=qid,
                    gate_type="Quadrant",
                    glm_type="Quadrant",
                    dimensions=list(loc.keys()),
                    layer=gate_node.layer,
                    name=qid,
                    parent_ids=[gate_node.id],
                    use_as_complement=False,
                    params={"hyperparams": quadrant_params, "params": quadrant_params},
                    custom_gates={
                        sid: sample_info.get("params", {}).get(qid, {})
                        for sid, sample_info in step_run.sample_outputs.items()
                        if sample_info.get("params")
                    }
                )
                # Add node with attributes to graph
                try:
                    strategy.add_node(quadrant_node)
                except ValueError as e:
                    step = qc.get_step("AddQuadrantNode")
                    step.flag = QCFlag.FAIL
                    step.add_reason("CYCLE_DETECTED", str(e))
                    return {}, qc

                gate_ids.append(quadrant_node.id)

        step_run.project_updates.append({"gating_strategy": strategy})

        # Populate evaluable_products: gating strategy is now updated with new gate(s)
        # Include metadata about parent dependencies for QC validation
        step_run.evaluable_products["gate_node"] = {gid: {} for gid in gate_ids}

        output_info["gate_node"] = gate_node.to_dict()
        del output_info["gate"]

        return output_info, qc

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        step_run = super().cleanup_step_run(step_run)
        # Remove params from sample outputs as they are persisted to the database
        # Keep only debugging/QC context information in outputs
        for sample_id, sample_info in step_run.sample_outputs.items():
            sample_info.pop("params", None)
        return step_run

    def _initialize_gate(self, gate_node: GateNode) -> Gate:
        """Initialize a Gate instance from a GateNode.

        This handles reconstruction of gates that may have batch-level fitted parameters.
        If the gate_node has params, use Gate.from_node() to properly restore them.
        Otherwise, initialize fresh (will be fitted during prepare_batch or run_sample).

        Validates that multi-parent gates are only BooleanGates (other gates must have
        single parent or no parent).

        Parameters
        ----------
        gate_node : GateNode
            Gate node from the gating strategy (may have empty params if new gate)

        Returns
        -------
        Gate
            Initialized gate instance, ready for fit() or apply()

        Raises
        ------
        ValueError
            If gate type is not found in registry, or if multiple parents on non-BooleanGate
        """
        # Validate multi-parent gates: only BooleanGates are allowed to have multiple parents
        if len(gate_node.parent_ids) > 1 and gate_node.gate_type != "Boolean":
            raise ValueError(
                f"MULTI_PARENT_NOT_ALLOWED: Gate '{gate_node.id}' has {len(gate_node.parent_ids)} parents. "
                f"Only BooleanGates support multiple parents. Other gates must have single parent or no parent."
            )

        try:
            gate_class = GateRegistry.get(gate_node.gate_type)
        except KeyError:
            raise ValueError(
                f"UNKNOWN_GATE_TYPE: Gate type '{gate_node.gate_type}' not recognized. "
                f"Available: {list(GateRegistry.list_gates().keys())}"
            )

        # If gate_node has saved params (batch-fitted), reconstruct with them
        if gate_node.params:
            return gate_class.from_node(gate_node)

        # Otherwise, initialize fresh with just hyperparams
        hyperparams = gate_node.params.get("hyperparams", {})
        return gate_class(
            gate_name=gate_node.name or gate_node.id,
            dimensions=gate_node.dimensions,
            use_as_complement=gate_node.use_as_complement,
            **hyperparams,
        )

    def _build_batch_adata(
        self,
        batch: BatchRef,
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
                parent_mask = self._get_parent_mask(gate_node.parent_ids, sample)
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

    def _get_parent_mask(self, parent_ids: Sequence[str], sample: SampleRef) -> NDArray[np.bool_]:
        """Load parent mask for a sample.

        Parameters
        ----------
        parent_ids : Sequence[str]
            List of parent gate IDs (should be 0 or 1; multi-parent only for BooleanGates)
        sample : SampleRef
            Sample to load parent masks for

        Returns
        -------
        NDArray[np.bool_]
            Combined parent mask (all True if no parents)

        Raises
        ------
        ValueError
            If multiple parents provided (batch fitting only supports single parent or no parent)
        """
        if not parent_ids:
            return np.ones(sample.n_events, dtype=bool)

        if len(parent_ids) > 1:
            raise ValueError(
                f"BATCH_FIT_MULTI_PARENT_ERROR: Batch fitting does not support gates with {len(parent_ids)} parents. "
                f"Only single-parent gates can use batch-level fitting. "
                f"For multi-parent gates (e.g., BooleanGates), use per-sample fitting (fit_on_batch=False)."
            )

        parent_dict = self.repo.load_gating_masks(
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
