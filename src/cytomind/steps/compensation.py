from __future__ import annotations
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
from flowutils.compensate import compensate, inverse_compensate

from cytomind.domain.qc import QCRunStatus, QCStepStatus, QCFlag
from .base import BaseStep
from . import register_step

if TYPE_CHECKING:
    from cytomind.domain.flow import CompensationRef
    from cytomind.domain.pipeline import SampleRef, StepRun
    from anndata import AnnData
else:
    CompensationRef = object
    AnnData = object
    QCStepStatus = object
    SampleRef = object
    StepRun = object


def apply_compensation(raw: AnnData, comp: CompensationRef, invert: bool = False) -> AnnData:
    """Apply or invert compensation on raw data.

    Parameters
    ----------
    raw : AnnData
        Raw or compensated data
    comp : CompensationRef
        Compensation reference with spillover matrix
    invert : bool, optional
        If True, undo compensation. If False, apply compensation (default).

    Returns
    -------
    AnnData
        Compensated or uncompensated data
    """
    spill = comp.matrix
    fluro = [raw.var.index.get_loc(det) for det in comp.detectors]
    events = np.asarray(raw.X)
    if invert:
        compensated_events = inverse_compensate(events, spill, fluro)
    else:
        compensated_events = compensate(events, spill, fluro)
    adata = raw.copy()
    adata.uns["compensation_id"] = None if invert else comp.id
    adata.X = compensated_events
    return adata


@register_step("compensate")
class CompensateStep(BaseStep):

    default_config = {"store_raw": True}  # whether to store raw data if we have to undo compensation to recover it

    def run_sample(self, sample_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:

        # Init results
        output_info: dict[str, Any] = {
            "comp_applied": None,
            "raw_recovered": False
        }
        qc = step_run.qc.get_sample_steps(sample_id)

        # 0) Validate sample exists in project
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return output_info, qc

        # 1) Load compensation
        step_comp = qc.get_step("load_compensation")
        if isinstance(step_run.config["comp_id"], str):
            comp_id = step_run.config["comp_id"]
        elif isinstance(step_run.config["comp_id"], Mapping):
            comp_id: str = step_run.config["comp_id"].get(sample.id, "")
        else:
            step_comp.flag = QCFlag.FAIL
            step_comp.add_reason(
                code="INVALID_COMP_ID",
                message="Invalid compensation ID format in config."
            )
            return output_info, qc

        try:
            comp = self.project.compensations[comp_id]
        except KeyError:
            step_comp.flag = QCFlag.FAIL
            step_comp.add_reason(
                code="COMP_NOT_FOUND",
                message=f"Compensation {comp_id} not found in project."
            )
            return output_info, qc

        # 2) Check if sample is already compensated
        try:
            comp_adata_path = self.repo.sample_adata_path(sample.id, layer='comp')
        except FileNotFoundError:
            comp_adata_path = None
        try:
            raw_adata_path = self.repo.sample_adata_path(sample.id, layer='raw')
        except FileNotFoundError:
            raw_adata_path = None
        if sample.compensation is not None and sample.compensation == comp.id:
            if not comp_adata_path:
                step_apply = qc.get_step("load_compensated_data")
                step_apply.flag = QCFlag.FAIL
                step_apply.add_reason(
                    code="MISSING_COMP_DATA",
                    message=(f"Sample {sample.id} is marked as compensated with compensation {comp.id}, "
                             "but 'comp' layer data is missing.")
                )
                return output_info, qc
            step_apply = qc.get_step("apply_compensation")
            step_apply.flag = QCFlag.PASS
            step_apply.add_reason(
                code="INFO",
                message=f"Sample {sample.id} is already compensated with compensation {comp.id}."
            )
            if self.config["store_raw"] and not raw_adata_path:
                self._undo_compensation(sample, qc)  # create raw layer (useful for qc/revision)
                output_info["raw_recovered"] = True
            return output_info, qc

        # 3) Get Raw Data
        if raw_adata_path:
            raw, _ = self.run_step(
                qc,
                "load_anndata_raw",
                self.repo._load_sample_layer,
                reason_code_fail="LOAD_ANNDATA_ERROR",
                sample_id=sample.id,
                layer='raw',
                backed=False
            )
        else:
            # If raw is missing, attempt to produce it by undoing any existing compensation
            raw, step_save_raw  = self._undo_compensation(sample, qc)
            if self.config["store_raw"] and step_save_raw.flag == QCFlag.PASS:
                output_info["raw_recovered"] = True

        if raw is None:
            return output_info, qc

        adata, step_apply = self.run_step(
            qc,
            "apply_compensation",
            apply_compensation,
            reason_code_fail="COMP_RUN_ERROR",
            raw=raw,
            comp=comp,
        )
        if not adata:
            return output_info, qc

        step_save = self.save_adata(sample, adata, qc, layer="comp")[1]
        if step_save.flag == QCFlag.FAIL:
            return output_info, qc

        output_info["comp_applied"] = sample.compensation = comp.id

        # Populate evaluable_products for QC evaluation
        if "compensation" not in step_run.evaluable_products:
            step_run.evaluable_products["compensation"] = {}
        if comp.id not in step_run.evaluable_products["compensation"]:
            step_run.evaluable_products["compensation"][comp.id] = {"sample_ids": []}
        step_run.evaluable_products["compensation"][comp.id]["sample_ids"].append(sample.id)

        return output_info, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        qc_iter = step_run.qc.sample_qc.items()
        samples = [self.project.samples[sid] for sid, qc in qc_iter if qc.overall_flag != QCFlag.FAIL]

        # Create "comp" dimension layer if not exists, copying from "raw" with raw provenance.
        dimensions = {}
        if "raw" in self.project.layers and "comp" not in self.project.layers:
            comp_dimensions = []
            for raw_dim in self.project.layers["raw"]:
                comp_dim = raw_dim.copy()
                comp_dim.source_layer = "raw"
                comp_dimensions.append(comp_dim)
            dimensions["comp"] = comp_dimensions

        self.repo.update_project_metadata(samples=samples, layers=dimensions)
        return step_run

    def _undo_compensation(self, sample: SampleRef, qc: QCRunStatus) -> tuple[AnnData | None, QCStepStatus]:
        """
        Attempt to undo an existing compensation on the sample's 'comp' layer and return
        an AnnData representing the recovered raw events. Does not save the result;
        QC steps are updated here as appropriate.
        """

        # 1) Load compensation
        step_ex_comp = qc.get_step("load_compensated_data")
        comp_id = sample.compensation
        if comp_id is None or comp_id not in self.project.compensations:
            step_ex_comp.flag = QCFlag.FAIL
            step_ex_comp.add_reason(
                code="COMP_NOT_FOUND",
                message="No existing compensation found on sample to invert."
            )
            return None, step_ex_comp
        comp = self.project.compensations[comp_id]


        # 2) Load 'comp' layer
        comp_adata, step_load_comp = self.run_step(
            qc,
            "load_anndata_comp",
            self.repo._load_sample_layer,
            reason_code_fail="LOAD_ANNDATA_ERROR",
            sample_id=sample.id,
            layer='comp',
            backed=False
        )

        if comp_adata is None:
            step_load_comp.flag = QCFlag.FAIL
            step_load_comp.add_reason(
                code="NO_INPUT_DATA",
                message="Neither 'raw' nor 'comp' layer present for this sample."
            )
            return None, step_load_comp

        if comp_adata.uns.get("compensation_id") != comp.id:
            step_load_comp.flag = QCFlag.FAIL
            step_load_comp.add_reason(
                code="COMP_MISMATCH",
                message=(f"Existing compensation on sample ({comp_adata.uns.get('compensation_id')})",
                         f"does not match expected ({comp.id}).")
            )
            return None, step_load_comp

        # 3) Invert compensation
        raw, step_invert = self.run_step(
            qc,
            "invert_compensation",
            apply_compensation,
            reason_code_fail="COMP_INVERT_ERROR",
            raw=comp_adata,
            comp=comp,
            invert=True,
        )

        if raw is not None and self.config["store_raw"]:
            # If raw is recovered successfully try to store for future revisions
            _, step_save_raw = self.save_adata(sample, raw, qc, layer="raw")
            if step_save_raw.flag == QCFlag.FAIL:
                step_save_raw.flag = QCFlag.WARN  # warn instead of fail since we still have the compensated data
            return raw, step_save_raw
        else:
            return raw, step_invert

