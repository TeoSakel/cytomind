from __future__ import annotations
from typing import Mapping, TYPE_CHECKING

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

from cytomind.domain.qc import QCRunStatus, QCStepStatus, QCFlag
from .base import BaseStep
from . import register_step

import numpy as np
from flowutils.compensate import compensate, inverse_compensate


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

    # QC/threshold parameters — can be overridden via step_run.config
    def run_sample(self, sample_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        qc = step_run.qc.get_sample_steps(sample_id)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return {}, qc

        input_files: list[str] = []
        output_files: list[str] = []

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
            return {}, qc

        try:
            comp = self.project.compensations[comp_id]
        except KeyError:
            step_comp.flag = QCFlag.FAIL
            step_comp.add_reason(
                code="COMP_NOT_FOUND",
                message=f"Compensation {comp_id} not found in project."
            )
            return {}, qc
        input_files.append(self.repo.spillover_path(comp.id).as_posix())

        # 2) Check if sample is already compensated
        comp_adata_path = self.repo.sample_adata_path(sample.id, layer='comp')
        if sample.compensation is not None and sample.compensation == comp.id:
            if not comp_adata_path.exists():
                step_apply = qc.get_step("load_compensated_data")
                step_apply.flag = QCFlag.FAIL
                step_apply.add_reason(
                    code="MISSING_COMP_DATA",
                    message=(f"Sample {sample.id} is marked as compensated with compensation {comp.id}, "
                             "but 'comp' layer data is missing.")
                )
                return {}, qc
            step_apply = qc.get_step("apply_compensation")
            step_apply.flag = QCFlag.PASS
            step_apply.add_reason(
                code="INFO",
                message=f"Sample {sample.id} is already compensated with compensation {comp.id}."
            )
            output_files.append(comp_adata_path.as_posix())
            output_info = {"input_files": input_files,
                           "output_files": output_files,
                           "comp_applied": (sample.id, comp.id)}
            return output_info, qc

        # 3) Get Raw Data
        raw_path = self.repo.sample_adata_path(sample.id, layer='raw')
        if raw_path.exists():
            raw, step_raw = self.run_step(
                qc,
                "load_anndata_raw",
                self.repo._load_sample_layer,
                reason_code_fail="LOAD_ANNDATA_ERROR",
                sample=sample,
                layer='raw',
                backed=False
            )
            input_files.append(raw_path.as_posix())
        else:
            # If raw is missing, attempt to produce it by undoing any existing compensation
            raw, step_raw  = self._undo_compensation(sample, qc)
            input_files.append(comp_adata_path.as_posix())

        if raw is None:
            return {}, qc

        adata, step_apply = self.run_step(
            qc,
            "apply_compensation",
            apply_compensation,
            reason_code_fail="COMP_RUN_ERROR",
            raw=raw,
            comp=comp,
        )
        if not adata:
            return {}, qc

        step_save = self.save_adata(sample, adata, qc, layer="comp")[1]
        if step_save.flag == QCFlag.FAIL:
            return {}, qc
        output_files.append(self.repo.sample_adata_path(sample.id, layer='comp').as_posix())

        sample.compensation = comp.id
        output_info = {"input_files": input_files,
                       "output_files": output_files,
                       "comp_applied": (sample.id, comp.id)}
        return output_info, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        qc_iter = step_run.qc.sample_qc.items()
        samples = {
            sid: self.project.samples[sid]
            for sid, qc in qc_iter if qc.overall_flag != QCFlag.FAIL
        }

        # Create "comp" dimension layer if not exists, copying from "raw" with use_comp=True
        dimensions = {}
        if "raw" in self.project.dimensions and "comp" not in self.project.dimensions:
            comp_dimensions = []
            for raw_dim in self.project.dimensions["raw"]:
                comp_dim = raw_dim.copy()
                comp_dim.use_comp = comp_dim.type in ("fluorescence", "spectral")
                comp_dimensions.append(comp_dim)
            dimensions["comp"] = comp_dimensions

        self.repo.update_project_metadata(samples=samples, dimensions=dimensions)
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
        return self.run_step(
            qc,
            "invert_compensation",
            apply_compensation,
            reason_code_fail="COMP_RUN_ERROR",
            raw=comp_adata,
            comp=comp,
            invert=True,
        )
