from __future__ import annotations
from typing import Iterable, Mapping, TYPE_CHECKING

from cytomind.domain.flow import DimensionDef
from cytomind.domain.transforms import TransformationRef, transform_registry
from cytomind.domain.qc import QCFlag, QCStepStatus, QCRunStatus
from .base import BaseStep
from . import register_step

import numpy as np
import pandas as pd
import anndata as ad

if TYPE_CHECKING:
    from cytomind.domain.pipeline import SampleRef, StepRun
else:
    SampleRef = object
    StepRun = object

@register_step("add_layer")
class AddLayerStep(BaseStep):
    config: dict[str, object] = {
        "layer": "xf",       # target layer name
        "sample_ids": [],    # list of sample ids to process
        "batch": None,       # optional batch id if sample_ids is not provided
        "default": True,     # whether to set the new layer as default for the samples
    }

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        qc = self._get_qc_run(sample_id, step_run)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return {}, qc

        # 1) Parse config
        layer = step_run.config["layer"]
        dimensions = sorted(self.project.dimensions[layer])

        # 2) Validate dimensions
        channels = self.project.panel_df.index.to_list()
        step_check = validate_dimensions(
            transformations=self.project.transformations,
            channels=channels,
            dimensions=dimensions,
            qc=qc
        )
        if step_check.flag == QCFlag.FAIL:
            return {}, qc

        # 3) load compensated data
        required = _get_required_channels(channels, dimensions)
        comp, step_load_comp = self.load_adata(sample, qc, layer="comp", select=required)
        if comp is None or comp.X is None:
            return {}, qc

        # 4) Apply transforms
        step_xform = qc.get_step("apply_transformations")
        new_X = np.zeros((comp.n_obs, len(dimensions)))
        for j, dim in enumerate(dimensions):
            dim.idx = j
            ref = self.project.transformations[dim.transform_id]
            transformation = transform_registry[ref.id](**ref.params)
            try:
                new_X[:, j] = transformation.apply(comp[:, dim.channel_id].X).squeeze() # pyright: ignore[reportArgumentType]
            except Exception as e:
                step_xform.flag = QCFlag.FAIL
                step_xform.add_reason(
                    code="TRANSFORM_APPLY_ERROR",
                    message=f"Error applying transformation {ref.id} to dimension {dim.id}: {e}"
                )
                return {}, qc

        # 5) Save results
        vars_df = pd.DataFrame.from_records([dim.to_record() for dim in dimensions])
        vars_df.set_index("id", drop=False, inplace=True)
        adata_xf = ad.AnnData(X=new_X, obs=comp.obs.copy(), var=vars_df) # pyright: ignore[reportArgumentType]
        step_save = self.save_adata(sample, adata_xf, qc, layer=layer)[1]
        if step_save.flag == QCFlag.FAIL:
            return {}, qc

        output_info = {
            "input_files": [self.repo.sample_adata_path(sample.id, layer='comp').as_posix()],
            "output_files": [self.repo.sample_adata_path(sample.id, layer=layer).as_posix()],
            "layer": layer,
        }
        return output_info, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        # dimesions are already updated in repo for this step to work
        default: bool = step_run.config.get("default", True)
        if not default:
            # layer is ready to use but not default, so nothing to do
            return step_run

        # updates sample default layer
        layer: str = step_run.config["layer"]
        samples_to_update: dict[str, SampleRef] = {}
        for sid, qc in step_run.qc.sample_qc.items():
            if qc.overall_flag != QCFlag.FAIL:
                sample = self.project.samples[sid]
                sample.default_layer = layer
                samples_to_update[sid] = sample

        self.repo.update_project_metadata(samples=samples_to_update)
        return step_run


@register_step("add_dimensions")
class AddDimensionsStep(BaseStep):

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        qc = self._get_qc_run(sample_id, step_run)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadFCS")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return {}, qc

        # 1) Parse config
        layer = step_run.config["layer"]
        new_dims = sorted(step_run.config.get("dimensions", []))
        if not new_dims:
            step = qc.get_step(f"update_dimensions_{layer}")
            step.flag = QCFlag.PASS
            step.add_reason("INFO", f"No new dimensions to add in layer {layer}.")
            return {}, qc

        # 2) Validate dimensions
        channels = self.project.panel_df.index.to_list()
        step_check = validate_dimensions(
            transformations=self.project.transformations,
            channels=channels,
            dimensions=new_dims,
            qc=qc
        )
        if step_check.flag == QCFlag.FAIL:
            return {}, qc

        # 3) load comp and target
        required = _get_required_channels(channels, new_dims)
        comp = self.load_adata(sample, qc, layer="comp", select=required)[0]
        if comp is None or comp.X is None:
            return {}, qc

        target_adata = self.load_adata(sample, qc, layer=layer)[0]
        if target_adata is None or target_adata.X is None:
            return {}, qc

        # 4) Apply transforms and build final X
        step_apply = qc.get_step("apply_transformations")
        # Prepare Updated Dimensions (adata.var)
        final_var = pd.DataFrame.from_records([dim.to_record() for dim in final_dim])
        final_var.set_index("id", drop=False, inplace=True)
        # Prepare Updated X matrix
        n_obs = target_adata.n_obs
        n_var_added = len(final_dim) - target_adata.n_vars
        final_X = np.concatenate([target_adata.X, np.zeros((n_obs, n_var_added))], axis=1) # pyright: ignore[reportCallIssue, reportArgumentType]
        # only compute newly updated dimensions
        for dim in new_dims:
            j = final_var.index.get_loc(dim.id)
            ref = self.project.transformations[dim.transform_id]
            transformation = transform_registry[ref.id](**ref.params)
            try:
                final_X[:, j] = transformation.apply(comp[:, dim.channel_id].X).squeeze() # pyright: ignore[reportArgumentType]
            except Exception as e:
                step_apply.flag = QCFlag.FAIL
                step_apply.add_reason("TRANSFORM_APPLY_ERROR",
                                      f"Error applying transformation {ref.id} to dimension {dim.id}: {e}")
                return {}, qc

        # 5) Save results
        adata_xf = ad.AnnData(X=final_X, obs=target_adata.obs.copy(), var=final_var) # pyright: ignore[reportArgumentType]
        step_save = self.save_adata(sample, adata_xf, qc, layer=layer)[1]
        if step_save.flag == QCFlag.FAIL:
            return {}, qc

        output_info = {
            "input_files": [self.repo.sample_adata_path(sample.id, layer='comp').as_posix()],
            "output_files": [self.repo.sample_adata_path(sample.id, layer=layer).as_posix()],
            "layer": layer,
        }
        return output_info, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        layer: str = step_run.config["layer"]
        final_dim: list[DimensionDef] = step_run.config["final_dimensions"]
        self.repo.update_project_metadata(dimensions={layer: final_dim})
        return step_run

    def merge_config(self, step_run: StepRun) -> dict:
        cfg = super().merge_config(step_run)
        cur_refs = {dim.id: dim.copy() for dim in self.project.dimensions[cfg["layer"]]}
        new_refs = [DimensionDef.from_dict(dim) for dim in cfg.get("dimensions", [])]
        for dim in new_refs:
            dim.idx = cur_refs[dim.id].idx if dim.id in cur_refs else len(cur_refs)
            cur_refs[dim.id] = dim
        cfg["dimensions"] = new_refs  # convert to list[DimensionDef]
        cfg["final_dimensions"] = sorted(list(cur_refs.values()))
        return cfg


def validate_dimensions(
    transformations: Mapping[str, TransformationRef],
    channels: Iterable[str],
    dimensions: list[DimensionDef],
    qc: QCRunStatus,
) -> QCStepStatus:
    step_check = qc.get_step("resolve_dimensions")

    # 1) uncompensated requirement
    missing_uncomp = [d.id for d in dimensions if not d.use_comp]
    if missing_uncomp:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="COMP_NOT_SUPPORTED",
            message=f"Some dimensions require uncompensated data, which is not supported.: {missing_uncomp}."
        )
        return step_check

    # 2) deterministic ordering & presence of channels on comp.var.index
    missing_channels = [
        ch for dim in dimensions for ch in dim.channel_id if ch not in channels
    ]
    if missing_channels:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason("CHANNEL_NOT_FOUND",
                              f"Some dimension channels not found in data var: {missing_channels}.")
        return step_check

    # 3) referenced transforms exist in project
    missing_transforms = [dim.transform_id for dim in dimensions if dim.transform_id not in transformations]
    if missing_transforms:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="TRANSFORM_NOT_FOUND",
            message=f"Some transformations not found in project: {missing_transforms}."
        )
        return step_check

    # 4) transform registry contains the referenced transform ids
    missing_registry = [
        dim.transform_id for dim in dimensions
        if transformations[dim.transform_id].id not in transform_registry
    ]
    if missing_registry:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="TRANSFORM_NOT_FOUND",
            message=f"Some transformations not found in transformation registry: {missing_registry}."
        )
        return step_check

    return step_check

def _get_required_channels(channels: list[str], dimensions: list[DimensionDef]) -> list[str]:
    required_set = {ch for dim in dimensions for ch in dim.channel_id}
    required = [ch for ch in channels if ch in required_set]
    return required
