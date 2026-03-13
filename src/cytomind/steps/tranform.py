from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

from cytomind.domain.flow import DimensionDef
from cytomind.domain.pipeline import StepRun
from cytomind.domain.transforms import build_transformer
from cytomind.domain.qc import QCFlag, QCStepStatus, QCRunStatus
from .base import BaseStep
from . import register_step

import numpy as np
import pandas as pd
import anndata as ad

if TYPE_CHECKING:
    from cytomind.domain.pipeline import SampleRef, StepRun
    from cytomind.domain.pipeline import Project
else:
    SampleRef = object
    StepRun = object
    Project = object

@register_step("add_layer")
class AddLayerStep(BaseStep):
    config: dict[str, object] = {
        "layer": "xf",       # target layer name
        "sample_ids": [],    # list of sample ids to process
        "batch": None,       # optional batch id if sample_ids is not provided
        "default": True,     # whether to set the new layer as default for the samples
    }

    def prepare_batch(self, batch_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        output, qc = super().prepare_batch(batch_id, step_run)
        if not output:
            return {}, qc

        raw_dimensions = step_run.config.get("dimensions", [])
        dimensions: list[DimensionDef] = [
            dim.copy() if isinstance(dim, DimensionDef) else DimensionDef.from_dict(dim)
            for dim in raw_dimensions
        ]
        if not dimensions:
            step = qc.get_step("resolve_dimensions")
            step.flag = QCFlag.FAIL
            step.add_reason(code="NO_DIMENSIONS",
                            message="No dimensions specified for the transformation.")
            return {}, qc

        step_check = validate_dimensions(
            project=self.project,
            target_layer="raw",
            dimensions=dimensions,
            qc=qc,
        )
        if step_check.flag == QCFlag.FAIL:
            return {}, qc

        required_by_layer: dict[str, set[str]] = {}
        dim_plan: dict[str, list[DimensionDef]] = {}
        for idx, dim in enumerate(dimensions):
            dim.idx = idx
            source_layer = dim.source_layer or "raw"
            dim_plan.setdefault(source_layer, []).append(dim)
            required_by_layer.setdefault(source_layer, set()).update(dim.source_dims)

        required_layer = {
            layer: sorted(dims)
            for layer, dims in required_by_layer.items()
            if dims
        }

        output = {
            "dimensions": sorted(dimensions),
            "dimension_plan": dim_plan,
            "required_layer": required_layer,
        }
        return output, qc

    def _get_batch_output(self, sample_id: str, step_run: StepRun) -> dict:
        for batch_id in step_run.inputs.get("batch_ids", []):
            batch = self.project.batches.get(batch_id)
            if batch is not None and sample_id in batch.sample_ids:
                return step_run.batch_outputs.get(batch_id, {})
        return {}

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        qc = step_run.qc.get_sample_steps(sample_id)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return {}, qc

        # 1) Parse config
        target_layer: str = step_run.config["layer"]
        batch_output = self._get_batch_output(sample_id, step_run)
        dimensions: list[DimensionDef] = batch_output["dimensions"]
        dim_plan: dict[str, list[DimensionDef]] = batch_output["dimension_plan"]
        required_layer: dict[str, list[str] | slice] = batch_output["required_layer"]


        n_events = sample.n_events
        new_X = np.zeros((n_events, len(dimensions)))
        step_xform = qc.get_step("apply_transformations")
        adata: ad.AnnData | None = None
        for source_layer, dims in dim_plan.items():
            select = required_layer[source_layer]
            adata, step = self.load_adata(sample=sample, qc=qc, layer=source_layer, select=select)
            if adata is None:
                return {}, qc
            for dim in dims:
                j = dim.idx
                transform_def = self.project.resolve_dimension_transform(dim)
                transformer = build_transformer(transform_def)
                try:
                    new_X[:, j] = transformer.apply(adata[:, dim.source_dims].X).squeeze() # pyright: ignore[reportArgumentType]
                except Exception as e:
                    step_xform.flag = QCFlag.FAIL
                    step_xform.add_reason(
                        code="TRANSFORM_APPLY_ERROR",
                        message=f"Error applying transformation {transform_def.id} to dimension {dim.id}: {e}"
                    )
                    return {}, qc

        if adata is None:
            # this should not happen since dimensions should have been validated to have source_dims, but just in case
            adata = ad.AnnData(X=new_X)

        # 3) Save results
        vars_df = pd.DataFrame.from_records([dim.to_record() for dim in sorted(dimensions)])
        vars_df.set_index("id", drop=False, inplace=True)
        adata_xf = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=vars_df) # pyright: ignore[reportArgumentType]
        self.save_adata(sample, adata_xf, qc, layer=target_layer)
        return {}, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        layer: str = step_run.config["layer"]
        dimensions: list[DimensionDef] = []
        for output in step_run.batch_outputs.values():
            if "dimensions" in output:
                dimensions = output["dimensions"]
                break
        if not dimensions:
            return step_run

        samples_to_update: list[SampleRef] = []
        default: bool = step_run.config.get("default", True)
        if default:
            for sid, qc in step_run.qc.sample_qc.items():
                if qc.overall_flag != QCFlag.FAIL:
                    sample = self.project.samples[sid]
                    sample.default_layer = layer
                    samples_to_update.append(sample)

        self.repo.update_project_metadata(
            layers={layer: sorted(dimensions)},
            samples=samples_to_update,
        )
        return step_run

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        step_run = super().cleanup_step_run(step_run)
        for batch_output in step_run.batch_outputs.values():
            batch_output.pop("dimensions", None)
            batch_output.pop("dimension_plan", None)
            batch_output.pop("required_layer", None)
        return step_run


@register_step("add_dimensions")
class AddDimensionsStep(BaseStep):

    def _resolve_dimensions_config(
        self,
        layer: str,
        raw_dimensions: Iterable[DimensionDef | dict],
        qc: QCRunStatus,
    ) -> tuple[list[DimensionDef], list[DimensionDef]] | None:
        step_check = qc.get_step("resolve_dimensions")
        if layer not in self.project.layers:
            step_check.flag = QCFlag.FAIL
            step_check.add_reason(
                code="LAYER_NOT_FOUND",
                message=f"Data layer {layer!r} does not exist. Use add_layer to create it first.",
            )
            return None

        current_dimensions = [dim.copy() for dim in self.project.layers[layer]]
        current_ids = {dim.id for dim in current_dimensions}
        new_dimensions = [
            dim.copy() if isinstance(dim, DimensionDef) else DimensionDef.from_dict(dim)
            for dim in raw_dimensions
        ]

        new_ids = [dim.id for dim in new_dimensions]
        duplicated_new_ids = sorted({dim_id for dim_id in new_ids if new_ids.count(dim_id) > 1})
        if duplicated_new_ids:
            step_check.flag = QCFlag.FAIL
            step_check.add_reason(
                code="DUPLICATE_DIMENSION_ID",
                message=f"New dimensions contain duplicate ids: {duplicated_new_ids}.",
            )
            return None

        existing_ids = sorted(current_ids.intersection(new_ids))
        if existing_ids:
            step_check.flag = QCFlag.FAIL
            step_check.add_reason(
                code="DIMENSION_ALREADY_EXISTS",
                message=(
                    f"Dimensions already exist in layer {layer!r}; add_dimensions only accepts new dimensions: "
                    f"{existing_ids}."
                ),
            )
            return None

        next_idx = len(current_dimensions)
        for dim in new_dimensions:
            dim.idx = next_idx
            next_idx += 1

        final_dimensions = sorted([*current_dimensions, *new_dimensions])
        return new_dimensions, final_dimensions

    def prepare_batch(self, batch_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        output, qc = super().prepare_batch(batch_id, step_run)
        if not output:
            return {}, qc

        layer: str = step_run.config["layer"]
        resolved = self._resolve_dimensions_config(
            layer=layer,
            raw_dimensions=step_run.config.get("dimensions", []),
            qc=qc,
        )
        if resolved is None:
            return {}, qc

        new_dims, final_dim = resolved
        new_dims = sorted(new_dims)
        if not new_dims:
            output = {
                "layer": layer,
                "new_dimensions": [],
                "final_dimensions": final_dim,
                "dimension_plan": {},
                "required_layer": {},
            }
            return output, qc

        step_check = validate_dimensions(
            project=self.project,
            target_layer=layer,
            dimensions=new_dims,
            qc=qc,
        )
        if step_check.flag == QCFlag.FAIL:
            return {}, qc

        required_by_layer: dict[str, set[str]] = {}
        dim_plan: dict[str, list[DimensionDef]] = {}
        for dim in new_dims:
            source_layer = dim.source_layer or layer
            dim_plan.setdefault(source_layer, []).append(dim)
            required_by_layer.setdefault(source_layer, set()).update(dim.source_dims)

        output = {
            "layer": layer,
            "new_dimensions": new_dims,
            "final_dimensions": final_dim,
            "dimension_plan": dim_plan,
            "required_layer": {
                source_layer: sorted(source_dims)
                for source_layer, source_dims in required_by_layer.items()
            },
        }
        return output, qc

    def _get_batch_output(self, sample_id: str, step_run: StepRun) -> dict:
        for batch_id in step_run.inputs.get("batch_ids", []):
            batch = self.project.batches.get(batch_id)
            if batch is not None and sample_id in batch.sample_ids:
                return step_run.batch_outputs.get(batch_id, {})
        return {}

    def run_sample(
        self,
        sample_id: str,
        step_run: StepRun,
    ) -> tuple[dict, QCRunStatus]:
        qc = step_run.qc.get_sample_steps(sample_id)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadFCS")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND",
                            message=f"Sample {sample_id} not found in project.")
            return {}, qc

        # 1) Parse config
        layer: str = step_run.config["layer"]
        batch_output = self._get_batch_output(sample_id, step_run)
        new_dims: list[DimensionDef] = batch_output["new_dimensions"]
        final_dim: list[DimensionDef] = batch_output["final_dimensions"]
        dim_plan: dict[str, list[DimensionDef]] = batch_output["dimension_plan"]
        required_layer: dict[str, list[str]] = batch_output["required_layer"]

        if not new_dims:
            step = qc.get_step(f"update_dimensions_{layer}")
            step.flag = QCFlag.PASS
            step.add_reason("INFO", f"No new dimensions to add in layer {layer}.")
            return {}, qc

        # 2) load target
        target_adata = self.load_adata(sample, qc, layer=layer)[0]
        if target_adata is None or target_adata.X is None:
            return {}, qc

        # 3) Apply transforms and build final X
        step_apply = qc.get_step("apply_transformations")
        final_var = pd.DataFrame.from_records([dim.to_record() for dim in final_dim])
        final_var.set_index("id", drop=False, inplace=True)
        n_obs = target_adata.n_obs
        n_var_added = len(final_dim) - target_adata.n_vars
        final_X = np.concatenate([target_adata.X, np.zeros((n_obs, n_var_added))], axis=1) # pyright: ignore[reportCallIssue, reportArgumentType]
        source_adatas: dict[str, ad.AnnData] = {layer: target_adata}
        for source_layer, dims in dim_plan.items():
            source_adata = source_adatas.get(source_layer)
            if source_adata is None:
                source_select = required_layer[source_layer]
                source_adata = self.load_adata(sample, qc, layer=source_layer, select=source_select)[0]
                if source_adata is None or source_adata.X is None:
                    return {}, qc
                source_adatas[source_layer] = source_adata

            for dim in dims:
                j = final_var.index.get_loc(dim.id)
                transform_def = self.project.resolve_dimension_transform(dim)
                transformer = build_transformer(transform_def)
                try:
                    final_X[:, j] = transformer.apply(source_adata[:, dim.source_dims].X).squeeze() # pyright: ignore[reportArgumentType]
                except Exception as e:
                    step_apply.flag = QCFlag.FAIL
                    step_apply.add_reason(
                        "TRANSFORM_APPLY_ERROR",
                        f"Error applying transformation {transform_def.id} to dimension {dim.id}: {e}"
                    )
                    return {}, qc

        # 4) Save results
        adata_xf = ad.AnnData(X=final_X, obs=target_adata.obs.copy(), var=final_var) # pyright: ignore[reportArgumentType]
        self.save_adata(sample, adata_xf, qc, layer=layer, overwrite=True)
        return {}, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        layer: str = step_run.config["layer"]
        final_dim: list[DimensionDef] | None = None
        for output in step_run.batch_outputs.values():
            if "final_dimensions" in output:
                final_dim = output["final_dimensions"]
                break
        if final_dim is None:
            return step_run
        self.repo.update_project_metadata(layers={layer: final_dim})
        return step_run

    def merge_config(self, step_run: StepRun) -> dict:
        return super().merge_config(step_run)

    def cleanup_step_run(self, step_run: StepRun) -> StepRun:
        step_run = super().cleanup_step_run(step_run)
        for batch_output in step_run.batch_outputs.values():
            batch_output.pop("new_dimensions", None)
            batch_output.pop("final_dimensions", None)
            batch_output.pop("dimension_plan", None)
            batch_output.pop("required_layer", None)
        return step_run


def validate_dimensions(
    project: Project,
    target_layer: str,
    dimensions: list[DimensionDef],
    qc: QCRunStatus,
) -> QCStepStatus:
    step_check = qc.get_step("resolve_dimensions")

    # 1) referenced transforms can be resolved
    unresolved_transforms: set[str] = set()
    for dim in dimensions:
        try:
            _ = project.resolve_dimension_transform(dim)
        except Exception:
            unresolved_transforms.add(dim.transform_id)

    if unresolved_transforms:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="TRANSFORM_NOT_FOUND",
            message=f"Some transformations could not be resolved: {sorted(unresolved_transforms)}."
        )
        return step_check

    # 2) source dimensions should be explicitly declared
    missing_sources = [dim.id for dim in dimensions if not dim.source_dims]
    if missing_sources:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="SOURCE_DIM_NOT_FOUND",
            message=f"Some dimensions have no source_dims declared: {sorted(missing_sources)}.",
        )
        return step_check

    # 3) referenced source layers and dimensions must exist in project metadata
    missing_layers: set[str] = set()
    missing_layer_dims: dict[str, list[str]] = {}
    for dim in dimensions:
        source_layer = dim.source_layer or target_layer
        layer_dims = {layer_dim.id for layer_dim in project.layers.get(source_layer, [])}
        if source_layer not in project.layers:
            missing_layers.add(source_layer)
            continue

        missing_dims = sorted(set(dim.source_dims) - layer_dims)
        if missing_dims:
            missing_layer_dims[dim.id] = missing_dims

    if missing_layers:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="SOURCE_LAYER_NOT_FOUND",
            message=f"Some dimensions reference missing source layers: {sorted(missing_layers)}.",
        )
        return step_check

    if missing_layer_dims:
        step_check.flag = QCFlag.FAIL
        step_check.add_reason(
            code="SOURCE_DIM_NOT_FOUND",
            message=f"Some dimensions reference source dims missing from their source layer: {missing_layer_dims}.",
        )
        return step_check

    return step_check
