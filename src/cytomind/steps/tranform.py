from __future__ import annotations
from turtle import st
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
    from cytomind.domain.pipeline import StepRun
    from cytomind.domain.pipeline import Project
else:
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

        # Handle empty sample: save zero-row adata from dimension names, leave ranges as None
        if n_events == 0:
            vars_df = pd.DataFrame.from_records([dim.to_record() for dim in sorted(dimensions)])
            vars_df.set_index("id", drop=False, inplace=True)
            adata_xf = ad.AnnData(X=np.zeros((0, len(dimensions))), var=vars_df) # pyright: ignore[reportArgumentType]
            self.save_adata(sample, adata_xf, qc, layer=target_layer)
            return {}, qc

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

        # Compute per-column min/max in one vectorized pass; columns with no valid value get NaN
        step = qc.get_step("compute_sample_ranges")
        try:
            valid_mask = ~(np.isnan(new_X) | np.isinf(new_X))
            has_valid = np.any(valid_mask, axis=0)
            min_vals = np.where(has_valid, np.min(new_X, axis=0, where=valid_mask, initial=np.inf), np.nan).tolist()
            max_vals = np.where(has_valid, np.max(new_X, axis=0, where=valid_mask, initial=-np.inf), np.nan).tolist()
        except Exception as e:
            step.flag = QCFlag.FAIL
            step.add_reason(
                code="RANGE_COMPUTE_ERROR",
                message=f"Error computing min/max ranges for sample {sample_id}: {e}"
            )
            return {}, qc

        # 2) Save results
        vars_df = pd.DataFrame.from_records([dim.to_record() for dim in sorted(dimensions)])
        vars_df["range_min"] = min_vals
        vars_df["range_max"] = max_vals
        vars_df.set_index("id", drop=False, inplace=True)
        adata_xf = ad.AnnData(X=new_X, obs=adata.obs.copy(), var=vars_df) # pyright: ignore[reportArgumentType]
        self.save_adata(sample, adata_xf, qc, layer=target_layer)
        return {"min_vals": min_vals, "max_vals": max_vals}, qc

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        """Aggregate ranges from all samples in the batch using min/max vectors."""

        output_info = step_run.batch_outputs.get(batch_id, {})
        updates = {}

        if "dimensions" not in output_info:
            return {}, qc

        target_layer = step_run.config["layer"]
        dimensions = _update_dim_ranges(step_run, qc, output_info["dimensions"])
        if dimensions is None:
            return {}, qc
        updates["layers"] = {target_layer: dimensions}

        # Persist default-layer sample updates through the standard project_updates path.
        if step_run.config.get("default", True):
            samples_to_update = []
            for sample_id, sample_qc in step_run.qc.sample_qc.items():
                if sample_qc.overall_flag == QCFlag.FAIL:
                    continue
                sample_ref = self.project.samples[sample_id]
                sample_ref.default_layer = target_layer
                samples_to_update.append(sample_ref)

            if samples_to_update:
                updates["samples"] = samples_to_update

        step_run.project_updates.append(updates)
        return {}, qc


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
            return {}, qc

        # 2) load target
        target_adata = self.load_adata(sample, qc, layer=layer)[0]
        if target_adata is None or target_adata.X is None:
            return {}, qc

        n_obs = target_adata.n_obs

        # Handle empty sample: return original adata with no ranges
        if n_obs == 0:
            final_var = pd.DataFrame.from_records([dim.to_record() for dim in final_dim])
            final_var.set_index("id", drop=False, inplace=True)
            adata_xf = ad.AnnData(X=np.zeros((0, len(final_dim))), obs=target_adata.obs.copy(), var=final_var) # pyright: ignore[reportArgumentType]
            self.save_adata(sample, adata_xf, qc, layer=layer, overwrite=True)
            return {}, qc

        # 3) Apply transforms and build final X
        step_apply = qc.get_step("apply_transformations")
        final_var = pd.DataFrame.from_records([dim.to_record() for dim in final_dim])
        final_var.set_index("id", drop=False, inplace=True)
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
                final_var.at[dim.id, "range_min"] = float(np.nanmin(final_X[:, j]))
                final_var.at[dim.id, "range_max"] = float(np.nanmax(final_X[:, j]))

        # 4) Save results
        adata_xf = ad.AnnData(X=final_X, obs=target_adata.obs.copy(), var=final_var) # pyright: ignore[reportArgumentType]
        self.save_adata(sample, adata_xf, qc, layer=layer, overwrite=True)

        # Compute per-column min/max for new dimensions only; columns with no valid value get NaN
        min_vals = final_var["range_min"].values.tolist()
        max_vals = final_var["range_max"].values.tolist()
        return {"min_vals": min_vals, "max_vals": max_vals}, qc

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        """Aggregate ranges for new dimensions across all samples in the batch."""
        output_info = step_run.batch_outputs.get(batch_id, {})
        if "final_dimensions" not in output_info:
            return {}, qc

        dimensions = _update_dim_ranges(step_run, qc, output_info["final_dimensions"])
        if dimensions is None:
            return {}, qc

        layer = step_run.config["layer"]
        step_run.project_updates.append({"layers": {layer: dimensions}})
        return {}, qc


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


def _update_dim_ranges(step_run: StepRun, qc: QCRunStatus, dimensions: list[DimensionDef]) -> list[DimensionDef] | None:
    # Stack per-sample min/max vectors into 2-D arrays and reduce along sample axis

    step = qc.get_step("aggregate_ranges")
    sample_mins, sample_maxs = [], []
    for sid, out in step_run.sample_outputs.items():
        if step_run.qc.sample_qc[sid].overall_flag == QCFlag.FAIL:
            continue

        if "min_vals" in out and "max_vals" in out:
            sample_mins.append(out.pop("min_vals"))
            sample_maxs.append(out.pop("max_vals"))

    if sample_mins:
        aggregated_min = np.nanmin(np.asarray(sample_mins), axis=0)  # (n_dms,)
        aggregated_max = np.nanmax(np.asarray(sample_maxs), axis=0)
        if len(dimensions) != len(aggregated_min):
            step.flag = QCFlag.WARN
            step.add_reason(
                code="RANGE_DIMENSION_MISMATCH",
                message=(
                    f"Number of dimensions ({len(dimensions)}) does not match length of aggregated range vectors "
                    f"({len(aggregated_min)}); cannot update dimension ranges."
                ),
            )
            return
        for min_val, max_val, dim in zip(aggregated_min, aggregated_max, dimensions):
            if not np.isnan(min_val):
                dim.range_min = float(min(min_val, dim.range_min) if dim.range_min is not None else min_val)
            if not np.isnan(max_val):
                dim.range_max = float(max(max_val, dim.range_max) if dim.range_max is not None else max_val)
    else:
        step.flag = QCFlag.WARN
        step.add_reason("NO_RANGES_CALCULATED", "No valid data found to calculate ranges.")

    return dimensions