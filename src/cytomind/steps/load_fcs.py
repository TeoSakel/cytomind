from __future__ import annotations
from turtle import update

from cytomind.domain.pipeline import StepRun
from cytomind.domain.qc import QCRunStatus, QCFlag
from .base import BaseStep
from . import register_step

import numpy as np
import flowkit as fk
import anndata as ad

@register_step("load_fcs")
class LoadFCS(BaseStep):
    """Load FCS files into AnnData format: samples/{sample_id}/raw.h5ad"""

    def run_sample(self, sample_id: str, step_run: StepRun) -> tuple[dict, QCRunStatus]:
        qc = step_run.qc.get_sample_steps(sample_id)
        try:
            sample = self.project.samples[sample_id]
        except KeyError:
            step = qc.get_step("LoadSample")
            step.flag = QCFlag.FAIL
            step.add_reason(code="SAMPLE_NOT_FOUND", message=f"Sample {sample_id} not found in project.")
            return {}, qc


        # Load FCS file using flowkit
        fcs, step_load = self.run_step(
            qc,
            "FlowKitLoad",
            func=lambda path: fk.Sample(path),
            reason_code_fail="FCSLoadError",
            path=sample.fcs_path
        )
        if not fcs: # fcs is None => step_load.flag == QCFlag.FAIL:
            return {}, qc

        # Get DataFrame of raw data with renamed channels
        fcs_df = fcs.as_dataframe(source='raw', subsample=False, col_multi_index=False)
        channel_renames = sample.rename.get("channel", {})
        fcs_df.rename(columns=channel_renames, inplace=True)

        # Check that all panel channels are present in FCS after renaming
        panel = self.project.panel_df
        panel_columns: set[str] = set(panel.index.tolist())
        fcs_columns = set(fcs_df.columns)
        missing = sorted(panel_columns - fcs_columns)
        if missing:
            step_load.flag = QCFlag.FAIL
            step_load.add_reason(
                code="CHANNELS_NOT_FOUND",
                message=f"Some channels not found in FCS file: {missing}.")
            return {}, qc

        # Load data matrix in panel order
        col_order = [pnn for pnn in panel.index.tolist() if pnn in fcs_df.columns]
        X = fcs_df[col_order].values

        # Create AnnData with panel (columns already in correct order)
        if sample.compensation:
            layer = 'comp'
            uns = {'sample_id': sample.id, 'compensation_id': sample.compensation}
            comp_evals = step_run.evaluable_products.setdefault("compensation", {})
            if sample.compensation in comp_evals:
                comp_evals[sample.compensation]["sample_ids"].append(sample.id)
            else:
                comp_evals[sample.compensation] = {"sample_ids": [sample.id]}

        else:
            layer = 'raw'
            uns = {'sample_id': sample.id}
            step_run.evaluable_products.setdefault("raw_data", {})[sample.id] = dict()
        var = self.repo.load_layer_df(layer)
        adata = ad.AnnData(X=X, var=var, uns=uns)

        step_save = self.save_adata(sample, adata, qc, layer=layer)[1]
        if step_save.flag == QCFlag.FAIL:
            return {}, qc

        # update sample metadata
        n_events = int(adata.n_obs)
        if n_events != fcs.event_count:
            step_load.flag = QCFlag.WARN
            step_load.add_reason(
                code="MALFORMED_FCS",
                message=(f"Event count mismatch between FCS file",
                         f"({fcs.event_count}) and loaded data ({n_events})."))

        output_info = {
            "n_events": n_events,
            "layer": layer,
        }
        # Compute actual per-channel min/max for range refinement
        if X.shape[0] > 0:
            valid_mask = ~(np.isnan(X) | np.isinf(X))
            has_valid = np.any(valid_mask, axis=0)
            output_info["min_vals"] = np.where(has_valid, np.min(X, axis=0, where=valid_mask, initial=np.inf), np.nan).tolist()
            output_info["max_vals"] = np.where(has_valid, np.max(X, axis=0, where=valid_mask, initial=-np.inf), np.nan).tolist()
        return output_info, qc

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        """Aggregate per-channel ranges within batch, update dim objects, store in batch output."""

        updates: dict = {}
        step = qc.get_step("aggregate_ranges")

        samples_to_update = []
        layer_ranges: dict[str, dict[str, list[float]]] = {}
        for sid, out in step_run.sample_outputs.items():
            sample_qc = step_run.qc.sample_qc.get(sid)
            if sample_qc is not None and sample_qc.overall_flag == QCFlag.FAIL:
                continue

            sample_ref = self.project.samples[sid]
            sample_ref.n_events = int(out["n_events"])
            samples_to_update.append(sample_ref)

            if "min_vals" not in out:
                continue
            lyr = out["layer"]
            layer_ranges.setdefault(lyr, {}).setdefault("mins", []).append(out.pop("min_vals"))
            layer_ranges.setdefault(lyr, {}).setdefault("maxs", []).append(out.pop("max_vals"))

        if samples_to_update:
            updates["samples"] = samples_to_update

        if not layer_ranges:
            step.flag = QCFlag.WARN
            step.add_reason("NO_RANGE_DATA", "No valid sample data found to compute ranges.")
            step_run.project_updates.append(updates)
            return {}, qc

        updated_layers: dict = {}
        for lyr, ranges in layer_ranges.items():
            agg_min = np.nanmin(np.asarray(ranges["mins"]), axis=0)
            agg_max = np.nanmax(np.asarray(ranges["maxs"]), axis=0)
            has_range = ~np.isnan(agg_min) & ~np.isnan(agg_max)
            dims = [dim.copy() for dim in self.project.layers[lyr]]
            for idx, dim in enumerate(dims):
                if has_range[idx]:
                    dim.range_min = float(np.min([agg_min[idx], dim.range_min]) if dim.range_min is not None else agg_min[idx])
                    dim.range_max = float(np.max([agg_max[idx], dim.range_max]) if dim.range_max is not None else agg_max[idx])
            updated_layers[lyr] = dims

        updates["layers"] = updated_layers
        step_run.project_updates.append(updates)
        return {}, qc
