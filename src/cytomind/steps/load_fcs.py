from __future__ import annotations

from cytomind.domain.pipeline import StepRun
from cytomind.domain.qc import QCRunStatus, QCFlag
from .base import BaseStep
from . import register_step

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
        panel_columns = set(panel.index.tolist())
        fcs_columns = set(fcs_df.columns)
        missing = list(panel_columns - fcs_columns)
        if missing:
            step_load.flag = QCFlag.FAIL
            step_load.add_reason(
                code="ChannelMissing",
                message=f"Sample {sample.id}: some panel channels not found in FCS file: {missing}.")
            return {}, qc

        # Load data matrix in panel order
        col_order = [pnn for pnn in panel.index.tolist() if pnn in fcs_df.columns]
        X = fcs_df[col_order].values

        # Create AnnData with panel (columns already in correct order)
        if sample.compensation:
            layer = 'comp'
            uns = {'compensation_id': sample.compensation}
            comp_evals = step_run.evaluable_products.setdefault("compensation", {})
            if sample.compensation in comp_evals:
                comp_evals[sample.compensation]["sample_ids"].append(sample.id)
            else:
                comp_evals[sample.compensation] = {"sample_ids": [sample.id]}

        else:
            layer = 'raw'
            uns = {}
            step_run.evaluable_products.setdefault("raw_data", {})[sample.id] = dict()
        var = self.repo.load_dimensions_df(layer)
        adata = ad.AnnData(X=X, var=var, uns=uns)

        step_save = self.save_adata(sample, adata, qc, layer=layer)[1]
        if step_save.flag == QCFlag.FAIL:
            return {}, qc

        # update sample metadata
        sample.n_events = adata.n_obs
        if sample.n_events != fcs.event_count:
            step_load.flag = QCFlag.WARN
            step_load.add_reason(
                code="MALFORMED_FCS",
                message=(f"Sample {sample.id}: event count mismatch between FCS file",
                         f"({fcs.event_count}) and loaded data ({sample.n_events})."))

        output_info = {
            "inputs": [sample.fcs_path.as_posix()],
            "n_events": sample.n_events,
            "outputs": [self.repo.sample_adata_path(sample.id, layer=layer).as_posix()],
        }
        return output_info, qc

    def update_project(self, step_run: StepRun) -> StepRun:
        samples = {
            sid: self.project.samples[sid]
            for sid, qc in step_run.qc.sample_qc.items() if qc.overall_flag != QCFlag.FAIL
        }
        for sid, ref in samples.items():
            ref.n_events = step_run.sample_outputs[sid].get("n_events", 0)
        self.repo.update_project_metadata(samples=samples)
        return step_run