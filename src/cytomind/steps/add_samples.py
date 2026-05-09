from __future__ import annotations
from typing import Any, Mapping

import hashlib
from datetime import datetime

import flowkit as fk

from cytomind.domain.flow import CompensationRef, ChannelRef, DimensionDef
from cytomind.domain.pipeline import SampleRef, StepRun, BatchRef
from cytomind.domain.qc import QCRunStatus, QCFlag
from .base import BaseStep
from . import register_step

__all__ = ["AddSamplesStep"]

@register_step("add_samples")
class AddSamplesStep(BaseStep):
    """
    Parse FCS files and build initial project registries (samples, compensations, panel).
    Each FCS file is treated as one sample; run_sample processes one file.
    Aggregation builds the deduplicated panel and compensation catalog.
    """

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

        # Parse config for channel mapping
        sample_info: dict[str, Any] = step_run.config.get("channel_mapping", {}).get(sample.fcs_path, {})
        if sample_info.get("drop", False):
            step_drop = qc.get_step("check_drop_status")
            step_drop.flag = QCFlag.WARN
            step_drop.add_reason(
                code="SAMPLE_DROPPED",
                message=f"Sample {sample.id} marked for drop in channel mapping (removed or error loading file)."
            )
            return {}, qc

        # 1) Parse FCS file
        step_load = qc.get_step("parse_fcs")
        fcs_path = sample.fcs_path.as_posix()  # for error messages
        try:
            fcs = fk.Sample(fcs_path)  # TODO: load only metadata not events
        except Exception as e:
            step_load.flag = QCFlag.FAIL
            step_load.add_reason(code="FCS_PARSE_ERROR",
                                 message=f"Error parsing FCS file {fcs_path}: {e}")
            return {}, qc

        # 2) Extract panel, compensation, metadata
        step_panel = qc.get_step("extract_panel")
        rename_info: dict[str, dict[str, str]] = sample_info.get("rename", {})
        try:
            panel = _extract_panel_from_fcs(fcs, rename_info=rename_info)
        except Exception as e:
            step_panel.flag = QCFlag.FAIL
            step_panel.add_reason(code="PANEL_EXTRACTION_ERROR",
                                  message=f"Error extracting panel from FCS file {fcs_path}: {e}")
            return {}, qc

        # 3) Parse metadata
        step_meta = qc.get_step("extract_metadata")
        try:
            meta = {
                "datatype": fcs.metadata.get("datatype", "N").upper(),
                "instrument": _get_instrument(fcs.metadata),
                "acquired_at": _parse_acquired_at(fcs.metadata),
                "timestep": float(fcs.metadata.get("timestep", "nan")),
            }
            meta.update(_parse_infcyt_fields(fcs.metadata))
        except Exception as e:
            step_meta.flag = QCFlag.FAIL
            step_meta.add_reason(
                code="METADATA_EXTRACTION_ERROR",
                message=f"Error extracting metadata from FCS file {fcs_path}: {e}"
            )
            return {}, qc
        if meta["datatype"] not in {"F", "D", "I", "A"}:
            step_meta.flag = QCFlag.WARN
            fcs_dtype = fcs.metadata.get("datatype")
            step_meta.add_reason(
                code="UNKNOWN_DATATYPE",
                message=(f"Unknown datatype '{fcs_dtype}' in FCS file {fcs_path}.",
                         "Expected one of F, D, I, A")
            )

        # 4) Parse compensation
        step_comp = qc.get_step("extract_compensation")

        detectors = [ch.pnn for ch in panel if ch.type == "fluorescence"]
        # some software use pnn:pns format for compensation labels
        infidetectors = [f"{ch.pnn}:{ch.pns}" for ch in panel if ch.type == "fluorescence"]

        comp_id: str | None = None
        comps_parsed: list[dict[str, Any]] = []
        for key, mat_txt in _iter_comp_records_from_metadata(fcs.metadata):
            if "0.-" in mat_txt:
                step_comp.flag = QCFlag.WARN
                step_comp.add_reason(
                    code="COMPENSATION_PARSE_WARNING",
                    message=(f"Compensation matrix '{key}' in FCS file {fcs_path} contains '0.-' values,",
                             "which may indicate an issue with how the matrix was stored in metadata.")
                )
                mat_txt = mat_txt.replace("0.-", "-0.")
            try:
                mat = fk.Matrix(mat_txt, detectors)
            except ValueError as e:
                if str(e).startswith("Matrix labels do not match given fluorescent labels"):
                    try:
                        mat = fk.Matrix(mat_txt, infidetectors)
                    except Exception as e2:
                        step_comp.flag = QCFlag.FAIL
                        step_comp.add_reason(
                            code="COMPENSATION_PARSE_ERROR",
                            message=("Error parsing compensation matrix",
                                     f"'{key}' from FCS file {fcs_path}: {e2}"))
                        return {}, qc
                else:
                    step_comp.flag = QCFlag.FAIL
                    step_comp.add_reason(
                        code="COMPENSATION_PARSE_ERROR",
                        message=f"Error parsing compensation matrix '{key}' from FCS file {fcs_path}: {e}"
                    )
                    return {}, qc
            comp_id_temp = _make_compensation_id(key, mat_txt)
            spill = mat.as_dataframe().rename(columns=lambda x: x.split(":")[0])
            comps_parsed.append({
                "id": comp_id_temp,
                "name": f"{sample.id}_{key}",
                "key": key,
                "mat_txt": mat_txt,
                "_spill": spill.to_dict(orient="list"), # pyright: ignore[reportArgumentType]
            })
            comp_id = comp_id_temp  # last one is the main comp

        comp_applied = fcs.metadata.get("apply compensation", "false").lower() == "true"
        if comp_applied:
            layer = "comp"
            if comp_id is None:
                step_comp.flag = QCFlag.WARN
                step_comp.add_reason(
                    code="COMPENSATION_MISSING",
                    message=("FCS metadata indicates compensation applied,",
                            "but no valid compensation matrix found.")
                )
            if meta["datatype"] == "I":
                step_comp.flag = QCFlag.WARN
                step_comp.add_reason(
                    code="COMPENSATION_INCONSISTENT",
                    message=("FCS metadata indicates compensation applied,",
                             "but data type is 'I' (compensated data usually 'F' or 'D').")
                )
        else:
            layer = "raw"
            if meta["datatype"] in {"F", "D"} and comp_id is not None:
                step_comp.flag = QCFlag.WARN
                step_comp.add_reason(
                    code="COMPENSATION_NOT_APPLIED",
                    message=("Compensation matrix found in FCS metadata,",
                             "but 'apply compensation' flag is false or missing.",
                             "Data type is 'F' or 'D', indicating compensated data.")
                )

        output_info = {
            "panel": [ch.to_record() for ch in panel],
            "compensations": comps_parsed,
            "sample_meta": {
                "id": sample.id,
                "fcs": sample.fcs_path.as_posix(),
                "default_layer": layer,
                "n_events": fcs.event_count,
                "compensation": comp_id,
                "rename": rename_info,
                "meta": meta,
            },
        }
        return output_info, qc

    def finalize_batch(self, batch_id: str, step_run: StepRun, qc: QCRunStatus) -> tuple[dict, QCRunStatus]:
        """
        Finalize batch by aggregating per-sample outputs.

        Deduplicates panel and compensations across samples, groups samples by panel,
        and populates project_updates with the aggregated data.
        """

        # Check batch existence
        step_get_outputs = qc.get_step("gather_sample_outputs")
        batch = self.project.batches[batch_id]

        missing_samples = [sid for sid in batch if sid not in step_run.sample_outputs]
        if missing_samples:
            step_get_outputs.flag = QCFlag.FAIL
            step_get_outputs.add_reason(
                code="SAMPLE_OUTPUTS_MISSING",
                message=(f"Some samples in batch {batch_id} are missing outputs",
                         f"from step run: {missing_samples}.")
            )
            return {}, qc

        # 1) Gather sample outputs
        sample_flags = step_run.qc.sample_flags
        outputs = {
            sid: step_run.sample_outputs[sid] for sid in batch
            if sample_flags[sid] == QCFlag.PASS
        }

        # 2) Extract Panel information and group samples by panel structure
        step_panel = qc.get_step("group_by_panel")
        panel_groups: dict[str, list[str]] = {}  # panel_hash -> [sample_ids]
        panel_cache: dict[str, list[ChannelRef]] = {}  # panel_hash -> panel

        for sample_id, out in outputs.items():
            panel_records = out.get("panel", [])
            if not panel_records:
                step_panel.flag = QCFlag.WARN
                step_panel.add_reason(code="MISSING_PANEL",
                                      message=f"Sample {sample_id} has no panel data; skipping.")
                continue
            panel = [ChannelRef.from_dict(ch) for ch in panel_records]
            panel_hash = _compute_panel_hash(panel)

            if panel_hash not in panel_groups:
                panel_groups[panel_hash] = []
                panel_cache[panel_hash] = panel
            panel_groups[panel_hash].append(sample_id)

        if not panel_groups:
            step_panel.flag = QCFlag.FAIL
            step_panel.add_reason("NO_PANELS", "No valid panels extracted from any sample.")
            return {}, qc

        # Identify the primary panel group.
        # If the project already has a main panel, prefer its hash over the majority group.
        existing_panel = self.project.panel_catalog.get("panel")
        if existing_panel:
            existing_hash = _compute_panel_hash(existing_panel)
            panel_hash = existing_hash if existing_hash in panel_groups else max(panel_groups.items(), key=lambda x: len(x[1]))[0]
        else:
            panel_hash = max(panel_groups.items(), key=lambda x: len(x[1]))[0]
        panel = panel_cache[panel_hash]
        sample_ids = panel_groups[panel_hash]

        if len(panel_groups) > 1:
            step_panel.flag = QCFlag.WARN
            step_panel.add_reason(
                code="MULTIPLE_PANELS",
                message=(f"Multiple panel groups detected ({len(panel_groups)}). "
                         f"Primary panel is {panel_hash[:8]} with {len(sample_ids)} samples.")
            )

        panel_groups_info = {
            "primary_hash": panel_hash,
            "groups": {
                ph: {
                    "sample_ids": sids,
                    "panel": [ch.to_record() for ch in panel_cache[ph]],
                    "is_primary": ph == panel_hash,
                    "n_samples": len(sids),
                }
                for ph, sids in panel_groups.items()
            },
        }

        # 3) Deduplicate compensations by (key, mat_txt)
        step_comp_dedup = qc.get_step("deduplicate_compensations")
        comp_dedupe: dict[tuple[str, str], CompensationRef] = {}
        for out in outputs.values():
            sid: str = out["sample_meta"]["id"]
            sid_comps: list[dict[str, Any]] = out.get("compensations", [])
            for comp_rec in sid_comps:
                dedupe_key = (comp_rec["key"], comp_rec["mat_txt"])
                if dedupe_key not in comp_dedupe:
                    existing_comp = self.project.compensations.get(comp_rec["id"])
                    comp_rec["source"] = "fcs"
                    comp_rec["batch"] = (list(existing_comp.batch) if existing_comp else []) + [sid]
                    comp_dedupe[dedupe_key] = CompensationRef.from_dict(comp_rec)
                else:
                    comp_dedupe[dedupe_key].batch.append(sid)

        compensations = list(comp_dedupe.values()) if comp_dedupe else []

        # 4) Build sample refs
        step_build_samples = qc.get_step("build_sample_refs")
        samples: list[SampleRef] = []
        for out in outputs.values():
            sm: dict[str, Any] = out["sample_meta"]
            samples.append(
                SampleRef(
                    id=sm["id"],
                    fcs=sm["fcs"],
                    default_layer=sm["default_layer"],
                    n_events=sm["n_events"],
                    compensation=sm["compensation"],
                    rename=sm.get("rename", {}),
                    meta=sm["meta"],
                )
            )

        # 5) Create batches
        step_create_batches = qc.get_step("create_batches")
        batches = []
        existing_panel_batch = self.project.batches.get("panel")
        panel_batch_samples = (existing_panel_batch.sample_ids if existing_panel_batch else set()) | set(sample_ids)
        batches.append(
            BatchRef(
                id="panel",
                sample_ids=panel_batch_samples,
                tags={"panel_group"},
                meta={"panel_hash": panel_hash}
            )
        )

        for ph, sids in panel_groups.items():
            if ph == panel_hash:
                continue
            bid = f"panel_{ph}"
            existing_batch = self.project.batches.get(bid)
            merged_sids = (existing_batch.sample_ids if existing_batch else set()) | set(sids)
            batches.append(
                BatchRef(
                    id=bid,
                    sample_ids=merged_sids,
                    tags={"panel_group"},
                    meta={"panel_hash": ph, "is_primary": False},
                )
            )

        # Create batches for each non-trivial compensation group
        n_comp_batches = 0
        for comp_ref in compensations:
            if len(comp_ref.batch) <= 1:
                continue  # skip trivial batches
            existing_comp_batch = self.project.batches.get(comp_ref.id)
            merged_sids = (existing_comp_batch.sample_ids if existing_comp_batch else set()) | set(comp_ref.batch)
            batches.append(
                BatchRef(
                    id=comp_ref.id,
                    sample_ids=merged_sids,
                    tags={"compensation_group"},
                    meta={"comp_id": comp_ref.id},
                )
            )
            n_comp_batches += 1

        # 6) Build dimensions for raw layer (one dimension per channel)
        step_build_dims = qc.get_step("build_dimensions")
        panel_dimensions = [
            DimensionDef(
                id=ch.pnn,
                source_dims=[ch.pnn],
                marker=ch.pns,
                type=ch.type,
                source_layer=None,
                transform_id="identity",
                idx=ch.idx,
            )
            for ch in panel
        ]

        # Build dimensions dict - always include raw, optionally include comp
        layers = {"raw": panel_dimensions}
        if any(sref.compensation is not None for sref in samples):
            # Create comp dimensions as identity transforms sourced from raw.
            comp_dimensions = [
                DimensionDef(
                    id=ch.pnn,
                    source_dims=[ch.pnn],
                    marker=ch.pns,
                    type=ch.type,
                    source_layer="raw",
                    transform_id="identity",
                    idx=ch.idx,
                )
                for ch in panel
            ]
            layers["comp"] = comp_dimensions

        # Set default dimension ranges from pnr (instrument amplifier range: 0 to pnr).
        # Build a lookup from channel pnn -> pnr using the primary panel.
        pnr_by_pnn = {ch.pnn: ch.pnr for ch in panel}
        missing_pnr: list[str] = []
        for layer_dims in layers.values():
            for dim in layer_dims:
                pnr = pnr_by_pnn.get(dim.id)
                if pnr is not None:
                    dim.range_min = 0.0
                    dim.range_max = float(pnr)
                else:
                    missing_pnr.append(dim.id)
        if missing_pnr:
            step_build_dims.flag = QCFlag.WARN
            step_build_dims.add_reason(
                code="PNR_MISSING",
                message=(f"pnr not set for {len(missing_pnr)} channel(s): {missing_pnr}. "
                         "Ranges will remain None until load_fcs is run or set manually.")
            )

        # Append project updates for this batch.
        # Only include panel catalog entries not already present in the project.
        existing_catalog_keys = set(self.project.panel_catalog.keys())
        panel_catalog = {
            f"panel_{ph}": panel_cache[ph]
            for ph in panel_groups
            if ph != panel_hash and f"panel_{ph}" not in existing_catalog_keys
        }
        if "panel" not in existing_catalog_keys:
            panel_catalog["panel"] = panel

        # Only include layers not already defined in the project.
        new_layers = {layer: dims for layer, dims in layers.items() if layer not in self.project.layers}

        step_run.project_updates.append({
            "panel_catalog": panel_catalog,
            "compensations": compensations,
            "samples": samples,
            "batches": batches,
            "layers": new_layers,
        })

        # Populate evaluable_products: these entities are fully initialized and ready for QC
        # Intentionally exclude compensations (they exist in registry but haven't been applied to sample data)
        step_run.evaluable_products["panel"] = {"panel": panel_groups_info}

        for sref in samples:
            out = step_run.sample_outputs.pop(sref.id)
            if sample_flags[sref.id] != QCFlag.PASS:
                step_run.sample_outputs[sref.id] = out["sample_meta"]


        # Store panel_groups_info in batch_outputs for revision handler
        return {}, qc

    def merge_config(self, step_run: StepRun) -> dict:
        batch_ids = step_run.inputs.get("batch_ids", [])
        if not batch_ids:
            raise ValueError("AddSampleStep requires a single batch_id as input.")
        if len(batch_ids) > 1:
            raise ValueError("AddSampleStep cannot merge multiple batch_ids; expected only one.")
        sample_ids = step_run.inputs.get("sample_ids", [])
        if sample_ids:
            raise ValueError("AddSampleStep does not support sample_ids input for merging; expected none.")

        return super().merge_config(step_run)

# ---- Helper functions (module-level) ----

def parse_channel_json_filtered(data: Mapping[str, Any]) -> dict[str, dict]:
    """
    Parse a channel JSON dictionary and extract drop/rename information for each FCS file.

    Parameters
    ----------
    data : dict
        Channel JSON data dictionary (already loaded from JSON file).

    Returns
    -------
    dict[str, dict]
        Dictionary mapping FCS file names to their processing information:
        {
            "sample.fcs": {
                "drop": bool,  # True if file should be dropped (removed or error)
                "rename": {
                    "channel": {old_name: current_name},  # Channel renames
                    "marker": {old_name: current_name}    # Marker renames
                }
            },
            ...
        }

    Examples
    --------
    >>> import json
    >>> with open("chanel_json_filtered.json") as f:
    ...     data = json.load(f)
    >>> info = parse_channel_json_filtered(data)
    >>> info["89349_Treg_PB_0.fcs"]
    {
        'drop': False,
        'rename': {
            'channel': {'FSC-A': 'FSC-A23', 'FITC-A': 'FITC-ABCD'},
            'marker': {'CD8': 'CD64'}
        }
    }
    """
    result = {}

    for fcs_file, file_info in data.items():
        # Determine if file should be dropped
        drop = file_info["removed"] or file_info["Error_loading_file"]

        # Extract channel and marker renames
        rename_channels = {}
        rename_markers = {}

        records = file_info["records"]
        for record in records:
            channel = record["channel"]
            channel_current = record["channel_current"]
            protein = record["protein"]
            protein_current = record["protein_current"]

            # Only include renames where current != previous and both are non-empty
            # Map old_name -> current_name
            if channel != channel_current:
                rename_channels[channel] = channel_current

            if protein != protein_current:
                rename_markers[protein] = protein_current

        result[fcs_file] = {
            "drop": drop,
            "rename": {
                "channel": rename_channels,
                "marker": rename_markers,
            }
        }

    return result


def _compute_panel_hash(panel: list[ChannelRef]) -> str:
    """
    Compute a fingerprint hash for a panel based on channel structure.
    Panels with identical pnn, pns, type, metric in same order get same hash.
    """
    # Build a stable string representation of the panel
    panel_str = ";".join(
        f"{ch.idx}:{ch.pnn}:{ch.pns}:{ch.type}:{ch.metric}"
        for ch in sorted(panel, key=lambda c: c.idx)
    )
    return hashlib.md5(panel_str.encode("utf-8")).hexdigest()[:8]


def _iter_comp_records_from_metadata(text: Mapping[str, str]):
    if "infcyt" in text:
        src = text.get("infinispill_src", "").strip()
        if src:
            yield ("infinispill_src", src)
        main = text.get("infinispill", "").strip()
        if main:
            yield ("infinispill", main)
        return
    for key in ("spill", "spillover", "comp"):
        val = text.get(key, "")
        if isinstance(val, str) and val.strip():
            yield (key, val.strip())
            return


def _make_compensation_id(comp_key: str, comp_value: str) -> str:
    h = hashlib.md5(f"{comp_key}:{comp_value}".encode("utf-8")).hexdigest()[:8]
    return f"comp_{h}"


def _get_instrument(text: Mapping[str, str]) -> str | None:
    for key in ("cyt", "cytsn", "inst"):
        val = text.get(key)
        if val:
            return val
    return None


def _parse_infcyt_fields(text: Mapping[str, str]) -> dict:
    if "infcyt" not in text:
        return {}
    return {
        "infversion": text.get("infversion", "").strip(),
        "infdate": text.get("infdate", "").strip(),
        "apply_compensation": text.get("apply compensation", "").strip().lower() == "true",
    }


def _parse_acquired_at(text: Mapping[str, str]) -> str | None:
    date_str = text.get("date")
    time_str = text.get("btim") or text.get("etim")
    if not date_str:
        return None
    date_formats = ["%d-%b-%Y", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"]
    parsed_date = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        return None
    if time_str:
        time_formats = ["%H:%M:%S", "%H:%M:%S.%f", "%H:%M"]
        parsed_time = None
        for tfmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, tfmt).time()
                break
            except ValueError:
                continue
        if parsed_time:
            dt = datetime.combine(parsed_date, parsed_time)
            return dt.isoformat()
    return datetime(parsed_date.year, parsed_date.month, parsed_date.day).isoformat()


def _extract_panel_from_fcs(fcs: fk.Sample, rename_info: Mapping[str, Mapping[str, str]] = {}) -> list[ChannelRef]:
    """
    Extract panel from FCS file and optionally apply channel/marker renames.

    Parameters
    ----------
    fcs : fk.Sample
        FlowKit Sample object
    rename_info : Mapping[str, Mapping[str, str]]
        Optional rename mappings with structure {"channel": {old: new}, "marker": {old: new}}

    Returns
    -------
    list[ChannelRef]
        List of channel references with renames applied if provided
    """
    channels = fcs.channels.copy()
    channels["type"] = "fluorescence"
    channels.loc[fcs.scatter_indices, "type"] = "scatter"
    if hasattr(fcs, "time_index") and fcs.time_index is not None:
        channels.loc[fcs.time_index, "type"] = "time"
    channels["metric"] = (
        channels.pnn.str.split("-").str[-1].str.lower().map({"a": "Area", "h": "Height", "w": "Width"})
    )
    channels.replace({"": None}, inplace=True)
    channels.rename(columns={"channel_number": "idx"}, inplace=True)

    # Apply renames to DataFrame before converting to ChannelRef
    if rename_info:
        channel_renames = rename_info.get("channel", {})
        marker_renames = rename_info.get("marker", {})

        if channel_renames:
            channels["pnn"] = channels["pnn"].replace(channel_renames)
        if marker_renames:
            channels["pns"] = channels["pns"].replace(marker_renames)

    channels.sort_values("idx", inplace=True)
    return [ChannelRef.from_dict(ch) for ch in channels.to_dict(orient="records")] # pyright: ignore[reportArgumentType]
