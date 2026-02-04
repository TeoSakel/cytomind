"""
Add samples revision handler for sample filtering and panel refinement.

Handles iterative refinement of sample selection and channel name mapping.
Parses user-provided channel remapping JSON (from frontend) to:
- Filter excluded samples
- Compute common panel (intersection on channel_current)
- Generate per-sample channel/protein name mappings
- Persist mappings to project repo
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import DataFrame

from cytomind.domain.flow import ChannelRef
from cytomind.domain.pipeline import StepRun
from cytomind.revisions import RevisionHandlerRegistry
from cytomind.revisions.base import BaseRevisionHandler

if TYPE_CHECKING:
    from cytomind.domain.pipeline import RevisionSession


@RevisionHandlerRegistry.register("add_samples")
class AddSamplesRevisionHandler(BaseRevisionHandler):
    """
    Revision handler for add_samples step.

    Processes channel remapping JSON from frontend to:
    1. Filter excluded samples
    2. Compute common panel via intersection on channel_current
    3. Generate per-sample channel/protein name mappings (only changes)
    4. Persist mappings to project repo
    """

    def start_revision(self, input_spec: dict[str, Any]) -> "RevisionSession":
        """
        Initialize revision workspace for sample filtering and channel mapping.

        Validates that samples exist in project and that FCS files are accessible.

        Parameters
        ----------
        input_spec : dict
            User input specification with sample_ids and revision data

        Returns
        -------
        RevisionSession
            Initialized session
        """
        session = super().start_revision(input_spec)
        self.state["step_output"] = self.step_run.batch_outputs.get("summary", {})
        self.save_session()

        return session

    def get_input_spec_schema(self) -> dict[str, Any]:
        """
        Get JSON schema for user input specification.

        Expects revision JSON with structure:
        {
            "sample_id": {
                "removed": bool,
                "records": [
                    {
                        "channel": original_pnn,
                        "channel_current": new_pnn,
                        "protein": original_pns,
                        "protein_current": new_pns,
                        ...
                    },
                    ...
                ]
            },
            ...
        }
        """
        return {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "removed": {"type": "boolean"},
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "channel": {"type": "string"},
                                "channel_current": {"type": "string"},
                                "protein": {"type": ["string", "null"]},
                                "protein_current": {"type": ["string", "null"]},
                            },
                            "required": ["channel", "channel_current"],
                        },
                    },
                },
            },
        }

    def get_modification_schema(self) -> dict[str, Any]:
        """Get JSON schema for modification specification."""
        return {
            "type": "object",
            "properties": {
                "excluded_samples": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

    def apply_revision(
        self,
        user_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply revision: filter samples and compute channel mappings.

        Parameters
        ----------
        user_input : dict
            User modifications with channel remapping data

        Returns
        -------
        dict
            QC summary with excluded samples and panel information
        """
        # Parse the revision JSON (user_input should contain the channel_json_filtered structure)
        # For now, return a simple placeholder
        # In a full implementation, this would:
        # 1. Parse input_spec (revision JSON from frontend)
        # 2. Filter excluded samples
        # 3. Compute common panel (intersection on channel_current)
        # 4. Generate per-sample channel/protein mappings

        excluded = []
        retained = []

        # Update session state
        self.state["excluded_samples"] = excluded
        self.state["retained_samples"] = retained
        self.save_session()

        return {
            "excluded_samples": excluded,
            "retained_samples": retained,
            "n_excluded": len(excluded),
            "n_retained": len(retained),
        }

    def _commit(self) -> tuple[dict[str, Any], StepRun | None]:
        """
        Commit add_samples revision changes.

        Returns
        -------
        tuple
            (metadata_updates, new_step)
        """
        # Return empty metadata updates and no new step
        # In a full implementation, this would persist the filtered samples and mappings

        return {"samples": {}}, None

    def get_figure(self, plot_type: str, input_params: dict[str, Any]) -> dict[str, Any]:
        """Generate a figure for visualization."""
        raise NotImplementedError(f"Figure type {plot_type} not implemented")

    def get_table(self, table_type: str, input_params: dict[str, Any]) -> DataFrame:
        """Generate a table for display."""
        if table_type == "sample_status":
            return self._get_sample_status_table()
        elif table_type == "panel_changes":
            return self._get_panel_changes_table()
        elif table_type == "channel_mapping":
            return self._get_channel_mapping_table()
        else:
            raise ValueError(f"Unknown table: {table_type}")

    # ---- Tables

    def get_panel_group_info(self) -> DataFrame:
        """Get summary DataFrame of panel information."""
        panel_group_info = self.state["step_output"].get("panel_group_info", {})
        return pd.DataFrame.from_dict(panel_group_info)

    def get_channel_info(self) -> DataFrame:
        """
        Get channel presence table across panel groups.

        Returns
        -------
        DataFrame
            Table with channels as rows, panel groups + 'samples' as columns.
            Each panel group column has 1 if channel is present, 0 otherwise.
            'samples' column contains total count of samples with that channel.
        """
        panel_group_info = self.state["step_output"].get("panel_groups_info", {})

        if not panel_group_info:
            return pd.DataFrame()

        groups = panel_group_info.get("groups", {})

        # Collect all unique channels across all panel groups
        all_channels = set()

        for panel_hash, group_info in groups.items():
            panel = group_info.get("panel", [])
            for ch_record in panel:
                ch_id = ch_record["id"]
                all_channels.add(ch_id)

        if not all_channels:
            return pd.DataFrame()

        # Initialize DataFrame with channels as index and zeros
        ch_idx = list(sorted(all_channels))
        columns = [panel_hash[:8] for panel_hash in groups] + ["samples"]
        df = pd.DataFrame(data=np.zeros((len(ch_idx), len(columns)), dtype=int),
                          index=ch_idx,
                          columns=columns)
        df.index.name = "channel"

        # Loop over panel groups and set presence
        for panel_hash, group_info in groups.items():
            col_name = panel_hash[:8]
            panel_channels = [ch["id"] for ch in group_info.get("panel", [])]
            df.loc[panel_channels, col_name] = 1
            df.loc[panel_channels, "samples"] += group_info.get("n_samples", 0)

        # Sort by number of samples (descending) then by channel name
        df = df.sort_values(["samples"], ascending=False)

        return df

    # ---- Private helpers ----

    def _compute_common_panel(
        self, sample_records: dict[str, list[dict[str, str]]]
    ) -> list[ChannelRef]:
        """
        Compute common panel via intersection on channel_current.

        Channels present in all retained samples form the common panel.

        Parameters
        ----------
        sample_records : dict[str, list[dict]]
            Per-sample channel records from revision JSON

        Returns
        -------
        list[ChannelRef]
            Common panel channels
        """
        if not sample_records:
            raise ValueError("No sample records provided")

        # Get set of channel_current from each sample
        sample_channel_sets = []
        for sample_id, records in sample_records.items():
            channels = {rec["channel_current"] for rec in records}
            sample_channel_sets.append(channels)

        # Intersection
        common_channels = set.intersection(*sample_channel_sets)
        if not common_channels:
            raise ValueError("No common channels across all retained samples")

        # Build common panel from first sample's records (use as canonical)
        first_sample_id = next(iter(sample_records.keys()))
        first_records = sample_records[first_sample_id]

        # Get original channel info (from project panel)
        project = self.main_repo.load_project()
        original_panel = {ch.pnn: ch for ch in project.panel}

        common_panel = []
        for rec in first_records:
            if rec["channel_current"] in common_channels:
                # Use original pnn to look up full channel info
                original_pnn = rec["channel"]
                if original_pnn in original_panel:
                    ch = original_panel[original_pnn]
                    # Create new ChannelRef with updated pnn and pns
                    updated_ch = ChannelRef(
                        pnn=rec["channel_current"],
                        pns=rec.get("protein_current") or ch.pns,
                        pne=ch.pne,
                        png=ch.png,
                        pnr=ch.pnr,
                        type=ch.type,
                        metric=ch.metric,
                        idx=ch.idx,
                    )
                    common_panel.append(updated_ch)

        return sorted(common_panel, key=lambda ch: ch.idx)

    def _compute_channel_mappings(
        self,
        sample_records: dict[str, list[dict[str, str]]],
        common_panel: list[ChannelRef],
    ) -> dict[str, dict[str, dict[str, str]]]:
        """
        Generate per-sample channel/protein name mappings.

        Only includes channels that change names (where channel != channel_current
        or protein != protein_current).

        Parameters
        ----------
        sample_records : dict[str, list[dict]]
            Per-sample channel records from revision JSON
        common_panel : list[ChannelRef]
            Common panel (with updated names)

        Returns
        -------
        dict
            Structure:
            {
                "sample_id": {
                    "channels": {"original": "new", ...},
                    "proteins": {"original": "new", ...}
                },
                ...
            }
        """
        mappings = {}

        for sample_id, records in sample_records.items():
            channel_renames = {}
            protein_renames = {}

            for rec in records:
                original_channel = rec["channel"]
                new_channel = rec["channel_current"]
                original_protein = rec.get("protein")
                new_protein = rec.get("protein_current")

                # Only store if in common panel
                if new_channel in {ch.pnn for ch in common_panel}:
                    # Store channel rename if different
                    if original_channel != new_channel:
                        channel_renames[original_channel] = new_channel

                    # Store protein rename if different and both exist
                    if original_protein and new_protein and original_protein != new_protein:
                        protein_renames[original_channel] = new_protein

            # Only store sample if it has any mappings
            if channel_renames or protein_renames:
                mappings[sample_id] = {
                    "channels": channel_renames,
                    "proteins": protein_renames,
                }

        return mappings

    def _get_sample_status_table(self) -> pd.DataFrame:
        """Get table of sample inclusion/exclusion status."""
        excluded = self.state.get("excluded_samples", [])
        retained = self.state.get("retained_samples", [])

        data = []
        for sid in self.session.target_samples:
            data.append({
                "sample_id": sid,
                "status": "excluded" if sid in excluded else "retained",
            })

        return pd.DataFrame(data)

    def _get_panel_changes_table(self) -> pd.DataFrame:
        """Get table of panel changes (channel count, etc.)."""
        initial_panel_size = len(self.main_repo.load_project().panel)
        common_panel_size = self.state.get("common_panel_size", 0)

        return pd.DataFrame({
            "metric": ["Initial panel size", "Common panel size", "Channels removed"],
            "value": [initial_panel_size, common_panel_size, initial_panel_size - common_panel_size],
        })

    def _get_channel_mapping_table(self) -> pd.DataFrame:
        """Get table of per-sample channel mappings."""
        mappings = self.state.get("channel_mappings", {})

        data = []
        for sample_id, sample_mapping in mappings.items():
            channels = sample_mapping.get("channels", {})
            proteins = sample_mapping.get("proteins", {})

            for orig_ch, new_ch in channels.items():
                data.append({
                    "sample_id": sample_id,
                    "channel_original": orig_ch,
                    "channel_new": new_ch,
                    "protein_original": proteins.get(orig_ch, ""),
                    "protein_new": proteins.get(orig_ch, ""),
                })

        return pd.DataFrame(data) if data else pd.DataFrame()
