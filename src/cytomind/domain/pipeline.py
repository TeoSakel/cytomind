"""
Structures to hold metadata required for the CytoMind pipeline management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Any
import json

import numpy as np
from pandas import DataFrame

from .flow import ChannelRef, DimensionDef, CompensationRef
from .transforms import TransformDef, make_default_transform_def
from .gates import GatingStrategyRef
from .qc import EntityQCStatus

__all__ = ["Project", "BatchRef", "SampleRef", "StepRun", "RevisionSession", "NumpyEncoder"]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy scalar types and arrays.

    Converts numpy integers and floats to their Python native equivalents
    for JSON serialization. Arrays are converted to lists.
    """

    def default(self, o):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


@dataclass
class Project:
    id: str
    panel_catalog: dict[str, list[ChannelRef]] = field(default_factory=dict)  # panel_id -> channels
    samples: dict[str, SampleRef] = field(default_factory=dict)
    batches: dict[str, BatchRef] = field(default_factory=dict)
    layers: dict[str, list[DimensionDef]] = field(default_factory=dict)  # layer -> dimensions
    compensations: dict[str, CompensationRef] = field(default_factory=dict)
    transforms: dict[str, TransformDef] = field(default_factory=dict)
    gating_strategy: GatingStrategyRef = field(default_factory=GatingStrategyRef)

    @property
    def panel(self) -> list[ChannelRef]:
        """Convenience property to get the default panel channels."""
        return self.panel_catalog.get("panel", [])

    @property
    def panel_df(self) -> DataFrame:
        if "raw" in self.layers:
            df = DataFrame.from_records([dim.to_record() for dim in self.layers["raw"]])
            df.set_index("id", inplace=True)
            return df
        panel = self.panel_catalog.get("panel", [])
        if not panel:
            return DataFrame()

        df = DataFrame.from_records([ch.to_record() for ch in panel])
        df.sort_values("idx", inplace=True)
        df.set_index("pnn", inplace=True, drop=False)
        return df.loc[:, df.notnull().any()]  # drop empty cols

    def get_channel_ref(self, channel_id: str, panel_id: str = "panel") -> ChannelRef | None:
        channels = self.panel_catalog.get(panel_id, [])
        for channel in channels:
            if channel.pnn == channel_id:
                return channel
        return None

    def register_transform(self, transform: TransformDef) -> str:
        if transform.id in self.transforms:
            existing = self.transforms[transform.id]
            if existing.type != transform.type or dict(existing.params) != dict(transform.params):
                raise ValueError(f"Conflicting transform definition for id '{transform.id}'.")
            return transform.id
        self.transforms[transform.id] = transform
        return transform.id

    def get_or_create_transform(
        self,
        transform_type: str,
        channel_id: str | None = None,
        panel_id: str = "panel",
        params_override: Mapping[str, Any] | None = None,
        tags: Iterable[str] = (),
    ) -> TransformDef:
        channel_ref = self.get_channel_ref(channel_id, panel_id=panel_id) if channel_id is not None else None
        transform = make_default_transform_def(
            transform_type=transform_type,
            channel_ref=channel_ref,
            params_override=params_override,
            tags=tags,
        )
        self.register_transform(transform)
        return self.transforms[transform.id]

    def resolve_transform(self, transform_id: str) -> TransformDef:
        if transform_id not in self.transforms:
            raise KeyError(f"Unknown transform id '{transform_id}'.")
        return self.transforms[transform_id]

    def resolve_dimension_transform(self, dimension: DimensionDef) -> TransformDef:
        return self.resolve_transform(dimension.transform_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Project":
        panel_catalog = {k: [ChannelRef.from_dict(ch) for ch in v] for k, v in data.get("panel_catalog", {}).items()}
        if not panel_catalog and data.get("panel"):
            panel_catalog = {"panel": [ChannelRef.from_dict(ch) for ch in data.get("panel", [])]}

        gating_strategy_data = data.get("gating_strategy")

        return cls(
            id=data["id"],
            samples={k: SampleRef.from_dict(v) for k, v in data.get("samples", {}).items()},
            panel_catalog=panel_catalog,
            layers={k: [DimensionDef.from_dict(dim) for dim in v] for k, v in data.get("layers", {}).items()},
            compensations={k: CompensationRef.from_dict(v) for k, v in data.get("compensations", {}).items()},
            transforms={k: TransformDef.from_dict(v) for k, v in data.get("transforms", {}).items()},
            batches={k: BatchRef.from_dict(v) for k, v in data.get("batches", {}).items()},
            gating_strategy=GatingStrategyRef.from_dict(gating_strategy_data) if gating_strategy_data else GatingStrategyRef(),
        )

    def to_dict(self) -> dict[str, Any]:
        def _serialize_list_catalog(catalog: Mapping[str, Iterable[Any]]) -> dict[str, list[dict[str, Any]]]:
            return {k: [item.to_dict() for item in v] for k, v in catalog.items()}

        return {
            "id": self.id,
            "samples": {k: v.to_dict() for k, v in self.samples.items()},
            "panel_catalog": _serialize_list_catalog(self.panel_catalog),
            "layers": _serialize_list_catalog(self.layers),
            "compensations": {k: v.to_dict() for k, v in self.compensations.items()},
            "transforms": {k: v.to_dict() for k, v in self.transforms.items()},
            "batches": {k: v.to_dict() for k, v in self.batches.items()},
            "gating_strategy": self.gating_strategy.to_dict(),
        }

@dataclass
class BatchRef:
    """
    Lightweight handle for a batch of samples.
    """

    id: str
    sample_ids: set[str]
    tags: set[str] = field(default_factory=set)  # optional tags to label where this batch belongs
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Iterate over sample_ids in the batch."""
        return iter(self.sample_ids)

    def __next__(self):
        """Get next sample_id in the batch."""
        yield from self.sample_ids

    def __len__(self):
        """Get number of samples in the batch."""
        return len(self.sample_ids)

    def __contains__(self, item: str | SampleRef):
        """Check if a sample_id is in the batch."""
        sample_id = item.id if isinstance(item, SampleRef) else item
        return sample_id in self.sample_ids

    def copy(self) -> "BatchRef":
        return BatchRef(
            id=self.id,
            sample_ids=self.sample_ids.copy(),
            tags=self.tags.copy(),
            meta=self.meta.copy(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchRef":
        return cls(
            id=data["id"],
            sample_ids=set(data.get("sample_ids", [])),
            tags=set(data.get("tags", [])),
            meta={k: v for k, v in data.items() if k not in {"id", "sample_ids", "tags", "root"}},
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "id": self.id,
            "sample_ids": list(self.sample_ids),
            "tags": list(self.tags),
        }
        base.update(self.meta)
        return base

    def subset(self, id:str, sample_ids: Iterable[str]) -> "BatchRef":
        """Create a new BatchRef that is a subset of the current one."""

        sample_ids = set(sample_ids)
        if not sample_ids.issubset(self.sample_ids):
            raise ValueError("Subset sample_ids must be a subset of the original batch's sample_ids.")

        tags = self.tags | {f"subset_of_{self.id}"} if self.tags else {f"subset_of_{self.id}"}
        return BatchRef(
            id=id,
            sample_ids=sample_ids,
            tags=tags,
            meta={sid: self.meta.get(sid, {}) for sid in sample_ids},
        )

    def drop_samples(self, sample_ids: Iterable[str]) -> "BatchRef":
        """Create a new BatchRef with specified sample_ids removed."""

        sample_ids = set(sample_ids)
        remaining = {sid for sid in self.sample_ids if sid not in sample_ids}

        return BatchRef(
            id=self.id,
            sample_ids=remaining,
            tags=self.tags.copy(),
            meta={sid: self.meta.get(sid, {}) for sid in remaining},
        )


@dataclass
class SampleRef:
    """
    Lightweight handle for a sample, with on-disk (backed) AnnData access.

    Notes
    -----
    - `fcs_path` and `anndata_path` are kept as strings so the dataclass is
      JSON-serializable via `dataclasses.asdict` or similar.
    - Internal `_fcs_path` / `_anndata_path` are Path objects for convenience.
    - `.adata` returns a *backed* AnnData (read_h5ad(..., backed="r")).
    - Use `.get_subset()` / `.get_random_subset()` to materialize small
      in-memory AnnData slices for gating, QC, etc.
    """

    id: str
    fcs: str                          # original FCS file name
    default_layer: str = "raw"        # default data layer
    n_events: int = -1                # number of events in fcs (if known)
    compensation: str | None = None   # map to CompensationRef id
    rename: dict[str, dict[str, str]] = field(default_factory=dict)  # {"channel": {old: new}, "marker": {old: new}}
    meta: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "SampleRef":
        return SampleRef(
            id=self.id,
            fcs=self.fcs,
            default_layer=self.default_layer,
            n_events=self.n_events,
            compensation=self.compensation,
            rename={k: v.copy() for k, v in self.rename.items()},
            meta=self.meta.copy(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleRef":
        non_meta_fields = {"id", "fcs", "default_layer", "n_events", "compensation", "rename", "status", "latest_steps"}
        return cls(
            id=data["id"],
            fcs=data["fcs"].as_posix() if isinstance(data["fcs"], Path) else str(data["fcs"]),
            default_layer=data.get("default_layer", "raw"),
            n_events=data.get("n_events", -1),
            compensation=data.get("compensation"),
            rename=data.get("rename", {}),
            meta={k: v for k, v in data.items() if k not in non_meta_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "id": self.id,
            "fcs": self.fcs,
            "default_layer": self.default_layer,
            "n_events": self.n_events,
            "compensation": self.compensation,
            "rename": self.rename,
        }
        base.update(self.meta)
        return base

    @property
    def fcs_path(self) -> Path:
        """Path object for the FCS file (for internal use)."""
        return Path(self.fcs)

@dataclass
class StepRun:
    """
    Mutable execution context for a step run, shared across all execution phases.

    The StepRun acts as a mutable container that phases populate incrementally:
    - prepare_batch() populates batch_outputs[batch_id] with batch context (priors, thresholds, etc.)
    - run_sample() reads from batch_outputs and populates sample_outputs[sample_id]
    - finalize_batch() reads from sample_outputs, aggregates, and populates project_updates
    - update_project() applies accumulated project_updates to the project

    Fields
    ------
    sample_outputs : dict[str, Any]
        Per-sample results keyed by sample_id. Populated by run_sample(),
        read by finalize_batch() for aggregation.

    batch_outputs : dict[str, Any]
        Per-batch results and context keyed by batch_id. Populated by
        prepare_batch() and finalize_batch(). read by run_sample() for batch context.

    project_updates : list[dict[str, Any]]
        List of project change dicts accumulated by phases (mainly finalize_batch()).
        Each dict contains keys for project registries (samples, dimensions, compensations, etc.).
        Applied sequentially by update_project() in the order they were added.
        Each batch can append its own update dict to this list.

    evaluable_products : dict[str, dict[str, Any]]
        Products explicitly marked as ready for QC evaluation by finalize_batch().
        Structure mirrors project_updates but only contains entities that have been
        fully processed and are safe to evaluate. Steps populate this during finalize_batch()
        to distinguish "created in registry" from "actually applicable/ready for QC".
        Example: AddSamplesStep includes samples but NOT compensations (registered but not applied).

    _qc: EntityQCStatus | None = None
        Quality control status for this step run. Initialized by BaseStep.run() and
        populated during execution. Contains per-sample QC data (qc.per_sample_steps)
        and aggregated summary (qc.summary).
    """
    id: str
    step_type: str
    status: str = "pending"                         # "pending" | "completed" | "failed"
    created_at: str = ""                            # ISO string
    config: dict[str, Any] = field(default_factory=dict)          # algorithmic knobs for this run
    inputs: dict[str, Any] = field(default_factory=dict)          # ids: sample_ids, batch_ids, etc.
    sample_outputs: dict[str, Any] = field(default_factory=dict)  # keyed by sample_id
    batch_outputs: dict[str, Any] = field(default_factory=dict)   # keyed by batch_id
    project_updates: list[dict[str, Any]] = field(default_factory=list)  # list of project changes to apply
    evaluable_products: dict[str, dict[str, Any]] = field(default_factory=dict)  # products ready for QC evaluation
    _qc: EntityQCStatus | None = None

    @property
    def qc(self) -> EntityQCStatus:
        if self._qc is None:
            self._qc = EntityQCStatus(
                entity_type="step",
                entity_id=self.id,
                context={}
            )
        return self._qc

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_type": self.step_type,
            "status": self.status,
            "created_at": self.created_at,
            "config": self.config,
            "inputs": self.inputs,
            "sample_outputs": self.sample_outputs,
            "batch_outputs": self.batch_outputs,
            "project_updates": self.project_updates,
            "evaluable_products": self.evaluable_products,
            "qc": self._qc.to_dict() if self._qc else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRun":
        qc_data = data.get("qc")
        qc = EntityQCStatus.from_dict(qc_data) if qc_data else None
        return cls(
            id=data["id"],
            step_type=data["step_type"],
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
            config=data.get("config", {}),
            inputs=data.get("inputs", {}),
            sample_outputs=data.get("sample_outputs", {}),
            batch_outputs=data.get("batch_outputs", {}),
            project_updates=data.get("project_updates", []),
            evaluable_products=data.get("evaluable_products", {}),
            _qc=qc,
        )


# ============================================================================
# Revision System - Abstract I/O Declarations
# ============================================================================

@dataclass
class RevisionSession:
    """
    Tracks the state of a revision session for a specific entity.

    A revision session creates an isolated workspace where users can iteratively
    refine entity data until satisfied, then commit changes back to main project.

    Supported entity types:
    - "compensation": Revise compensation matrices
    - "gating_strategy": Revise gating strategies
    - "step": Edit step configuration for rerun
    """
    id: str                              # revision session id (e.g., "rev_001")
    entity_type: str                     # Entity type (e.g., "compensation", "gating_strategy", "step")
    status: str                           # "active" | "ready_to_commit" | "committed" | "cancelled"
    entity_id: str | None = None         # The entity being revised (e.g., "comp_001", "step_0003")
    created_at: str = ""                 # ISO timestamp of when the revision session was created
    updated_at: str = ""                 # ISO timestamp of the last update to the revision session

    # Handler state
    handler_state: dict[str, Any] = field(default_factory=dict)  # internal state for the revision handler

    # User selections for what to revise
    context: dict[str, Any] = field(default_factory=dict)  # context for the revision (arguments used to generate artifacts etc)
    revision_history: list[dict[str, Any]] = field(default_factory=list)  # History of apply_revision calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "status": self.status,
            "entity_id": self.entity_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "handler_state": self.handler_state,
            "context": self.context,
            "revision_history": self.revision_history,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RevisionSession":
        return cls(
            id=data["id"],
            entity_type=data["entity_type"],
            status=data["status"],
            entity_id=data.get("entity_id"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            handler_state=data.get("handler_state", {}),
            context=data.get("context", {}),
            revision_history=data.get("revision_history", []),
        )