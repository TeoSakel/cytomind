"""
Structures to hold metadata required for the CytoMind pipeline management.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping, Any

from anndata import AnnData
from pandas import DataFrame

from .flow import *
from .gates import GatingStrategyRef
from .qc import EntityQCStatus

__all__ = ["Project", "BatchRef", "SampleRef", "StepRun", "ResourceSpec", "RevisionSession"]

@dataclass
class Project:
    id: str
    panel: list[ChannelRef] = field(default_factory=list)  # list of all channels in the panel
    samples: dict[str, SampleRef] = field(default_factory=dict)
    batches: dict[str, BatchRef] = field(default_factory=dict)
    dimensions: dict[str, list[DimensionDef]] = field(default_factory=dict)  # layer -> dimensions
    compensations: dict[str, CompensationRef] = field(default_factory=dict)
    transformations: dict[str, TransformationRef] = field(default_factory=dict)
    gating_strategies: dict[str, GatingStrategyRef] = field(default_factory=dict)

    @property
    def panel_df(self) -> DataFrame:
        if "raw" in self.dimensions:
            df = DataFrame.from_records([dim.to_record() for dim in self.dimensions["raw"]])
            df.set_index("id", inplace=True)
            return df
        df = DataFrame.from_records([ch.to_record() for ch in self.panel])
        df.sort_values("idx", inplace=True)
        df.set_index("pnn", inplace=True, drop=False)
        return df.loc[:, df.notnull().any()]  # drop empty cols

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Project":

        return cls(
            id=data["id"],
            samples={k: SampleRef.from_record(v) for k, v in data.get("samples", {}).items()},
            panel=[ChannelRef.from_record(ch) for ch in data.get("panel", [])],
            dimensions={k: [DimensionDef.from_dict(dim) for dim in v] for k, v in data.get("dimensions", {}).items()},
            compensations={k: CompensationRef.from_record(v) for k, v in data.get("compensations", {}).items()},
            transformations={k: TransformationRef.from_record(v) for k, v in data.get("transformations", {}).items()},
            batches={k: BatchRef.from_record(v) for k, v in data.get("batches", {}).items()},
            gating_strategies={k: GatingStrategyRef.from_dict(v) for k, v in data.get("gating_strategies", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "samples": {k: v.to_dict() for k, v in self.samples.items()},
            "panel": [ch.to_dict() for ch in self.panel],
            "dimensions": {
                k: [dim.to_dict() for dim in v]
                for k, v in self.dimensions.items()
            },
            "compensations": {k: v.to_dict() for k, v in self.compensations.items()},
            "transformations": {k: v.to_dict() for k, v in self.transformations.items()},
            "batches": {k: v.to_dict() for k, v in self.batches.items()},
            "gating_strategies": {k: v.to_dict() for k, v in self.gating_strategies.items()},
        }

@dataclass
class BatchRef:
    """
    Lightweight handle for a batch of samples.
    """

    id: str
    sample_ids: list[str]
    tags: list[str] = field(default_factory=list)  # optional tags to label where this batch belongs
    meta: dict[str, Any] = field(default_factory=dict)
    _adata_backed: dict[str, AnnData] = field(init=False, repr=False, hash=False, default_factory=dict)

    def __iter__(self):
        """Iterate over sample_ids in the batch."""
        return iter(self.sample_ids)

    def __next__(self):
        """Get next sample_id in the batch."""
        yield from self.sample_ids

    def __len__(self):
        """Get number of samples in the batch."""
        return len(self.sample_ids)

    def copy(self) -> "BatchRef":
        return BatchRef(
            id=self.id,
            sample_ids=self.sample_ids.copy(),
            tags=self.tags.copy(),
            meta=self.meta.copy(),
        )

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "BatchRef":
        return cls(
            id=data["id"],
            sample_ids=list(data.get("sample_ids", [])),
            tags=list(data.get("tags", [])),
            meta={k: v for k, v in data.items() if k not in {"id", "sample_ids", "tags", "root"}},
        )

    def to_dict(self) -> dict[str, Any]:
        base = {
            "id": self.id,
            "sample_ids": self.sample_ids,
            "tags": self.tags,
        }
        base.update(self.meta)
        return base


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
    def from_record(cls, data: Mapping[str, Any]) -> "SampleRef":
        non_meta_fields = {"id", "fcs", "root", "default_layer", "n_events", "compensation", "rename", "status", "latest_steps"}
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
        base = asdict(self)
        # Serialize QC object
        if self._qc:
            base["qc"] = self._qc.to_dict()
        else:
            base["qc"] = None
        return base

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRun":
        qc_data = data.get("qc")
        qc = EntityQCStatus.from_dict(qc_data) if qc_data else None
        return cls(
            id=data["id"],
            step_type=data["step_type"],
            config=data.get("config", {}),
            inputs=data.get("inputs", {}),
            sample_outputs=data.get("sample_outputs", {}),
            batch_outputs=data.get("batch_outputs", {}),
            project_updates=data.get("project_updates", []),
            _qc=qc,
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
        )


# ============================================================================
# Revision System - Abstract I/O Declarations
# ============================================================================

@dataclass
class ResourceSpec:
    """
    Abstract specification of what resources a revision step needs to read/write.

    The actual paths are resolved by ProjectRepository based on the abstract
    resource identifiers (sample_id, layer name, metadata file type, etc.).
    """
    # Project Metadata
    samples: dict[str, SampleRef] = field(default_factory=dict)
    dimensions: dict[str, list[DimensionDef]] = field(default_factory=dict)  # layer -> dimensions
    compensations: dict[str, CompensationRef] = field(default_factory=dict)
    transformations: dict[str, TransformationRef] = field(default_factory=dict)
    batches: dict[str, BatchRef] = field(default_factory=dict)
    # Steps to rerun
    steps: list[StepRun] = field(default_factory=list)  # steps that need to be rerun

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {
                k: [dim.to_dict() for dim in v]
                for k, v in self.dimensions.items()
            },
            "compensations": {k: v.to_dict() for k, v in self.compensations.items()},
            "transformations": {k: v.to_dict() for k, v in self.transformations.items()},
            "batches": self.batches,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceSpec":
        steps_data = data.get("steps", [])
        steps = []
        for step_dict in steps_data:
            step = StepRun(
                id=step_dict["id"],
                step_type=step_dict["step_type"],
                config=step_dict.get("config", {}),
                inputs=step_dict.get("inputs", {}),
                sample_outputs=step_dict.get("sample_outputs", {}),
                batch_outputs=step_dict.get("batch_outputs", {}),
                project_updates=step_dict.get("project_updates", []),
                qc=EntityQCStatus.from_dict(step_dict["qc"]) if step_dict.get("qc") else None,
                status=step_dict.get("status", "pending"),
                created_at=step_dict.get("created_at", ""),
            )
            steps.append(step)

        return cls(
            samples={sid: SampleRef.from_record(ref) for sid, ref in data.get("samples", {}).items()},
            dimensions={
                k: [DimensionDef.from_dict(dim) for dim in v]
                for k, v in data.get("dimensions", {}).items()
            },
            compensations={k: CompensationRef.from_record(v) for k, v in data.get("compensations", {}).items()},
            transformations={k: TransformationRef.from_record(v) for k, v in data.get("transformations", {}).items()},
            batches=data.get("batches", []),
            steps=steps,
        )


@dataclass
class RevisionSession:
    """
    Tracks the state of a revision session for a specific step.

    A revision session creates an isolated workspace where users can iteratively
    refine step results until satisfied, then commit changes back to main project.
    """
    id: str                              # revision session id (e.g., "rev_001")
    parent_step_id: str                  # The step being revised (e.g., "step_0003")
    parent_step_type: str                # Step type (e.g., "compensate")
    state: str                           # "active" | "ready_to_commit" | "committed" | "cancelled"
    created_at: str
    updated_at: str

    # Handler state
    handler_state: dict[str, Any] = field(default_factory=dict)  # internal state for the revision handler

    # User selections for what to revise
    target_samples: list[str] = field(default_factory=list)
    input_spec: dict[str, Any] = field(default_factory=dict)  # User's revision specification
    revision_history: list[dict[str, Any]] = field(default_factory=list)  # History of apply_revision calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_step_id": self.parent_step_id,
            "parent_step_type": self.parent_step_type,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "handler_state": self.handler_state,
            "target_samples": self.target_samples,
            "input_spec": self.input_spec,
            "revision_history": self.revision_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RevisionSession":
        return cls(
            id=data["id"],
            parent_step_id=data["parent_step_id"],
            parent_step_type=data["parent_step_type"],
            state=data["state"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            handler_state=data.get("handler_state", {}),
            target_samples=data.get("target_samples", []),
            input_spec=data.get("input_spec", {}),
            revision_history=data.get("revision_history", []),
        )