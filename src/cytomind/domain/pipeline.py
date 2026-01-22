"""
Structures to hold metadata required for the CytoMind pipeline management.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Iterable, Sequence, Any, TYPE_CHECKING

from anndata import AnnData
from pandas import DataFrame

from cytomind.domain.flow import *

if TYPE_CHECKING:
    Numeric = int | float

@dataclass
class Project:
    id: str
    panel: list[ChannelRef] = field(default_factory=list)  # list of all channels in the panel
    samples: dict[str, SampleRef] = field(default_factory=dict)
    batches: dict[str, BatchRef] = field(default_factory=dict)
    dimensions: dict[str, list[DimensionDef]] = field(default_factory=dict)  # layer -> dimensions
    compensations: dict[str, CompensationRef] = field(default_factory=dict)
    transformations: dict[str, TransformationRef] = field(default_factory=dict)

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
    id: str
    step_type: str          # "parse_fcs", "compensate", "transform", ...
    config: dict[str, Any]  # algorithmic knobs for this run
    inputs: dict[str, Any]  # ids: sample_ids, comp_id, etc.
    outputs: dict[str, Any] = field(default_factory=dict)
    per_sample_qc: dict[str, QCRunStatus] = field(default_factory=dict)
    qc_summary: dict[str, Any] = field(default_factory=dict)
    project_updates: set = field(default_factory=set)
    status: str = "pending"  # "pending" | "completed" | "failed"
    created_at: str = ""     # ISO string

    def to_dict(self) -> dict[str, Any]:
        base = asdict(self)
        # set is not JSON-serializable
        base["per_sample_qc"] = {
            sid: qc.to_dict() for sid, qc in self.per_sample_qc.items()
        }
        base["project_updates"] = list(self.project_updates)
        return base


class QCFlag(str, Enum):
    """Quality control FLAGS for gates and transformations."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    @classmethod
    def from_str(cls, value: str) -> "QCFlag":
        try:
            return cls(value.upper())
        except ValueError as e:
            raise ValueError(f"Invalid QCFlag: {value!r}") from e

    @staticmethod
    def combine(flags: Iterable["QCFlag"]) -> "QCFlag":
        """Combine multiple flags into a single overall flag (FAIL > WARN > PASS)."""
        flags = set(flags)
        if not flags:
            return QCFlag.PASS
        if QCFlag.FAIL in flags:
            return QCFlag.FAIL
        if QCFlag.WARN in flags:
            return QCFlag.WARN
        return QCFlag.PASS

@dataclass
class QCPlotRef:
    """
    Lightweight reference to a QC diagnostic plot (and optional data behind it).
    """

    id: str                       # unique within this QC context
    kind: str                     # "histogram", "scatter", "density2d", ...
    path: str                     # relative path to the plot file from project_root
    thumb_path: str | None = None  # optional smaller preview
    data_path: str | None = None   # optional path to data used to create the plot
    meta: dict[str, Any] = field(default_factory=dict)  # edits to default plot behavior

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QCPlotRef":
        return cls(
            id=d["id"],
            kind=d["kind"],
            path=d["path"],
            thumb_path=d.get("thumb_path"),
            data_path=d.get("data_path"),
            meta=dict(d.get("meta", {})),
        )

@dataclass
class QCTestRecord:
    """
    Structured record of a single QC test with metrics, thresholds, and results.

    This captures the full context of a QC test including what was measured,
    what thresholds were applied, and the outcome. Multiple test records can
    be linked to a single reason code in QCReasonDetail.
    """
    test_type: str                                                          # "compensation_channel", "compensation_pair", etc.
    test_name: str                                                          # unique name for this test instance
    metadata: dict[str, Any] = field(default_factory=dict)                  # context (channel names, indices, etc.)
    metrics: dict[str, Numeric] = field(default_factory=dict)               # measured values (p_neg, sigma, etc.)
    thresholds: dict[str, Sequence[Numeric]] = field(default_factory=dict)  # threshold parameters used
    status: str = "PENDING"                                                 # "PASS", "WARN", "FAIL", "SKIP"
    message: str = ""                                                       # human-readable summary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QCTestRecord":
        return cls(
            test_type=data["test_type"],
            test_name=data["test_name"],
            metadata=data.get("metadata", {}),
            metrics=data.get("metrics", {}),
            thresholds=data.get("thresholds", {}),
            status=data.get("status", "PENDING"),
            message=data.get("message", "")
        )

@dataclass
class QCStepStatus:
    """QC info for a single step instance (step_id & sample_id)."""

    flag: QCFlag = QCFlag.PASS
    # reasons: mapping reason_code -> {"messages": list[str], "tests": list[QCTestRecord]}
    reasons: dict[str, dict[str, list]] = field(default_factory=dict)
    plots: list["QCPlotRef"] = field(default_factory=list)

    def add_reason(
        self,
        code: str,
        message: str | Iterable[str] = [],
        test: QCTestRecord | Iterable[QCTestRecord] = [],
    ) -> None:
        """Add a reason code with optional message and/or test record to this step's QC."""
        if code not in self.reasons:
            self.reasons[code] = {"messages": [], "tests": []}

        if isinstance(message, str):
            if message:  # Only add non-empty messages
                self.reasons[code]["messages"].append(message)
        else:
            self.reasons[code]["messages"].extend(message)

        if isinstance(test, QCTestRecord):
            self.reasons[code]["tests"].append(test)
        else:
            self.reasons[code]["tests"].extend(test)

    def iter_tests(self) -> Iterable[tuple[str, QCTestRecord]]:
        """Iterate over all QCTestRecords in this step's QC."""
        for code, detail in self.reasons.items():
            for test in detail.get("tests", []):
                yield code, test

    def add_plot(self, plot: "QCPlotRef") -> None:
        self.plots.append(plot)

    def to_dict(self) -> dict[str, Any]:
        # serialize tests to dicts
        serialized_reasons: dict[str, dict] = {}
        for code, detail in self.reasons.items():
            serialized_reasons[code] = {
                "messages": list(detail.get("messages", [])),
                "tests": [t.to_dict() for t in detail.get("tests", [])],
            }
        return {
            "flag": self.flag.value,
            "reasons": serialized_reasons,
            "plots": [p.to_dict() for p in self.plots],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCStepStatus":
        reasons_data = data.get("reasons", {}) or {}
        reasons = {}
        for code, detail_data in reasons_data.items():
            # detail_data expected to be {"messages": [...], "tests": [{...}, ...]}
            msgs = list(detail_data.get("messages", []))
            tests = [QCTestRecord.from_dict(t) for t in detail_data.get("tests", [])]
            reasons[code] = {"messages": msgs, "tests": tests}

        return cls(
            flag=QCFlag.from_str(data.get("flag", "PASS")),
            reasons=reasons,
            plots=[QCPlotRef.from_dict(d) for d in data.get("plots", [])],
        )

@dataclass
class QCRunStatus:
    """
    QC report for a single (step_run_id, sample_id) pair.
    This is the in-memory representation of per-sample QC for a pipeline step.
    """

    sample_id: str    # which sample this QC is for
    step_run_id: str  # which step generated this QC
    steps: OrderedDict[str, QCStepStatus] = field(default_factory=OrderedDict)  # accumulated QC reports from previous steps

    # ---- step management ----------------------------------------------------

    def get_step(self, step_name: str) -> QCStepStatus:
        if step_name not in self.steps:
            self.steps[step_name] = QCStepStatus()
        return self.steps[step_name]

    def __getitem__(self, step_name: str) -> QCStepStatus:
        return self.steps[step_name]

    def __setitem__(self, step_name: str, step: QCStepStatus) -> None:
        if not isinstance(step, QCStepStatus):
            raise ValueError("step must be a QCStepStatus instance")
        self.steps[step_name] = step

    def __iter__(self):
        """Iterate over QC steps."""
        return iter(self.steps.items())

    def __next__(self):
        """Get next QC step."""
        yield from self.steps.items()

    def __len__(self):
        """Get number of QC steps."""
        return len(self.steps)

    # ---- aggregation & serialization ---------------------------------------

    @property
    def overall_flag(self) -> QCFlag:
        return QCFlag.combine([s.flag for s in self.steps.values()])

    @property
    def all_reason_codes(self) -> list[str]:
        return list(set(code for step in self.steps.values() for code in step.reasons))

    @property
    def all_messages(self) -> list[str]:
        out: set[str] = set()
        for step in self.steps.values():
            for detail in step.reasons.values():
                out.update(detail.get("messages", []))
        return list(out)

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "step_run_id": self.step_run_id,
            "status": {
                "overall": self.overall_flag.value,
                "reason_codes": self.all_reason_codes,
                "steps": {
                    step_name: step.to_dict()
                    for step_name, step in self.steps.items()
                },
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCRunStatus":
        status = data.get("status", {}) or {}
        steps_raw = status.get("steps", {}) or {}
        steps = OrderedDict()
        for name, step_dict in steps_raw.items():
            steps[name] = QCStepStatus.from_dict(step_dict)

        inst = cls(
            step_run_id=data.get("step_run_id", ""),
            sample_id=data.get("sample_id", ""),
            steps=steps,
        )
        return inst


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
                outputs=step_dict.get("outputs", {}),
                per_sample_qc=step_dict.get("per_sample_qc", {}),
                qc_summary=step_dict.get("qc_summary", {}),
                project_updates=set(step_dict.get("project_updates", [])),
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