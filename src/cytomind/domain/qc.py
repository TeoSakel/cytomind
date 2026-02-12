from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Hashable, Mapping, Iterable, Sequence, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import floating
    Numeric = int | float | floating[Any]
else:
    Numeric = object

class QCFlag(str, Enum):
    """Quality control FLAGS for gates and transformations."""
    PASS = "PASS"
    WARN = "WARN"
    SEVERE = "SEVERE"
    FAIL = "FAIL"

    @classmethod
    def from_str(cls, value: str) -> "QCFlag":
        try:
            return cls(value.upper())
        except ValueError as e:
            raise ValueError(f"Invalid QCFlag: {value!r}") from e

    @staticmethod
    def combine(flags: Iterable["QCFlag"]) -> "QCFlag":
        """Combine multiple flags into a single overall flag (FAIL > SEVERE > WARN > PASS)."""
        flags = set(flags)
        if not flags:
            return QCFlag.PASS
        if QCFlag.FAIL in flags:
            return QCFlag.FAIL
        if QCFlag.SEVERE in flags:
            return QCFlag.SEVERE
        if QCFlag.WARN in flags:
            return QCFlag.WARN
        return QCFlag.PASS

@dataclass
class QCTestRecord:
    """
    Structured record of a single QC test with metrics, thresholds, and results.

    Best Hybrid QC Pattern:
    - During execution: Steps emit test_type, test_name, metadata, metrics with status="PENDING"
    - After execution: QC Evaluators apply thresholds, assign status (PASS/WARN/FAIL), add message

    This allows:
    - Steps to capture ephemeral metrics without QC logic
    - QC criteria to be adjusted and re-evaluated without re-running pipeline
    - Centralized threshold management in evaluators
    """
    test_type: str                                                          # "gate_fit", "compensation_channel", etc.
    test_name: str                                                          # unique name for this test instance
    metadata: dict[str, Any] = field(default_factory=dict)                  # context (gate_id, channel, cutpoint, bounds, etc.)
    metrics: dict[str, Numeric] = field(default_factory=dict)               # measured values (proportion_passing, r_squared, p_neg, etc.)
    thresholds: dict[str, Sequence[Numeric]] = field(default_factory=dict)  # threshold parameters (set by evaluator)
    status: str = "PENDING"                                                 # "PENDING" (execution) → "PASS"/"WARN"/"SEVERE"/"FAIL"/"SKIP" (evaluator)
    message: str = ""                                                       # human-readable summary (set by evaluator)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def flag(self) -> QCFlag:
        """Get QCFlag enum from status string."""
        if self.status == "PENDING":
            return QCFlag.WARN # Treat PENDING as WARN for flag combination purposes

        if self.status == "SKIP":
            return QCFlag.PASS # Treat SKIP as PASS for flag combination purposes

        return QCFlag.from_str(self.status)

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
    """
    QC info for a single (conceptual) step.
    Ideally, every step should correspond to a single test, with one or few reasons to fail but we allow multiple for flexibility.
    """

    flag: QCFlag = QCFlag.PASS
    # reasons: mapping reason_code -> {"messages": list[str], "tests": list[QCTestRecord]}
    reasons: dict[str, dict[str, set]] = field(default_factory=dict)
    tests: dict[Hashable, QCTestRecord] = field(default_factory=dict)

    def add_test(self, key: Hashable, test: QCTestRecord) -> None:
        self.tests[key] = test
        self.flag = QCFlag.combine([self.flag, QCFlag.from_str(test.status)])

    def add_reason(
        self,
        code: str,
        message: str | Iterable[str] = [],
        test: Mapping[Hashable, QCTestRecord] = {},
    ) -> None:
        """Add a reason code with optional message and/or test record to this step's QC."""
        if code not in self.reasons:
            self.reasons[code] = {"messages": set(), "tests": set()}

        if isinstance(message, str):
            if message:  # Only add non-empty messages
                self.reasons[code]["messages"].add(message)
        else:
            self.reasons[code]["messages"].update(message)

        for key, test_record in test.items():
            self.reasons[code]["tests"].add(key)
            self.add_test(key, test_record)

    def to_dict(self) -> dict[str, Any]:
        # serialize tests to dicts
        serialized_reasons: dict[str, dict] = {}
        for code, detail in self.reasons.items():
            serialized_reasons[code] = {
                "messages": list(detail.get("messages", set())),
                "tests": list(detail.get("tests", set())),
            }
        return {
            "flag": self.flag.value,
            "reasons": serialized_reasons,
            "tests": {key: test.to_dict() for key, test in self.tests.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCStepStatus":
        reasons = data.get("reasons", {}) or {}
        tests = {key: QCTestRecord.from_dict(test_dict) for key, test_dict in data.get("tests", {}).items()}
        for reason_detail in reasons.values():
            reason_detail["messages"] = set(reason_detail.get("messages", []))
            reason_detail["tests"] = set(reason_detail.get("tests", []))
            for test_key in reason_detail["tests"]:
                if test_key not in tests:
                    raise ValueError(f"Reason references test key {test_key} which is not in tests dict")
        test_flag = QCFlag.combine([test.flag for test in tests.values() if test.status != "PENDING" and test.status != "SKIP"])
        return cls(
            flag=QCFlag.from_str(data.get("flag", test_flag.value)),
            reasons=reasons,
            tests=tests,
        )

@dataclass
class QCRunStatus:
    """
    Simple wrapper around OrderedDict[str, QCStepStatus] for a single sample's QC steps.

    This is what steps work with - they call .get_step() to add/access their QC info.
    Used as the value type in EntityQCStatus.per_sample_steps.
    """
    steps: OrderedDict[str, QCStepStatus] = field(default_factory=OrderedDict)

    def get_step(self, step_name: str) -> QCStepStatus:
        """Get or create a step status."""
        if step_name not in self.steps:
            self.steps[step_name] = QCStepStatus()
        return self.steps[step_name]

    def add_step(self, step_name: str, status: QCStepStatus) -> None:
        """Add a step status."""
        self.steps[step_name] = status

    def __getitem__(self, key: str) -> QCStepStatus:
        """Access step by name."""
        return self.steps[key]

    def __setitem__(self, key: str, value: QCStepStatus) -> None:
        """Set step by name."""
        self.steps[key] = value

    def __iter__(self):
        """Iterate over step names and statuses."""
        return iter(self.steps.items())

    def __len__(self):
        """Get number of steps."""
        return len(self.steps)

    def __contains__(self, key: str) -> bool:
        """Check if step exists."""
        return key in self.steps

    @property
    def overall_flag(self) -> QCFlag:
        """Combined flag from all steps."""
        flags = [step.flag for step in self.steps.values()]
        return QCFlag.combine(flags) if flags else QCFlag.PASS

    @property
    def all_reason_codes(self) -> list[str]:
        codes = set()
        for step in self.steps.values():
            codes.update(step.reasons.keys())
        return list(codes)

    @property
    def all_messages(self) -> list[str]:
        out: set[str] = set()
        for step in self.steps.values():
            for detail in step.reasons.values():
                out.update(detail.get("messages", []))
        return list(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            step_name: step.to_dict()
            for step_name, step in self.steps.items()
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCRunStatus":
        """Deserialize from dict."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        steps = OrderedDict(
            (name, QCStepStatus.from_dict(step_dict))
            for name, step_dict in data.items()
        )
        return cls(steps=steps)


@dataclass
class EntityQCStatus:
    """
    QC report for an entity evaluation across multiple samples.

    Each entry in sample_qc is a QCRunStatus (wrapper around OrderedDict[str, QCStepStatus]).
    Samples are treated as keys for subsetting/aggregating, not entities themselves.

    Entity types include: "step", "compensation", "gating_strategy", etc.
    """
    entity_type: str  # "step", "compensation", "gating_strategy", etc.
    entity_id: str    # step_run_id, comp_id, strategy_id, etc.
    context: dict[str, Any] = field(default_factory=dict)  # evaluation context
    batch_qc: QCRunStatus = field(default_factory=QCRunStatus)  # optional batch-level QC
    sample_qc: dict[str, QCRunStatus] = field(default_factory=dict)  # sample_id -> QCRunStatus
    summary: dict[str, Any] = field(default_factory=dict)  # aggregated metrics
    generated_at: str = ""

    def get_sample_steps(self, sample_id: str) -> QCRunStatus:
        """Get or create QCRunStatus for a sample."""
        if sample_id not in self.sample_qc:
            self.sample_qc[sample_id] = QCRunStatus()
        return self.sample_qc[sample_id]

    def iter_sample_tests(self, sample_id: str) -> Iterable[tuple[str, Hashable, QCTestRecord]]:
        """Iterate over all tests for a specific sample and all steps."""
        for step_name, step in self.sample_qc[sample_id].steps.items():
            for test_key, test in step.tests.items():
                yield step_name, test_key, test

    @property
    def overall_flag(self) -> QCFlag:
        """Combined flag from summary or computed from all samples/steps."""
        if "overall_flag" in self.summary:
            return QCFlag.from_str(self.summary["overall_flag"])
        # Combine across all samples and steps
        all_flags = [
            step.flag
            for qc_run in self.sample_qc.values()
            for step in qc_run.steps.values()
        ]
        if self.batch_qc.steps:
            all_flags.append(self.batch_qc.overall_flag)
        return QCFlag.combine(all_flags) if all_flags else QCFlag.PASS

    def sample_flags(self) -> dict[str, QCFlag]:
        """Get overall flag per sample."""
        return {
            sample_id: qc_run.overall_flag
            for sample_id, qc_run in self.sample_qc.items()
        }

    @property
    def all_reason_codes(self) -> list[str]:
        codes = set()
        for qc_run in [self.batch_qc] + list(self.sample_qc.values()):
            for step in qc_run.steps.values():
                codes.update(step.reasons.keys())
        return list(codes)

    @property
    def all_messages(self) -> list[str]:
        out: set[str] = set()
        for qc_run in [self.batch_qc] + list(self.sample_qc.values()):
            for step in qc_run.steps.values():
                for detail in step.reasons.values():
                    out.update(detail.get("messages", []))
        return list(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        # Serialize per-sample steps
        serialized_per_sample: dict[str, dict[str, Any]] = {}
        for sample_id, qc_run in self.sample_qc.items():
            serialized_per_sample[sample_id] = qc_run.to_dict()

        result = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "context": self.context,
            "batch_qc": self.batch_qc.to_dict(),
            "sample_qc": serialized_per_sample,
            "summary": self.summary,
            "generated_at": self.generated_at,
        }

        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityQCStatus":
        """Deserialize from dict."""
        status = data.get("status")
        if status is None:
            status = {
                "batch_qc": data.get("batch_qc", {}),
                "sample_qc": data.get("sample_qc", {}),
                "steps": data.get("steps", {}),
            }
        status = status or {}

        # Handle batch qc
        batch_qc_data = status.get("batch_qc", {})
        batch_qc = QCRunStatus.from_dict(batch_qc_data) if batch_qc_data else QCRunStatus()

        # Handle per-sample steps (new format)
        per_sample_raw = status.get("sample_qc", {})
        per_sample_steps: dict[str, QCRunStatus] = {}
        for sample_id, sample_steps_dict in per_sample_raw.items():
            per_sample_steps[sample_id] = QCRunStatus.from_dict(sample_steps_dict)

        # Handle legacy flat steps format
        legacy_steps_raw = status.get("steps", {})
        if legacy_steps_raw and not per_sample_raw:
            # Legacy single-sample format: convert to per-sample
            sample_id = data.get("sample_id", "")
            if sample_id:
                per_sample_steps[sample_id] = QCRunStatus.from_dict(legacy_steps_raw)

        return cls(
            entity_type=data.get("entity_type", ""),
            entity_id=data.get("entity_id", ""),
            context=data.get("context", {}),
            summary=data.get("summary", {}),
            generated_at=data.get("generated_at", ""),
            batch_qc=batch_qc,
            sample_qc=per_sample_steps,
        )
