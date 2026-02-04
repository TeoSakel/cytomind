from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Mapping, Iterable, Sequence, Any, TYPE_CHECKING

if TYPE_CHECKING:
    Numeric = int | float
else:
    Numeric = object

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

