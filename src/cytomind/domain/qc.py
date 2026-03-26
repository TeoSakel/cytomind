from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Mapping, Iterable, Any, TYPE_CHECKING, cast


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
    SKIP = "SKIP"

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
        flags.discard(QCFlag.SKIP)
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
    id: dict[str, Any]                                                      # unique identifier for this test record
    test_type: str                                                          # to group related tests. Suggested format: "{entity}_{level}_{scope}" e.g. "compensation_sample_channel", "gating_strategy_step", etc.
    test_name: str                                                          # the name of the test within the QC evaluator
    targets: dict[str, Any] = field(default_factory=dict)                   # target identifiers (sample_id, compensation_id, strategy_id, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)                  # context (gate_id, channel, cutpoint, bounds, etc.)
    metrics: dict[str, Numeric] = field(default_factory=dict)               # measured values (proportion_passing, r_squared, p_neg, etc.)
    thresholds: dict[str, dict[str, tuple[Numeric | None, Numeric | None]]] = field(default_factory=dict)
    # standardized threshold parameters: {metric_name: {"warn": (low, high), "severe": (low, high)}}
    status: str = "PENDING"                                                 # "PENDING" (execution) → "PASS"/"WARN"/"SEVERE"/"FAIL"/"SKIP" (evaluator)
    message: str = ""                                                       # human-readable summary (set by evaluator)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def iter_records(self, metadata: bool = True, thresholds: bool = True) -> Iterable[dict[str, Any]]:
        """Flatten to a single dict for tabular representation, combining id, metadata, metrics, and status."""
        base = self.id.copy()  # start with id fields
        if metadata:
            base.update(self.metadata)
        base["status"] = self.status

        for metric_name, metric_value in self.metrics.items():
            record = base.copy()
            record["metric"] = metric_name
            record["value"] = metric_value
            if thresholds:
                low, high = self.thresholds.get(metric_name, {}).get("warn", (None, None))
                if low is not None: record["warn_low"] = low
                if high is not None: record["warn_high"] = high
                low, high = self.thresholds.get(metric_name, {}).get("severe", (None, None))
                if low is not None: record["severe_low"] = low
                if high is not None: record["severe_high"] = high
            yield record

    @property
    def key(self) -> tuple:
        """Unique key for this test record, derived from id dict."""
        # return tuple(sorted(self.id.values()))
        return tuple(self.id.values())

    @property
    def flag(self) -> QCFlag:
        """Get QCFlag enum from status string."""
        if self.status == "PENDING":
            return QCFlag.WARN # Treat PENDING as WARN for flag combination purposes

        if self.status == "SKIP":
            return QCFlag.PASS # Treat SKIP as PASS for flag combination purposes

        return QCFlag.from_str(self.status)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCTestRecord":
        return cls(
            test_type=data["test_type"],
            test_name=data["test_name"],
            id=data["id"],
            targets=data.get("targets", {}),
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
    # reasons: mapping reason_code -> {"messages": set[str], "tests": set[tuple]}
    reasons: dict[str, dict[str, set[str | tuple]]] = field(default_factory=dict)
    tests: dict[tuple, QCTestRecord] = field(default_factory=dict)

    def add_test(self, test: QCTestRecord) -> None:
        self.tests[test.key] = test
        self.flag = QCFlag.combine([self.flag, QCFlag.from_str(test.status)])

    def add_reason(
        self,
        code: str,
        message: str | Iterable[str] = [],
        tests: Iterable[QCTestRecord] = [],
    ) -> None:
        """Add a reason code with optional message and/or test record to this step's QC."""
        if code not in self.reasons:
            self.reasons[code] = {"messages": set(), "tests": set()}

        if isinstance(message, str):
            if message:  # Only add non-empty messages
                self.reasons[code]["messages"].add(message)
        else:
            self.reasons[code]["messages"].update(message)

        for test in tests:
            self.reasons[code]["tests"].add(test.key)
            self.add_test(test)

    def to_dict(self) -> dict[str, Any]:
        # serialize tests to dicts
        serialized_reasons: dict[str, dict[str, list[str | tuple]]] = {}
        for code, detail in self.reasons.items():
            serialized_reasons[code] = {
                "messages": list(detail.get("messages", set())),
                "tests": list(detail.get("tests", set())),
            }
        return {
            "flag": self.flag.value,
            "reasons": serialized_reasons,
            "tests": [test.to_dict() for test in self.tests.values()],
        }

    def merge(self, other: "QCStepStatus") -> None:
        self.flag = QCFlag.combine([self.flag, other.flag])
        self.tests.update(other.tests)
        for code, payload in other.reasons.items():
            messages = payload["messages"]
            tests = payload["tests"]
            if code in self.reasons:
                self.reasons[code]["messages"].update(messages)
                self.reasons[code]["tests"].update(tests)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QCStepStatus":
        reasons = data.get("reasons", {}) or {}
        tests = {}
        for test in data.get("tests", []):
            test_record = QCTestRecord.from_dict(test)
            tests[test_record.key] = test_record
        for reason_detail in reasons.values():
            reason_detail["messages"] = set(reason_detail.get("messages", []))
            reason_detail["tests"] = set(map(tuple, reason_detail.get("tests", [])))
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

    @property
    def flag(self) -> QCFlag:
        """Combined flag from all steps."""
        flags = [step.flag for step in self.steps.values()]
        return QCFlag.combine(flags) if flags else QCFlag.PASS

    def get_step(self, step_name: str) -> QCStepStatus:
        """Get or create a step status."""
        if step_name not in self.steps:
            self.steps[step_name] = QCStepStatus()
        return self.steps[step_name]

    def update(self, other: QCRunStatus, merge: bool = False) -> None:
        if not merge:
            existing = [step_name for step_name in other.steps if step_name in self.steps]
            if existing:
                raise ValueError(f"Steps {existing} already exist. Use get_step() to modify existing steps or ensure no overlap when updating.")
        for step_name, step_status in other.steps.items():
            if step_name in self.steps:
                self.steps[step_name].merge(step_status)
            else:
                self.steps[step_name] = step_status

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

    def __bool__(self):
        return bool(self.steps)

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
                messages = list(detail.get("messages", []))
                messages = cast(list[str], messages)  # avoid type confusion with tests which are tuples
                out.update(messages)
        return list(out)

    @property
    def all_tests(self) -> dict[tuple, QCTestRecord]:
        tests = {}
        for step in self.steps.values():
            tests.update(step.tests)
        return tests

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
    artifacts: dict[str, Any] = field(default_factory=dict)  # QC artifacts (plots, tables, etc.)
    generated_at: str = ""
    updated_at: str = ""

    def get_sample_steps(self, sample_id: str) -> QCRunStatus:
        """Get or create QCRunStatus for a sample."""
        if sample_id not in self.sample_qc:
            self.sample_qc[sample_id] = QCRunStatus()
        return self.sample_qc[sample_id]

    def iter_sample_tests(self, sample_id: str, filter_by: Mapping[str, Any] | None = None) -> Iterable[tuple[tuple[str, tuple], QCTestRecord]]:
        """Iterate over all tests for a specific sample and all steps, optionally filtered by test id fields."""
        filter_by = filter_by or {}

        def keep_test(test: QCTestRecord) -> bool:

            container = (list, set, tuple)
            for k, v in filter_by.items():
                test_v = test.id.get(k)

                if isinstance(v, container) and isinstance(test_v, container):
                    # If either is a set, compare as sets (order-insensitive)
                    if isinstance(v, set) or isinstance(test_v, set):
                        if set(test_v) != set(v):
                            return False
                    else:
                        if test_v != v:
                            return False

                # If filter value is a container, test membership of test_v in v
                elif isinstance(v, container):
                    if test_v not in v:
                        return False

                # Otherwise simple equality
                else:
                    if test_v != v:
                        return False

            return True

        for step_name, step in self.sample_qc[sample_id].steps.items():
            for test_key, test in step.tests.items():
                if keep_test(test):
                    yield (step_name, test_key), test

        for step_name, step in self.batch_qc.steps.items():
             for test_key, test in step.tests.items():
                batch_sample_id = test.id.get("sample_id")
                if batch_sample_id and batch_sample_id != sample_id:
                    continue
                if keep_test(test):
                    yield (step_name, test_key), test

    def update_sample_steps(self, sample_id: str, steps: QCRunStatus | Iterable[tuple[str, QCStepStatus]], merge: bool = False) -> None:
        sample_steps = self.get_sample_steps(sample_id)
        if not isinstance(steps, QCRunStatus):
            steps = QCRunStatus(OrderedDict(steps))
        sample_steps.update(steps, merge=merge)

    def update_batch_steps(self, steps: QCRunStatus | Iterable[tuple[str, QCStepStatus]], merge: bool = False) -> None:
        if not isinstance(steps, QCRunStatus):
            steps = QCRunStatus(OrderedDict(steps))
        self.batch_qc.update(steps, merge=merge)

    @property
    def overall_flag(self) -> QCFlag:
        # Combine across all samples and steps
        all_flags = [
            step.flag
            for qc_run in self.sample_qc.values()
            for step in qc_run.steps.values()
        ]
        if self.batch_qc.steps:
            all_flags.append(self.batch_qc.overall_flag)
        return QCFlag.combine(all_flags) if all_flags else QCFlag.PASS

    @property
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
    def all_messages(self) -> list[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for qc_run in [self.batch_qc] + list(self.sample_qc.values()):
            for step in qc_run.steps.values():
                for code, detail in step.reasons.items():
                    messages = list(detail.get("messages", []))
                    messages = cast(list[str], messages)  # avoid type confusion with tests which are tuples
                    out.update((code, message) for message in messages)
        return list(out)

    def group_samples_by_flag(self) -> dict[str, list[str]]:
        """Group sample IDs by their overall flag."""
        groups: dict[str, list[str]] = {}
        for sample_id, qc_run in self.sample_qc.items():
            flag = qc_run.overall_flag.value
            groups.setdefault(flag, []).append(sample_id)
        return groups

    def group_samples_by_reason_code(self) -> dict[str, list[str]]:
        """Group sample IDs by reason codes (a sample may appear in multiple groups if it has multiple reason codes)."""
        groups: dict[str, list[str]] = {}
        for sample_id, qc_run in self.sample_qc.items():
            for step in qc_run.steps.values():
                for reason_code in step.reasons.keys():
                    groups.setdefault(reason_code, []).append(sample_id)
        return groups

    def group_samples_by_message(self) -> dict[tuple[str, str], list[str]]:
        """Group sample IDs by code x message (a sample may appear in multiple groups if it has multiple messages)."""
        groups: dict[tuple[str, str], list[str]] = {}
        for sample_id, qc_run in self.sample_qc.items():
            for step in qc_run.steps.values():
                for code, detail in step.reasons.items():
                    for message in detail.get("messages", []):
                        message = cast(str, message)  # avoid type confusion with tests which are tuples
                        groups.setdefault((code, message), []).append(sample_id)
        return groups

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
            "artifacts": self.artifacts,
            "generated_at": self.generated_at,
            "updated_at": self.updated_at,
        }

        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityQCStatus":
        """Deserialize from dict."""

        # Handle batch qc
        batch_qc_data = data.get("batch_qc", {})
        batch_qc = QCRunStatus.from_dict(batch_qc_data) if batch_qc_data else QCRunStatus()

        # Handle per-sample steps (new format)
        per_sample_raw = data.get("sample_qc", {})
        per_sample_steps: dict[str, QCRunStatus] = {}
        for sample_id, sample_steps_dict in per_sample_raw.items():
            per_sample_steps[sample_id] = QCRunStatus.from_dict(sample_steps_dict)

        return cls(
            entity_type=data["entity_type"],
            entity_id=data["entity_id"],
            context=data.get("context", {}),
            summary=data.get("summary", {}),
            artifacts=data.get("artifacts", {}),
            batch_qc=batch_qc,
            sample_qc=per_sample_steps,
            generated_at=data.get("generated_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def __repr__(self) -> str:
        return f"EntityQCStatus(entity_type={self.entity_type!r}, entity_id={self.entity_id!r}, overall_flag={self.overall_flag.value!r})"

    def iter_tests(
        self,
        filter_by: Mapping[str, Any] | None = None,
        sample_ids: Iterable[str] | None = None,
        with_batch_tests: bool | None = None,
    ) -> Iterable[QCTestRecord]:
        """
        Iterate over all tests for this entity, optionally filtered by test id fields and/or sample IDs.

        filter_by: dict of {id_key: id_value} to filter tests by their id fields (e.g. {"test_type": "compensation_sample_channel"})
        sample_ids: optional list of sample IDs to include (if None, include all samples)
        """

        if with_batch_tests is None:
            with_batch_tests = sample_ids is None  # include batch tests if not filtering by sample

        sample_ids = set(sample_ids if sample_ids is not None else self.sample_qc.keys())
        for sample_id in sample_ids:
            for _, test in self.iter_sample_tests(sample_id, filter_by=filter_by):
                if with_batch_tests or test.id.get("sample_id") in sample_ids:
                    yield test

    def iter_test_records(
        self,
        filter_by: Mapping[str, Any] | None = None,
        sample_ids: Iterable[str] | None = None,
        with_thresholds: bool = True,
        with_metadata: bool = True,
        with_batch_tests: bool | None = None,
    ) -> Iterable[dict[str, Any]]:
        """
        Iterate over all test records for this entity, optionally filtered by test id fields and/or sample IDs.

        filter_by: dict of {id_key: id_value} to filter tests by their id fields (e.g. {"test_type": "compensation_sample_channel"})
        sample_ids: optional list of sample IDs to include (if None, include all samples)
        """

        for test in self.iter_tests(filter_by=filter_by, sample_ids=sample_ids, with_batch_tests=with_batch_tests):
            yield from test.iter_records(metadata=with_metadata, thresholds=with_thresholds)