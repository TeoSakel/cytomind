from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("CytoMind")
except PackageNotFoundError:
    __version__ = "(local)"

del PackageNotFoundError
del version

from cytomind.domain.flow import CompensationRef, DimensionDef, ChannelRef, TransformationRef
from cytomind.domain.pipeline import Project, SampleRef, StepRun, QCRunStatus
from cytomind.infra.repo import ProjectRepository
from cytomind.infra.pipeline import InteractivePipeline
from cytomind.steps import StepRegistry

__all__ = [
    "ProjectRepository",
    "InteractivePipeline",
    "StepRegistry",
    "Project",
    "SampleRef",
    "StepRun",
    "QCRunStatus",
    "CompensationRef",
    "DimensionDef",
    "ChannelRef",
    "TransformationRef",
]