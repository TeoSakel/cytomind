"""
Domain-level constants used across flow cytometry modules.

GatingML version can be overridden via the environment variable
`CYTOMIND_GATINGML_VERSION`, otherwise defaults to "2.0".
"""

import os
from typing import Any, Final, Iterable, Protocol, Mapping, runtime_checkable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

# Single source of truth for GatingML version used in exports/definitions
GML_VERSION: Final[str] = os.getenv("CYTOMIND_GATINGML_VERSION", "2.0")

PathLike = Path | str
BooleanArray = NDArray[np.bool_]
FloatArray = NDArray[np.floating]
MaskLike = BooleanArray | NDArray[np.int_] | slice

@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> dict:
        """Convert the object to a dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Serializable":
        """Create an instance of the class from a dictionary representation."""
        ...

ProjectMetadata = Serializable | list[Serializable] | dict[str, Serializable] | dict[str, list[Serializable]]