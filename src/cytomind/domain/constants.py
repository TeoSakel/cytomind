"""
Domain-level constants used across flow cytometry modules.

GatingML version can be overridden via the environment variable
`CYTOMIND_GATINGML_VERSION`, otherwise defaults to "2.0".
"""

import os
from typing import Final
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

# Single source of truth for GatingML version used in exports/definitions
GML_VERSION: Final[str] = os.getenv("CYTOMIND_GATINGML_VERSION", "2.0")

PathLike = Path | str
MaskLike = NDArray[np.bool_] | NDArray[np.int_] | slice