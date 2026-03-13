"""
Flow cytometry domain classes and references.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, TYPE_CHECKING
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from flowutils.compensate import get_spill

from cytomind.utils import spillover_df_to_string

if TYPE_CHECKING:
    from .constants import PathLike
    from numpy.typing import NDArray
    FloatArray = NDArray[np.float64]
else:
    PathLike = object
    FloatArray = object

@dataclass
class CompensationRef:
    id: str       # unique id
    _spill: dict[str, list[float]]
    name: str | None = None    # human-readable name
    source: str | None = None  # fcs | user | computation step
    batch: list[str] = field(hash=False, default_factory=list)   # list of fcs files specifying this compensation

    @classmethod
    def generate_id(cls, matrix: str | pd.DataFrame) -> str:
        """
        Generate a unique id for the compensation matrix based on its content.
        Also validate that the matrix is a proper spillover matrix.
        """
        if isinstance(matrix, str):
            try:
                # Try to parse as spillover matrix string
                _ = get_spill(matrix)
                codec = matrix.encode()
            except Exception as e:
                # Fallback: treat as file path
                path = Path(matrix)
                if path.exists():
                    df = pd.read_csv(path, index_col=False)
                    codec = spillover_df_to_string(df).encode()
                else:
                    raise ValueError("Invalid spillover matrix string or file path.") from e
        elif isinstance(matrix, pd.DataFrame):
            codec = spillover_df_to_string(matrix).encode()
        else:
            raise TypeError("matrix must be a string or a pandas DataFrame")

        # Create a hash based on the matrix values
        return "comp_" + hashlib.md5(codec).hexdigest()[:8]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str | None = None,
        source: str | None = None,
        batch: list[str] = []
    ) -> "CompensationRef":
        comp_id = cls.generate_id(df)  # also performs validation

        return CompensationRef(
            id=comp_id,
            name=name,
            source=source,
            batch=batch,
            _spill=df.to_dict(orient="list"), # pyright: ignore[reportArgumentType]
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompensationRef":
        return CompensationRef(
            id=data["id"],
            _spill=data.get("_spill", None),
            name=data.get("name", None),
            source=data.get("source", None),
            batch=data.get("batch", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def spill(self) -> pd.DataFrame:
        df = pd.DataFrame(self._spill)
        df.index = df.columns
        return df

    @property
    def matrix(self) -> FloatArray:
        return self.spill.values

    @property
    def detectors(self) -> list[str]:
        return list(self._spill.keys())


@dataclass(frozen=True)
class ChannelRef:
    idx: int        # index in data var
    pnn: str        # laser name
    pns: str | None # marker name
    pne: tuple[float, float] | None  # amplification type (log, linear)
    png: float | None                # ampliefier gain
    pnr: float | None                # amplifier range
    metric: str | None    # "area", "height", "width"
    type: str             # "fluorescence", "scatter", "time", ...

    def copy(self) -> "ChannelRef":
        return ChannelRef(
            idx=self.idx,
            pnn=self.pnn,
            pns=self.pns,
            pne=self.pne,
            png=self.png,
            pnr=self.pnr,
            metric=self.metric,
            type=self.type,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "pnn": self.pnn,
            "pns": self.pns,
            "pne_decades": self.pne[0] if self.pne is not None else None,
            "pne_offset": self.pne[1] if self.pne is not None else None,
            "png": self.png,
            "pnr": self.pnr,
            "metric": self.metric,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelRef":
        return ChannelRef(
            idx=data["idx"],
            pnn=data["pnn"],
            pns=data.get("pns", None),
            pne=tuple(data["pne"]) if data.get("pne", None) is not None else None,
            png=data.get("png", None),
            pnr=data.get("pnr", None),
            metric=data.get("metric", None),
            type=data["type"],
        )


@dataclass
class DimensionDef:
    id: str                        # unique id for the dimension = var.index
    source_dims: list[str]         # upstream dimensions used as transform inputs
    marker: str | None             # human-readable marker name
    type: str                      # "fluorescence", "scatter", "time", ...
    source_layer: str | None = None  # source layer containing source_dims
    transform_id: str = "identity"  # link to TransformDef
    idx: int | None = None         # resolved index in data var

    def copy(self) -> "DimensionDef":
        return DimensionDef(
            id=self.id,
            source_dims=self.source_dims.copy(),
            marker=self.marker,
            type=self.type,
            source_layer=self.source_layer,
            transform_id=self.transform_id,
            idx=self.idx,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DimensionDef":
        return DimensionDef(
            id=data["id"],
            source_dims=list(data.get("source_dims", [])),
            marker=data.get("marker", None),
            type=data["type"],
            source_layer=data.get("source_layer", None),
            transform_id=data.get("transform_id", "identity"),
            idx=data.get("idx", None),
        )


    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_dims": ",".join(self.source_dims),
            "marker": self.marker,
            "type": self.type,
            "source_layer": self.source_layer,
            "transform_id": self.transform_id,
            "idx": self.idx,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DimensionDef":
        return DimensionDef(
            id=record["id"],
            source_dims=record["source_dims"].split(",") if record.get("source_dims") else [],
            marker=record.get("marker"),
            type=record["type"],
            source_layer=record.get("source_layer", None),
            transform_id=record.get("transform_id", "identity"),
            idx=record.get("idx", None),
        )

    # ---- comparison operations: order by idx, None goes last ----
    def _cmp_key(self) -> float:
        return float("inf") if self.idx is None else float(self.idx)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DimensionDef):
            raise NotImplementedError
        return self._cmp_key() < other._cmp_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimensionDef):
            raise NotImplementedError
        return (
            self.source_dims == other.source_dims and
            self.transform_id == other.transform_id and
            self.marker == other.marker and  # probably not needed
            self.source_layer == other.source_layer
        )