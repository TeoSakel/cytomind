"""
Flow cytometry domain classes and references.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Hashable
from pathlib import Path
import hashlib

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, read_csv
from flowutils.compensate import get_spill

from cytomind.utils import spillover_df_to_string

@dataclass
class CompensationRef:
    id: str       # unique id
    name: str | None = None    # human-readable name
    source: str | None = None  # fcs | user | computation step
    path: str | None = None  # path to csv file with spillover matrix
    batch: list[str] = field(hash=False, default_factory=list)   # list of fcs files specifying this compensation
    _spill: DataFrame | None = field(repr=False, hash=False, default=None)

    @classmethod
    def generate_id(cls, matrix: str | DataFrame) -> str:
        """Generate a unique id for the compensation matrix based on its content."""
        if isinstance(matrix, str):
            try:
                # Try to parse as spillover matrix string
                _ = get_spill(matrix)
                codec = matrix.encode()
            except Exception as e:
                # Fallback: treat as file path
                path = Path(matrix)
                if path.exists():
                    df = read_csv(path, index_col=False)
                    codec = spillover_df_to_string(df).encode()
                else:
                    raise ValueError("Invalid spillover matrix string or file path.") from e
        elif isinstance(matrix, DataFrame):
            codec = spillover_df_to_string(matrix).encode()
        else:
            raise TypeError("matrix must be a string or a pandas DataFrame")

        # Create a hash based on the matrix values
        return "comp_" + hashlib.md5(codec).hexdigest()[:8]

    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        name: str | None = None,
        source: str | None = None,
        path: str | None = None,
        batch: list[str] = []
    ) -> "CompensationRef":
        comp_id = cls.generate_id(df)
        if path is not None:
            csv_path = Path(path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

        return CompensationRef(
            id=comp_id,
            name=name,
            source=source,
            path=path,
            batch=batch,
            _spill=df,
        )

    def copy(self) -> "CompensationRef":
        return CompensationRef(
            id=self.id,
            name=self.name,
            source=self.source,
            path=self.path,
            batch=self.batch.copy(),
            _spill=self._spill.copy() if isinstance(self._spill, DataFrame) else self._spill,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CompensationRef":
        ref = CompensationRef(
            id=record["id"],
            name=record["name"],
            source=record["source"],
            path=record.get("path", None),
            batch=record.get("batch", []),
        )

        return ref

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "batch": self.batch,
            "path": self.path,
        }

    @property
    def spill(self) -> DataFrame:
        if isinstance(self._spill, DataFrame):
            if self._spill.index is None:
                self._spill.index = self._spill.columns
            return self._spill

        if self.path is not None:
            csv_path = Path(self.path)
            if csv_path.exists():
                df = read_csv(csv_path, index_col=False)
                df.index = df.columns
                self._spill = df
            else:
                raise FileNotFoundError(f"Compensation spill file not found: {self.path}")
        else:
            raise ValueError("Compensation spill matrix is not loaded and no path is provided.")

        return self._spill

    @spill.setter
    def spill(self, value: DataFrame) -> None:
        self._spill = value
        if self.path is not None:
            value.to_csv(self.path, index=False)

    @property
    def matrix(self) -> NDArray[np.float64]:
        return self.spill.values

    @property
    def detectors(self) -> list[str]:
        return self.spill.columns.tolist()


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
    def from_record(cls, record: Mapping[Hashable, Any]) -> "ChannelRef":
        return ChannelRef(
            idx=record["idx"],
            pnn=record["pnn"],
            pns=record.get("pns", None),
            pne=tuple(record["pne"]) if record.get("pne", None) is not None else None,
            png=record.get("png", None),
            pnr=record.get("pnr", None),
            metric=record.get("metric", None),
            type=record["type"],
        )

@dataclass
class TransformationRef:
    id: str      # key to transformation registry
    type: str    # human readable "logicle", "asinh", ...
    params: dict[str, Any] = field(default_factory=dict) # parameters for the transform

    def copy(self) -> "TransformationRef":
        return TransformationRef(
            id=self.id,
            type=self.type,
            params=self.params.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TransformationRef":
        return TransformationRef(
            id=record["id"],
            type=record["type"],
            params=record.get("params", {}),
        )

@dataclass
class DimensionDef:
    id: str                        # unique id for the dimension = var.index
    channel_id: list[str]          # link to ChannelRefs
    marker: str | None             # human-readable marker name
    type: str                      # "fluorescence", "scatter", "time", ...
    use_comp: bool                 # True/False
    transform_id: str = "identity" # link to TransformationRef
    idx: int | None = None         # resolved index in data var

    def copy(self) -> "DimensionDef":
        return DimensionDef(
            id=self.id,
            channel_id=self.channel_id.copy(),
            marker=self.marker,
            type=self.type,
            use_comp=self.use_comp,
            transform_id=self.transform_id,
            idx=self.idx,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DimensionDef":
        return DimensionDef(
            id=data["id"],
            channel_id=data["channel_id"],
            marker=data.get("marker", None),
            type=data["type"],
            use_comp=data["use_comp"],
            transform_id=data.get("transform_id", "identity"),
            idx=data.get("idx", None),
        )


    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "channel_id": ",".join(self.channel_id),
            "marker": self.marker,
            "type": self.type,
            "use_comp": self.use_comp,
            "transform_id": self.transform_id,
            "idx": self.idx,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DimensionDef":
        return DimensionDef(
            id=record["id"],
            channel_id=record["channel_id"].split(","),
            marker=record.get("marker"),
            type=record["type"],
            use_comp=record["use_comp"],
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
            self.channel_id == other.channel_id and
            self.transform_id == other.transform_id and
            self.marker == other.marker and  # probably not needed
            self.use_comp == other.use_comp
        )