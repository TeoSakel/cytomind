"""
Gate Node QC Evaluator.

Performs QC analysis on individual gates within their gating strategy context.
Evaluates event counts, ratios, fitting quality, and outlier detection for a single gate.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, Sequence, TYPE_CHECKING, cast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go

from cytomind.domain.qc import EntityQCStatus, QCFlag, QCStepStatus, QCTestRecord
from cytomind.gates import GateRegistry, Gate
from cytomind.utils import now_iso

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester, _ScalarOutlierTester

if TYPE_CHECKING:
    from cytomind.domain.constants import BooleanArray, FloatArray, PathLike
    from cytomind.domain.gates import GateNode
    from cytomind.infra.dataloader import UnifiedDataLoader
else:
    BooleanArray = object
    FloatArray = object
    PathLike = object
    GateNode = object
    UnifiedDataLoader = object


_GATE_SPACE_ARTIFACT_VERSION = 3


# ============================================================================
# Gate Space Geometry
# ============================================================================

class GateSpaceGeometry:
    """Owns gate-space basis generation, mask evaluation, pairwise distance state,
    centrality scoring, and artifact serialization/deserialization.

    Lifecycle
    ---------
    - :meth:`ensure` — primary entry point: load-or-build with update policy applied.
    - :meth:`build_from_entity` — full build from scratch.
    - :meth:`load` — restore from persisted artifacts.
    - :meth:`save` — persist artifacts and update ``entity_qc.artifacts`` metadata.
    - :meth:`centrality_by_sample` — compute per-sample centrality from the stored distance state.
    """

    artifact_key = "gate_space_geometry"
    _info_fields = ("entity_id", "gate_type", "layer", "dimensions", "lower_bounds", "upper_bounds", "resolution", "seed")
    # Maximum number of random evaluation points for dims >= 3
    _MAX_RANDOM_EVAL_POINTS = 2 ** 20

    # --------- Init and Helpers to allow reloading ---------

    @classmethod
    def create_artifact(
        cls,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        sample_ids: list[str],
        resolution: int = 256,
        seed: int | None = 42,
    ) -> "GateSpaceGeometry":

        layer=dataloader.load_data("project").layers[entity.layer]

        # `resolution` is interpreted as points-per-axis for up to 3D grids.
        # For dims > 3 we will randomly sample the space using `resolution**3`
        # points (capped by _MAX_RANDOM_EVAL_POINTS).
        if resolution <= 64:
            raise ValueError("Resolution (points per axis) must be greater than 64 for meaningful gate space geometry.")

        dims_by_id = {dim.id: dim for dim in layer}
        low: list[float] = []
        high: list[float] = []
        for dim_id in entity.dimensions:
            dim = dims_by_id.get(dim_id)
            if dim is None or dim.range_min is None or dim.range_max is None:
                raise ValueError(f"Dimension {dim_id} is missing or does not have valid range bounds in layer {entity.layer}.")
            low.append(float(dim.range_min))
            high.append(float(dim.range_max))

        n_dims = len(entity.dimensions)
        info = {
            "entity_id": entity.id,
            "gate_type": entity.gate_type,
            "layer": entity.layer,
            "dimensions": entity.dimensions,
            "lower_bounds": low,
            "upper_bounds": high,
            "resolution": resolution,
            "seed": seed if n_dims >= 3 else None,
        }

        missing = [field for field in cls._info_fields if field not in info]
        if missing:
            raise ValueError(f"Info dictionary is missing required fields: {missing}")
        gsg = cls(info=info, gate_node=entity, sample_ids=sample_ids)
        gsg.save(entity_qc=entity_qc, dataloader=dataloader)
        return gsg

    @classmethod
    def _validate_gate(cls, gate_node: GateNode, info: Mapping[str, Any]) -> None:
        if gate_node.id != info["entity_id"]:
            raise ValueError(f"Gate ID mismatch: expected {info['entity_id']}, got {gate_node.id}")
        if gate_node.gate_type != info["gate_type"]:
            raise ValueError(f"Gate type mismatch: expected {info['gate_type']}, got {gate_node.gate_type}")
        if gate_node.layer != info["layer"]:
            raise ValueError(f"Layer mismatch: expected {info['layer']}, got {gate_node.layer}")
        if tuple(gate_node.dimensions) != tuple(info["dimensions"]):
            raise ValueError(f"Dimensions mismatch: expected {info['dimensions']}, got {gate_node.dimensions}")

    def __init__(
        self,
        *,
        info: Mapping[str, Any],
        gate_node: GateNode,
        sample_ids: list[str] | None = None,
        # Arguments below are meant to be passed internally when loading from artifacts or patching, not by external callers.
        eval_points: ad.AnnData | FloatArray | None = None,
        masks: dict[str, BooleanArray] | None = None,
        dist_matrix: FloatArray | None = None,
    ) -> None:

        # Basis metadata
        self.gate_node = gate_node
        self.info = {field: info[field] for field in self._info_fields}
        # samples |> gates == masks
        self.sample_hashes: dict[str, str] = {}  # map sample_id to hash of gate params for that sample
        self.gates: dict[str, int] = {}  # map gate param hash to index in mask_matrix rows
        self._centrality: FloatArray | None = None
        self._distances: FloatArray | None = None

        # Coarse Grain Gate-Space
        self._make_eval_points() if eval_points is None else self._update_eval_points(eval_points)
        # Mask = Sample Geometry
        if masks is None:
            self._compute_masks()
            if dist_matrix is not None:
                warnings.warn("Provided dist_matrix will be ignored since masks were not provided and had to be computed.")
            self._compute_distances_full()
        else:
            original_keys = list(masks.keys())
            self._validate_masks(masks=masks)
            new_keys = list(self.masks.keys())
            # If mask keys unchanged and a distance matrix was provided, coerce it
            if original_keys == new_keys and dist_matrix is not None:
                self._condense_distances(dist_matrix=dist_matrix)
            else:
                # Keys changed. If a dist_matrix was provided for the original
                # keyset, inflate/deflate it to match the new keyset and only
                # compute distances for newly added masks. This keeps work
                # minimal compared to a full rebuild.
                if dist_matrix is not None and original_keys:
                    new_condensed = self._update_distance_matrix(dist_matrix, original_keys)
                    if new_condensed is not None:
                        self._distances = new_condensed
                        self._centrality = None
                    else:
                        self._compute_distances_full()
                else:
                    # No dist_matrix available for inflation -> full recompute
                    if dist_matrix is not None:
                        warnings.warn(
                            "Provided dist_matrix will be ignored since the set of mask keys has changed. "
                            "Distances will be recomputed based on the new masks."
                        )
                    self._compute_distances_full()

        if sample_ids is not None:
            self._add_missing_samples(sample_ids)

    def _add_missing_samples(self, sample_ids: list[str]) -> None:
        missing = list(set(sample_ids) - set(self.sample_hashes.keys()))
        if missing and "__batch__" not in self.sample_hashes:
            raise ValueError(f"Missing samples: {missing} with no default gate configuration found for gate {self.gate_node.id}.")
        batch_key = self.sample_hashes["__batch__"]
        for sid in missing:
            self.sample_hashes[sid] = batch_key

    def _make_eval_points(self):
        dims = len(self.dimensions)
        # `resolution` is points per axis for up to 3 dimensions. For 1/2/3
        # dimensions create a regular grid. For higher dimensions randomly
        # sample `resolution**3` points (capped) to avoid combinatorial explosion.
        if dims == 1:
            X = np.linspace(self.low[0], self.high[0], self.resolution, dtype=np.float32).reshape(-1, 1)
        elif dims == 2:
            x = np.linspace(self.low[0], self.high[0], self.resolution, dtype=np.float32)
            y = np.linspace(self.low[1], self.high[1], self.resolution, dtype=np.float32)
            gx, gy = np.meshgrid(x, y, indexing="xy")
            X = np.column_stack((gx.ravel(), gy.ravel())).astype(np.float32, copy=False)
        else:
            # For 3D and higher dimensions random-sample the space. Use
            # resolution**3 points (independent of the actual dimension count)
            # but cap to _MAX_RANDOM_EVAL_POINTS to avoid explosion.
            rng = np.random.default_rng(self.seed)
            n_points = min(self.resolution ** 3, self._MAX_RANDOM_EVAL_POINTS)
            X = rng.uniform(low=self.low, high=self.high, size=(int(n_points), dims)).astype(np.float32)

        var_df = {
            "id": list(self.dimensions),
            "lower_bound": self.low,
            "upper_bound": self.high,
        }
        adata = ad.AnnData(X=X, var=pd.DataFrame(var_df).set_index("id", drop=True))
        self.eval_points = adata

    def _update_eval_points(self, eval_points: ad.AnnData | FloatArray):
        if isinstance(eval_points, np.ndarray):
            if eval_points.ndim != 2:
                raise ValueError("eval_points array must be 2D.")
            if eval_points.shape[1] != len(self.dimensions):
                raise ValueError(
                    f"eval_points array must have {len(self.dimensions)} columns corresponding to gate dimensions."
                )
            self.eval_points = ad.AnnData(X=eval_points.astype(np.float32, copy=False))
            self.eval_points.var_names = list(self.dimensions)
        else:
            if not set(self.dimensions).issubset(set(eval_points.var_names)):
                raise ValueError("Eval points must include all gate dimensions as var_names.")
            self.eval_points = eval_points[:, list(self.dimensions)].copy()

        self.eval_points.var["lower_bound"] = self.low
        self.eval_points.var["upper_bound"] = self.high

        # Ensure that bounds are valid
        point_mins = self.eval_points.X.min(axis=0) # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        if np.any(point_mins < np.asarray(self.low)):
            warnings.warn(
                "Eval points have minimum values outside the specified lower bounds. "
                "Expanding low bounds to include eval points."
            )
            self.low = point_mins.tolist()

        point_maxs = self.eval_points.X.max(axis=0) # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        if np.any(point_maxs > np.asarray(self.high)):
            warnings.warn(
                "Eval points have maximum values outside the specified upper bounds. "
                "Expanding high bounds to include eval points."
            )
            self.high = point_maxs.tolist()

        # Normalize stored `resolution` to mean points-per-axis for up to 3D.
        # If eval_points were provided externally, estimate the per-axis
        # resolution by taking the n'th root where n = min(dimensions, 3).
        per_axis = int(round(self.eval_points.n_obs ** (1.0 / min(len(self.dimensions), 3))))
        self.info["resolution"] = max(1, per_axis)

    def get_mask(self, gate: Gate, parent_mask: dict[str, BooleanArray] | None = None) -> BooleanArray:
        if parent_mask is None:
            parent_mask = {"root": np.ones(self.n_eval_points, dtype=bool)}
        mask = gate.apply(self.eval_points, parent_mask)
        if len(mask) > 1:
            raise ValueError(
                f"Expected a single mask for gate {gate.gate_name}, but got {len(mask)} masks from apply()"
            )
        return next(iter(mask.values()))

    def _compute_masks(self):
        gate_cls = GateRegistry.get(self.gate_node.gate_type)
        self.masks: dict[str, BooleanArray] = {}
        parent_mask = {"root": np.ones(self.n_eval_points, dtype=bool)}

        # First enumerate custom (sample-specific) gates and assign indices
        for i, sample_id in enumerate(self.gate_node.custom_gates):
            gate = gate_cls.from_node(self.gate_node, sample_id=sample_id)
            try:
                key = hex(hash(gate))
            except Exception:
                continue
            self.sample_hashes[sample_id] = key
            self.masks[key] = self.get_mask(gate, parent_mask)
            self.gates[key] = i

        # Now try to add the default/batch gate, placing it after custom gates
        # so it receives index `i+1` (or 0 when there are no custom gates).
        try:
            default_gate = gate_cls.from_node(self.gate_node)
            default_key = hex(hash(default_gate))
        except Exception:
            default_key = None

        if default_key is not None:
            self.sample_hashes["__batch__"] = default_key
            if default_key not in self.masks:
                self.masks[default_key] = self.get_mask(default_gate, parent_mask) # pyright: ignore[reportPossiblyUnboundVariable]
            if default_key not in self.gates:
                ix = 0 if not self.gates else max(self.gates.values()) + 1
                self.gates[default_key] = ix

    def _validate_masks(self, masks: dict[str, BooleanArray]):

        # Ensure that mask formats are correct and consistent with eval points
        gate_idx: dict[str, int] = {}
        for i, (key, mask) in enumerate(masks.items()):
            if not isinstance(key, str):
                raise ValueError(f"Mask key {key} is not a string.")
            if mask.shape != (self.n_eval_points,):
                raise ValueError(
                    f"Mask for key {key} has shape {mask.shape}, expected ({self.n_eval_points},)."
                )
            if mask.dtype != bool:
                raise ValueError(f"Mask for key {key} has dtype {mask.dtype}, expected bool.")
            gate_idx[key] = i

        gate_cls = GateRegistry.get(self.gate_node.gate_type)

        self.masks = dict(masks)  # make a copy to ensure internal consistency
        sample_hashes: dict[str, str] = {}
        parent_mask = {"root": np.ones(self.n_eval_points, dtype=bool)}

        # Try to get default gate
        gate = gate_cls.from_node(self.gate_node)
        try:
            key = hex(hash(gate))
        except Exception:
            key = None

        if key is not None:
            sample_hashes["__batch__"] = key
            if key not in self.masks:
                self.masks[key] = self.get_mask(gate, parent_mask)

        for sample_id in self.gate_node.custom_gates:
            gate = gate_cls.from_node(self.gate_node, sample_id=sample_id)
            key = hex(hash(gate))
            sample_hashes[sample_id] = key
            if key not in self.masks:
                self.masks[key] = self.get_mask(gate, parent_mask)

        self.gates.update(gate_idx)
        self.sample_hashes.update(sample_hashes)
        inactive_masks = list(set(self.masks.keys()) - set(self.sample_hashes.values()))
        for inactive in inactive_masks:
            del self.masks[inactive]
            del self.gates[inactive]

    @staticmethod
    def _condensed_index(a: int, b: int) -> int:
        """Return index into condensed lower-triangular array for pair (a,b)."""
        if a == b:
            raise ValueError("No condensed index for identical indices")
        if a < b:
            a, b = b, a
        return a * (a - 1) // 2 + b

    def _compute_distances_full(self) -> None:
        """Full recompute of pairwise distances using float32 + BLAS."""
        mask_matrix = self.mask_matrix
        n_entries = self.n_entries
        if n_entries == 0:
            self._distances = np.zeros(0, dtype=np.float32)
            self._centrality = None
            return

        mask_f32 = np.ascontiguousarray(mask_matrix.astype(np.float32))
        support = np.sum(mask_matrix, axis=1, dtype=float)

        intersections = mask_f32 @ mask_f32.T
        unions = support[:, None] + support[None, :] - intersections
        distance_matrix = 1.0 - np.divide(
            intersections,
            unions,
            out=np.ones_like(intersections, dtype=float),
            where=unions > 0,
        )
        self._distances = distance_matrix[np.tril_indices(n_entries, k=-1)]
        self._centrality = None

    @staticmethod
    def inflate_distances(distances: Sequence[float] | FloatArray | None, N: int) -> np.ndarray:
        """Expand a condensed distance array (or an NxN matrix) to a full symmetric matrix.

        Returns an NxN float matrix.
        """
        if distances is None:
            return np.zeros((N, N), dtype=np.float32)

        arr = np.asarray(distances, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != (N * (N - 1) // 2):
                raise ValueError("Condensed distances length does not match N")
            mat = np.zeros((N, N), dtype=np.float32)
            mat[np.tril_indices(N, k=-1)] = arr
            return mat + mat.T
        elif arr.ndim == 2:
            if arr.shape != (N, N):
                raise ValueError("Square distance matrix shape does not match N")
            return arr.copy()
        else:
            raise ValueError("Invalid distances array")

    def _update_distance_matrix(
        self,
        dist_matrix: Sequence[float] | FloatArray,
        original_keys: list[str]
    ) -> np.ndarray | None:
        """Inflate/deflate a condensed distance array from original_keys -> new_keys.

        If `dist_matrix` is None or invalid, returns None to signal caller to
        perform a full recompute. Otherwise returns a condensed distance array
        matching `new_keys` order. Uses `self.masks` for mask lookups.
        """

        try:
            old_full = self.inflate_distances(dist_matrix, len(original_keys))
        except Exception:
            return None

        # derive new_keys from current instance masks
        new_keys = list(self.masks.keys())
        L = len(new_keys)
        new_full = np.zeros((L, L), dtype=float)

        old_idx = {k: i for i, k in enumerate(original_keys)}
        new_idx = {k: i for i, k in enumerate(new_keys)}

        # copy common block
        common = [k for k in new_keys if k in old_idx]
        if common:
            old_pos = [old_idx[k] for k in common]
            new_pos = [new_idx[k] for k in common]
            new_full[np.ix_(new_pos, new_pos)] = old_full[np.ix_(old_pos, old_pos)]

        # compute distances for added keys
        added = [new_idx[k] for k in new_keys if k not in old_idx]
        if added:
            mask_matrix = np.vstack([self.masks[k] for k in new_keys]).astype(np.float32)
            mask_matrix = np.ascontiguousarray(mask_matrix)
            intersections = mask_matrix[added, :] @ mask_matrix.T  # (k, L)
            support = np.sum(mask_matrix, axis=1)
            sup_added = support[added][:, None]
            unions = sup_added + support[None, :] - intersections
            with np.errstate(divide='ignore', invalid='ignore'):
                dists = 1.0 - np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
            for r, i in enumerate(added):
                new_full[i, :] = dists[r]
                new_full[:, i] = dists[r]

        return new_full[np.tril_indices(L, k=-1)]

    def _compute_distances_partial(self, changed_keys: Sequence[str]) -> None:
        """Partial update: recompute distances only for pairs involving changed mask keys."""

        changed = [k for k in changed_keys if k in self.masks]
        if not changed:
            # No masks affected => distances unchanged, keep cached centrality
            return

        N = self.n_entries
        if N == 0:
            self._distances = np.zeros(0, dtype=np.float32)
            self._centrality = None
            return

        mask_f32 = np.ascontiguousarray(self.mask_matrix.astype(np.float32))
        support: np.ndarray = mask_f32.sum(axis=1)

        # Ensure _distances exists and has correct length
        if self._distances is None or len(self._distances) != self.n_distances:
            # fall back to full recompute
            self._compute_distances_full()
            return

        changed_indices = [self.gates[k] for k in changed]
        intersections_sub = mask_f32[changed_indices, :] @ mask_f32.T  # shape (k, N)

        # Vectorized update for pairs involving changed indices.
        # Build flattened pair arrays (A,B) for all changed -> all entries,
        # mask out diagonal (i==j), compute intersections/unions in one pass,
        # then compute condensed indices and write distances.
        changed_idx_arr = np.asarray(changed_indices, dtype=int)
        k = changed_idx_arr.size
        if k:
            # intersections_sub has shape (k, N) in row-major order
            inter_flat = intersections_sub.ravel()  # length k * N
            A = np.repeat(changed_idx_arr, N)
            B = np.tile(np.arange(N, dtype=int), k)
            neq = A != B
            if np.any(neq):
                A_mask = A[neq]
                B_mask = B[neq]
                inter_vals = inter_flat[neq].astype(np.float32)
                union = support[A_mask] + support[B_mask] - inter_vals
                with np.errstate(divide='ignore', invalid='ignore'):
                    dvals = np.where(union > 0, 1.0 - (inter_vals / union), 0.0)
                # Ensure condensed index uses a >= b
                swap = A_mask < B_mask
                a = np.where(swap, B_mask, A_mask)
                b = np.where(swap, A_mask, B_mask)
                ci = (a * (a - 1) // 2 + b).astype(int)
                self._distances[ci] = dvals.astype(np.float32)

        self._centrality = None

    def _compute_distances(self, changed_keys: Sequence[str] | None = None) -> None:
        """Compatibility wrapper: dispatch to full or partial implementation."""
        if changed_keys:
            return self._compute_distances_partial(changed_keys)
        return self._compute_distances_full()

    def _condense_distances(self, dist_matrix: Sequence[float] | FloatArray):
        distances = np.asarray(dist_matrix, dtype=np.float32)
        N = self.n_entries

        if distances.ndim == 1:
            if distances.shape[0] != self.n_distances:
                raise ValueError(
                    f"Provided dist_matrix must have length {self.n_distances} for condensed distance matrix."
                )
            self._distances = distances.copy()
        elif distances.ndim == 2:
            if distances.shape != (N, N) or \
                not np.allclose(distances, distances.T, atol=1e-6) or \
                not np.allclose(np.diag(distances), 0.0, atol=1e-6):
                raise ValueError(
                    f"Provided dist_matrix must be a symmetric {N}x{N} matrix with zeros on the diagonal."
                )
            self._distances = distances[np.tril_indices(N, k=-1)]
        else:
            raise ValueError("dist_matrix must be either a 1D condensed distance array or a 2D square matrix.")

        if not np.all(self._distances >= 0):
            # Zeros are allowed because different configuration can produce identical masks
            raise ValueError("All distances must be non-negative.")

    # --------- Properties for key dimensions of the gate space geometry ---------

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(self.info["dimensions"])

    @property
    def resolution(self) -> int:
        return self.info["resolution"]

    @property
    def low(self) -> list[float]:
        return self.info["lower_bounds"]

    @low.setter
    def low(self, new_low: list[float]) -> None:
        if len(new_low) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} lower bound values, got {len(new_low)}.")
        self.info["lower_bounds"] = new_low
        self.eval_points.var["lower_bound"] = new_low

    @property
    def high(self) -> list[float]:
        return self.info["upper_bounds"]

    @high.setter
    def high(self, new_high: list[float]) -> None:
        if len(new_high) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} upper bound values, got {len(new_high)}.")
        self.info["upper_bounds"] = new_high
        self.eval_points.var["upper_bound"] = new_high

    @property
    def seed(self) -> int | None:
        return self.info.get("seed")

    @property
    def n_eval_points(self) -> int:
        return self.eval_points.n_obs

    @property
    def n_entries(self) -> int:
        return len(self.masks)

    @property
    def mask_matrix(self) -> BooleanArray:
        # TODO: move it to eval_points.obs ?
        if self.masks:
            return np.vstack(list(self.masks.values()))
        return np.zeros((0, self.n_eval_points), dtype=bool)

    @property
    def mean_mask(self) -> FloatArray:
        """Return the mean mask geometry as a float array of shape (n_eval_points,)."""
        if self.n_entries == 0:
            return np.zeros(self.n_eval_points)
        weights = np.asarray(self.sample_per_gate())
        return np.average(self.mask_matrix, axis=0, weights=weights)

    @property
    def n_distances(self) -> int:
        N = self.n_entries
        return N * (N - 1) // 2

    @property
    def distance_matrix(self) -> FloatArray:
        N = self.n_entries
        return self.inflate_distances(self._distances, N)

    @property
    def centrality(self) -> FloatArray:
        if self._centrality is None:
            self._centrality = self._compute_centrality()
        return self._centrality

    def _compute_centrality(self) -> FloatArray:
        if self.n_entries == 0:
            return np.zeros(0, dtype=np.float32)
        dist_mat = self.distance_matrix
        return np.sum(dist_mat, axis=1) / (self.n_entries - 1)

    def get_sample_mask(self, sample_id: str) -> BooleanArray:
        try:
            key = self.sample_hashes[sample_id]
            return self.masks[key]
        except KeyError as e:
            raise ValueError(f'No mask found for sample_id: "{sample_id}".') from e

    def _sample_index(self, sample_id: str) -> int:
        try:
            key = self.sample_hashes[sample_id]
            return self.gates[key]
        except KeyError as e:
            raise ValueError(f'No gate index found for sample_id: "{sample_id}".') from e

    def centrality_by_sample(self) -> dict[str, float]:
        return {sid: self.centrality[self.gates[key]] for sid, key in self.sample_hashes.items() if sid != "__batch__"}

    def sample_per_gate(self) -> list[int]:
        """Return a mapping of sample_id to gate hash key for all samples with masks."""
        gate_count = [0 for k in self.gates]
        for sid, key in self.sample_hashes.items():
            if sid == "__batch__": continue
            idx = self.gates[key]
            gate_count[idx] += 1
        return gate_count

    def plot(
        self,
        *,
        dimensions: list[str] | None = None,
        title: str | None = None,
        overlay_gate: bool = False,
        gate_sample_id: str | None = None,
        output_path: Path | None = None,
    ) -> go.Figure:
        """Render a heatmap representation of the gate-space mean mask.

        - 1D: simple bar plot with one bar per eval point (intensity = mean mask)
        - 2D: heatmap with one box per eval grid cell (intensity = mean mask)
        - >2D: scatter projection onto first two dims with color = mean mask

        If ``overlay_gate`` is True and ``gate_sample_id`` is provided the
        gate mask for that sample will be drawn on top of the heatmap.
        """
        dims = len(self.dimensions)
        x = np.asarray(self.eval_points.X)
        if x is None:
            raise ValueError("Eval points X matrix is missing.")
        mean = np.asarray(self.mean_mask)

        fig = go.Figure()

        if dims == 1:
            xs = x[:, 0]
            fig.add_trace(go.Bar(x=xs, y=mean, marker=dict(color=mean, colorscale="Viridis")))
            fig.update_layout(xaxis_title=self.dimensions[0], yaxis_title="Mean mask")
            if overlay_gate and gate_sample_id is not None:
                try:
                    mask = self.get_sample_mask(gate_sample_id)
                    # find contiguous True segments
                    m = np.asarray(mask, dtype=bool)
                    if m.any():
                        # determine half-step for x extents
                        if len(xs) > 1:
                            half = float((xs[1] - xs[0]) / 2.0)
                        else:
                            half = 0.0
                        starts = np.where(np.logical_and(~np.concatenate([[False], m[:-1]]), m))[0]
                        ends = np.where(np.logical_and(m, ~np.concatenate([m[1:], [False]])))[0]
                        for s, e in zip(starts, ends):
                            x0 = float(xs[s] - half)
                            x1 = float(xs[e] + half)
                            fig.add_shape(type="rect", x0=x0, x1=x1, y0=0.0, y1=1.0, xref="x", yref="y", line=dict(color="red", width=2), fillcolor="rgba(0,0,0,0)")
                except Exception:
                    pass

        elif dims == 2:
            # Pivot points into a regular grid for heatmap
            df = pd.DataFrame({
                "x": x[:, 0],
                "y": x[:, 1],
                "v": mean,
            })
            pivot = df.pivot(index="y", columns="x", values="v")
            # Ensure axis order is increasing
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot[pivot.columns.sort_values()]
            z = np.asarray(pivot.values)
            fig.add_trace(go.Heatmap(
                x=list(pivot.columns),
                y=list(pivot.index),
                z=z,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0
            ))
            fig.update_layout(xaxis_title=self.dimensions[0], yaxis_title=self.dimensions[1])

            if overlay_gate and gate_sample_id is not None:
                mask = self.get_sample_mask(gate_sample_id).astype(float)
                dfm = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "m": mask})
                pm = dfm.pivot(index="y", columns="x", values="m")
                pm = pm.sort_index(ascending=True)
                pm = pm[pm.columns.sort_values()]
                mz = np.asarray(pm.values)
                # Draw only the contour line at 0.5 (no filled interior)
                fig.add_trace(go.Contour(
                    x=list(pm.columns),
                    y=list(pm.index),
                    z=mz,
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1, coloring="lines", type="constraint"),
                    line=dict(color="red", width=2),
                    hoverinfo="skip",
                ))

        else:
            # For higher dims, project onto first two dims and scatter
            xs = x[:, 0]
            ys = x[:, 1] if x.shape[1] > 1 else np.zeros_like(xs)
            fig.add_trace(go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=mean, colorscale="Viridis", size=4),
                hoverinfo="skip"
            ))
            fig.update_layout(
                xaxis_title=self.dimensions[0],
                yaxis_title=(self.dimensions[1] if len(self.dimensions) > 1 else "dim2")
            )
            if overlay_gate and gate_sample_id is not None:
                try:
                    mask = self.get_sample_mask(gate_sample_id).astype(bool)
                    # plot open markers for masked points so interior remains visually transparent
                    fig.add_trace(go.Scattergl(
                        x=xs[mask],
                        y=ys[mask],
                        mode="markers",
                        marker=dict(color="red", size=3, symbol="circle-open", opacity=0.6),
                        name="gate_mask"
                    ))
                except Exception:
                    pass

        if title:
            fig.update_layout(title=title)

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig

    # ------------------------------------------------------------------
    # Artifact I/O helpers
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
    ) -> None:
        """Persist all gate-space artifacts and update ``entity_qc.artifacts``."""
        artifact_dir = dataloader.get_entity_qc_artifact_path(
            entity_type="gate_node",
            entity_id=self.info["entity_id"],
            artifact_key=self.artifact_key
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        info = dict(self.info)  # copy to avoid mutating in-memory state
        info["sample_hashes"] = self.sample_hashes
        # gate_index will be reconstructed masks in `._validate_masks()`, so we don't need to persist it
        info["distances"] = self._distances

        # Save eval_points as a single .h5ad file and store masks in `.obs`
        adata = self.eval_points.copy()
        # Ensure obs contains mask columns: one column per mask key, rows == n_eval_points
        if self.masks:
            obs_df = pd.DataFrame({k: np.asarray(v, dtype=bool) for k, v in self.masks.items()})
        else:
            obs_df = pd.DataFrame(index=range(self.n_eval_points))
        adata.obs = obs_df

        h5ad_name = "eval_points.h5ad"
        dataloader._save_h5ad(adata, artifact_dir / h5ad_name)

        # Record pointer to h5ad in info (do not inline large arrays)
        info["eval_points_h5ad"] = h5ad_name

        # Persist remaining metadata
        dataloader._save_json(artifact_dir / "info.json", info)

        entity_qc.artifacts[self.artifact_key] = {
            "artifact_type": self.artifact_key,
            "entity_id": info["entity_id"],
            "schema_version": _GATE_SPACE_ARTIFACT_VERSION,
            "updated_at": now_iso(),
        }

    @classmethod
    def load(
        cls,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader,
        entity: GateNode,
    ) -> "GateSpaceGeometry | None":
        """Load persisted gate-space state.  Returns ``None`` if absent or corrupt."""

        artifact_key = cls.artifact_key
        artifact_dir = dataloader.get_entity_qc_artifact_path(
            entity_type="gate_node",
            entity_id=entity.id,
            artifact_key=artifact_key
        )

        try:
            metadata: dict[str, str] = entity_qc.artifacts[artifact_key]
        except KeyError:
            return None

        assert metadata["artifact_type"] == artifact_key, \
            f"Expected artifact_type {artifact_key}, got {metadata['artifact_type']}"

        try:
            info = dataloader._load_json(artifact_dir / "info.json")
        except Exception:
            entity_qc.artifacts.pop(artifact_key, None)
            return None
        info = cast(Mapping[str, Any], info)  # for type checking purposes only

        cls._validate_gate(entity, info)

        # Load eval_points H5AD (which stores masks in `.obs`)
        h5ad_name = info.get("eval_points_h5ad", "eval_points.h5ad")
        try:
            eval_points = dataloader._load_h5ad(artifact_dir / h5ad_name)
        except Exception:
            # If we fail to load the heavy artifact, consider artifact corrupt
            entity_qc.artifacts.pop(artifact_key, None)
            return None

        # Extract masks from AnnData.obs -- columns correspond to mask keys
        masks: dict[str, BooleanArray] = {}
        if hasattr(eval_points, "obs") and isinstance(eval_points.obs, pd.DataFrame):
            for col in eval_points.obs.columns:
                masks[col] = np.asarray(eval_points.obs[col].values, dtype=bool)

        raw_dist = info.get("distances")
        dist_matrix = None
        if raw_dist is not None:
            dist_matrix = np.asarray(raw_dist, dtype=float)

        return cls(
            info=info,
            gate_node=entity,
            eval_points=eval_points,
            masks=masks,
            dist_matrix=dist_matrix,
        )

    @classmethod
    def update(
        cls,
        *,
        entity: GateNode,
        dataloader: UnifiedDataLoader,
        entity_qc: EntityQCStatus,
        sample_ids: list[str],
        config: Mapping[str, Any] | None = None,
    ):
        """Update gate-space geometry based on the current gate configuration and provided sample IDs.

        This method applies the following update policy:
        - If the gate type is Boolean or Quadrant, or if the config policy is "skip", no updates are made.
        - If an artifact exists but the gate configuration has changed (as determined by hash keys), the artifact is deleted and rebuilt from scratch.
        - If an artifact exists and the gate configuration has not changed, only the distance matrix is recomputed to reflect any changes in eval points or masks.
        - If no artifact exists, a new one is created from scratch.
        """

        config = config or {}
        resolution=config.get("gate_space_resolution", 1024)
        seed=config.get("gate_space_seed", 42)
        artifact = cls.load(entity_qc=entity_qc, dataloader=dataloader, entity=entity)
        if artifact is None or resolution != artifact.resolution or seed != artifact.seed:
            artifact = cls.create_artifact(
                entity=entity,
                entity_qc=entity_qc,
                dataloader=dataloader,
                sample_ids=sample_ids,
                resolution=resolution,
                seed=seed,
            )
            return
        # For update we rebuild the GateSpaceGeometry object using the
        # current artifact state (masks/eval_points) and the existing
        # condensed distances. The heavy inflate/deflate logic lives in
        # ``__init__`` after ``_validate_masks`` so we delegate to that
        # machinery by re-instantiating the object.
        artifact._add_missing_samples(sample_ids)

        # Ensure masks exist for any newly required gate configs
        for sample_id in sample_ids:
            try:
                gate = dataloader.load_gate(entity, sample_id=sample_id)
                key = hex(hash(gate))
            except Exception:
                key = None
            if key is None:
                continue
            artifact.sample_hashes[sample_id] = key
            if key not in artifact.masks:
                artifact.masks[key] = artifact.get_mask(gate) # pyright: ignore[reportPossiblyUnboundVariable]

        # Default gate
        try:
            default_gate = dataloader.load_gate(entity)
            default_key = hex(hash(default_gate))
        except Exception:
            default_key = None

        if default_key is not None:
            artifact.sample_hashes.setdefault("__batch__", default_key)
            if default_key not in artifact.masks:
                artifact.masks[default_key] = artifact.get_mask(default_gate) # pyright: ignore[reportPossiblyUnboundVariable]

        # Rebuild via constructor to let __init__ handle inflation/deflation
        new_artifact = cls(
            info=artifact.info,
            gate_node=entity,
            sample_ids=sample_ids,
            eval_points=artifact.eval_points,
            masks=artifact.masks,
            dist_matrix=artifact._distances
        )
        new_artifact.save(entity_qc=entity_qc, dataloader=dataloader)


# ============================================================================
# QC Test Classes
# ============================================================================

class GateMaskRatioTest(QCTester):
    """Test for event counts and ratios in a gated region."""

    test_type = "gatenode"
    test_name = "gatenode_event_count"
    target_keys = ("entity_id", "sample_id")
    meta_keys = ("parent_id", )
    default_config = {}
    meta_fields = [
        ("parent_id", "Parent gate ID(s)"),
        ("n_events_total", "Total number of events"),
        ("n_events_parent", "Number of events in parent gate"),
        ("n_events_passing", "Number of events passing this gate"),
    ]
    metric_fields = [
        ("ratio_total", "Proportion of events passing gate relative to total"),
        ("ratio_parent", "Proportion of events passing gate relative to parent"),
    ]
    default_thresholds = {
        "ratio_total": {"warn": (0.0, 1.0), "severe": (None, None)},
        "ratio_parent": {"warn": (0.0, 1.0), "severe": (None, None)},
    }
    plot_type = "bar"
    plot_description = "Gate event counts and ratios relative to total and parent gate"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config=config, thresholds=thresholds)

    def fit(
        self,
        targets: dict[str, Any],
        entity: GateNode,
        masks: dict[str, BooleanArray],
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all gates in the mask dict.

        Yields one QCTestRecord per gate.

        Parameters
        ----------
        targets : dict[str, Any]
            Base target identifiers (entity_id, sample_id)
        entity : GateNode
            Gate node entity for the test.
        masks : dict[str, BooleanArray]
            Mapping of gate IDs to boolean masks for ratio calculation.
        **kwargs
            Additional test-specific parameters.
        """

        metadata = self.metadata.copy()  # Start with default metadata
        # Calculate basic metrics
        entity_id = entity.id
        mask = masks[entity_id]
        n_passing = int(np.sum(mask))
        n_total = int(mask.shape[0])

        # Gate Parent Mask. If multiple (Boolean Gate) combine with OR and concatenate parent IDs:
        parent_ids = [pid for pid in entity.parent_ids if pid != "root"]
        if parent_ids:
            parent_id = "|".join(sorted(parent_ids))
            parent_mask = np.logical_or.reduce([masks[parent] for parent in parent_ids])
            if parent_mask.shape[0] != n_total:
                raise ValueError(
                    f"Mask length mismatch for gate {targets['entity_id']}: "
                    f"gate mask length={n_total}, parent mask length={parent_mask.shape[0]}"
                )
        else:
            parent_id = "root"
            parent_mask = np.ones_like(mask)

        metadata["parent_id"] = parent_id
        metadata["n_events_total"] = n_total
        metadata["n_events_parent"] = n_parent = int(np.sum(parent_mask))
        metadata["n_events_passing"] = n_passing

        if n_total == 0:
            yield QCTestRecord(
                id=self.make_key(targets=targets, metadata=metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={
                    "ratio_total": float("nan"),
                    "ratio_parent": float("nan"),
                },
                thresholds=self.thresholds.copy(),
                status="SKIP",
                message="No events to test.",
            )
            return

        ratio_total = float(n_passing / n_total)
        ratio_parent = float("nan")
        n_passing_in_parent = float(np.sum(mask & parent_mask))
        ratio_parent = n_passing_in_parent / n_parent if n_parent else float("nan")

        test = QCTestRecord(
            id=self.make_key(targets=targets, metadata=metadata),
            test_type=self.test_type,
            test_name=self.test_name,
            targets=targets,
            metadata=metadata,
            metrics={
                "ratio_total": ratio_total,
                "ratio_parent": ratio_parent,
            },
            thresholds=self.thresholds.copy(),
            status="PENDING",
        )

        yield test

    def plot(
        self,
        test: QCTestRecord,
        *,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Generate diagnostic plot for gate mask event counts.

        Creates two normalized stacked bars (height 1.0):
        1. Total view: shows gate/parent/other proportions relative to total events
        2. Parent view: shows gate/non-gate proportions relative to parent events

        Both bars overlay their corresponding ratio thresholds.
        """
        entity_id = test.targets.get("entity_id", "Unknown")
        parent_id = test.metadata.get("parent_id", "Unknown")

        n_total = test.metadata.get("n_events_total", 0)
        n_parent = test.metadata.get("n_events_parent", 0)
        n_passing = test.metadata.get("n_events_passing", 0)

        ratio_total = test.metrics.get("ratio_total", 0.0)
        ratio_parent = test.metrics.get("ratio_parent", 0.0)

        # Calculate event counts for stacked bars
        # For parent view: need events in gate AND parent
        if not np.isnan(ratio_parent) and n_parent > 0:
            n_passing_in_parent = int(ratio_parent * n_parent)
        else:
            n_passing_in_parent = 0

        # Normalized proportions for first bar (Total view)
        if n_total > 0:
            prop_gate_total = n_passing_in_parent / n_total
            prop_parent_not_gate_total = (n_parent - n_passing_in_parent) / n_total
            prop_not_parent_total = (n_total - n_parent) / n_total
        else:
            prop_gate_total = prop_parent_not_gate_total = prop_not_parent_total = 0

        # Normalized proportions for second bar (Parent view)
        if n_parent > 0:
            prop_gate_parent = ratio_parent
            prop_parent_not_gate_parent = 1.0 - ratio_parent
        else:
            prop_gate_parent = prop_parent_not_gate_parent = 0

        # Create figure
        fig = go.Figure()

        # First bar: Total view (3 parts)
        fig.add_trace(go.Bar(
            name="Not in parent",
            x=["Total View"],
            y=[prop_not_parent_total],
            marker=dict(color="rgba(211, 211, 211, 0.7)"),
            text=f"Other<br>{prop_not_parent_total:.1%}",
            textposition="inside",
            hovertemplate="<b>Not in parent</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Parent (not gate)",
            x=["Total View"],
            y=[prop_parent_not_gate_total],
            marker=dict(color="rgba(100, 149, 237, 0.7)"),
            text=f"Parent<br>{prop_parent_not_gate_total:.1%}",
            textposition="inside",
            hovertemplate="<b>In parent, not in gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Gate",
            x=["Total View"],
            y=[prop_gate_total],
            marker=dict(color="rgba(60, 179, 113, 0.7)"),
            text=f"Gate<br>{prop_gate_total:.1%}",
            textposition="inside",
            hovertemplate="<b>In gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        # Second bar: Parent view (2 parts)
        fig.add_trace(go.Bar(
            name="Parent (not gate)",
            x=["Parent View"],
            y=[prop_parent_not_gate_parent],
            marker=dict(color="rgba(100, 149, 237, 0.7)"),
            text=f"Parent<br>{prop_parent_not_gate_parent:.1%}",
            textposition="inside",
            showlegend=False,
            hovertemplate="<b>In parent, not in gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        fig.add_trace(go.Bar(
            name="Gate",
            x=["Parent View"],
            y=[prop_gate_parent],
            marker=dict(color="rgba(60, 179, 113, 0.7)"),
            text=f"Gate<br>{prop_gate_parent:.1%}",
            textposition="inside",
            showlegend=False,
            hovertemplate="<b>In gate</b><br>Proportion: %{y:.2%}<extra></extra>",
        ))

        # Add threshold lines
        thresholds = test.thresholds or {}

        # Total ratio threshold (on first bar)
        if "ratio_total" in thresholds and "warn" in thresholds["ratio_total"]:
            warn_range = thresholds["ratio_total"]["warn"]
            if warn_range and len(warn_range) == 2 and warn_range[0] is not None:
                fig.add_shape(
                    type="line",
                    x0=-0.4, x1=0.4,  # First bar position
                    y0=warn_range[0], y1=warn_range[0],
                    line=dict(color="orange", width=3, dash="dash"),
                    xref="x", yref="y",
                )
            if warn_range and len(warn_range) == 2 and warn_range[1] is not None and warn_range[1] < 1.0:
                fig.add_shape(
                    type="line",
                    x0=-0.4, x1=0.4,  # First bar position
                    y0=warn_range[1], y1=warn_range[1],
                    line=dict(color="orange", width=3, dash="dash"),
                    xref="x", yref="y",
                )

        # Parent ratio threshold (on second bar)
        if "ratio_parent" in thresholds and "warn" in thresholds["ratio_parent"]:
            warn_range = thresholds["ratio_parent"]["warn"]
            if warn_range and len(warn_range) == 2 and warn_range[0] is not None:
                fig.add_shape(
                    type="line",
                    x0=0.6, x1=1.4,  # Second bar position
                    y0=warn_range[0], y1=warn_range[0],
                    line=dict(color="red", width=3, dash="dash"),
                    xref="x", yref="y",
                )
            if warn_range and len(warn_range) == 2 and warn_range[1] is not None and warn_range[1] < 1.0:
                fig.add_shape(
                    type="line",
                    x0=0.6, x1=1.4,  # Second bar position
                    y0=warn_range[1], y1=warn_range[1],
                    line=dict(color="red", width=3, dash="dash"),
                    xref="x", yref="y",
                )

        # Update layout
        title = f"Gate: {entity_id} (parent: {parent_id})<br>" \
                f"<sub>Total events: {n_total:,} | Parent events: {n_parent:,} | " \
                f"Gate events: {n_passing_in_parent:,}</sub>"

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="Proportion",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            barmode="stack",
            hovermode="closest",
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


class GateFitDiagnosticTest(QCTester):
    """Test for gate fitting quality based on diagnostics.

    Evaluates gate fitting diagnostics (e.g., r_squared, residual_std, n_outliers)
    for gates that have been fitted via automated methods. Skips gates with manual
    parameter overrides (which lack diagnostics).

    Gate types can define their own expected diagnostics and thresholds via
    _get_gate_type_config().
    """

    test_type = "gatenode"
    test_name = "gatenode_fit_quality"
    target_keys = ("entity_id", "sample_id")
    meta_keys = ("parent_id", )
    default_config = {}
    meta_fields = [
           ("gate_type", "Gate type"),
           ("parent_id", "Parent gate ID(s)"),
       ]
    metric_fields = [
           ("r_squared", "R-squared value of the fit"),
           ("residual_std", "Standard deviation of residuals"),
           ("n_outliers", "Number of outlier points"),
       ]
    default_thresholds = {
           "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
           "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
           "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
       }
    plot_type = ""  # No plot for now
    plot_description = "Gate fitting quality diagnostics"

    # Gate type configuration registry
    _gate_type_configs = {
        "ols_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
                   "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
                   "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
            },
        },
        "wls_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.7, None), "severe": (0.5, None)},
                   "residual_std": {"warn": (None, 1.0), "severe": (None, 1.5)},
                   "n_outliers": {"warn": (None, 100), "severe": (None, 150)},
            },
        },
        "logistic_regression": {
            "expected_diagnostics": ["r_squared", "residual_std", "n_outliers"],
            "thresholds": {
                   "r_squared": {"warn": (0.6, None), "severe": (0.4, None)},
                   "residual_std": {"warn": (None, 1.5), "severe": (None, 2.0)},
                   "n_outliers": {"warn": (None, 150), "severe": (None, 200)},
            },
        },
        "min_density": {
            "expected_diagnostics": [],  # Per-dimension diagnostics only
            "thresholds": {},
        },
    }

    def fit(
        self,
        targets: dict[str, Any],
        entity: GateNode,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for gate fitting diagnostics.

        Yields one QCTestRecord per diagnostic key found in the gate node.
        Yields SKIP records if diagnostics are missing (manual gate override).

        Parameters
        ----------
        targets : dict[str, Any]
            Target identifiers (sample_id, entity_id)
        **kwargs
            Additional test-specific parameters (gate_node, sample_id, entity, etc.)
        """
        sample_id: str = targets["sample_id"]
        thresholds = self.thresholds.copy()
        metadata = self.metadata.copy()
        metadata["gate_type"] = entity.gate_type
        # Add parent_id for comopatibility with older test records
        metadata["parent_id"] = "|".join(sorted(entity.parent_ids)) if entity.parent_ids else "root"

        # Extract diagnostics from gate node
        node_params = entity.get_params_for_sample(sample_id)
        diagnostics: dict[str, Any] = node_params.get("diagnostics", {})

        gate_config = self._gate_type_configs.get(entity.gate_type, {})
        thresholds.update(gate_config.pop("thresholds", {}))  # Override thresholds with gate-type-specific values if available
        metadata.update(gate_config)  # Add expected diagnostics and thresholds to metadata
        for diag_key, diag_value in diagnostics.items():
            try:
                metric_value = float(diag_value)
            except (TypeError, ValueError):
                # Skip non-scalar diagnostics
                continue

            yield QCTestRecord(
                id=self.make_key(targets=targets, metadata=metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={diag_key: metric_value},
                thresholds=thresholds,
                status="PENDING",
            )


@EntityQCEvaluatorRegistry.register("gate_node")
class GateNodeQCEvaluator(EntityQCEvaluator):
    """QC evaluator for individual gate nodes within a gating strategy.

    Evaluates gate performance within its strategy context, including:
    - Event counts and ratios (total and parent-relative)
    - Gate fitting quality diagnostics
    - Sample-level outlier detection

    Uses the project-level gating strategy implicitly.
    """

    entity_type = "gate_node"
    _supported_tables = {
        "event_metrics": {
            "description": "Event counts and ratios per sample for this gate",
            "input_params": {}
        },
        "fitting_quality": {
            "description": "Gate fitting diagnostics across samples",
            "input_params": {}
        },
    }
    _supported_figures = {}  # Test plots are auto-discovered from registered tests
    _gate_space_artifact_name = "gate_space"

    default_config = {
        "ratio_total_min": 0.0,
        "ratio_total_max": 1.0,
        "ratio_parent_min": 0.0,
        "ratio_parent_max": 1.0,
        # Gate fitting quality thresholds
        "r_squared": 0.7,
        "residual_std": 1.0,
        "n_outliers": 100,
        # Outlier detection config/thresholds
        "min_samples": 6,
        "outlier_method": "iqr",
        "outlier_thresholds": {
            "warn": (-1.5, 1.5),
            "severe": (-3.0, 3.0),
        },
        "use_mad": True,
        # Full-gate geometry outlier config
        "gate_space_min_samples": 6,
        "gate_space_outlier_method": "zscore",
        "gate_space_resolution": 256,
        "gate_space_thresholds": {
            "warn": (-3.0, 3.0),
            "severe": (-5.0, 5.0),
        },
        "gate_space_seed": 42,
        "gate_space_update_policy": "incremental",
    }

    @staticmethod
    def _normalize_outlier_thresholds(raw: Any) -> dict[str, Any]:
        """Normalize evaluator threshold config to QCTester threshold schema.

        Accepted inputs:
        - None: use tester defaults
        - {"warn": (...), "severe": (...)}
        - {"outlier_score": {"warn": (...), "severe": (...)}}
        - (low, high): interpreted as warn only; severe is filled by tester defaults
        """
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            if "outlier_score" in raw:
                return {"outlier_score": raw["outlier_score"]}
            if "warn" in raw or "severe" in raw:
                return {"outlier_score": dict(raw)}
            return dict(raw)
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            return {"outlier_score": {"warn": tuple(raw)}}
        raise ValueError(
            "Invalid thresholds config. Expected None, (low, high), "
            "{'warn': (...)}, or {'outlier_score': {'warn': (...), 'severe': (...)}}."
        )

    @classmethod
    def get_tests(cls, entity: GateNode | None = None) -> dict[str, type[QCTester]]:
        """Return dictionary of test classes for gate node QC.

        Reuses the same test classes as GatingStrategyQCEvaluator.

        Parameters
        ----------
        entity : GateNode | None
            Gate node entity (optional, can be used for entity-specific tests)

        Returns
        -------
        dict[str, type[QCTester]]
            Mapping of test_name → QCTester subclass
        """

        if entity is None:
            raise ValueError("Entity must be provided to get tests for gate node QC.")

        # Base testers
        testers = [GateMaskRatioTest, GateFitDiagnosticTest]

        default_thresholds = cls._normalize_outlier_thresholds(cls.default_config.get("outlier_thresholds"))
        extra_meta_keys = ("parent_id", )
        extra_meta_fields = [("parent_id", "Parent gate ID(s)")]
        default_config = {
            "min_samples": cls.default_config["min_samples"],
            "outlier_method": cls.default_config["outlier_method"],
            "use_mad": cls.default_config["use_mad"],
        }

        for metric_type in ("masks", "diagnostics", "params"):
            testers.append(
                _ScalarOutlierTester.from_defaults(
                    entity=entity,
                    metric_type=metric_type,
                    extra_meta_keys=extra_meta_keys,
                    extra_meta_fields=extra_meta_fields,
                    config=default_config,
                    thresholds=default_thresholds,
                )
            )

        if entity.gate_type not in {"Boolean", "Quadrant"}:
            default_config = {
                "min_samples": cls.default_config["gate_space_min_samples"],
                "outlier_method": cls.default_config["gate_space_outlier_method"],
                "use_mad": cls.default_config["use_mad"],
                "gate_type": entity.gate_type,
                "resolution": cls.default_config["gate_space_resolution"],
            }
            default_thresholds = cls._normalize_outlier_thresholds(cls.default_config["gate_space_thresholds"])
            testers.append(_ScalarOutlierTester.from_defaults(
                entity=entity,
                metric_type="geometry",
                extra_meta_keys=extra_meta_keys,
                extra_meta_fields=extra_meta_fields,
                config=default_config,
                thresholds=default_thresholds,
            ))

        return {tester.test_name: tester for tester in testers}

    def required_layer(self, entity: GateNode | None = None) -> str | None:
        """Return the required AnnData layer for gate node QC."""
        if entity is None:
            return None
        return entity.layer

    def load_entity(
        self,
        dataloader: UnifiedDataLoader,
        entity_id: Hashable,
        context: dict[str, Any] | None = None
    ) -> GateNode:
        """Load a gate node from the dataloader.
        """
        return dataloader.load_gate_node(node_id=str(entity_id))

    def prepare_artifacts(
        self,
        entity: Any,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:

        dataloader_context = dataloader_context or {}
        context = context or {}

        config = self.config.copy()
        config.update(context)
        sample_ids = list(
            context.get("sample_ids")
            or dataloader_context.get("sample_ids")
            or entity_qc.sample_qc.keys()
        )

        if dataloader is None:
            raise ValueError("dataloader must be provided to load gate masks")

        if entity.gate_type in {"Boolean", "Quadrant"} or config.get("gate_space_update_policy") == "skip":
            return

        GateSpaceGeometry.update(
            entity=entity,
            entity_qc=entity_qc,
            dataloader=dataloader,
            sample_ids=sample_ids,
            config=config,
        )

    def update_sample_qc(
        self,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Evaluate gate node QC against sample data.

        Parameters
        ----------
        entity : GateNode
            The gate node entity to evaluate
        entity_qc : EntityQCStatus
            QC status to update
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading sample data
        dataloader_context : dict[str, Any] | None
            Optional context with sample_ids, layer, etc.
        context : dict[str, Any]
            Optional evaluation context

        Returns
        -------
        None
        """
        # Default context to empty dict to avoid None checks
        dataloader_context = dataloader_context or {}
        context = context or {}

        config = self.config.copy()
        config.update(context)
        sample_ids = list(
            context.get("sample_ids")
            or dataloader_context.get("sample_ids")
            or entity_qc.sample_qc.keys()
        )
        entity_qc.context = {
            **config,
            "sample_ids": sample_ids,
        }

        # Dataloader is required to load masks for QC evaluation.
        if dataloader is None:
            raise ValueError("dataloader must be provided to load gate masks")

        # Main evaluation
        for sample_id in sample_ids:
            self._evaluate_gate_node(
                entity=entity,
                entity_qc=entity_qc,
                sample_id=sample_id,
                config=config,
                dataloader=dataloader,
            )

    def _evaluate_gate_node(
        self,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        sample_id: str,
        config: Mapping[str, Any],
        dataloader: UnifiedDataLoader,
    ) -> None:
        """Evaluate the gate node for a single sample.

        Parameters
        ----------
        entity : GateNode
            The gate node to evaluate
        entity_qc : EntityQCStatus
            QC status to update
        sample_id : str
            Sample identifier
        config : Mapping[str, Any]
            Evaluation config with threshold settings
        dataloader : UnifiedDataLoader
            Dataloader for loading gate masks
        """
        sample_steps = entity_qc.get_sample_steps(sample_id)
        mask_step = sample_steps.get_step("GATE_QC_MASK")
        diagnostics_step = sample_steps.get_step("GATE_QC_DIAGNOSTICS")
        gate_targets = {
            "sample_id": sample_id,
            "entity_id": entity.id,
        }

        self._evaluate_gate_mask(
            entity=entity,
            sample_id=sample_id,
            config=config,
            dataloader=dataloader,
            step=mask_step,
            gate_targets=gate_targets,
        )

        self._evaluate_gate_diagnostics(
            entity=entity,
            config=config,
            step=diagnostics_step,
            gate_targets=gate_targets,
        )

    def _evaluate_gate_mask(
        self,
        entity: GateNode,
        sample_id: str,
        config: Mapping[str, Any],
        dataloader: UnifiedDataLoader,
        step: QCStepStatus,
        gate_targets: dict[str, str],
    ) -> None:
        """Evaluate mask-derived gate metrics for a single sample."""
        gate_id = gate_targets["entity_id"]
        masks_to_load = [pid for pid in entity.parent_ids if pid != "root"]
        masks_to_load.append(gate_id)

        try:
            masks = dataloader.load_masks(
                sample_id=sample_id,
                gate_ids=masks_to_load,
            )
        except FileNotFoundError as e:
            step.flag = QCFlag.FAIL
            step.add_reason(
                code="GATE_MASKS_MISSING",
                message=f"Required gate masks are missing for sample {sample_id}: {e}",
            )
            return

        event_tester = GateMaskRatioTest(
            config={},
            thresholds={
                "ratio_total": {
                    "warn": (config["ratio_total_min"], config["ratio_total_max"]),
                    "severe": (None, None),
                },
                "ratio_parent": {
                    "warn": (config["ratio_parent_min"], config["ratio_parent_max"]),
                    "severe": (None, None),
                },
            }
        )

        for classified_test in event_tester.fit_classify(
            targets=gate_targets,
            entity=entity,
            masks=masks,
        ):
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                step.add_reason(
                    code=f"GATING_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                step.add_test(classified_test)

    def _evaluate_gate_diagnostics(
        self,
        entity: GateNode,
        config: Mapping[str, Any],
        step: QCStepStatus,
        gate_targets: dict[str, str],
    ) -> None:
        """Evaluate fitted-gate diagnostics for a single sample."""
        fitting_tester = GateFitDiagnosticTest(
            thresholds={
                "r_squared": {"warn": (config["r_squared"], None), "severe": (None, None)},
                "residual_std": {"warn": (None, config["residual_std"]), "severe": (None, None)},
                "n_outliers": {"warn": (None, config["n_outliers"]), "severe": (None, None)},
            }
        )

        for classified_test in fitting_tester.fit_classify(
            targets=gate_targets,
            entity=entity,
        ):
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                step.add_reason(
                    code=f"GATE_FITTING_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                step.add_test(classified_test)

    def update_batch_qc(
        self,
        entity: GateNode,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update batch-level QC tests for gate node.

        Runs outlier detection across samples for this gate's event metrics.
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided to load gate masks and geometry for batch QC")

        context = context or {}
        config = self.config.copy()
        config.update(context)

        # Get basic QC
        sample_ids = list(entity_qc.sample_qc.keys())
        batch_step = entity_qc.batch_qc.get_step("GATE_NODE_BATCH_QC")
        param_metrics, diag_metrics = self._collect_sample_params(gate_node=entity, sample_ids=sample_ids)
        mask_metrics = self._collect_mask_metrics(entity_qc, gate_id=entity.id)
        sample_metrics: dict[str, dict[str, list[float]]] = {
            "diagnostics": diag_metrics,
            "masks": mask_metrics,
        }

        if entity.custom_gates:
            sample_metrics["params"] = param_metrics

        # Run outlier detection for this gate
        method = config["outlier_method"]
        outlier_thresholds = self._normalize_outlier_thresholds(
            config.get("outlier_thresholds", config.get("thresholds"))
        )
        outlier_config = {
            "min_samples": config["min_samples"],
            "outlier_method": method,
            "use_mad": config["use_mad"],
            "parent_id": "|".join(sorted(entity.parent_ids)) if entity.parent_ids else "root",
        }
        testers = self.get_tests(entity)
        for metric_type, metrics_by_name in sample_metrics.items():
            test_name = f"gatenode_{metric_type}_outlier"
            tester_class = testers.get(test_name)
            if tester_class is None: continue  # No tester defined for this metric type
            for metric_name, metric_series in metrics_by_name.items():
                outlier_tester = tester_class(outlier_config, thresholds=outlier_thresholds)
                outlier_targets = {
                    "entity_id": entity.id,
                    "metric_type": metric_type,
                    "metric_name": metric_name
                }
                sample_values = {sid: metric_series[idx] for idx, sid in enumerate(sample_ids)}
                for classified_test in outlier_tester.fit_classify(
                    targets=outlier_targets,
                    sample_values=sample_values,
                ):
                    # Add to batch step
                    if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                        batch_step.add_reason(
                            code=f"GATE_OUTLIER_{classified_test.status}",
                            message=classified_test.message,
                            tests=[classified_test],
                        )
                    else:
                        batch_step.add_test(classified_test)

        # Compare full gate spaces across samples using the cached coarse-grained geometry.
        gate_space_geometry = GateSpaceGeometry.load(entity_qc=entity_qc, dataloader=dataloader, entity=entity)
        if gate_space_geometry is None: return

        # Build Tester
        gate_space_targets = {
            "entity_id": entity.id,
            "metric_type": "geometry",
            "metric_name": "centrality_score"
        }
        gate_space_config = {
            "min_samples": config["gate_space_min_samples"],
            "outlier_method": config["gate_space_outlier_method"],
            "use_mad": config["use_mad"],
            "parent_id": "|".join(sorted(entity.parent_ids)) if entity.parent_ids else "root",
            "gate_type": entity.gate_type,
            "resolution": config["gate_space_resolution"],
        }
        gate_space_thresholds = self._normalize_outlier_thresholds(
            config["gate_space_thresholds"]
        )
        centrality = gate_space_geometry.centrality_by_sample()
        space_tester = testers["gatenode_geometry_outlier"](gate_space_config, thresholds=gate_space_thresholds)
        for classified_test in space_tester.fit_classify(
            targets=gate_space_targets,
            sample_values=centrality,
        ):
            if classified_test.status in {"WARN", "SEVERE", "FAIL"}:
                batch_step.add_reason(
                    code=f"GATE_GEOMETRY_OUTLIER_{classified_test.status}",
                    message=classified_test.message,
                    tests=[classified_test],
                )
            else:
                batch_step.add_test(classified_test)

    def _collect_mask_metrics(
        self,
        entity_qc: EntityQCStatus,
        gate_id: str,
    ) -> dict[str, list[float]]:
        """Collect scalar mask metrics in metric-centric format.

        Returns
        -------
        dict[str, list[float]]
            Mapping ``metric_name -> [value_per_sample, ...]`` aligned with
            ``sample_ids``. Missing values are filled with ``NaN``.
        """
        sample_ids = list(entity_qc.sample_qc.keys())
        n_samples = len(sample_ids)
        metrics_by_name: dict[str, list[float]] = {}

        for idx, sample_id in enumerate(sample_ids):
            for (step_name, test_key), test_record in entity_qc.iter_sample_tests(sample_id):
                if (test_record.test_type == GateMaskRatioTest.test_type and
                    test_record.test_name == GateMaskRatioTest.test_name and
                    test_record.targets["entity_id"] == gate_id):

                    for metric_key, metric_value in test_record.metrics.items():
                        metric_series = metrics_by_name.setdefault(
                            metric_key,
                            [float("nan")] * n_samples,
                        )
                        try:
                            metric_series[idx] = float(metric_value)
                        except (TypeError, ValueError):
                            continue

        return metrics_by_name

    @staticmethod
    def _collect_sample_params(
        gate_node: GateNode,
        sample_ids: list[str] | None = None,
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        """Collect scalar gate metrics across samples in metric-centric format.

        Expected shape (by convention, not enforced by all gates):
        - node_params["params"]:      {metric_name: scalar | array}
        - node_params["diagnostics"]: {metric_name: scalar | array}

        Arrays/non-scalars are ignored for now to keep this path strictly 1D.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            (
                {"params": {metric_name: [value_per_sample, ...]}},
                {"diagnostics": {metric_name: [value_per_sample, ...]}},
            )
            Metric series are aligned with "sample_ids".
        """
        sample_ids = sample_ids or list(gate_node.custom_gates.keys())
        n_samples = len(sample_ids)

        sample_params: dict[str, list[float]] = {}
        sample_diagnostics: dict[str, list[float]] = {}

        for idx, sample_id in enumerate(sample_ids):
            node_params = gate_node.get_params_for_sample(sample_id)
            for section_name, sink in (("params", sample_params), ("diagnostics", sample_diagnostics)):
                section = node_params.get(section_name, {})
                if not isinstance(section, Mapping):
                    continue

                for metric_name, metric_value in section.items():
                    # Skip array-like values until multi-dimensional tests are implemented.
                    if isinstance(metric_value, (list, tuple, dict, np.ndarray)):
                        continue

                    metric_series = sink.setdefault(metric_name, [float("nan")] * n_samples)
                    try:
                        metric_series[idx] = float(metric_value)
                    except (TypeError, ValueError):
                        # Keep NaN for non-numeric values.
                        continue

        return sample_params, sample_diagnostics

    def _plot_gate_with_ratio(
        self,
        gate_node: GateNode,
        dataloader: UnifiedDataLoader,
        sample_id: str,
        dimensions: list[str] | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Render a gate-level diagnostic plot using the Gate.plot API with
        `show_ratio=True` enabled.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            QC status for the gate node entity
        dataloader : UnifiedDataLoader
            Data loader used to fetch sample AnnData and gate node
        sample_id : str
            Sample identifier to render
        test : QCTestRecord | None
            Optional test record (unused, provided for symmetry)
        **kwargs : Any
            Forwarded to `Gate.plot`
        """
        if dataloader is None or sample_id is None:
            raise ValueError("Dataloader and sample_id are required to generate gate plot")

        layer = self.required_layer(gate_node)
        events = dataloader.load_adata(sample_id=sample_id, layer=layer)
        gate = dataloader.load_gate(gate_node, sample_id=sample_id)

        return gate.plot(
            events=events,
            mask=None,
            dimensions=dimensions,
            show_ratio=True,
            **kwargs
        )

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Mapping[str, Any] | QCTestRecord,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Generate plot for a gate-level test (sample or batch).

        Looks up the `QCTestRecord` in `entity_qc` using `test_key`, instantiates
        the corresponding `QCTester` from the record and delegates to its
        `plot()` method. For batch outlier tests this collects the per-sample
        metric series and passes `sample_values` + `metric_name` to the tester.
        """
        if dataloader is None:
            raise ValueError("Dataloader is required to generate figures for gate node QC tests")
        dataloader_context = dataloader_context or {}
        entity = self.load_entity(dataloader, entity_qc.entity_id)

        # Parse the test_key once up-front (supports QCTestRecord or mapping/tuple)
        if isinstance(test_key, QCTestRecord):
            tester_class, test_key_dict = self._parse_test_key(test_key.id, entity=entity)
        else:
            tester_class, test_key_dict = self._parse_test_key(test_key, entity=entity)

        # Find stored QCTestRecord (prefer provided record)
        tester: QCTester | type[QCTester]
        if isinstance(test_key, QCTestRecord):
            test = test_key
            tester = tester_class.from_dict(test)
        else:
            test: QCTestRecord | None = None
            if test_key_dict["test_type"] == "gatenode":
                sid: str = test_key_dict["sample_id"]
                try:
                    qc = entity_qc.sample_qc[sid]
                except KeyError:
                    raise KeyError(f"Sample {sid} not found in "
                                   f"QC status for entity {entity_qc.entity_id}")
            elif test_key_dict["test_type"] == "gatenode_batch_outlier":
                qc = entity_qc.batch_qc
            else:
                raise ValueError(f"Unsupported test_type '{test_key_dict['test_type']}'"
                                 " for gate node QC figure generation")

            test_key_tuple = tuple(tester_class.make_key(test_key_dict).values())
            if step_id is not None:
                step = qc.steps[step_id]
                test = step.tests[test_key_tuple]
            else:
                for step in qc.steps.values():
                    if test_key_tuple in step.tests:
                        test = step.tests[test_key_tuple]
                        break

            if test is None:
                raise KeyError(f"Test {test_key_tuple} not found in "
                               f"QC status for entity {entity_qc.entity_id}")

            tester = tester_class.from_dict(test)

        if isinstance(tester, GateMaskRatioTest):
            sample_id = test_key_dict["sample_id"]
            return self._plot_gate_with_ratio(
                gate_node=entity,
                dataloader=dataloader,
                sample_id=sample_id,
                test=test,
                **kwargs
            )

        if isinstance(tester, _ScalarOutlierTester):
            metric_name = test_key_dict["metric_name"]
            metric_type = test_key_dict.get("metric_type")
            sample_values = self._collect_batch_sample_values(
                entity_qc=entity_qc,
                test=test,
                gate_node=entity,
                dataloader=dataloader
            )
            # Special-case: for geometry batch tests allow returning a gated heatmap
            if metric_type == "geometry" and kwargs.get("plot_gate"):
                gs = GateSpaceGeometry.load(entity_qc=entity_qc, dataloader=dataloader, entity=entity)
                if gs is None:
                    raise ValueError(f"Gate space geometry artifact not found for entity {entity_qc.entity_id}")
                title = f"Gate-space mean mask: {entity.id}"
                return gs.plot(title=title, overlay_gate=True, gate_sample_id=test_key_dict["sample_id"], output_path=kwargs.get("output_path"))

            return tester.plot(test=test, sample_values=sample_values, metric_name=metric_name)

        raise NotImplementedError("Figure generation for non-batch tests is not implemented yet")

    def _collect_batch_sample_values(
        self,
        entity_qc: EntityQCStatus,
        test: QCTestRecord,
        gate_node: GateNode,
        dataloader: UnifiedDataLoader,
    ) -> dict[str, float]:
        """Collect per-sample scalar values corresponding to a batch outlier test.

        Attempts to reuse existing collectors:
        - `mask` metrics via `_collect_mask_metrics`
        - `params` / `diagnostics` via `_collect_sample_params` (requires `dataloader` to load gate node)

        Falls back to scanning per-sample tests if the specialized collector can't be used.
        """

        # Get metric identifiers from test metadata/targets
        try:
            metric_name = test.id["metric_name"]
            metric_type = test.id["metric_type"]
        except (KeyError, TypeError):
            raise ValueError("Invalid test key: missing 'metric_name' or 'metric_type' in test metadata")

        # Use mask collector when available
        if metric_type == "geometry":
            gs = GateSpaceGeometry.load(entity_qc=entity_qc, dataloader=dataloader, entity=gate_node)
            if gs is None:
                raise ValueError(f"Gate space geometry artifact not found for entity {entity_qc.entity_id}")
            return gs.centrality_by_sample()

        if metric_type == "masks":
            metrics_by_name = self._collect_mask_metrics(entity_qc=entity_qc, gate_id=entity_qc.entity_id)
        elif metric_type in {"params", "diagnostics"}:
            params, diags = self._collect_sample_params(gate_node, sample_ids=list(entity_qc.sample_qc.keys()))
            metrics_by_name = params if metric_type == "params" else diags
        else:
            raise ValueError(f"Unsupported metric_type '{metric_type}' for batch outlier test")

        series = metrics_by_name[metric_name]
        sample_ids = list(entity_qc.sample_qc.keys())
        return dict(zip(sample_ids, series))
