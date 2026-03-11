"""
Unified DataLoader for all I/O operations.

UnifiedDataLoader owns all I/O business logic and is configured with base directories.
This eliminates coupling to ProjectRepository and centralizes file operations.

Architecture:
- UnifiedDataLoader: Consolidated implementation handling both repo and workspace I/O
  - Configurable with root_dir, optional fallback_root for workspace-first pattern
  - Handles all I/O: AnnData, masks, gates, metadata with optional parsing
  - Includes visualization subset caching management

Handlers only configure loaders with paths—they don't micromanage I/O.
Visualization subset caching is managed by dataloader for test/debug workflows.
"""
from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence, TypeVar, TYPE_CHECKING
import warnings
from pathlib import Path
from shutil import rmtree
import json

import numpy as np
import anndata as ad

from cytomind.domain.pipeline import NumpyEncoder
from cytomind.domain.gates import GateNode, GatingStrategyRef
from cytomind.utils import rlencode, rldecode, now_iso

if TYPE_CHECKING:
    from cytomind.domain.constants import PathLike, MaskLike
    from numpy.typing import NDArray
    BooleanArray = NDArray[np.bool_]
    BooleanMask = BooleanArray | Sequence[bool]
    JSONSerializable = dict[str, Any] | list | str | int | float | bool | None
    R = TypeVar("R")
    HandlerDictType = Mapping[str, tuple[Callable[[Any], R], Callable[[R], dict | list] | None]]
else:
    PathLike = object
    MaskLike = object
    BooleanArray = object
    BooleanMask = BooleanArray | Sequence[bool]
    JSONSerializable = object
    HandlerDictType = object
    R = object


class UnifiedDataLoader:
    """
    Single domain-agnostic I/O class for all data access patterns.

    Consolidates path resolution logic and adds generic metadata I/O.
    Configured with simple parameters (no schema object):
    - root_dir: primary directory (repo or workspace)
    - fallback_root: optional secondary directory for fallback reads
    - save_to_root: if True, save to root_dir; else save to fallback_root

    All path resolution delegates to provided path pattern mappings.
    Handles all I/O: AnnData, masks, gates, metadata with optional parsing/serialization.
    """

    def __init__(
        self,
        path_scheme: Mapping[str, str],
        root_dir: PathLike,
        fallback: "UnifiedDataLoader" | PathLike | None = None,
        data_handlers: HandlerDictType = {},
        viz_cache_dir: PathLike | None = None,
    ):
        """
        Initialize UnifiedDataLoader with direct configuration.

        Parameters
        ----------
        path_scheme : Mapping[str, str]
            Mapping of path templates for all operations
        root_dir : PathLike
            Primary root directory (repo root or workspace)
        fallback : "UnifiedDataLoader" | PathLike | None
            Secondary read-only root for workspace-first fallback pattern.
            If None, no fallback is available.
        viz_cache_dir : PathLike | None
            Directory for visualization subset caching.
            If None, defaults to root_dir/viz_cache.
        """
        self.root_dir = Path(root_dir)
        if isinstance(fallback, UnifiedDataLoader):
            self.fallback_root = fallback.root_dir
            # Inherit and override path_scheme and data_handlers from fallback
            fallback_scheme = fallback.path_scheme.copy()
            fallback_scheme.update(path_scheme)
            self.path_scheme = fallback_scheme
            fallback_handlers = fallback.data_handlers.copy()
            fallback_handlers.update(data_handlers)
            self.data_handlers = fallback_handlers
        else:
            self.fallback_root = Path(fallback) if fallback else None
            self.path_scheme = dict(path_scheme)
            self.data_handlers = dict(data_handlers)

        # Set viz_cache_dir default
        if viz_cache_dir is None:
            viz_cache_dir = self.root_dir / "viz_cache"

        self.viz_cache_dir = Path(viz_cache_dir) if viz_cache_dir else None
        self._viz_metadata = {}
        self._viz_key_separator = ":"

        # Load viz metadata if caching is enabled
        if self.viz_cache_dir:
            self._load_viz_metadata()

    # ========== Path Resolution (Core Logic) ==========

    def _pattern(self, name: str) -> str:
        """Resolve a pattern by name from mapping, attribute, or raw string."""
        if isinstance(self.path_scheme, Mapping) and name in self.path_scheme:
            return self.path_scheme[name]
        if hasattr(self.path_scheme, name):
            return getattr(self.path_scheme, name)
        raise ValueError(f"Pattern {name!r} not found in path_scheme mapping or attributes")

    def _resolve_read_path(self, pattern: str, **kwargs) -> Path:
        """Resolve a read path with workspace-first fallback if configured."""
        relative_path = pattern.format(**kwargs)

        primary_path = self.root_dir / relative_path
        if primary_path.exists():
            return primary_path

        if self.fallback_root:
            fallback_path = self.fallback_root / relative_path
            if fallback_path.exists():
                return fallback_path

        raise FileNotFoundError(f"Path not found: {primary_path.as_posix()}")

    def _resolve_write_path(self, pattern: str, **kwargs) -> Path:
        # Always write on root_dir. Keep it as an option for future change
        relative_path = pattern.format(**kwargs)
        return self.root_dir / relative_path

    # ========== AnnData I/O ==========

    def load_adata(
        self,
        sample_id: str,
        layer: str | None = None,
        mask: MaskLike = slice(None),
        select: Sequence[str] | slice = slice(None),
        backed: bool | Literal["r", "r+"] = False,
        **context: Any,
    ) -> ad.AnnData:
        """Load AnnData with optional masking and column selection."""
        path = self._resolve_read_path(
            self._pattern("sample_adata"),
            sample_id=sample_id,
            layer=layer,
        )
        return self._load_h5ad(path, mask=mask, select=select, backed=backed)

    def save_adata(
        self,
        sample_id: str,
        layer: str,
        adata: ad.AnnData,
        overwrite: bool = True,
        **context: Any,
    ) -> None:
        """Save AnnData to resolved path."""
        path = self._resolve_write_path(
            self._pattern("sample_adata"),
            sample_id=sample_id,
            layer=layer,
        )
        self._save_h5ad(adata, path, overwrite=overwrite, **context)

    # ========== Gating Mask I/O ==========

    def load_masks(
        self,
        sample_id: str,
        gate_ids: Iterable[str] | None = None,
        parse_func: Callable[[Any], BooleanArray] = rldecode,
        **context: Any,
    ) -> dict[str, BooleanArray]:
        """Load gating masks for a sample."""
        masks = {}

        if gate_ids is None:
            gate_ids = self._get_all_gate_ids()

        for gate_id in gate_ids:
            path = self._resolve_read_path(
                self._pattern("gating_mask"),
                mask_id=gate_id,
                sample_id=sample_id,
            )
            masks[gate_id] = self._load_mask_array(path, parse_func=parse_func)

        return masks

    def save_masks(
        self,
        sample_id: str,
        masks: Mapping[str, BooleanMask],
        serialize_func: Callable[[BooleanMask], Any] = rlencode,
        overwrite: bool = True,
        **context: Any,
    ) -> None:
        """Save gating masks to resolved location."""

        for gate_id, mask in masks.items():
            path = self._resolve_write_path(
                self._pattern("gating_mask"),
                mask_id=gate_id,
                sample_id=sample_id,
            )
            self._save_mask_array(path, mask, serialize_func=serialize_func, overwrite=overwrite)

    # ========== Gate Node I/O ==========

    def load_gate_node(
        self,
        node_id: str,
        parse_func: Callable[[dict], GateNode] = GateNode.from_dict,
        **context: Any,
    ) -> GateNode:
        """
        Load gate node definition from JSON.

        Parameters
        ----------
        node_id : str
            Gate node identifier
        parse_func : Callable[[dict], GateNode], optional
            Function to parse the loaded dict to a GateNode
        **context : dict
            Optional context

        Returns
        -------
        GateNode
            Gate node definition (parsed via parse_func)
        """
        return self.load_data(
            entity="gate_node",
            parse_func=parse_func,
            node_id=node_id,
            **context
        )

    def save_gate_node(
        self,
        node: GateNode,
        serialize_func: Callable[[GateNode], dict] = GateNode.to_dict,
        overwrite: bool = True,
        **context: Any,
    ) -> None:
        """
        Save gate node definition to JSON.

        Parameters
        ----------
        node : GateNode
            Gate node to save (will be serialized if serialize_func provided)
        serialize_func : Callable[[GateNode], dict], optional
            Function to serialize node to dict. Defaults to GateNode.to_dict.
        **context : dict
            Optional context (e.g., overwrite flags)
        """
        self.save_data(
            pattern="gate_node",
            data=node,
            serialize_func=serialize_func,
            overwrite=overwrite,
            node_id=node.id,
            **context
        )

    def load_gating_strategy(
        self,
        parse_func: Callable[[dict], GatingStrategyRef] = GatingStrategyRef.from_dict,
        **context: Any
    ) -> GatingStrategyRef:
        """
        Load gating strategy definition from JSON.

        Parameters
        ----------
        parse_func : Callable[[dict], GatingStrategyRef]
            Function to parse the loaded dict to a GatingStrategyRef
        **context : dict
            Optional context

        Returns
        -------
        GatingStrategyRef
            Gating strategy definition (parsed via parse_func)
        """
        return self.load_data(
            entity="gating_strategy",
            parse_func=parse_func,
            **context
        )

    def save_gating_strategy(
        self,
        strategy: GatingStrategyRef,
        serialize_func: Callable[[GatingStrategyRef], dict] = GatingStrategyRef.to_dict,
        overwrite: bool = True,
        **context: Any
    ) -> None:
        """
        Save gating strategy definition to JSON.

        Parameters
        ----------
        strategy : GatingStrategyRef
            Gating strategy to save (will be serialized if serialize_func provided)
        serialize_func : Callable[[GatingStrategyRef], dict], optional
            Function to serialize strategy to dict. Defaults to GatingStrategyRef.to_dict.
        **context : dict
            Optional context (e.g., overwrite flags)
        """
        self.save_data(
            pattern="gating_strategy",
            data=strategy,
            serialize_func=serialize_func,
            overwrite=overwrite,
            **context
        )

    # ========== Generic Data I/O ==========

    def load_data(self, entity: str, parse_func: Callable[[Any], R] | None = None, **kwargs) -> R:
        """Load data from a pattern and optionally parse it.

        Parameters
        ----------
        entity : str
            Entity name (mapping key or attribute) or raw pattern string.
        parse_func : Callable[[dict | list], Any] | None, optional
            Function to parse the loaded dict or list. If None, returns raw dict or list (default).
        **kwargs
            Format arguments for pattern.

        Returns
        -------
        Any
            Parsed result (via parse_func) or raw dict if no parse_func.
        """
        pattern = self._pattern(entity)
        path = self._resolve_read_path(pattern, **kwargs)
        data = self._load_json(path) if path.suffix == ".json" else path  # leave it to parse_func to handle non-JSON data if needed
        if parse_func is None:
            parse_func, _ = self.data_handlers.get(entity, (lambda x: x, None))  # default to identity if no handler

        return parse_func(data) # pyright: ignore[reportArgumentType]

    def save_data(self, pattern: str, data: Any, serialize_func: Callable[[Any], dict[str, Any] | list] | None = None, overwrite: bool = True, **kwargs) -> Path:
        """Save data to a pattern, optionally serializing first.

        Parameters
        ----------
        pattern : str
            Pattern name (mapping key or attribute) or raw pattern string.
        data : Any
            Data to save (will be serialized if serialize_func provided).
        serialize_func : Callable[[Any], dict | list] | None, optional
            Function to serialize data to dict or list before saving.
            If None uses default from data_handlers or tries to transform data with to_dict(). Defaults to None.
        overwrite : bool, optional
            Whether to overwrite existing file. Defaults to True.
        **kwargs
            Format arguments for pattern.

        Returns
        -------
        Path
            Path to the saved metadata file.
        """
        actual_pattern = self._pattern(pattern)
        path = self._resolve_write_path(actual_pattern, **kwargs)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{pattern} file already exists at {path.as_posix()}. Use overwrite=True to replace.")

        if serialize_func is None:
            _, default_serialize_func = self.data_handlers.get(pattern, (None, None))
            serialize_func = default_serialize_func or self._serialize_to_container  # default to to_dict if available, else identity

        to_save = serialize_func(data)
        self._save_json(path, to_save)
        return path

    def remove_data(self, pattern: str, **kwargs) -> None:
        """
        Remove a file or directory based on a path pattern.

        Automatically detects whether the target is a file or directory
        and uses the appropriate removal method (unlink for files,
        rmtree for directories).

        Parameters
        ----------
        pattern : str
            Pattern name (mapping key or attribute) from path_scheme.
        **kwargs
            Format arguments for the pattern (e.g., sample_id, batch_id).

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist.
        """
        actual_pattern = self._pattern(pattern)
        relative_path = actual_pattern.format(**kwargs)
        path = self.root_dir / relative_path

        if not path.exists():
            # Path does not exist, nothing to remove
            return

        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    # ========== Visualization Metadata Management ==========

    @property
    def _viz_metadata_path(self) -> Path:
        """Path to viz subset metadata file."""
        if not self.viz_cache_dir:
            raise RuntimeError("Visualization caching is disabled (viz_cache_dir is None)")
        return self.viz_cache_dir / "viz_metadata.json"

    def _load_viz_metadata(self) -> None:
        """Load viz subset metadata from cache."""
        if not self.viz_cache_dir:
            return
        try:
            self._viz_metadata = json.loads(self._viz_metadata_path.read_text())
        except FileNotFoundError:
            self._viz_metadata = {}

    def _save_viz_metadata(self) -> None:
        """Persist viz subset metadata."""
        if not self.viz_cache_dir or not self._viz_metadata:
            return
        self._save_json(self._viz_metadata_path, self._viz_metadata)

    # ========== Visualization Subset Implementation ==========

    def get_or_create_viz_subset(
        self,
        sample_id: str,
        layer: str,
        n_subset: int | None = None,
        seed: int | None = None,
        n_total: int | None = None,
        **context: Any
    ) -> ad.AnnData:
        """
        Generic implementation for visualization subset creation.

        Parameters
        ----------
        sample_id : str
            Sample identifier
        layer : str
            Layer name
        n_subset : int | None, optional
            Number of events to subset
        seed : int | None, optional
            Random seed
        n_total : int | None, optional
            Total number of events (not used in this implementation)
        **context : dict
            Optional context:
            - mask_id : str, optional (default: "root")
            - mask : MaskLike, optional (default: slice(None))
            - select : Sequence[str] | slice, optional (default: slice(None))
        """

        if not self.viz_cache_dir:
            raise RuntimeError("Visualization caching is disabled")

        # Extract context parameters with defaults
        mask_id = context.get("mask_id", "root")
        mask = context.get("mask", slice(None))
        select = context.get("select", slice(None))

        if mask_id == "root" and mask != slice(None):
            raise ValueError("Cannot specify a mask when mask_id is 'root'")
        subset_key = self.make_viz_data_key(sample_id, layer, mask_id, n_subset or 0)

        # Check if already materialized
        if subset_key in self._viz_metadata:
            subset_path = self._viz_metadata[subset_key]["path"]
            return self._load_h5ad(Path(subset_path), mask=mask, select=select)

        # Ensure n_subset and seed are provided for creating new cache
        if n_subset is None:
            raise ValueError("n_subset is required when creating a new viz cache entry")
        if seed is None:
            raise ValueError("seed is required when creating a new viz cache entry")

        # Load full data keep all columns to faciliate select in future calls
        full_data = self.load_adata(sample_id, layer, mask=mask)
        n_total = full_data.n_obs

        # Subsample if needed
        if n_total > n_subset:
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_total, n_subset, replace=False)
            indices.sort()
            subset = full_data[indices, :].to_memory(copy=True)
        else:
            if n_total < n_subset:
                warnings.warn(
                    f"Requested subset size {n_subset} exceeds total events {n_total} "
                    f"for sample {sample_id}. Using all events."
                )
            subset = full_data.to_memory(copy=True)

        try:
            full_data.file.close()
            del full_data
        except Exception:
            pass

        # Save to disk for future access
        self.save_viz_subset(
            adata=subset,
            sample_id=sample_id,
            layer=layer,
            mask_id=mask_id,
            seed=seed,
            n_total=n_total,
            overwrite=False  # should not be a problem otherwise it would have returned from cache at the start of this method
        )

        return subset[:, select]

    def make_viz_data_key(self, sample_id: str, layer: str, mask_id: str, n_subset: int) -> str:
        """Helper to generate viz subset key."""
        return self._viz_key_separator.join((sample_id, layer, mask_id, str(n_subset)))

    def save_viz_subset(
        self,
        adata: ad.AnnData,
        sample_id: str,
        layer: str,
        mask_id: str,
        seed: int | None = None,
        n_total: int | None = None,
        overwrite: bool = True
    ) -> Path:
        """Save a visualization subset with a specific key."""

        if self.viz_cache_dir is None:
            raise RuntimeError("Visualization caching is disabled (viz_cache_dir is None)")

        n_subset = adata.n_obs
        key = self.make_viz_data_key(sample_id, layer, mask_id, n_subset)
        subset_path = self.viz_cache_dir / f"{key.replace(self._viz_key_separator, '_')}.h5ad"
        if subset_path.exists() and not overwrite:
            raise FileExistsError(f"Viz subset already exists at {subset_path.as_posix()}. Use overwrite=True to replace.")
        if key in self._viz_metadata and not overwrite:
            raise FileExistsError(f"Viz metadata already contains key {key}. Use overwrite=True to replace.")

        self._save_h5ad(adata, subset_path, overwrite=overwrite)
        self._viz_metadata[key] = {
            "sample_id": sample_id,
            "layer": layer,
            "mask_id": mask_id,
            "n_subset": n_subset,
            "seed": seed,
            "n_total": n_total,
            "created_at": now_iso(),
            "path": subset_path.as_posix()
        }
        self._save_viz_metadata()

        return subset_path

    def invalidate_viz_cache(
        self,
        sample_ids: Iterable[str],
        layer: str,
        **context: Any,
    ) -> None:
        """
        Invalidate cached visualization subsets.

        Call after modifying data in a layer to clear stale cached subsets.

        Parameters
        ----------
        sample_ids : Iterable[str]
            Sample IDs whose data was modified
        layer : str
            Layer that was modified
        **context : dict
            Optional additional context
        """
        if self.viz_cache_dir is None:
            return  # Nothing to invalidate if caching is disabled

        # Collect keys to remove
        sample_id_set = set(sample_ids)
        keys_to_remove = [
            key for key, metadata in self._viz_metadata.items()
            if metadata["sample_id"] in sample_id_set and metadata["layer"] == layer
        ]

        # Remove from metadata and delete files
        for key in keys_to_remove:
            metadata = self._viz_metadata.pop(key, None)
            if metadata is not None:
                path = Path(metadata["path"])
                if path.exists():
                    path.unlink()

        # Save updated metadata if changes were made
        if keys_to_remove:
            self._save_viz_metadata()

    # ========== Workspace Generation ==========

    def generate_workspace(self, session_id: str) -> Path:
        """Generate a new workspace path for an entity."""
        relative_path = self._pattern("revision_workspace_dir").format(session_id=session_id)
        workspace_path = self.root_dir / relative_path
        if workspace_path.exists():
            raise FileExistsError(f"Workspace already exists at {workspace_path.as_posix()}")
        workspace_path.mkdir(parents=True)
        return workspace_path

    def iter_workspace_revisions(self) -> Iterator[Path]:
        """Iterate over existing workspace revisions for an entity type."""
        relative_path = self._pattern("workspaces_dir")
        search_path = self.root_dir / relative_path
        if not search_path.exists():
            return
        for path in search_path.iterdir():
            if path.is_dir():
                yield path

    # ========== Low-level H5AD I/O Helpers ==========

    @staticmethod
    def _load_h5ad(
        path: PathLike,
        mask: MaskLike = slice(None),
        select: Sequence[str] | slice = slice(None),
        backed: bool | Literal["r", "r+"] = False,
    ) -> ad.AnnData:
        """Load AnnData from disk with optional subsetting and backing.

        Parameters
        ----------
        path : PathLike
            Path to the h5ad file
        mask : MaskLike
            Row selection (observations)
        select : Sequence[str] | slice
            Column selection (variables)
        backed : bool | Literal["r", "r+"]
            Backing mode. False for in-memory (default), True or "r" for read-only,
            "r+" for read-write. When backed, returns a view of the file.

        Returns
        -------
        ad.AnnData
            Loaded AnnData (in-memory or backed)
        """
        # Normalize backed parameter
        if backed is True:
            backed_mode = "r"
        elif backed is False:
            backed_mode = False
        else:
            backed_mode = backed

        # If not backed, use existing logic (load with temporary backing, then to memory)
        if not backed_mode:
            adata = ad.read_h5ad(path, backed="r")
            try:
                result = adata[mask, select].to_memory(copy=True)
            finally:
                # Ensure file is always closed, even if to_memory() fails
                if hasattr(adata, 'file') and adata.file is not None:
                    try:
                        adata.file.close()
                    except Exception as e:
                        # Log but don't fail - data was already loaded
                        warnings.warn(f"Failed to close h5ad file {path}: {e}")
                del adata

            return result

        # If backed, return backed instance (possibly with selection)
        adata = ad.read_h5ad(path, backed=backed_mode)
        if mask is not slice(None) or select is not slice(None):
            adata = adata[mask, select]

        return adata

    @staticmethod
    def _save_h5ad(adata: ad.AnnData, path: PathLike, overwrite: bool = False, **kwargs) -> None:
        """Save AnnData to disk, handling backed objects."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists at {path.as_posix()}. Use overwrite=True to replace.")
        path.parent.mkdir(parents=True, exist_ok=True)

        if adata.isbacked:
            # Convert backed to in-memory before writing
            in_memory = adata.to_memory(copy=False)
            try:
                adata.file.close()
            except Exception:
                pass
            in_memory.write_h5ad(path, **kwargs)
        else:
            adata.write_h5ad(path, **kwargs)

    # ========== Low-level Mask I/O Helpers ==========

    @staticmethod
    def _load_mask_array(path: PathLike, parse_func: Callable[[Any], BooleanArray]) -> BooleanArray:
        """Load and decode a run-length encoded mask."""

        path = Path(path)
        encoded = np.load(path, allow_pickle=True)
        return parse_func(encoded)

    @staticmethod
    def _save_mask_array(path: PathLike, mask: BooleanMask, serialize_func: Callable[[BooleanMask], Any], overwrite: bool = False) -> None:
        """Save and encode a mask with run-length encoding."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Mask already exists at {path.as_posix()}. "
                "Use overwrite=True to replace."
            )

        encoded = serialize_func(mask)
        np.save(path, encoded)

    # ========== Helper: Get All Gate IDs ==========

    def _get_all_gate_ids(self) -> Iterator[str]:
        """Get all gate IDs for a strategy (checks both root and fallback)."""
        relative_masks_dir = self._pattern("gating_strategy_masks_dir")
        root_masks_dir = self.root_dir / relative_masks_dir
        if root_masks_dir.exists():
            yield from  (d.name for d in root_masks_dir.iterdir() if d.is_dir())

        if self.fallback_root:
            fallback_masks_dir = self.fallback_root / relative_masks_dir
            if fallback_masks_dir.exists():
                yield from (d.name for d in fallback_masks_dir.iterdir() if d.is_dir())

    # ========== Helper Methods ==========

    @staticmethod
    def _load_json(path: PathLike) -> JSONSerializable:
        """Load JSON file."""
        path = Path(path)
        return json.loads(path.read_text())

    @staticmethod
    def _save_json(path: PathLike, data: JSONSerializable) -> None:
        """Save JSON file with NumpyEncoder for numpy types."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    @staticmethod
    def _serialize_to_container(data: Any) -> dict | list:
        """Default serialization function for generic data."""
        if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
            return data.to_dict()  # type: ignore
        elif isinstance(data, (dict, list)):
            return data
        else:
            raise TypeError(f"Data of type {type(data)} is not serializable by default. Provide a custom serialize_func.")
