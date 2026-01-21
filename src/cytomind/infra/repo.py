from __future__ import annotations
from typing import Any, Iterable, Iterator, Sequence, Mapping, Literal
from numpy.typing import NDArray

from pathlib import Path
from shutil import rmtree

import json
import warnings

import numpy as np
import pandas as pd
import anndata as ad

from cytomind.domain.flow import CompensationRef, ChannelRef, DimensionDef, TransformationRef
from cytomind.domain.pipeline import Project, SampleRef, StepRun, BatchRef
from cytomind.domain.transforms import get_default_transformations

PathLike = Path | str
MaskLike = NDArray[np.bool_] | NDArray[np.int_] | slice

class ProjectRepository:
    """A repository for loading/saving project data."""

    @classmethod
    def init_new_project(cls, root: PathLike, name: str | None = None) -> "ProjectRepository":
        """
        Create a minimal project structure on disk.

        Parameters
        ----------
        root : PathLike
            Filesystem path where the project will be created.
        name : str | None
            Optional project name; if None the directory name is used.

        Returns
        -------
        ProjectRepository
            Initialized repository instance pointing at the new project root.
        """
        root = Path(root)
        repo = cls(root)

        # Create an empty project record
        project = Project(
            id=name or root.name,
            samples={},
            panel=[],
            compensations={},
            transformations=get_default_transformations(),
        )
        repo.save_project(project, deep_copy=True)
        return repo

    def __init__(self, root: PathLike, name: str | None = None):
        """
        Initialize repository for an existing or new project root.

        If the project does not exist on disk a minimal project.json is created.

        Parameters
        ----------
        root : PathLike
            Path to the project directory.
        name : str | None
            Optional project name used when creating a new project. Ignored when
            loading an existing project.
        """
        self.root = Path(root)
        if not self.project_config_path.exists():
            # First time init: create project with given name
            self.root.mkdir(parents=True, exist_ok=True)
            project = Project(
                id=name or self.root.name,
                samples={},
                panel=[],
                compensations={},
                transformations=get_default_transformations(),
            )
            self.save_project(project, deep_copy=True)
        elif name is not None:
            warnings.warn("Project name is ignored when loading existing project from disk.")


        # Initialize step counter
        self._step_counter = 0
        if self.steps_dir.exists():
            for step_dir in self.steps_dir.iterdir():
                if step_dir.is_dir():
                    self._step_counter += 1

    # ---------- Directory paths ----------

    @property
    def transformations_path(self) -> Path:
        """
        Path to the transformations JSON file.

        Returns
        -------
        Path
            Absolute path to 'transformations.json' inside the project root.
        """
        return self.root / "transformations.json"

    # -------------- Project I/O -------------

    @property
    def project_config_path(self) -> Path:
        """
        Path to the main project configuration file.

        Returns
        -------
        Path
            Absolute path to 'project.json' inside the project root.
        """
        return self.root / "project.json"

    def load_project(self) -> Project:
        """
        Load the project metadata from disk.

        Reads project.json and reconstructs CompensationRef spill paths.

        Returns
        -------
        Project
            Deserialized Project domain object.

        Raises
        ------
        FileNotFoundError
            If the project configuration file does not exist.
        """
        proj_meta = self._read_json(self.project_config_path)
        return Project.from_dict(proj_meta)

    def save_project(self, project: Project, deep_copy: bool = True) -> None:
        """
        Persist project metadata and optionally write related artifacts.

        Parameters
        ----------
        project : Project
            Project domain object to persist.
        deep_copy : bool
            If True, also write per-sample metadata, batches, panel, compensations,
            dimensions and transformations to disk.
        """
        self._write_json(self.project_config_path, project.to_dict())
        if deep_copy:
            for sample in project.samples.values():
                self._write_sample_meta(sample)
            for batch in project.batches.values():
                self._write_batch_meta(batch)
            self._save_panel(project.panel)
            self._update_comp_catalog(project.compensations.values())
            self._write_dimensions(project.dimensions)
            self._write_transformations(project.transformations)

    def update_project_metadata(
        self,
        samples: Mapping[str, SampleRef] = {},
        panel: Sequence[ChannelRef] = [],
        dimensions: Mapping[str, list[DimensionDef]] = {},
        compensations: Iterable[CompensationRef] = [],
        transformations: Mapping[str, TransformationRef] = {},
        batches: Mapping[str, BatchRef] = {},
        drop_samples: Sequence[str] = [],
        drop_batches: Sequence[str] = [],
    ) -> None:
        """
        Merge and persist updates to the project's registries.

        Only non-empty keyword arguments will be merged into the on-disk project metadata.

        Parameters
        ----------
        samples : Mapping[str, SampleRef], optional
            Mapping of sample_id -> SampleRef to add/update.
        panel : Sequence[ChannelRef], optional
            Channel definitions for the project's panel.
        dimensions : Mapping[str, list[DimensionDef]], optional
            Data layer dimension definitions to add/update.
        compensations : Iterable[CompensationRef], optional
            Compensation references to add/update.
        transformations : Mapping[str, TransformationRef], optional
            Transformation references to add/update.
        batches : Mapping[str, BatchRef], optional
            Batch references to add/update.
        drop_samples : Sequence[str], optional
            List of sample IDs to remove from the project. This will delete sample
            directories and remove references from batches.
        drop_batches : Sequence[str], optional
            List of batch IDs to remove from the project. This will delete batch directories.
        """
        project = self.load_project()
        update_needed = False

        # Handle sample deletions first
        drop_samples = list(drop_samples)
        if drop_samples:
            for sample_id in drop_samples:
                # Remove sample directory and all its contents
                sample_dir = self.sample_path(sample_id)
                if sample_dir.exists():
                    rmtree(sample_dir)

                # Remove from project samples dict
                project.samples.pop(sample_id, None)

            # Remove samples from batches
            drop_samples_set = set(drop_samples)
            for batch in project.batches.values():
                original_count = len(batch.sample_ids)
                batch.sample_ids = [sid for sid in batch.sample_ids if sid not in drop_samples_set]
                if len(batch.sample_ids) < original_count:
                    self._write_batch_meta(batch)

            update_needed = True

        # Handle sample additions/updates
        if samples:
            for sample in samples.values():
                self._write_sample_meta(sample)
            project.samples.update(samples)
            update_needed = True

        panel = list(panel)
        if panel:
            project.panel = panel
            self._save_panel(panel)
            update_needed = True
        if compensations:
            project.compensations = self._update_comp_catalog(compensations)
            update_needed = True
        if dimensions:
            project.dimensions.update(dimensions)
            update_needed = True
            self._write_dimensions(project.dimensions)
        if transformations:
            project.transformations.update(transformations)
            update_needed = True
            self._write_transformations(project.transformations)

        # Handle batch deletions first
        drop_batches = list(drop_batches)
        if drop_batches:
            for batch_id in drop_batches:
                # Remove batch directory if it exists
                batch_dir = self.batch_path(batch_id)
                if batch_dir.exists():
                    rmtree(batch_dir)

                # Remove from project batches dict
                project.batches.pop(batch_id, None)

            update_needed = True

        # Handle batch additions/updates
        if batches:
            for batch in batches.values():
                self._write_batch_meta(batch)
            project.batches.update(batches)
            update_needed = True

        if update_needed:
            self._write_json(self.project_config_path, project.to_dict())

    # ------------- Panel I/O -----------------

    @property
    def panel_path(self) -> Path:
        """
        Path to the panel CSV file.

        Returns
        -------
        Path
            Absolute path to 'panel.csv' inside the project root.
        """
        return self.root / "panel.csv"

    def _save_panel(self, panel: Iterable[ChannelRef]) -> None:
        """
        Write channel panel to disk as CSV.

        Parameters
        ----------
        panel : Iterable[ChannelRef]
            Iterable of ChannelRef objects describing the panel.
        """
        panel_path = self.panel_path
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records([vars(ch) for ch in panel]).to_csv(panel_path, index=False)

    def _load_channelRefs(self) -> list[ChannelRef]:
        """
        Load ChannelRef objects from the panel CSV.

        Returns
        -------
        list[ChannelRef]
            Deserialized list of ChannelRef instances.
        """
        panel = pd.read_csv(self.panel_path, index_col=False)
        return [ChannelRef(**{str(k): v for k, v in ch.items()}) for ch in panel.to_dict(orient="records")]

    # ------------- Channel Mapping I/O -----------------

    @property
    def channel_mapping_path(self) -> Path:
        """
        Path to the channel mapping JSON file.

        Returns
        -------
        Path
            Absolute path to 'channel_mapping.json' inside the project root.
        """
        return self.root / "channel_mapping.json"

    def load_channel_mapping(self) -> dict[str, dict[str, dict[str, str]]]:
        """
        Load channel name mappings from disk.

        Returns structure:
        {
            "sample_id": {
                "channels": {"original_pnn": "new_pnn", ...},
                "proteins": {"original_pnn": "new_pns", ...}
            },
            ...
        }

        Returns
        -------
        dict
            Channel mapping dictionary, empty dict if file doesn't exist.
        """
        if not self.channel_mapping_path.exists():
            return {}
        return self._read_json(self.channel_mapping_path)

    def save_channel_mapping(
        self,
        mapping: dict[str, dict[str, dict[str, str]]],
    ) -> None:
        """
        Persist channel name mappings to disk.

        Parameters
        ----------
        mapping : dict
            Channel mapping structure with per-sample channel/protein renames.
            Only includes channels that change names.
        """
        self.channel_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(self.channel_mapping_path, mapping)

    # ------------- Dimension I/O -----------------

    @property
    def dimensions_path(self) -> Path:
        """
        Path to the dimensions JSON file.

        Returns
        -------
        Path
            Absolute path to 'dimensions.json' inside the project root.
        """
        return self.root / "dimensions.json"

    def load_dimensions(self) -> dict[str, list[DimensionDef]]:
        """
        Load data layer dimension definitions.

        Returns
        -------
        dict[str, list[DimensionDef]]
            Mapping of layer name -> list of DimensionDef objects.
        """
        if not self.dimensions_path.exists():
            return {}
        catalog = self._read_json(self.dimensions_path)
        return {layer: [DimensionDef.from_dict(dim) for dim in dims] for layer, dims in catalog.items()}

    def load_dimensions_df(self, layer: str) -> pd.DataFrame:
        """
        Load data layer dimensions as a DataFrame.

        Parameters
        ----------
        layer : str
            Data layer name.

        Returns
        -------
        pd.DataFrame
            DataFrame of dimension definitions for the specified layer.

        Raises
        ------
        KeyError
            If the specified layer does not exist.
        """
        catalog = self.load_dimensions()
        if layer not in catalog:
            raise KeyError(f"Data layer '{layer}' not found in dimensions catalog.")
        df = pd.DataFrame.from_records([dim.to_record() for dim in catalog[layer]])
        return df.set_index("id", drop=False).sort_values("idx")

    def _write_dimensions(self, dimensions: Mapping[str, list[DimensionDef]]) -> None:
        """
        Write the dimensions catalog to disk.

        Parameters
        ----------
        dimensions : Mapping[str, list[DimensionDef]]
            Mapping of layer -> list of DimensionDef instances or dict-like records.
        """
        self.dimensions_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.dimensions_path,
            {layer: [vars(dim) for dim in dims] for layer, dims in dimensions.items()}
        )

    def add_data_layer(self, layer: str, dimensions: Iterable[Mapping[str, Any]]) -> None:
        """
        Create a new data layer with provided dimensions.

        Parameters
        ----------
        layer : str
            Name of the new data layer.
        dimensions : Iterable[Mapping[str, Any]]
            Sequence of dimension records (dict-like) to create DimensionDef instances.

        Raises
        ------
        ValueError
            If the named layer already exists.
        """
        catalog = self.load_dimensions()
        if layer in catalog:
            raise ValueError(f"Data layer '{layer}' already exists. Call update_layer_dimensions instead.")
        dim_refs = [DimensionDef.from_dict(dim) for dim in dimensions]
        for i, dim in enumerate(dim_refs):
            dim.idx = i
        self.update_project_metadata(dimensions={layer: dim_refs})

    def update_layer_dimensions(self, layer: str, dimensions: Iterable[Mapping[str, Any]]) -> None:
        """
        Add or update dimensions for an existing data layer.

        If the layer does not exist it will be created.

        Parameters
        ----------
        layer : str
            Name of the data layer to update.
        dimensions : Iterable[Mapping[str, Any]]
            Sequence of dimension records (dict-like) to be added or updated.
        """
        catalog = self.load_dimensions()
        if layer not in catalog:
            warnings.warn(f"Data layer '{layer}' does not exist. Calling add_data_layer instead.")
            self.add_data_layer(layer, dimensions)
            return

        cur_refs = {dim.id: dim for dim in catalog[layer]}
        new_refs = [DimensionDef.from_dict(dim) for dim in dimensions]
        for dim in new_refs:
            dim.idx = cur_refs[dim.id].idx if dim.id in cur_refs else len(cur_refs)
            cur_refs[dim.id] = dim

        catalog[layer] = sorted(list(cur_refs.values()))
        self.update_project_metadata(dimensions=catalog)

    def load_transformations(self) -> dict[str, TransformationRef]:
        """
        Load transformation references from disk.

        Returns
        -------
        dict[str, TransformationRef]
            Mapping of transformation id -> TransformationRef.
        """
        if not self.transformations_path.exists():
            return {}
        catalog = self._read_json(self.transformations_path)
        return {k: TransformationRef(**v) for k, v in catalog.items()}

    def _write_transformations(self, transformations: Mapping[str, TransformationRef]) -> None:
        """
        Persist transformation references to disk.

        Parameters
        ----------
        transformations : Mapping[str, TransformationRef]
            Mapping of id -> TransformationRef to persist.
        """
        self.transformations_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.transformations_path,
            {k: v.to_dict() for k, v in transformations.items()}
        )

    # ------------- Sample I/O -----------------

    @property
    def samples_dir(self) -> Path:
        """Absolute path to samples directory."""
        return self.root / "samples"

    def sample_path(self, sample: SampleRef | str) -> Path:
        """
        Get absolute path to a sample directory.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample.

        Returns
        -------
        Path
            Absolute path to the sample directory within the project.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample
        return self.samples_dir / sample_id

    def sample_config_path(self, sample: SampleRef | str) -> Path:
        """
        Path to a sample's metadata JSON file.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample.

        Returns
        -------
        Path
            Absolute path to '<sample_id>/sample.json'.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample
        return self.sample_path(sample_id) / "sample.json"

    def sample_adata_path(self, sample: SampleRef | str, layer: str) -> Path:
        """
        Path to a sample's AnnData file for a given layer.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample.
        layer : str
            Data layer name.

        Returns
        -------
        Path
            Absolute path to '<sample_id>/<layer>.h5ad'.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample
        return self.sample_path(sample_id) / f"{layer}.h5ad"

    def sample_relpath(self, sample: SampleRef | str) -> Path:
        """
        Relative path of the sample directory with respect to project root.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample.

        Returns
        -------
        Path
            Relative path object.
        """
        sample_id = sample.id if isinstance(sample, SampleRef) else sample
        abs_path = self.sample_path(sample_id)
        return abs_path.relative_to(self.root)

    def iter_sample_dirs(self) -> Iterator[Path]:
        """
        Yield sample directories present in the project.

        Returns
        -------
        Iterator[Path]
            Iterator over Path objects for each sample directory.
        """
        if self.samples_dir.exists():
            yield from (p for p in self.samples_dir.iterdir() if p.is_dir())
        else:
            yield from []

    def load_samples(self) -> dict[str, SampleRef]:
        """
        Load all sample metadata as SampleRef objects.

        Returns
        -------
        dict[str, SampleRef]
            Mapping of sample_id -> SampleRef.
        """
        samples: dict[str, SampleRef] = {}
        for sample_dir in self.iter_sample_dirs():
            s = self.load_sample_meta(sample_dir.name)
            samples[s.id] = s
        return samples

    def load_sample_meta(self, sample_id: str) -> SampleRef:
        """
        Load a single sample's metadata record.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample to load.

        Returns
        -------
        SampleRef
            Deserialized SampleRef instance.

        Raises
        ------
        FileNotFoundError
            If the sample metadata file does not exist.
        """
        path = self.sample_config_path(sample_id)
        config = self._read_json(path)
        config["root"] = self.sample_path(sample_id)
        return SampleRef.from_record(config)

    def _write_sample_meta(self, sample: SampleRef) -> None:
        """
        Persist sample metadata to disk.

        Parameters
        ----------
        sample : SampleRef
            Sample reference to write.
        """
        path = self.sample_config_path(sample.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, sample.to_dict())

    def _load_sample_layer(self, sample_id: str, layer: str, backed: bool | Literal["r", "r+"] = "r") -> ad.AnnData:
        """
        Load an AnnData object for a sample layer (possibly backed).

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        layer : str
            Data layer name.
        backed : bool or {'r','r+'}, optional
            Whether to open the AnnData in backed mode. Defaults to "r".

        Returns
        -------
        anndata.AnnData
            AnnData view (may be backed). Caller is responsible for converting to memory if needed.

        Raises
        ------
        FileNotFoundError
            If the requested layer file does not exist.
        """
        path = self.sample_path(sample_id) / f"{layer}.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"Sample {sample_id} layer '{layer}' data file not found at {path.as_posix()}")
        ondisk = ad.read_h5ad(path, backed=backed)
        return ondisk

    def load_sample_adata(
        self,
        sample_id: str,
        layer: str,
        mask: MaskLike = slice(None),
        select: Sequence[str] | slice = slice(None),
    ) -> ad.AnnData:
        """
        Load a sample AnnData into memory applying optional row/column selection.

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        layer : str
            Data layer name to load.
        mask : MaskLike, optional
            Boolean or integer index or slice to select rows (observations). Defaults to slice(None).
        select : Sequence[str] or slice, optional
            Sequence of var_names or slice to select columns. Defaults to slice(None).

        Returns
        -------
        anndata.AnnData
            Materialized in-memory AnnData object.
        """
        ondisk = self._load_sample_layer(sample_id, layer)
        adata = ondisk[mask, select].to_memory(copy=True)
        try:
            ondisk.file.close()
            del ondisk
        except Exception:
            pass
        return adata

    def save_sample_adata(self, sample_id: str, layer: str, adata: ad.AnnData, **kwargs) -> None:
        """
        Save an AnnData for a sample's layer.

        Handles both backed and in-memory AnnData objects.

        Parameters
        ----------
        sample_id : str
            Sample identifier.
        layer : str
            Data layer name.
        adata : anndata.AnnData
            AnnData object to persist.
        **kwargs
            Additional keyword arguments passed to AnnData.write_h5ad.
        """
        path = self.sample_path(sample_id) / f"{layer}.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        if adata.isbacked and adata.filename == path.as_posix():
            flush = adata.to_memory(copy=False)
            try:
                adata.file.close()
            except Exception:
                pass
            del adata
            flush.write_h5ad(path, **kwargs)
        else:
            adata.write_h5ad(path, **kwargs)

    def drop_samples(self, sample_ids: Sequence[str]) -> None:
        """
        Remove samples from the project.

        This method removes sample data files, sample metadata, and removes
        sample references from the project and any batches that contain them.

        Parameters
        ----------
        sample_ids : Sequence[str]
            List of sample identifiers to remove from the project.

        Raises
        ------
        ValueError
            If any sample_id does not exist in the project.
        """

        project = self.load_project()
        sample_ids_set = set(sample_ids)

        # Validate that all samples exist
        missing_samples = sample_ids_set - set(project.samples.keys())
        if missing_samples:
            raise ValueError(f"Sample(s) not found in project: {', '.join(sorted(missing_samples))}")

        # Remove samples from project
        for sample_id in sample_ids_set:
            # Remove sample directory and all its contents
            sample_dir = self.sample_path(sample_id)
            if sample_dir.exists():
                rmtree(sample_dir)

            # Remove from project samples dict
            project.samples.pop(sample_id, None)

        # Remove samples from batches
        for batch in project.batches.values():
            original_count = len(batch.sample_ids)
            batch.sample_ids = [sid for sid in batch.sample_ids if sid not in sample_ids_set]
            if len(batch.sample_ids) < original_count:
                self._write_batch_meta(batch)

        # Compensations and panel are not sample-specific, so no changes needed there.

        # Save updated project metadata
        self.save_project(project, deep_copy=False)


    # ------------- Batch I/O -----------------

    @property
    def batch_dir(self) -> Path:
        """
        Path to batches directory.

        Returns
        -------
        Path
            Absolute path to the project's 'batches' directory.
        """
        return self.root / "batches"

    def batch_path(self, batch_id: str) -> Path:
        """
        Get path to a batch directory.

        Parameters
        ----------
        batch_id : str
            Batch identifier.

        Returns
        -------
        Path
            Absolute path to the batch directory.
        """
        return self.batch_dir / batch_id

    def batch_config_path(self, batch_id: str) -> Path:
        """
        Path to a batch's metadata JSON file.

        Parameters
        ----------
        batch_id : str
            Batch identifier.

        Returns
        -------
        Path
            Absolute path to '<batch_id>/batch.json'.
        """
        return self.batch_path(batch_id) / "batch.json"

    def _load_batches(self) -> dict[str, BatchRef]:
        """
        Load all batch metadata present on disk.

        Returns
        -------
        dict[str, BatchRef]
            Mapping of batch_id -> BatchRef.
        """
        batches: dict[str, BatchRef] = {}
        if not self.batch_dir.exists():
            return batches
        for batch_dir in self.batch_dir.iterdir():
            if not batch_dir.is_dir():
                continue
            b = self.load_batch_meta(batch_dir.name)
            batches[b.id] = b
        return batches

    def load_batch_meta(self, batch_id: str) -> BatchRef:
        """
        Load a batch's metadata record.

        Parameters
        ----------
        batch_id : str
            Batch identifier.

        Returns
        -------
        BatchRef
            Deserialized BatchRef instance.
        """
        path = self.batch_config_path(batch_id)
        config = self._read_json(path)
        config["root"] = self.batch_path(batch_id)
        return BatchRef.from_record(config)

    def _write_batch_meta(self, batch: BatchRef) -> None:
        """
        Persist batch metadata to disk.

        Parameters
        ----------
        batch : BatchRef
            Batch reference to persist.
        """
        path = self.batch_config_path(batch.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, batch.to_dict())

    def build_batch_adata(
        self,
        batch,                       # BatchRef
        layer: str,
        n_events_per_sample: int = 10_000,
        channels: Sequence[str] | None = None,
        mask_dict: Mapping[str, NDArray[np.bool_]] = {},
        seed: int | None = None,
    ) -> ad.AnnData:
        """
        Build an AnnData pooling events from all samples in `batch`,
        with up to `n_events_per_sample` events per sample.

        Parameters
        ----------
        batch : BatchRef
            Batch whose sample_ids will be pooled.
        n_events_per_sample : int
            Max number of events drawn from each sample.
        channels : list of str, optional
            Channel names (var_names) to keep. If None, keep all channels.
        mask_dict : Mapping[str, MaskLike], optional
            Mapping from sample_id to a boolean mask (np.ndarray) of length n_obs indicating which events to keep.
            This is applied *before* downsampling.
        layer : str
            Use this layer as X in the pooled AnnData.
        seed : int or None
            Random seed for reproducible subsampling.

        Returns
        -------
        batch_adata : AnnData
            Concatenated AnnData with events from all samples.
            Has a column `sample_id` in .obs.
        """
        rng = np.random.default_rng(seed)

        per_sample_adatas: list[ad.AnnData] = []

        for sample_id in batch.sample_ids:
            # 1) Load sample AnnData (backed or in-memory)
            sample_view = self._load_sample_layer(sample_id, layer=layer)

            mask = np.asarray(mask_dict.get(sample_id, np.ones(sample_view.n_obs, dtype=bool)))
            mask = np.where(mask)[0]
            n_events = len(mask)
            # If nothing left after filtering, skip this sample
            if n_events == 0:
                continue

            # 4) downsample to n_events_per_sample
            if n_events > n_events_per_sample:
                mask = rng.choice(mask, size=n_events_per_sample, replace=False)
                n_events = n_events_per_sample

            # This copy materializes the subset into memory (small)
            sample_sub = sample_view[mask, channels].to_memory(copy=True)
            try:
                sample_view.file.close()
                del sample_view
            except Exception:
                pass

            # 6) annotate sample_id in obs for provenance
            sample_sub.obs["sample_id"] = sample_id

            per_sample_adatas.append(sample_sub)

        if not per_sample_adatas:
            raise ValueError("No events collected for this batch (all empty after filtering?)")

        # 7) concatenate all subsampled adatas
        batch_adata = ad.concat(
            per_sample_adatas,
            axis=0,
            join="inner",      # keep only channels common to all samples
            label="sample_id", # also use sample_id as key label
            keys=[a.obs["sample_id"].iloc[0] for a in per_sample_adatas],
            index_unique="-",  # "<sample_id>-<event_index>"
        )

        return batch_adata

    # --------- Compensation I/O ----------

    @property
    def compensation_dir(self) -> Path:
        """
        Path to compensations directory.

        Returns
        -------
        Path
            Absolute path to the project's 'compensations' directory.
        """
        return self.root / "compensations"

    @property
    def comp_catalog_path(self) -> Path:
        """
        Path to the compensation catalog JSON file.

        Returns
        -------
        Path
            Absolute path to 'compensations/catalog.json'.
        """
        return self.compensation_dir / "catalog.json"

    def get_comp_catalog(self) -> dict[str, CompensationRef]:
        """
        Load the compensation catalog and attach spillover file paths.

        Returns
        -------
        dict[str, CompensationRef]
            Mapping comp_id -> CompensationRef.
        """
        if not self.comp_catalog_path.exists():
            return {}
        catalog = self._read_json(self.comp_catalog_path)
        return {comp_id: CompensationRef.from_record(config) for comp_id, config in catalog.items()}

    @property
    def spillover_dir(self) -> Path:
        """
        Path to compensation spillover matrices directory.

        Returns
        -------
        Path
            Absolute path to 'compensations/matrices'.
        """
        return self.compensation_dir / "matrices"

    def spillover_path(self, comp: CompensationRef | str) -> Path:
        """
        Path to a compensation spillover CSV for a given compensation id.

        Parameters
        ----------
        comp_id : str
            Compensation identifier.

        Returns
        -------
        Path
            Absolute path to 'compensations/matrices/<comp_id>.csv'.
        """

        comp_id = comp.id if isinstance(comp, CompensationRef) else comp
        return self.spillover_dir / f"{comp_id}.csv"

    def get_spill_df(self, comp: CompensationRef | str) -> pd.DataFrame:
        """
        Retrieve a compensation matrix DataFrame.

        Parameters
        ----------
        comp : CompensationRef or str
            CompensationRef instance or comp_id string.

        Returns
        -------
        pandas.DataFrame
            DataFrame representing the spillover/compensation matrix.

        Raises
        ------
        FileNotFoundError
            If the referenced spillover CSV does not exist when comp is a string id.
        """
        if isinstance(comp, str):
            path = self.spillover_path(comp)
            if not path.exists():
                raise FileNotFoundError(f"Compensation spillover file not found: {path.as_posix()}")
            df = pd.read_csv(path, index_col=False)
        elif isinstance(comp, CompensationRef):
            df = comp.spill.copy()
        else:
            raise ValueError(f"Unsupported comp type: {type(comp)}")
        df.index = df.columns
        return df

    def _update_comp_catalog(self, comp_refs: Iterable[CompensationRef]) -> dict[str, CompensationRef]:
        """
        Merge and persist compensation catalog entries.

        Parameters
        ----------
        comp_refs : Iterable[CompensationRef]
            New or updated compensation references to merge into the catalog.
        """
        self.compensation_dir.mkdir(parents=True, exist_ok=True)
        catalog = self.get_comp_catalog()
        comps: dict[str, CompensationRef] = {}
        for ref in comp_refs:
            if not isinstance(ref, CompensationRef):
                raise TypeError("comps values must be CompensationRef instances.")
            path = self.spillover_path(ref.id)
            if not path.exists():
                ref = self._save_compensation_ref(ref)
            ref.path = path.as_posix()  # Ensure path is set correctly
            comps[ref.id] = ref
        catalog.update(comps)
        self._write_json(
            self.comp_catalog_path,
            {comp_id: comp.to_dict() for comp_id, comp in catalog.items()}
        )
        return catalog

    def add_compensation(
        self,
        spillover: pd.DataFrame,
        name: str | None = None,
        source: str | None = None,
        batch: list[str] = [],
    ) -> str:
        """
        Write a single compensation spillover CSV and update the catalog.

        Parameters
        ----------
        spillover : pd.DataFrame
            Spillover/compensation matrix DataFrame.
        name : str | None
            Optional human-readable name for the compensation.
        source : str | None
            Optional source for the compensation.
        batch : list[str]
            Optional list of batch identifiers for the compensation.

        Raises
        ------
        ValueError
            If the spillover DataFrame is invalid.

        Returns
        -------
        str
            Identifier of the newly created CompensationRef.
        """
        if not isinstance(spillover, pd.DataFrame):
            raise TypeError("Spillover must be a pandas DataFrame.")

        ref = CompensationRef.from_dataframe(df = spillover.copy(), name=name, source=source, batch=batch)
        self._add_compensation_refs([ref])
        return ref.id

    def _save_compensation_ref(self, comp_ref: CompensationRef) -> CompensationRef:
        """
        Write a single compensation spillover CSV and update the catalog.

        Parameters
        ----------
        comp_ref : CompensationRef
            CompensationRef instance with attached spillover matrix.

        Raises
        ------
        ValueError
            If the CompensationRef is missing an attached spillover matrix.
        """
        if not isinstance(comp_ref, CompensationRef):
            raise TypeError("comp_ref must be a CompensationRef instance.")
        path = self.spillover_path(comp_ref.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        comp_ref.spill.to_csv(path, index=False)
        comp_ref.path = path.as_posix()
        return comp_ref

    def _add_compensation_refs(self, comp_refs: Iterable[CompensationRef]) -> None:
        """
        Write multiple compensation spillover CSVs and update the catalog.

        Parameters
        ----------
        comp_refs : Iterable[CompensationRef]
            Iterable of CompensationRef instances with attached spillover matrices.

        Raises
        ------
        ValueError
            If any CompensationRef is missing an attached spillover matrix.
        """
        comps: list[CompensationRef] = []
        for comp_ref in comp_refs:
            saved_ref = self._save_compensation_ref(comp_ref)
            comps.append(saved_ref)
        self._update_comp_catalog(comps)


    def update_compensations(self, comp_refs: Iterable[CompensationRef]) -> None:
        """
        Write compensation spillover CSVs and update the catalog.

        Parameters
        ----------
        comp_refs : Iterable[CompensationRef]
            Iterable of CompensationRef instances with attached spillover matrices.

        Raises
        ------
        ValueError
            If any CompensationRef is missing an attached spillover matrix.
        """
        comps: list[CompensationRef] = []
        self.spillover_dir.mkdir(parents=True, exist_ok=True)
        for ref in comp_refs:
            path = self.spillover_path(ref.id)
            ref.spill.to_csv(path, index=False)
            ref.path = path.as_posix()
            comps.append(ref)
        self._update_comp_catalog(comps)

    # ---------- Pipeline I/O ----------

    @property
    def step_counter(self) -> int:
        """
        Number of saved step runs.

        Returns
        -------
        int
            Integer step counter used to generate new step IDs.
        """
        return self._step_counter

    @property
    def steps_dir(self) -> Path:
        """
        Path to steps directory.

        Returns
        -------
        Path
            Absolute path to the project's 'steps' directory.
        """
        return self.root / "steps"

    def step_dir(self, step_id: str) -> Path:
        """
        Path to a specific step run directory.

        Parameters
        ----------
        step_id : str
            Step run identifier.

        Returns
        -------
        Path
            Absolute path to the step directory.
        """
        return self.steps_dir / step_id

    def revisions_dir(self, step_id: str) -> Path:
        """
        Path to the revisions directory for a specific step run.

        Parameters
        ----------
        step_id : str
            Step run identifier.

        Returns
        -------
        Path
            Absolute path to the step revisions directory.
        """
        return self.step_dir(step_id) / "revisions"

    def save_step_run(self, step: StepRun) -> None:
        """
        Persist a StepRun record to disk.

        Creates a directory for the step id and writes a JSON file.

        Parameters
        ----------
        step : StepRun
            StepRun domain object to persist.
        """
        step_dir = self.step_dir(step.id)
        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / f"{step.id}_{step.step_type}.json"
        self._write_json(path, step.to_dict())
        self._step_counter += 1

    def load_step_run(self, step_run_id: str) -> StepRun:
        """
        Load a StepRun record from disk.

        Parameters
        ----------
        step_run_id : str
            Step run identifier (e.g., "step_0001").

        Returns
        -------
        StepRun
            Loaded StepRun domain object.

        Raises
        ------
        FileNotFoundError
            If the step run directory or JSON file does not exist.
        ValueError
            If multiple or no JSON files found in step directory.
        """
        step_dir = self.step_dir(step_run_id)
        if not step_dir.exists():
            raise FileNotFoundError(f"Step run directory not found: {step_dir}")

        # Find JSON file in step directory
        json_files = list(step_dir.glob(f"{step_run_id}_*.json"))
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON file found for step run: {step_run_id}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found for step run: {step_run_id}")

        data = self._read_json(json_files[0])
        return StepRun(
            id=data["id"],
            step_type=data["step_type"],
            config=data.get("config", {}),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            per_sample_qc=data.get("per_sample_qc", {}),
            qc_summary=data.get("qc_summary", {}),
            project_updates=set(data.get("project_updates", [])),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", "")
        )

    def list_step_run_ids(self) -> list[str]:
        """
        List all step run IDs in the project.

        Returns
        -------
        list[str]
            List of step run identifiers sorted by name.
        """
        if not self.steps_dir.exists():
            return []

        step_ids = []
        for step_dir in self.steps_dir.iterdir():
            if step_dir.is_dir():
                step_ids.append(step_dir.name)

        return sorted(step_ids)

    def list_step_runs(self, step_type: str | None = None) -> list[StepRun]:
        """
        List all step runs in the project, optionally filtered by step type.

        Parameters
        ----------
        step_type : str | None
            Optional step type filter (e.g., "compensate", "load_fcs").
            If None, returns all step runs.

        Returns
        -------
        list[StepRun]
            List of StepRun objects sorted by step_run_id.
        """
        step_runs = []
        for step_id in self.list_step_run_ids():
            try:
                step_run = self.load_step_run(step_id)
                if step_type is None or step_run.step_type == step_type:
                    step_runs.append(step_run)
            except (FileNotFoundError, ValueError, KeyError) as e:
                warnings.warn(f"Failed to load step run {step_id}: {e}")
                continue

        return step_runs

    # ---------- tiny JSON helpers ----------

    @staticmethod
    def _read_json(path: Path) -> dict:
        """
        Read and return JSON content from a file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Returns
        -------
        dict
            Parsed JSON object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file contents are not valid JSON.
        """
        with path.open() as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        """
        Write a JSON-serializable dictionary to a file.

        Parameters
        ----------
        path : Path
            Destination file path.
        data : dict
            Data to serialize to JSON.
        """
        with path.open("w") as f:
            json.dump(data, f, indent=2)