from __future__ import annotations
from typing import Iterable, Sequence, Mapping, Any, TYPE_CHECKING

from cytomind.revisions.base import BaseRevisionHandler

if TYPE_CHECKING:
    from cytomind.domain.flow import DimensionDef
    from cytomind.domain.pipeline import RevisionSession
else:
    DimensionDef = object
    RevisionSession = object

from pathlib import Path

from cytomind.domain.pipeline import StepRun, BatchRef, SampleRef
from .repo import ProjectRepository
from cytomind.steps import StepRegistry
from cytomind.qc import QCEvaluatorRegistry
from cytomind.revisions import RevisionHandlerRegistry
from cytomind.utils import now_iso

PathLike = Path | str


class InteractivePipeline:
    def __init__(self, project_root: PathLike, name: str | None = None):
        """
        Initialize an InteractivePipeline with step, QC, and revision components.

        Parameters
        ----------
        project_root : PathLike
            Path to the project root directory.
        name : str | None
            Optional project name.
        """
        self.repo = ProjectRepository(project_root, name=name)
        self._active_revision_handlers: dict[str, BaseRevisionHandler] = {}

    # ---- Step lifecycle ----

    def run_step(self, step_type: str, config: dict, inputs: dict) -> StepRun:
        """
        Execute complete step lifecycle: compute → evaluate QC → persist.

        Parameters
        ----------
        step_type : str
            The registered step type name.
        config : dict
            Configuration dictionary passed to the step implementation.
        inputs : dict
            Inputs dictionary passed to the step implementation.

        Returns
        -------
        StepRun
            Completed step run with outputs and detailed QC.

        Raises
        ------
        ValueError
            If the given step_type is not registered.
        """
        # 1. Create step run
        step_run = StepRun(
            id=self._next_step_id(),
            step_type=step_type,
            config=config,
            inputs=inputs,
            outputs={},
            qc_summary={},
            per_sample_qc={},
            project_updates=set(),
            status="pending",
            created_at=now_iso(),
        )

        step_run = self._execute_step(step_run)
        return step_run

    def _execute_step(self, step_run: StepRun) -> StepRun:
        """Execute step computation (computation phase only)."""
        if step_run.status != "pending":
            raise ValueError(
                f"Cannot run step {step_run.id!r} with status {step_run.status!r}, must be 'pending'."
            )

        step_class = StepRegistry.get(step_run.step_type)
        if not step_class:
            known_steps = ", ".join(repr(k) for k in StepRegistry.keys())
            raise ValueError(f"Unknown step type {step_run.step_type!r}, known steps are: {known_steps}")

        step_run = step_class(repo=self.repo).run(step_run)
        self.repo.save_step_run(step_run)

        # 3. QC Evaluation phase (if evaluator available)
        qc_evaluator_class = QCEvaluatorRegistry.get(step_run.step_type)
        if qc_evaluator_class:
            qc_evaluator = qc_evaluator_class()
            step_run = qc_evaluator.run_step_qc(self.repo, step_run)
            self.repo.save_step_run(step_run)
        return step_run

    # ---- Review & Revision ----

    def review_step(self, step_run_id: str) -> StepRun:
        """
        Get review summary for user inspection.

        Parameters
        ----------
        step_run_id : str
            Step to review (e.g., "step_0003")

        Returns
        -------
        StepRun
            User-facing summary with tables, metrics, recommendations
        """
        step_run = self.repo.load_step_run(step_run_id)
        qc_evaluator_class = QCEvaluatorRegistry.get(step_run.step_type)

        if qc_evaluator_class:
            qc_evaluator = qc_evaluator_class()
            return qc_evaluator.run_step_qc(self.repo, step_run)

        # Fallback if no evaluator
        return step_run

    def start_revision(self, step_run_id: str, input_spec: dict[str, Any]) -> BaseRevisionHandler:
        """
        Initialize revision workspace for iterative refinement.

        Parameters
        ----------
        step_run_id : str
            Step to revise (e.g., "step_0003")
        input_spec : dict
            Specification of inputs for revision handler

        Returns
        -------
        BaseRevisionHandler
            Initialized handler for managing the revision session
        """
        step_run = self.repo.load_step_run(step_run_id)

        # Get handler and QC evaluator
        revision_handler_class = RevisionHandlerRegistry.get(step_run.step_type)
        qc_evaluator_class = QCEvaluatorRegistry.get(step_run.step_type)

        if not revision_handler_class:
            raise ValueError(f"No revision handler for step type '{step_run.step_type}'")

        # Generate session ID
        rev_dir = self.repo.revisions_dir(step_run.id)
        num_rev = len(list(rev_dir.iterdir())) + 1 if rev_dir.exists() else 1
        session_id = f"rev_{num_rev:03d}"
        workspace = rev_dir / session_id

        # Instantiate evaluator and handler
        qc_evaluator = qc_evaluator_class() if qc_evaluator_class else None
        handler = revision_handler_class(
            step_run=step_run,
            main_repo=self.repo,
            workspace=workspace,
            session=session_id,
            qc_evaluator=qc_evaluator)

        # Initialize session
        session = handler.start_revision(input_spec)

        # Store handler for later use
        self._active_revision_handlers[session.id] = handler

        return handler

    def commit_revision(self, handler: BaseRevisionHandler) -> StepRun | None:
        """
        Finalize revision and optionally execute new step.

        Parameters
        ----------
        handler : BaseRevisionHandler
            The revision handler managing the session

        Returns
        -------
        StepRun | None
            New step run if handler returned a new step to execute, else None
        """
        session = handler.session
        if session is None:
            raise RuntimeError("Session not initialized")

        # Get metadata updates and optional new step
        metadata_updates, new_step = handler._commit()

        # Apply metadata to main project
        self.repo.update_project_metadata(**metadata_updates)

        # Clean up
        handler.cleanup_workspace()
        if session.id in self._active_revision_handlers:
            del self._active_revision_handlers[session.id]

        # If handler produced a new step, run it
        if new_step:
            new_step = self._execute_step(new_step)

        return new_step

    # ---- Helpers ----

    def _next_step_id(self) -> str:
        """Generate the next sequential step identifier."""
        n_steps = self.repo.step_counter + 1
        return f"step_{n_steps:04d}"

    # --- Convenience Methods for Common Operations ----

    def add_samples(self, samples: Mapping[str, PathLike], config: dict[str, Any] = {}, channel_mapping: dict[str, dict] | None = None) -> StepRun:
        """
        Initialize a new project by parsing FCS files and building registries.

        Parameters
        ----------
        samples : Mapping[str, PathLike]
            Mapping from sample_id to FCS file path.
        config : dict[str, Any]
            Additional configuration for the add_samples step.
        channel_mapping : dict[str, dict] | None
            Optional channel mapping data (from parse_channel_json_filtered).
            Maps FCS filenames to drop/rename information.

        Returns
        -------
        StepRun
            The completed add_samples step run.
        """

        # The step will create SampleRef instances internally, but we need placeholders for BaseStep.run
        # Temporarily register minimal samples in project so run can iterate
        samples_dict = {sid: SampleRef(id=sid, fcs=Path(fcs).as_posix()) for sid, fcs in samples.items()}
        batch_dict = {
            "summary": BatchRef(
                id="summary",
                sample_ids=list(samples_dict.keys()),
                tags=["all_samples"],
                meta={},
            )
        }
        self.repo.update_project_metadata(samples=samples_dict, batches=batch_dict)

        # Add channel_mapping to config if provided
        step_config = dict(config)
        if channel_mapping is not None:
            step_config["channel_mapping"] = channel_mapping

        return self.run_step(
            step_type="add_samples",
            config=step_config,
            inputs={"sample_ids": list(samples_dict.keys()),
                    "batch_ids": ["summary"]},
        )

    def load_fcs(self, sample_ids: Sequence[str] | None = None) -> StepRun:
        """
        Load FCS files for the specified samples.

        Parameters
        ----------
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to load. If None, loads all samples.

        Returns
        -------
        StepRun
            The completed load_fcs step run.
        """
        if sample_ids is None:
            sample_ids = [p.name for p in self.repo.iter_sample_dirs()]
        return self.run_step(
            step_type="load_fcs",
            config={},
            inputs={"sample_ids": list(sample_ids)},
        )

    def compensate_samples(
        self,
        comp_id: str | Mapping[str, str],
        sample_ids: Sequence[str] | None = None
    ) -> StepRun:
        """
        Compensates the specified samples using the given compensation ID(s).

        Parameters
        ----------
        comp_id : str | Mapping[str, str]
            The compensation ID or a mapping from sample ID to compensation ID.
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to compensate. If None, compensates all samples.

        Returns
        -------
        StepRun
            The completed compensate step run.
        """
        if sample_ids is None:
            if isinstance(comp_id, Mapping):
                sample_ids = list(comp_id.keys())
            else:
                sample_ids = [p.name for p in self.repo.iter_sample_dirs()]
        sample_ids = list(sample_ids)
        step_comp = self.run_step(
            step_type="compensate",
            config={"comp_id": comp_id},
            inputs={"sample_ids": sample_ids}
        )
        return step_comp

    def add_layer(
        self,
        layer: str,
        dimensions: Iterable[Mapping[str, Any]] | None = None,
        sample_ids: Iterable[str] | None = None,
        default: bool = False
    ) -> StepRun:
        """
        Adds a new data layer to the specified samples.

        Parameters
        ----------
        layer : str
            The name of the new data layer to add.
        dimensions : Iterable[Mapping[str, Any]] | None
            A list of dimension definitions to create the new layer. If None, the layer will be created empty.
        sample_ids : Iterable[str] | None
            Optional list of sample IDs to which the new layer will be applied. If None, applies to all samples.
        default : bool
            If True, sets the new layer as the default data layer for the samples. Defaults to False.

        Returns
        -------
        StepRun
            The completed add_layer step run.
        """
        if sample_ids is None:
            sample_ids = [p.name for p in self.repo.iter_sample_dirs()]

        catalog = self.repo.load_dimensions()
        if layer in catalog and dimensions is not None:
            raise ValueError(f"Data layer {layer!r} already exists use add_dimensions instead.")

        if layer not in catalog:
            if dimensions is None:
                raise ValueError(f"Data layer {layer!r} does not exist. Provide dimensions to create it.")
            self.repo.add_data_layer(layer, dimensions=dimensions)

        return self.run_step(
            step_type="add_layer",
            config={"layer": layer, "default": default},
            inputs={"sample_ids": list(sample_ids)},
        )

    def add_dimensions(
        self,
        layer: str,
        dimensions: Sequence[Mapping[str, Any]],
        sample_ids: Sequence[str] | None = None
    ) -> StepRun:
        """
        Adds dimensions to an existing data layer.

        Parameters
        ----------
        layer : str
            The data layer to which dimensions will be added.
        dimensions : Sequence[Mapping[str, Any]]
            A list of dimension definitions to add.
        sample_ids : Sequence[str] | None
            Optional list of sample IDs to which the new dimensions will be applied. If None, applies to all samples.

        Returns
        -------
        StepRun
            The completed add_dimensions step run.
        """
        if sample_ids is None:
            sample_ids = [p.name for p in self.repo.iter_sample_dirs()]
        return self.run_step(
            step_type="add_dimensions",
            config={"layer": layer, "dimensions": dimensions},
            inputs={"sample_ids": list(sample_ids)},
        )

    def add_batch(
        self,
        batch_id: str,
        sample_ids: Iterable[str],
        tags: Iterable[str] | None = None,
        **meta,
    ) -> BatchRef:
        """
        Group existing samples into a batch for subsequent analysis.

        Parameters
        ----------
        sample_ids : Iterable[str]
            List of sample IDs to include in the batch.
        batch_id : str
            Optional batch identifier. If None, auto-generated.
        tags : Iterable[str] | None
            Optional tags to label the batch.
        **meta
            Additional metadata to store with the batch.

        Returns
        -------
        BatchRef
            The created batch reference.
        """

        project = self.repo.load_project()
        if not batch_id or batch_id in project.batches:
            raise ValueError(f"Batch ID {batch_id!r} is invalid or already exists.")

        sample_ids = list(sample_ids)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Duplicate sample IDs in batch.")
        if len(sample_ids) < 2:
            raise ValueError("At least two samples are required to create a batch.")

        batch = BatchRef(
            id=batch_id,
            sample_ids=sample_ids,
            tags=list(tags or []),
            meta=dict(meta),
        )

        # Add batch to project and persist
        project.batches[batch.id] = batch
        self.repo.update_project_metadata(batches=project.batches)
        return batch
