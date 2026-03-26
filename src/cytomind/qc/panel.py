"""
Panel QC evaluator and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, TYPE_CHECKING

from Levenshtein import distance as levenshtein_distance
import numpy as np
from scipy.optimize import linear_sum_assignment

from cytomind.domain.flow import ChannelRef
from cytomind.domain.pipeline import BatchRef, Project
from cytomind.domain.qc import EntityQCStatus, QCFlag, QCTestRecord

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
    from cytomind.domain.constants import PathLike
    from cytomind.infra.dataloader import UnifiedDataLoader
else:
    Figure = object
    PathLike = object
    UnifiedDataLoader = object


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _is_substring_match(lhs: str, rhs: str) -> bool:
    return lhs in rhs or rhs in lhs


@dataclass(frozen=True)
class _PairScore:
    score: float    # normalized levenshtein similarity score between 0 and 1
    method: str     # method used to determine the match (e.g., "exact", "substring", "edit")
    eligible: bool  # whether this pair is considered an eligible match based on the scoring thresholds


def _pair_score(
    lhs: str,
    rhs: str,
    *,
    min_score_cutoff: float,
    substring_bonus: float,
    ineligible_penalty: float = 0.75,
    weights: tuple[int, int, int] = (1, 1, 1),
) -> _PairScore:

    # check validity of parameters
    if not 0 <= min_score_cutoff <= 1.0:
        raise ValueError("min_score_cutoff must be between 0 and 1")
    if not 0 <= substring_bonus <= 1.0:
        raise ValueError("substring_bonus must be between 0 and 1")
    if not 0 <= ineligible_penalty <= 1.0:
        raise ValueError("ineligible_penalty must be between 0 and 1")

    lhs_norm = _normalize_name(lhs)
    rhs_norm = _normalize_name(rhs)

    if lhs_norm == rhs_norm:
        return _PairScore(score=1.0, method="exact", eligible=True)

    max_dist = len(lhs_norm) + len(rhs_norm)
    distance_cutoff = round((1 - min_score_cutoff) * max_dist)
    min_weight = min(weights)
    norm_weights = (w / min_weight for w in weights)  # normalize weights to ensure consistent scoring
    edit_distance = levenshtein_distance(lhs_norm, rhs_norm, weights=norm_weights, score_cutoff=distance_cutoff) # pyright: ignore[reportArgumentType]
    score = 1. - edit_distance / max_dist

    is_substring = _is_substring_match(lhs_norm, rhs_norm)
    if is_substring: score = min(0.99, score + substring_bonus)

    eligible = bool(is_substring or edit_distance > distance_cutoff)
    if not eligible: score *= ineligible_penalty

    method = "substring" if is_substring else "edit"
    return _PairScore(score=float(score), method=method, eligible=eligible)



def _score_matrix(
    source_items: list[str],
    reference_items: list[str],
    *,
    max_edit_distance: float,
    substring_bonus: float,
    ineligible_penalty: float = 0.75,
    matrix_cache: dict[tuple[tuple[str, ...], tuple[str, ...], float, float], np.ndarray] | None = None,
    pair_cache: dict[tuple[str, str, float, float], _PairScore] | None = None,
) -> np.ndarray:
    if matrix_cache is None:
        matrix_cache = {}
    if pair_cache is None:
        pair_cache = {}

    key = (tuple(source_items), tuple(reference_items), max_edit_distance, substring_bonus)
    if key in matrix_cache:
        return matrix_cache[key]

    matrix = np.zeros((len(source_items), len(reference_items)), dtype=float)
    for i, src in enumerate(source_items):
        for j, ref in enumerate(reference_items):
            cache_key = (src, ref, max_edit_distance, substring_bonus)
            if cache_key not in pair_cache:
                pair_cache[cache_key] = _pair_score(
                    src,
                    ref,
                    min_score_cutoff=max_edit_distance,
                    substring_bonus=substring_bonus,
                    ineligible_penalty=ineligible_penalty,
                )
            matrix[i, j] = pair_cache[cache_key].score

    matrix_cache[key] = matrix
    return matrix


def _optimal_assignment(scores: np.ndarray) -> list[tuple[int, int, float]]:
    if scores.size == 0:
        return []

    n_rows, n_cols = scores.shape
    if n_rows == 0 or n_cols == 0:
        return []

    if linear_sum_assignment is None:
        # Fallback greedy matching when scipy is unavailable.
        used_cols: set[int] = set()
        matched: list[tuple[int, int, float]] = []
        for i in range(n_rows):
            best_j = None
            best_score = -1.0
            for j in range(n_cols):
                if j in used_cols:
                    continue
                if scores[i, j] > best_score:
                    best_score = float(scores[i, j])
                    best_j = j
            if best_j is None:
                continue
            used_cols.add(best_j)
            matched.append((i, best_j, float(scores[i, best_j])))
        return matched

    row_ind, col_ind = linear_sum_assignment(-scores)
    return [(int(r), int(c), float(scores[r, c])) for r, c in zip(row_ind, col_ind)]



class _BaseMatchTest(QCTester):
    # Subclasses must define target_keys to include panel_id, ref_panel_id, and item-specific ID
    default_config = {
        "max_edit_distance": 0.35,
        "substring_bonus": 0.12,
        "ineligible_penalty": 0.75,
        "high_confidence_threshold": 0.9,
        "medium_confidence_threshold": 0.75,
    }
    meta_fields = [
           ("item", "Item being matched (e.g., channel or marker name)"),
           ("reference_panel_id", "ID of the reference panel"),
           ("closest_match", "Closest matching item in reference panel"),
           ("match_method", "Method used for matching (exact, fuzzy, etc.)"),
           ("score", "Match score"),
           ("has_eligible_match", "Whether an eligible match was found"),
       ]
    metric_fields = [
           ("score", "Similarity score of the best match"),
           ("is_exact", "Whether the match is exact (0 or 1)"),
           ("has_eligible_match", "Whether an eligible match exists (0 or 1)"),
       ]
    default_thresholds = {
           "score": {"warn": (0.80, None), "severe": (0.60, None)},
       }

    def fit(
        self,
        *,
        source_items: list[str],
        reference_items: list[str],
        targets: dict[str, Any],
        reference_panel_id: str,
        pair_cache: dict[tuple[str, str, float, float], _PairScore] | None = None,
        **kwargs: Any,
    ) -> Iterable[QCTestRecord]:
        max_edit_distance = float(self.metadata["max_edit_distance"])
        substring_bonus = float(self.metadata["substring_bonus"])
        ineligible_penalty = float(self.metadata.get("ineligible_penalty", 0.75))
        pair_cache = pair_cache if pair_cache is not None else {}

        # Get item key name from target_keys (last element should be channel_id or marker_id)
        item_key = self.target_keys[-1] if self.target_keys else "item_id"

        for item in source_items:  # iterate over channels/markers in the panel
            closest_match, pair_score, has_eligible = self._best_match_for_item(
                item,
                reference_items,
                max_edit_distance=max_edit_distance,
                substring_bonus=substring_bonus,
                ineligible_penalty=ineligible_penalty,
                pair_cache=pair_cache,
            )

            score = 0.0 if pair_score is None else float(pair_score.score)

            # Build targets dict for this specific item
            item_targets = {
                **targets,  # Include panel_id from input
                "ref_panel_id": reference_panel_id,
                item_key: item,
            }

            metadata = {
                "item": item,
                "reference_panel_id": reference_panel_id,
                "closest_match": closest_match,
                "match_method": None if pair_score is None else pair_score.method,
                "score": score,
                "has_eligible_match": has_eligible,
            }
            metrics: dict[str, Any] = {
                "score": score,
                "is_exact": float(pair_score is not None and pair_score.method == "exact"),
                "has_eligible_match": float(has_eligible),
            }
            yield QCTestRecord(
                id=self.make_key(targets=item_targets, metadata=metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=item_targets,
                metadata=metadata,
                metrics=metrics,
                thresholds=self.thresholds,
                status="PENDING",
            )

    def classify(self, test: QCTestRecord, **kwargs: Any) -> QCTestRecord:
        if not test.metadata.get("closest_match"):
            test.status = "SEVERE"
            test.message = "No reference items available for matching."
            return test

        score = float(test.metrics.get("score", 0.0))
        method = str(test.metadata.get("match_method", ""))
        has_eligible = bool(test.metadata.get("has_eligible_match", False))

        if method == "exact":
            test.status = "PASS"
            test.message = "Exact match found."
            return test

        # Get thresholds from nested structure
        severe_bounds = test.thresholds["score"]["severe"]
        warn_bounds = test.thresholds["score"]["warn"]
        severe_threshold, warn_threshold = severe_bounds[0], warn_bounds[0]
        if severe_threshold is None or warn_threshold is None:
            raise ValueError("Thresholds for 'score' must be defined for both warn and severe levels.")

        if not has_eligible or score < severe_threshold:
            test.status = "SEVERE"
            test.message = "No robust inexact match found under current edit-distance cutoff."
            return test

        if score >= warn_threshold:
            test.status = "WARN"
            test.message = "Inexact match found with moderate confidence."
        else:
            test.status = "SEVERE"
            test.message = "Inexact match found but similarity is low."

        return test

    def plot(self, test: QCTestRecord, *, adata: Any, output_path: PathLike | None = None, **kwargs: Any) -> Figure:
        raise NotImplementedError("Match tests do not implement plotting.")

    @staticmethod
    def _best_match_for_item(
        source_item: str,
        reference_items: list[str],
        *,
        max_edit_distance: float,
        substring_bonus: float,
        ineligible_penalty: float = 0.75,
        pair_cache: dict[tuple[str, str, float, float], _PairScore] | None = None,
    ) -> tuple[str | None, _PairScore | None, bool]:
        if not reference_items:
            return None, None, False

        if pair_cache is None:
            pair_cache = {}

        scored: list[tuple[str, _PairScore]] = []
        for item in reference_items:
            key = (source_item, item, max_edit_distance, substring_bonus)
            if key not in pair_cache:
                pair_cache[key] = _pair_score(
                    source_item,
                    item,
                    min_score_cutoff=max_edit_distance,
                    substring_bonus=substring_bonus,
                    ineligible_penalty=ineligible_penalty,
                )
            scored.append((item, pair_cache[key]))

        eligible = [(item, ps) for item, ps in scored if ps.eligible]
        pool = eligible if eligible else scored
        best_item, best_pair = max(pool, key=lambda x: (x[1].score, -x[1].edit_distance))
        return best_item, best_pair, bool(eligible)



class _BaseSimilarityTest(QCTester):
    target_keys = ("panel_id",)
    meta_keys = ("reference_panel_id",)
    default_config = {
        "max_edit_distance": 0.35,
        "substring_bonus": 0.12,
        "shared_score_cutoff": 0.60,
        "ineligible_penalty": 0.75,
        "high_confidence_threshold": 0.9,
        "medium_confidence_threshold": 0.75,
    }
    meta_fields = [
        ("reference_panel_id", "Reference panel ID"),
        ("shared_score_cutoff", "Minimum score for shared items"),
        ("matched_pairs", "List of matched item pairs"),
    ]
    metric_fields = [
        ("n_source", "Number of items in source panel"),
        ("n_reference", "Number of items in reference panel"),
        ("n_shared", "Number of shared items"),
        ("n_unique_source", "Number of items unique to source"),
        ("n_unique_reference", "Number of items unique to reference"),
        ("similarity_score", "Overall panel similarity score"),
    ]
    default_thresholds = {
        "similarity_score": {"warn": (0.85, None), "severe": (0.65, None)},
    }

    def fit(
        self,
        *,
        source_items: list[str],
        reference_items: list[str],
        targets: dict[str, Any],
        reference_panel_id: str,
        pair_cache: dict[tuple[str, str, float, float], _PairScore] | None = None,
        matrix_cache: dict[tuple[tuple[str, ...], tuple[str, ...], float, float], np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Iterable[QCTestRecord]:
        max_edit_distance = float(self.metadata["max_edit_distance"])
        substring_bonus = float(self.metadata["substring_bonus"])
        shared_score_cutoff = float(self.metadata["shared_score_cutoff"])
        ineligible_penalty = float(self.metadata.get("ineligible_penalty", 0.75))
        high_confidence_threshold = float(self.metadata.get("high_confidence_threshold", 0.9))
        medium_confidence_threshold = float(self.metadata.get("medium_confidence_threshold", 0.75))
        pair_cache = pair_cache if pair_cache is not None else {}
        matrix_cache = matrix_cache if matrix_cache is not None else {}

        metadata = {
            "reference_panel_id": reference_panel_id,
            "shared_score_cutoff": shared_score_cutoff,
        }

        if not source_items and not reference_items:
            metrics: dict[str, Any] = {
                "n_source": 0,
                "n_reference": 0,
                "n_shared": 0,
                "n_unique_source": 0,
                "n_unique_reference": 0,
                "similarity_score": 1.0,
            }
            yield QCTestRecord(
                id=self.make_key(targets=targets, metadata=metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics=metrics,
                thresholds=self.thresholds,
                status="SKIP",
                message="Both source and reference panels are empty.",
            )
            return

        score_matrix = _score_matrix(
            source_items,
            reference_items,
            max_edit_distance=max_edit_distance,
            substring_bonus=substring_bonus,
            ineligible_penalty=ineligible_penalty,
            high_confidence_threshold=high_confidence_threshold,
            medium_confidence_threshold=medium_confidence_threshold,
            matrix_cache=matrix_cache,
            pair_cache=pair_cache,
        )
        assignments = _optimal_assignment(score_matrix)

        shared = int(sum(1 for _, _, score in assignments if score >= shared_score_cutoff))
        n_source = len(source_items)
        n_reference = len(reference_items)
        n_unique_source = int(max(n_source - shared, 0))
        n_unique_reference = int(max(n_reference - shared, 0))
        denom = max(n_source, n_reference, 1)
        similarity_score = float(sum(score for _, _, score in assignments) / float(denom))

        metadata["matched_pairs"] = [
            {
                "from": source_items[i],
                "to": reference_items[j],
                "score": score,
            }
            for i, j, score in assignments
        ]

        metrics: dict[str, Any] = {
            "n_source": float(n_source),
            "n_reference": float(n_reference),
            "n_shared": float(shared),
            "n_unique_source": float(n_unique_source),
            "n_unique_reference": float(n_unique_reference),
            "similarity_score": similarity_score,
        }
        yield QCTestRecord(
            id=self.make_key(targets=targets, metadata=metadata),
            test_type=self.test_type,
            test_name=self.test_name,
            targets=targets,
            metadata=metadata,
            metrics=metrics,
            thresholds=self.thresholds,
            status="PENDING",
        )

    def classify(self, test: QCTestRecord, **kwargs: Any) -> QCTestRecord:
        if test.status == "SKIP":
            return test

        # Use base class classify for threshold checking
        test = super().classify(test, **kwargs)

        # Adjust message based on classification
        if test.status == "PASS":
            test.message = "Panel is highly similar to the reference panel."
        elif test.status == "WARN":
            test.message = "Panel has moderate divergence from the reference panel."
        elif test.status == "SEVERE":
            test.message = "Panel strongly diverges from the reference panel."

        return test

    def plot(self, adata: Any, test: QCTestRecord, output_path: PathLike | None = None, **kwargs: Any) -> Figure:
        raise NotImplementedError("Similarity tests do not implement plotting.")


class MatchChannelTest(_BaseMatchTest):
    test_type = "panel_channel"
    test_name = "match_channel"
    target_keys = ("panel_id", "ref_panel_id", "channel_id")


class PanelChannelSimilarityTest(_BaseSimilarityTest):
    test_type = "panel_channel_set"
    test_name = "panel_channel_similarity"


class MatchMarkerTest(_BaseMatchTest):
    test_type = "panel_marker"
    test_name = "match_marker"
    target_keys = ("panel_id", "ref_panel_id", "marker_id")


class PanelMarkerSimilarityTest(_BaseSimilarityTest):
    test_type = "panel_marker_set"
    test_name = "panel_marker_similarity"


# @EntityQCEvaluatorRegistry.register("panel")
class PanelQCEvaluator(EntityQCEvaluator):
    entity_type = "panel"
    _supported_tables = {
        "panel_channel": {
            "description": "Per-item channel matching to the primary panel",
            "input_params": {},
        },
        "panel_marker": {
            "description": "Per-item marker matching to the primary panel",
            "input_params": {},
        },
        "panel_channel_set": {
            "description": "Panel-level channel similarity metrics",
            "input_params": {},
        },
        "panel_marker_set": {
            "description": "Panel-level marker similarity metrics",
            "input_params": {},
        },
    }
    default_config = {
        "primary_panel_id": None,
        "max_edit_distance": 0.35,
        "substring_bonus": 0.12,
        "shared_score_cutoff": 0.60,
        "ineligible_penalty": 0.75,
        "high_confidence_threshold": 0.9,
        "medium_confidence_threshold": 0.75,
        "warn_match_score": 0.80,
        "severe_match_score": 0.60,
        "warn_similarity": 0.85,
        "severe_similarity": 0.65,
    }

    @classmethod
    def get_tests(cls, entity: Any = None) -> dict[str, type[QCTester]]:
        return {
            "match_channel": MatchChannelTest,
            "panel_channel_similarity": PanelChannelSimilarityTest,
            "match_marker": MatchMarkerTest,
            "panel_marker_similarity": PanelMarkerSimilarityTest,
        }

    def required_layer(self, entity: Any = None) -> str | None:
        return None

    def load_entity(self, dataloader: UnifiedDataLoader, entity_id: Hashable, context: dict[str, Any] | None = None) -> dict[str, Any]:
        project: Project = dataloader.load_data("project")
        return {
            "id": str(entity_id),
            "panel_catalog": project.panel_catalog,
            "batches": project.batches,
        }

    def update_sample_qc(
        self,
        entity: Mapping[str, Any],
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EntityQCStatus:

        context = context or {}
        # Panel QC is batch-level by design; per-sample QC remains empty.
        entity_qc.sample_qc = {}
        cfg = self.config.copy()
        cfg.update(context or {})
        entity_qc.context = cfg
        return entity_qc

    def update_batch_qc(
        self,
        entity: Mapping[str, Any],
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> EntityQCStatus:
        context = context or {}
        cfg = self.config.copy()
        cfg.update(context or {})
        ref_panel_id = cfg.setdefault("ref_panel_id", "panel")
        entity_qc.context = cfg

        panel_id = str(entity["id"])
        if panel_id == ref_panel_id:
            # If the current panel is the reference panel, we can skip all tests.
            step = entity_qc.batch_qc.get_step("PANEL_QC")
            step.flag = QCFlag.PASS
            step.add_reason("REF_PANEL", "This panel is the reference panel for comparison.")
            return entity_qc

        panel_catalog: dict[str, list[ChannelRef]] = dict(entity.get("panel_catalog", {}))
        batches: dict[str, BatchRef] = dict(entity.get("batches", {}))

        panel = panel_catalog[panel_id]
        panel_samples = sorted(batches[panel_id].sample_ids)
        channel_names = self._extract_channel_names(panel)
        marker_names = self._extract_marker_names(panel)

        reference_panel = panel_catalog[ref_panel_id]
        reference_channels = self._extract_channel_names(reference_panel)
        reference_markers = self._extract_marker_names(reference_panel)

        pair_cache: dict[tuple[str, str, float, float], _PairScore] = {}
        matrix_cache: dict[tuple[tuple[str, ...], tuple[str, ...], float, float], np.ndarray] = {}

        step = entity_qc.batch_qc.get_step("PANEL_QC")
        targets = {"panel_id": panel_id}
        channel_matcher = MatchChannelTest(
            config={
                "max_edit_distance": cfg["max_edit_distance"],
                "substring_bonus": cfg["substring_bonus"],
                "ineligible_penalty": cfg["ineligible_penalty"],
                "high_confidence_threshold": cfg["high_confidence_threshold"],
                "medium_confidence_threshold": cfg["medium_confidence_threshold"],
            },
            thresholds={
                "warn_score": cfg["warn_match_score"],
                "severe_score": cfg["severe_match_score"],
            },
        )
        for test in channel_matcher.fit_classify(
            source_items=channel_names,
            reference_items=reference_channels,
            targets=targets,
            reference_panel_id=ref_panel_id,
            pair_cache=pair_cache,
        ):
            self._add_batch_test(step=step, test=test)

        channel_similarity = PanelChannelSimilarityTest(
            config={
                "max_edit_distance": cfg["max_edit_distance"],
                "substring_bonus": cfg["substring_bonus"],
                "shared_score_cutoff": cfg["shared_score_cutoff"],
                "ineligible_penalty": cfg["ineligible_penalty"],
                "high_confidence_threshold": cfg["high_confidence_threshold"],
                "medium_confidence_threshold": cfg["medium_confidence_threshold"],
            },
            thresholds={
                "warn_similarity": cfg["warn_similarity"],
                "severe_similarity": cfg["severe_similarity"],
            },
        )
        for test in channel_similarity.fit_classify(
            source_items=channel_names,
            reference_items=reference_channels,
            targets=targets,
            reference_panel_id=ref_panel_id,
            pair_cache=pair_cache,
            matrix_cache=matrix_cache,
        ):
            test.metadata["sample_ids"] = panel_samples
            self._add_batch_test(step=step, test=test)

        marker_matcher = MatchMarkerTest(
            config={
                "max_edit_distance": cfg["max_edit_distance"],
                "substring_bonus": cfg["substring_bonus"],
                "ineligible_penalty": cfg["ineligible_penalty"],
                "high_confidence_threshold": cfg["high_confidence_threshold"],
                "medium_confidence_threshold": cfg["medium_confidence_threshold"],
            },
            thresholds={
                "warn_score": cfg["warn_match_score"],
                "severe_score": cfg["severe_match_score"],
            },
        )
        for test in marker_matcher.fit_classify(
            source_items=marker_names,
            reference_items=reference_markers,
            targets=targets,
            reference_panel_id=ref_panel_id,
            pair_cache=pair_cache,
        ):
            test.metadata["sample_ids"] = panel_samples
            self._add_batch_test(step=step, test=test)

        marker_similarity = PanelMarkerSimilarityTest(
            config={
                "max_edit_distance": cfg["max_edit_distance"],
                "substring_bonus": cfg["substring_bonus"],
                "shared_score_cutoff": cfg["shared_score_cutoff"],
                "ineligible_penalty": cfg["ineligible_penalty"],
                "high_confidence_threshold": cfg["high_confidence_threshold"],
                "medium_confidence_threshold": cfg["medium_confidence_threshold"],
            },
            thresholds={
                "warn_similarity": cfg["warn_similarity"],
                "severe_similarity": cfg["severe_similarity"],
            },
        )
        for test in marker_similarity.fit_classify(
            source_items=marker_names,
            reference_items=reference_markers,
            targets=targets,
            reference_panel_id=ref_panel_id,
            pair_cache=pair_cache,
            matrix_cache=matrix_cache,
        ):
            test.metadata["sample_ids"] = panel_samples
            self._add_batch_test(step=step, test=test)

        return entity_qc

    def summarize_entity_qc(self, entity_qc: EntityQCStatus) -> dict[str, Any]:
        return {
            "primary_panel_id": entity_qc.context.get("primary_panel_id"),
            "n_panels": len(entity_qc.context.get("panel_sample_counts", {})),
        }

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Mapping[str, Any],
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
        figure_dir: PathLike | None = None,
        **kwargs: Any,
    ) -> Figure:
        raise NotImplementedError("Panel QC currently exposes tables/records but no figures.")

    @staticmethod
    def _add_batch_test(step: Any, test: QCTestRecord) -> None:
        if test.status in {"WARN", "SEVERE", "FAIL"}:
            step.add_reason(
                code=f"PANEL_{test.status}",
                message=test.message,
                tests=[test],
            )
        else:
            step.add_test(test)

    @staticmethod
    def _extract_channel_names(panel: Iterable[ChannelRef]) -> list[str]:
        return [ch.pnn for ch in panel if ch.pnn]

    @staticmethod
    def _extract_marker_names(panel: Iterable[ChannelRef]) -> list[str]:
        return [ch.pns for ch in panel if ch.pns]
