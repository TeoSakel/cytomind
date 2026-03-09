"""
QC evaluator registry and decorators.

Provides a unified registry for all QC evaluators (domain entities and steps).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EntityQCEvaluator
else:
    EntityQCEvaluator = object

class EntityQCEvaluatorRegistry:
    """Unified registry for all QC evaluators (entities and steps)."""

    _evaluators: dict[str, tuple[int, type[EntityQCEvaluator]]] = {}
    _next_priority: int = 0

    @classmethod
    def register(cls, entity_type: str, priority: int | None = None):
        """
        Decorator to register a QC evaluator for an entity type.

        Priority determines evaluation order in evaluate_step_products.
        Lower values are evaluated first.

        By default, priority is automatically assigned based on registration order,
        which follows import order in __init__.py. To change evaluation order for
        core evaluators, reorder the imports at the bottom of this file.

        For custom/plugin evaluators, specify priority explicitly to control where
        they fit in the evaluation sequence (e.g., priority=15 runs after panel but
        before compensation).

        Register evaluators by entity type (not step type).

        For entity-oriented design:
        - Domain entities (compensation, gating_strategy) register by entity type
        - Steps always use StepQCEvaluator (entity_type="step")
        - Products created by steps are evaluated by their entity-type evaluators

        Parameters
        ----------
        entity_type : str
            The entity type identifier
        priority : int | None
            Optional explicit priority. If None, auto-assigns based on registration order.

        Usage:
            # Core evaluator (auto priority from import order):
            @EntityQCEvaluatorRegistry.register("compensation")
            class CompensationQCEvaluator(EntityQCEvaluator):
                entity_type = "compensation"
                ...

            # Custom/plugin evaluator (explicit priority):
            @EntityQCEvaluatorRegistry.register("custom_layer", priority=15)
            class CustomLayerQCEvaluator(EntityQCEvaluator):
                entity_type = "custom_layer"
                ...
        """

        def decorator(evaluator_class: type[EntityQCEvaluator]) -> type[EntityQCEvaluator]:
            if priority is None:
                assigned_priority = cls._next_priority
                cls._next_priority += 10
            else:
                assigned_priority = priority
            cls._evaluators[entity_type] = (assigned_priority, evaluator_class)
            return evaluator_class

        return decorator

    @classmethod
    def get(cls, entity_type: str) -> type[EntityQCEvaluator] | None:
        """Get QC evaluator class for entity type, or None if not registered."""
        registered = cls._evaluators.get(entity_type)
        if not registered:
            return None
        return registered[1]

    @classmethod
    def get_priority(cls, entity_type: str) -> int | None:
        """Get evaluator priority for an entity type, or None if not registered."""
        registered = cls._evaluators.get(entity_type)
        if not registered:
            return None
        return registered[0]

    @classmethod
    def iter_evaluators(cls) -> list[tuple[str, type[EntityQCEvaluator]]]:
        """Iterate evaluators ordered by priority, then registration order."""
        ordered = sorted(cls._evaluators.items(), key=lambda item: item[1][0])
        return [(entity_type, evaluator_class) for entity_type, (_, evaluator_class) in ordered]

    @classmethod
    def list_evaluators(cls) -> dict[str, type[EntityQCEvaluator]]:
        """List registered evaluators ordered by priority, then registration order."""
        return {entity_type: evaluator_class for entity_type, evaluator_class in cls.iter_evaluators()}


# Import evaluators to trigger registration
from . import panel, compensation, gate_node, gating_strategy, step
