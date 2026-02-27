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

    _evaluators: dict[str, type[EntityQCEvaluator]] = {}

    @classmethod
    def register(cls, entity_type: str):
        """
        Decorator to register a QC evaluator for an entity type.

        Register evaluators by entity type (not step type).

        For entity-oriented design:
        - Domain entities (compensation, gating_strategy) register by entity type
        - Steps always use StepQCEvaluator (entity_type="step")
        - Products created by steps are evaluated by their entity-type evaluators

        Usage:
            @EntityQCEvaluatorRegistry.register("compensation")
            class CompensationQCEvaluator(EntityQCEvaluator):
                entity_type = "compensation"
                ...
        """

        def decorator(evaluator_class: type[EntityQCEvaluator]) -> type[EntityQCEvaluator]:
            cls._evaluators[entity_type] = evaluator_class
            return evaluator_class

        return decorator

    @classmethod
    def get(cls, entity_type: str) -> type[EntityQCEvaluator] | None:
        """Get QC evaluator class for entity type, or None if not registered."""
        return cls._evaluators.get(entity_type)

    @classmethod
    def list_evaluators(cls) -> dict[str, type[EntityQCEvaluator]]:
        """List all registered evaluators."""
        return cls._evaluators.copy()


# Import evaluators to trigger registration
from . import compensation, gating_strategy, step
