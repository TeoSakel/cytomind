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

        Works for both domain entities (compensation, gating_strategy) and
        step entities (compensate, universal_gates, etc.).

        Usage:
            @EntityQCEvaluatorRegistry.register("compensation")
            class CompensationQCEvaluator(EntityQCEvaluator):
                entity_type = "compensation"
                ...

            @EntityQCEvaluatorRegistry.register("compensate")  # step type
            class CompensationStepQCEvaluator(EntityQCEvaluator):
                entity_type = "step"
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
from . import compensation  # noqa: E402, F401
from .step import StepQCEvaluator
from . import gates  # noqa: E402, F401
