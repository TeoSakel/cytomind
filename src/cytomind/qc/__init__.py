"""
QC evaluator registry and decorators.

Provides a unified registry for step implementations, QC evaluators, and revision handlers.
"""
from __future__ import annotations
from .base import StepQCEvaluator


class QCEvaluatorRegistry:
    """Registry for QC evaluators keyed by step type"""

    _evaluators: dict[str, type[StepQCEvaluator]] = {}

    @classmethod
    def register(cls, step_type: str):
        """
        Decorator to register a QC evaluator for a step type.

        Usage:
            @QCEvaluatorRegistry.register("compensate")
            class CompensationQCEvaluator(StepQCEvaluator):
                ...
        """

        def decorator(evaluator_class: type[StepQCEvaluator]) -> type[StepQCEvaluator]:
            cls._evaluators[step_type] = evaluator_class
            return evaluator_class

        return decorator

    @classmethod
    def get(cls, step_type: str) -> type[StepQCEvaluator] | None:
        """Get QC evaluator class for step type, or None if not registered"""
        return cls._evaluators.get(step_type)

    @classmethod
    def list_evaluators(cls) -> dict[str, type[StepQCEvaluator]]:
        """List all registered evaluators"""
        return cls._evaluators.copy()


# Import evaluators to trigger registration
from . import compensation  # noqa: E402, F401
