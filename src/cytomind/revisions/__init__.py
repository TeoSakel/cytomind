from __future__ import annotations

from collections.abc import Callable
from .base import BaseRevisionHandler

__all__ = ["RevisionHandlerRegistry", "BaseRevisionHandler"]


class RevisionHandlerRegistry:
    """Registry for revision handlers with decorator pattern."""

    _handlers: dict[str, type[BaseRevisionHandler]] = {}

    @classmethod
    def register(cls, step_type: str) -> Callable[[type[BaseRevisionHandler]], type[BaseRevisionHandler]]:
        """
        Decorator to register a revision handler for a specific step type.

        Parameters
        ----------
        step_type : str
            The step type this handler supports (e.g., "compensate", "gate")

        Examples
        --------
        >>> @RevisionHandlerRegistry.register("compensate")
        >>> class CompensationRevisionHandler(BaseRevisionHandler):
        ...     pass
        """
        def decorator(handler_cls: type[BaseRevisionHandler]) -> type[BaseRevisionHandler]:
            if step_type in cls._handlers:
                raise ValueError(f"Duplicate handler for step type {step_type!r}")
            cls._handlers[step_type] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def get(cls, step_type: str) -> type[BaseRevisionHandler] | None:
        """Get handler class for a step type."""
        return cls._handlers.get(step_type)

    @classmethod
    def list_handlers(cls) -> dict[str, type[BaseRevisionHandler]]:
        """List all registered handlers."""
        return cls._handlers.copy()


# Import concrete handlers so their decorators run and populate registry
from . import compensation
from . import add_samples
