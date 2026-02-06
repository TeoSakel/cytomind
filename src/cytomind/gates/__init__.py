from __future__ import annotations

from collections.abc import Callable
from .base import Gate

__all__ = ["GateRegistry", "Gate"]


class GateRegistry:
    """Registry for gate classes with decorator pattern."""

    _gates: dict[str, type[Gate]] = {}

    @classmethod
    def register(cls, gate_type: str) -> Callable[[type[Gate]], type[Gate]]:
        """
        Decorator to register a gate class.

        Parameters
        ----------
        gate_type : str
            The gate type identifier (e.g., "RectangleGate", "QuadrantGate")

        Examples
        --------
        >>> @GateRegistry.register("RectangleGate")
        >>> class RectangleGate(Gate):
        ...     pass
        """
        def decorator(gate_cls: type[Gate]) -> type[Gate]:
            if gate_type in cls._gates:
                raise ValueError(f"Duplicate gate type {gate_type!r}")
            cls._gates[gate_type] = gate_cls
            return gate_cls
        return decorator

    @classmethod
    def get(cls, gate_type: str) -> type[Gate]:
        """Get gate class for a gate type."""
        return cls._gates[gate_type]

    @classmethod
    def list_gates(cls) -> dict[str, type[Gate]]:
        """List all registered gates."""
        return cls._gates.copy()


# Import concrete gates so their decorators run and populate registry
from . import glm_gates, regression_gates, mindensity_gates
