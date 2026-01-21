from __future__ import annotations

from collections.abc import Callable
from .base import BaseStep


__all__ = ["StepRegistry"]

StepRegistry: dict[str, type[BaseStep]] = {}

def register_step(name: str) -> Callable[[type[BaseStep]], type[BaseStep]]:
    def decorator(cls: type[BaseStep]) -> type[BaseStep]:
        if name in StepRegistry:
            raise ValueError(f"Duplicate step name {name!r}")
        StepRegistry[name] = cls
        return cls
    return decorator

# Import concrete steps so their decorators run and populate StepRegistry
from . import add_samples, compensation, load_fcs, tranform


del Callable
del BaseStep
