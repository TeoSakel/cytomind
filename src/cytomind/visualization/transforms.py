from __future__ import annotations
from typing import Sequence

import numpy as np

from cytomind.domain.flow import TransformationRef
from cytomind.domain.transforms import transform_registry, get_default_transformations

__all__ = ["apply_transform"]


def _coerce_ref(name_or_ref: str | TransformationRef | None) -> TransformationRef:
    """Normalize input to a TransformationRef using registry defaults."""
    if isinstance(name_or_ref, TransformationRef):
        return name_or_ref

    key = name_or_ref or "identity"
    defaults = get_default_transformations()
    if key in defaults:
        return defaults[key]

    raise ValueError(f"Unknown transformation '{key}'. Available: {sorted(defaults.keys())}")


def apply_transform(
    values: np.ndarray,
    transformation: str | TransformationRef = "identity",
) -> np.ndarray:
    """Apply a visualization transform using the shared transformation registry.

    Parameters
    ----------
    values : np.ndarray
        1D array of channel values.
    transformation : str | TransformationRef | None
        Transformation key or ref; defaults to "logicle" if None.
    """

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    ref = _coerce_ref(transformation)
    if ref.type == "identity":
        return values

    t_cls = transform_registry.get(ref.id)
    if t_cls is None:
        raise ValueError(f"Unknown transformation '{ref.id}'. Available: {sorted(transform_registry.keys())}")

    try:
        transformer = t_cls(**(ref.params or {})) if ref.params else t_cls()
    except TypeError as exc:  # constructor mismatch
        raise TypeError(f"Failed to construct transformer '{ref.id}' with params {ref.params}: {exc}") from exc

    # Most transforms expect a 2D array; reshape to (n, 1) and flatten back
    arr2d = values.reshape(-1, 1)
    transformed = transformer.apply(arr2d)
    if isinstance(transformed, Sequence) and not isinstance(transformed, np.ndarray):
        transformed = np.asarray(transformed)
    if transformed.ndim > 1:
        transformed = transformed[:, 0]
    return transformed
