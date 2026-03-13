from __future__ import annotations
from typing import Sequence

import numpy as np

from cytomind.domain.transforms import TransformDef, build_transformer, make_default_transform_def

__all__ = ["apply_transform"]


def _coerce_ref(name_or_ref: str | TransformDef | None) -> TransformDef:
    """Normalize input to a TransformDef using registry defaults."""
    if isinstance(name_or_ref, TransformDef):
        return name_or_ref

    key = name_or_ref or "identity"
    try:
        return make_default_transform_def(transform_type=key)
    except ValueError as exc:
        from cytomind.domain.transforms import transform_registry
        available = sorted(transform_registry.keys())
        raise ValueError(f"Unknown transformation '{key}'. Available: {available}") from exc


def apply_transform(
    values: np.ndarray,
    transformation: str | TransformDef = "identity",
) -> np.ndarray:
    """Apply a visualization transform using the shared transformation registry.

    Parameters
    ----------
    values : np.ndarray
        1D array of channel values.
    transformation : str | TransformDef | None
        Transformation key or ref; defaults to "logicle" if None.
    """

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    ref = _coerce_ref(transformation)
    if ref.type == "identity":
        return values

    transformer = build_transformer(ref)

    # Most transforms expect a 2D array; reshape to (n, 1) and flatten back
    arr2d = values.reshape(-1, 1)
    transformed = transformer.apply(arr2d)
    if isinstance(transformed, Sequence) and not isinstance(transformed, np.ndarray):
        transformed = np.asarray(transformed)
    if transformed.ndim > 1:
        transformed = transformed[:, 0]
    return transformed
