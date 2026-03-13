"""Flow cytometry transformation definitions, factories, and runtime registry."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import warnings
from typing import Any, Callable, Iterable, Mapping, Protocol, TYPE_CHECKING

from flowkit import transforms as xf
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cytomind.domain.flow import ChannelRef

__all__ = [
    "TransformDef",
    "TransformLike",
    "SCALE_TRANSFORM_TYPES",
    "transform_registry",
    "register_transform_type",
    "build_transformer",
    "make_default_transform_def",
    "register_custom_transform",
]

class TransformLike(Protocol):
    def apply(self, events: NDArray) -> NDArray: ...
    def inverse(self, events: NDArray) -> NDArray: ...

class IdentityTransform:
    def apply(self, events: NDArray) -> NDArray: return events
    def inverse(self, events: NDArray) -> NDArray: return events

class RatioTransform:
    """fratio(x, y, A, B, C) = A * ((x - B) / (y - C))"""
    def __init__(self, param_a: float = 1.0, param_b: float = 0.0, param_c: float = 0.0) -> None:
        self.A = param_a
        self.B = param_b
        self.C = param_c

    def apply(self, events: NDArray) -> NDArray:
        # TODO: use string indices?
        x = events[:, 0]
        y = events[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.A * ((x - self.B) / (y - self.C))
            ratio[np.isnan(ratio)] = 0.0
            ratio[np.isinf(ratio)] = 0.0
        return ratio.reshape(-1, 1)

    def inverse(self, events: NDArray) -> NDArray:
        raise NotImplementedError("RatioTransform is not reversible.")


SCALE_TRANSFORM_TYPES: frozenset[str] = frozenset(
    {
        "identity",
        "flin",
        "log",
        "asinh",
        "logicle",
        "hyperlog",
        "flojo_asinh",
        "flojo_log",
        "flojo_biexp",
        "fratio",
    }
)


transform_registry: dict[str, type[TransformLike]] = {
    "identity": IdentityTransform,
    "flin": xf.LinearTransform,
    "log": xf.LogTransform,
    "flojo_log": xf.WSPLogTransform,  # pyright: ignore[reportAssignmentType]
    "flojo_biexp": xf.WSPBiexTransform,
    "asinh": xf.AsinhTransform,
    "logicle": xf.LogicleTransform,
    "hyperlog": xf.HyperlogTransform,
    "fratio": RatioTransform,
}


@dataclass(frozen=True)
class TransformDef:
    id: str
    type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        _validate_transform_params(self.type, self.params)

    @classmethod
    def generate_id(cls, transform_type: str, params: Mapping[str, Any]) -> str:
        payload = {
            "type": transform_type,
            "params": _normalize_for_hash(params),
        }
        digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
        return f"{transform_type}_{digest}"

    @classmethod
    def create(
        cls,
        transform_type: str,
        params: Mapping[str, Any] | None = None,
        tags: Iterable[str] = (),
        id: str | None = None,
    ) -> "TransformDef":
        transform_params = dict(params or {})
        all_tags = set(tags)
        if transform_type in SCALE_TRANSFORM_TYPES:
            all_tags.add("scale")
        return cls(
            id=id or cls.generate_id(transform_type, transform_params),
            type=transform_type,
            params=transform_params,
            tags=tuple(all_tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "params": dict(self.params),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransformDef":
        transform_type = str(data["type"])
        params = dict(data.get("params", {}))
        if "id" in data:
            transform_id = str(data["id"])
        else:
            transform_id = cls.generate_id(transform_type=transform_type, params=params)
        tags = tuple(data.get("tags", []))
        if not tags and transform_type in SCALE_TRANSFORM_TYPES:
            tags = ("scale",)
        return cls(id=transform_id, type=transform_type, params=params, tags=tags)


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, np.ndarray):
        return [_normalize_for_hash(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_hash(v) for v in value]
    return value


def _coerce_float_param(transform_type: str, params: Mapping[str, Any], aliases: tuple[str, ...]) -> float:
    for name in aliases:
        if name in params:
            try:
                return float(params[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Transformation '{transform_type}' parameter '{name}' must be numeric; got {params[name]!r}."
                ) from exc
    raise ValueError(
        f"Transformation '{transform_type}' is missing required parameter; expected one of {aliases}."
    )


def _validate_transform_params(transform_type: str, params: Mapping[str, Any]) -> None:
    # GatingML 2.0 Chapter 6 and Chapter 8 validity conditions.
    if transform_type == "flin":
        t = _coerce_float_param(transform_type, params, ("param_t", "t"))
        a = _coerce_float_param(transform_type, params, ("param_a", "a"))
        if t <= 0:
            raise ValueError("Transformation 'flin' requires T > 0.")
        if a < 0:
            raise ValueError("Transformation 'flin' requires A >= 0.")
        if a > t:
            raise ValueError("Transformation 'flin' requires A <= T.")
        return

    if transform_type == "log":
        t = _coerce_float_param(transform_type, params, ("param_t", "t"))
        m = _coerce_float_param(transform_type, params, ("param_m", "m"))
        if t <= 0:
            raise ValueError("Transformation 'log' requires T > 0.")
        if m <= 0:
            raise ValueError("Transformation 'log' requires M > 0.")
        return

    if transform_type == "asinh":
        t = _coerce_float_param(transform_type, params, ("param_t", "t"))
        m = _coerce_float_param(transform_type, params, ("param_m", "m"))
        a = _coerce_float_param(transform_type, params, ("param_a", "a"))
        if t <= 0:
            raise ValueError("Transformation 'asinh' requires T > 0.")
        if m <= 0:
            raise ValueError("Transformation 'asinh' requires M > 0.")
        if a < 0:
            raise ValueError("Transformation 'asinh' requires A >= 0.")
        if a > m:
            raise ValueError("Transformation 'asinh' requires A <= M.")
        return

    if transform_type in {"logicle", "hyperlog"}:
        t = _coerce_float_param(transform_type, params, ("param_t", "t"))
        m = _coerce_float_param(transform_type, params, ("param_m", "m"))
        a = _coerce_float_param(transform_type, params, ("param_a", "a"))
        w = _coerce_float_param(transform_type, params, ("param_w", "w"))
        if t <= 0:
            raise ValueError(f"Transformation '{transform_type}' requires T > 0.")
        if m <= 0:
            raise ValueError(f"Transformation '{transform_type}' requires M > 0.")
        if transform_type == "logicle" and not (0.0 <= w <= (m / 2.0)):
            raise ValueError("Transformation 'logicle' requires 0 <= W <= M/2.")
        if transform_type == "hyperlog" and not (0.0 < w <= (m / 2.0)):
            raise ValueError("Transformation 'hyperlog' requires 0 < W <= M/2.")
        if not (-w <= a <= (m - 2.0 * w)):
            raise ValueError(f"Transformation '{transform_type}' requires -W <= A <= M - 2W.")
        return

    if transform_type == "fratio":
        # Chapter 8 expresses validity via XML schema; enforce numeric parameter presence.
        _coerce_float_param(transform_type, params, ("param_a", "a"))
        _coerce_float_param(transform_type, params, ("param_b", "b"))
        _coerce_float_param(transform_type, params, ("param_c", "c"))


def register_transform_type(type_name: str, transform_cls: type[TransformLike]) -> None:
    if type_name in transform_registry:
        raise ValueError(f"Transformation type '{type_name}' is already registered.")
    transform_registry[type_name] = transform_cls


def build_transformer(transform: TransformDef) -> TransformLike:
    t_cls = transform_registry.get(transform.type)
    if t_cls is None:
        raise ValueError(f"Unknown transformation '{transform.type}'. Available: {sorted(transform_registry.keys())}")
    try:
        return t_cls(**dict(transform.params)) if transform.params else t_cls()
    except TypeError as exc:
        raise TypeError(
            f"Failed to construct transformer '{transform.type}' with params {dict(transform.params)}: {exc}"
        ) from exc


def make_default_transform_def(
    transform_type: str,
    channel_ref: ChannelRef | None = None,
    params_override: Mapping[str, float] | None = None,
    tags: Iterable[str] = (),
) -> TransformDef:
    defaults = {
        "t": 262144.0,  # 2^18, common upper bound for 18-bit cytometry data
        "w": 0.5,
        "m": 4.5,
        "a": 0.0,
    }

    if channel_ref is None or channel_ref.pnr is None:
        warnings.warn(
            "ChannelRef or channel pnr is missing; using default param_t = 2^18 (262144.0).",
            stacklevel=2,
        )
        param_t = defaults["t"]
    else:
        param_t = float(channel_ref.pnr)

    params_by_t_id: dict[str, dict[str, float]] = {
        "log":         {"param_t": param_t, "param_m": defaults["m"]},
        "flin":        {"param_t": param_t, "param_a": defaults["a"]},
        "asinh":       {"param_t": param_t, "param_m": defaults["m"], "param_a": defaults["a"]},
        "logicle":     {"param_t": param_t, "param_m": defaults["m"], "param_a": defaults["a"], "param_w": defaults["w"]},
        "hyperlog":    {"param_t": param_t, "param_m": defaults["m"], "param_a": defaults["a"], "param_w": defaults["w"]},
        "flojo_asinh": {"param_t": param_t, "param_m": 4.0, "param_a": 0.7},
        "fratio":      {"param_a": 1.0, "param_b": 0.0, "param_c": 0.0},
        "flojo_log":   {"offset": 1.0, "decades": defaults["m"]},
        "flojo_biexp": {"negative": 0.0, "width": -10.0, "positive": 4.418540, "max_value": param_t},
        "identity":    {},
    }

    if transform_type not in params_by_t_id:
        raise ValueError(
            f"Unknown transformation '{transform_type}'. Available: {sorted(params_by_t_id.keys())}"
        )

    params = params_by_t_id[transform_type]
    if params_override:
        params.update(dict(params_override))

    transform_id: str | None = None
    if "param_t" in params:
        t_val = params["param_t"]
        t_exp = math.log2(t_val) if t_val > 0 else float("nan")
        if t_val > 0 and math.isclose(t_exp, round(t_exp), rel_tol=0.0, abs_tol=1e-9):
            transform_id = f"{transform_type}_{int(round(t_exp))}"
        else:
            warnings.warn(
                f"param_t={t_val} is not a power of 2 for transformation '{transform_type}'.",
                stacklevel=2,
            )

    return TransformDef.create(transform_type=transform_type, params=params, tags=tags, id=transform_id)


def register_custom_transform(
    transform_type: str,
    apply_fn: Callable[[NDArray, dict[str, Any]], NDArray],
    default_params: dict[str, Any],
    inverse_fn: Callable[[NDArray, dict[str, Any]], NDArray] | None = None,
    tags: Iterable[str] = (),
) -> TransformDef:

    def _inverse(events: NDArray, params: dict[str, Any]) -> NDArray:
        if inverse_fn is None:
            raise NotImplementedError(f"No inverse defined for transformation '{transform_type}'.")
        return inverse_fn(events, params)

    class _CustomTransform:
        def __init__(self, **params: Any) -> None:
            self._params = params

        def apply(self, events: NDArray) -> NDArray:
            return apply_fn(events, self._params)

        def inverse(self, events: NDArray) -> NDArray:
            return _inverse(events, self._params)

    _CustomTransform.__name__ = transform_type
    register_transform_type(type_name=transform_type, transform_cls=_CustomTransform)

    return TransformDef.create(transform_type=transform_type, params=default_params, tags=tags)

