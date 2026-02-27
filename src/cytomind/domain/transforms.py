"""Flow cytometry data transformations and their registry."""
from __future__ import annotations
from collections import namedtuple
from flowkit import transforms as xf
from typing import Protocol, Type
from numpy.typing import NDArray
from .flow import TransformationRef

import numpy as np

__all__ = ["transform_registry", "get_default_transformations"]

TransParams = namedtuple('params', ['t', 'w', 'm', 'a'])
param = TransParams(
    t = 262144.0, # Upper Bound 2^18 for 18-bit data
    w = 0.5,      # number of decades in linear range
    m = 4.5,      # number of decades the true logarithmic scale approaches at the high end of the scale
    a = 0.0,      # Lower Bound (additional negative decades)
)

class TransformLike(Protocol):
    def apply(self, events: NDArray) -> NDArray: ...
    def inverse(self, events: NDArray) -> NDArray: ...

class IdentityTransform:
    """Identity transform: returns input as is."""
    def apply(self, events: NDArray) -> NDArray:
        return events

    def inverse(self, events: NDArray) -> NDArray:
        return events

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
        raise NotImplementedError("Inverse of RatioTransform is not defined.")

transform_registry: dict[str, Type[TransformLike]] = {
    "identity": IdentityTransform,
    "flin": xf.LinearTransform,
    "log": xf.LogTransform,
    "flojo_log": xf.WSPLogTransform, # pyright: ignore[reportAssignmentType]
    "flojo_biexp": xf.WSPBiexTransform,
    "asinh": xf.AsinhTransform,
    "logicle": xf.LogicleTransform,
    "hyperlog": xf.HyperlogTransform,
    "fratio": RatioTransform,
}

def get_default_transformations() -> dict[str, TransformationRef]:
    # TODO: get as input a ChannelRef to customize params per channel
    #       using PnB to set param_t
    transformations =  (
        TransformationRef(id="log",         type="log",      params={"param_t": param.t, "param_m": param.m}),
        TransformationRef(id="flin",        type="linear",   params={"param_t": param.t,                     "param_a": param.a}),
        TransformationRef(id="asinh",       type="asinh",    params={"param_t": param.t, "param_m": param.m, "param_a": param.a}),
        TransformationRef(id="logicle",     type="logicle",  params={"param_t": param.t, "param_m": param.m, "param_a": param.a, "param_w": param.w}),
        TransformationRef(id="hyperlog",    type="hyperlog", params={"param_t": param.t, "param_m": param.m, "param_a": param.a, "param_w": param.w}),
        TransformationRef(id="flojo_asinh", type="asinh",    params={"param_t": 12000.,  "param_m": 4.,      "param_a": 0.7}),
        TransformationRef(id="fratio",      type="ratio",    params={"param_a": 1., "param_b": 0., "param_c": 0.}),
        TransformationRef(id="flojo_log",   type="log",      params={"offset": 1., "decades": param.m}),
        TransformationRef(id="flojo_biexp", type="biexp",    params={"negative": 0., "width": -10., "positive": 4.418540, "max_value": 262144.000029}),
        TransformationRef(id="identity",    type="identity", params={}),
      )
    return {tref.id: tref for tref in transformations}
