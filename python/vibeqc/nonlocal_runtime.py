"""Bounded native fixed-grid execution for VV10/rVV10 pair mathematics.

This is a primitive runtime, not a public KS capability. It intentionally stays
separate from Calculator until the MethodIR/native KS contribution seam and
backend-specific capability gates are qualified.
"""

from __future__ import annotations

import ctypes
import typing
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

import numpy as np
from vibeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec

from . import _native


@dataclass(frozen=True)
class NonlocalRuntimeDiagnostic:
    """Owned numeric capacity and deterministic pair-work evidence."""

    backend: str
    workspace_bytes: int
    maximum_bytes: int
    pair_evaluations: int
    point_count: int
    tile_points: int


@dataclass(frozen=True)
class NonlocalFixedGridResult:
    """One fixed-grid nonlocal energy plus requested derivative families."""

    energy: float
    vrho: np.ndarray | None
    vsigma: np.ndarray | None
    point_derivative: np.ndarray | None
    weight_derivative: np.ndarray | None
    backend: str


def _positive_uint32(value: typing.Any, label: str) -> int:
    if type(value) is not int or not 0 < value < 2**32:
        raise ValueError(f"{label} must be a positive uint32 integer")
    return value


def _backend_name(value: int) -> str:
    if value == _native.BACKEND_CPU_REFERENCE:
        return "cpu"
    if value == _native.BACKEND_CUDA:
        return "cuda"
    return f"backend-{value}"


def _array(value: typing.Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError(f"{label} must be real")
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{label} requires shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


class NonlocalFixedGridPlan:
    """Persistent O(N_grid)-workspace CPU plan for VV10/rVV10 pair execution."""

    def __init__(
        self,
        spec: NonlocalCorrelationSpec,
        point_count: int,
        *,
        coefficient: Fraction = Fraction(1),
        tile_points: int = 256,
        maximum_bytes: int = 256 << 20,
        device: str = "cpu",
    ) -> None:
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("spec must be NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("coefficient must be a positive Fraction")
        point_count = _positive_uint32(point_count, "point_count")
        if point_count > (2**32 - 1) // 3:
            raise ValueError(
                "point_count exceeds the flattened-coordinate uint32 domain"
            )
        tile_points = _positive_uint32(tile_points, "tile_points")
        if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
            raise ValueError("maximum_bytes must be a positive uint64 integer")
        if device != "cpu":
            raise NotImplementedError("native VV10 CUDA lowerer is not yet qualified")

        self.spec = spec
        self.coefficient = coefficient
        self.point_count = point_count
        self._library = _native.load_library()
        if not hasattr(self._library, "vibeqc_nonlocal_plan_prepare"):
            raise RuntimeError(
                "loaded VIBEQC library does not expose native nonlocal execution"
            )
        self._context = ctypes.c_void_p()
        self._plan = ctypes.c_void_p()
        context_descriptor = _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            0,
            _native.BACKEND_CPU_REFERENCE,
        )
        variant = {
            "vv10": _native.NONLOCAL_VV10,
            "rvv10": _native.NONLOCAL_RVV10,
        }.get(spec.variant)
        if variant is None:
            raise NotImplementedError(f"unsupported nonlocal variant {spec.variant!r}")
        model = _native.NonlocalDescriptor(
            ctypes.sizeof(_native.NonlocalDescriptor),
            _native.ABI_VERSION,
            variant,
            float(spec.b),
            float(spec.c),
            float(coefficient),
            point_count,
            tile_points,
            maximum_bytes,
        )
        try:
            _native.check(
                self._library,
                self._library.vibeqc_context_create(
                    ctypes.byref(context_descriptor), ctypes.byref(self._context)
                ),
            )
            _native.check(
                self._library,
                self._library.vibeqc_nonlocal_plan_prepare(
                    self._context, ctypes.byref(model), ctypes.byref(self._plan)
                ),
                context=self._context,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._plan.value:
            self._library.vibeqc_nonlocal_plan_destroy(self._plan)
            self._plan.value = None
        if self._context.value:
            self._library.vibeqc_context_destroy(self._context)
            self._context.value = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self._plan.value or not self._context.value:
            raise RuntimeError("native nonlocal plan is closed")

    def diagnostic(self) -> NonlocalRuntimeDiagnostic:
        self._require_open()
        native = _native.NonlocalRuntimeDiagnostic(
            ctypes.sizeof(_native.NonlocalRuntimeDiagnostic),
            _native.ABI_VERSION,
        )
        _native.check(
            self._library,
            self._library.vibeqc_nonlocal_plan_get_diagnostic(
                self._plan, ctypes.byref(native)
            ),
            context=self._context,
        )
        return NonlocalRuntimeDiagnostic(
            backend=_backend_name(native.backend),
            workspace_bytes=int(native.workspace_bytes),
            maximum_bytes=int(native.maximum_bytes),
            pair_evaluations=int(native.pair_evaluations),
            point_count=int(native.point_count),
            tile_points=int(native.tile_points),
        )

    def execute(
        self,
        coordinates: typing.Any,
        weights: typing.Any,
        density: typing.Any,
        density_gradient: typing.Any,
        *,
        features: bool = True,
        geometry: bool = False,
    ) -> NonlocalFixedGridResult:
        self._require_open()
        if type(features) is not bool or type(geometry) is not bool:
            raise TypeError("features and geometry must be boolean")
        n = self.point_count
        coordinates = _array(coordinates, (n, 3), "coordinates")
        weights = _array(weights, (n,), "weights")
        density = _array(density, (n,), "density")
        density_gradient = _array(density_gradient, (n, 3), "density_gradient")
        if np.any(density <= 0):
            raise ValueError("native VV10 execution requires strictly positive density")

        vrho = np.empty(n, dtype=np.float64) if features else None
        vsigma = np.empty(n, dtype=np.float64) if features else None
        point = np.empty((n, 3), dtype=np.float64) if geometry else None
        weight = np.empty(n, dtype=np.float64) if geometry else None

        def pointer(value: np.ndarray | None) -> typing.Any:
            return (
                None
                if value is None
                else value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )

        input_descriptor = _native.NonlocalInputDescriptor(
            ctypes.sizeof(_native.NonlocalInputDescriptor),
            _native.ABI_VERSION,
            pointer(coordinates),
            coordinates.size,
            pointer(weights),
            weights.size,
            pointer(density),
            density.size,
            pointer(density_gradient),
            density_gradient.size,
        )
        output = _native.NonlocalResultDescriptor(
            ctypes.sizeof(_native.NonlocalResultDescriptor),
            _native.ABI_VERSION,
            0.0,
            pointer(vrho),
            0 if vrho is None else vrho.size,
            pointer(vsigma),
            0 if vsigma is None else vsigma.size,
            pointer(point),
            0 if point is None else point.size,
            pointer(weight),
            0 if weight is None else weight.size,
            0,
        )
        _native.check(
            self._library,
            self._library.vibeqc_nonlocal_plan_execute(
                self._plan, ctypes.byref(input_descriptor), ctypes.byref(output)
            ),
            context=self._context,
        )
        return NonlocalFixedGridResult(
            energy=float(output.energy),
            vrho=None if vrho is None else vrho.copy(),
            vsigma=None if vsigma is None else vsigma.copy(),
            point_derivative=None if point is None else point.copy(),
            weight_derivative=None if weight is None else weight.copy(),
            backend=_backend_name(output.executed_backend),
        )
