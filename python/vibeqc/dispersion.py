"""Production standalone D3(BJ) correction runtime.

The correction is deliberately separate from SCF/Fock equations. A named
MethodIR such as PBE-D3(BJ) selects and identities the correction, while this
module evaluates only the additive geometry-dependent D3 energy and dE/dR.
"""

from __future__ import annotations

import ctypes
import typing
from dataclasses import dataclass
from typing import Self

import numpy as np
from vibeqc_compiler.method import (
    DispersionCorrectionPrimitive,
    MethodIR,
    MethodSpec,
    resolve_method,
)

from . import _native

if typing.TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class D3CorrectionResult:
    """One correction-only D3 batch result."""

    status: int
    energy: float
    gradient: np.ndarray | None
    backend: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class D3RuntimeDiagnostic:
    """Persistent owner evidence plus MethodIR/correction identities."""

    backend: str
    plan_host_bytes: int
    execution_host_bytes: int
    device_bytes: int
    table_bytes: int
    workspace_bytes: int
    maximum_bytes: int
    total_atoms: int
    system_count: int
    maximum_atoms: int
    method_ir_identity: str
    correction_identity: str
    table_sha256: str
    radii_sha256: str


def _backend_name(value: int) -> str:
    if value == _native.BACKEND_CPU_REFERENCE:
        return "cpu"
    if value == _native.BACKEND_CUDA:
        return "cuda"
    return f"backend-{value}"


def _method_ir(method: str | MethodSpec | MethodIR) -> MethodIR:
    if isinstance(method, MethodIR):
        return method
    return resolve_method(method)


def _correction(graph: MethodIR) -> DispersionCorrectionPrimitive:
    nodes = [
        primitive
        for primitive in graph.primitives
        if isinstance(primitive, DispersionCorrectionPrimitive)
    ]
    if len(nodes) != 1:
        raise ValueError(
            "D3 correction execution requires a MethodIR with exactly one "
            "DispersionCorrectionPrimitive"
        )
    return nodes[0]


def _normalize_system(value: typing.Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        atomic_numbers, coordinates = value
    except (TypeError, ValueError) as error:
        raise TypeError(
            "each D3 system must be (atomic_numbers, coordinates)"
        ) from error

    z_raw = np.asarray(atomic_numbers)
    if z_raw.ndim != 1 or z_raw.size == 0:
        raise ValueError("D3 atomic numbers must be a non-empty one-dimensional array")
    if not np.issubdtype(z_raw.dtype, np.integer):
        raise TypeError("D3 atomic numbers must be integers")
    if np.any(z_raw < 1) or np.any(z_raw > 86):
        raise NotImplementedError("production D3(BJ) supports H through Rn only")

    z = np.ascontiguousarray(z_raw, dtype=np.int32)
    if np.iscomplexobj(coordinates):
        raise TypeError("D3 coordinates must be real")
    xyz = np.ascontiguousarray(coordinates, dtype=np.float64)
    if xyz.shape != (z.size, 3):
        raise ValueError("D3 coordinates must have shape (natoms, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("prepared D3 coordinates must be finite")
    return z, xyz


class D3CorrectionBatch:
    """Persistent correction-only D3(BJ) owner for ragged CPU/CUDA fleets."""

    def __init__(
        self,
        method: str | MethodSpec | MethodIR,
        systems: Sequence,
        *,
        device: str = "cpu",
        device_id: int = 0,
        maximum_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
            raise ValueError("maximum_bytes must be a positive uint64 integer")

        graph = _method_ir(method)
        correction = _correction(graph)
        spec = correction.specification
        normalized = tuple(_normalize_system(system) for system in systems)
        if not normalized:
            raise ValueError("D3 correction batch requires at least one system")

        self.method_ir = graph
        self.method_ir_identity = graph.identity
        self.correction_identity = spec.identity
        self._counts = tuple(int(z.size) for z, _ in normalized)
        self._library = _native.load_library(
            device="cuda" if device == "cuda" else None,
            device_id=device_id,
        )
        if not hasattr(self._library, "vibeqc_d3_batch_prepare"):
            raise RuntimeError("loaded VIBEQC library does not expose production D3")

        table_sha256 = self._library.vibeqc_d3_table_sha256().decode("ascii")
        radii_sha256 = self._library.vibeqc_d3_radii_sha256().decode("ascii")
        if table_sha256 != spec.table_sha256 or radii_sha256 != spec.radii_sha256:
            raise NotImplementedError(
                "MethodIR D3 data identity does not match the compiled production tables"
            )
        self.table_sha256 = table_sha256
        self.radii_sha256 = radii_sha256

        backend = (
            _native.BACKEND_CUDA if device == "cuda" else _native.BACKEND_CPU_REFERENCE
        )
        context_descriptor = _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            int(device_id),
            backend,
        )
        self._context = ctypes.c_void_p()
        self._batch = ctypes.c_void_p()
        try:
            _native.check(
                self._library,
                self._library.vibeqc_context_create(
                    ctypes.byref(context_descriptor), ctypes.byref(self._context)
                ),
            )

            z_owners: list[np.ndarray] = []
            xyz_owners: list[np.ndarray] = []
            native_systems: list[_native.D3SystemDescriptor] = []
            for z, xyz in normalized:
                z_owners.append(z)
                xyz_owners.append(xyz)
                native_systems.append(
                    _native.D3SystemDescriptor(
                        ctypes.sizeof(_native.D3SystemDescriptor),
                        _native.ABI_VERSION,
                        z.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        z.size,
                    )
                )

            system_array_type = _native.D3SystemDescriptor * len(native_systems)
            model = _native.D3BjDescriptor(
                ctypes.sizeof(_native.D3BjDescriptor),
                _native.ABI_VERSION,
                _native.D3_DAMPING_BJ,
                spec.s6,
                spec.s8,
                spec.a1,
                spec.a2,
                spec.s9,
                0.0 if spec.cn_cutoff is None else spec.cn_cutoff,
                0.0 if spec.pair_cutoff is None else spec.pair_cutoff,
                spec.pair_switch_width,
                maximum_bytes,
            )
            _native.check(
                self._library,
                self._library.vibeqc_d3_batch_prepare(
                    self._context,
                    system_array_type(*native_systems),
                    len(native_systems),
                    ctypes.byref(model),
                    ctypes.byref(self._batch),
                ),
                context=self._context,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._batch.value:
            self._library.vibeqc_d3_batch_destroy(self._batch)
            self._batch.value = None
        if self._context.value:
            self._library.vibeqc_context_destroy(self._context)
            self._context.value = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self._batch.value or not self._context.value:
            raise RuntimeError("D3 correction batch is closed")

    def diagnostic(self) -> D3RuntimeDiagnostic:
        self._require_open()
        native = _native.D3RuntimeDiagnostic(
            ctypes.sizeof(_native.D3RuntimeDiagnostic),
            _native.ABI_VERSION,
        )
        _native.check(
            self._library,
            self._library.vibeqc_d3_batch_get_diagnostic(
                self._batch, ctypes.byref(native)
            ),
            context=self._context,
        )
        return D3RuntimeDiagnostic(
            backend=_backend_name(native.backend),
            plan_host_bytes=int(native.plan_host_bytes),
            execution_host_bytes=int(native.execution_host_bytes),
            device_bytes=int(native.device_bytes),
            table_bytes=int(native.table_bytes),
            workspace_bytes=int(native.workspace_bytes),
            maximum_bytes=int(native.maximum_bytes),
            total_atoms=int(native.total_atoms),
            system_count=int(native.system_count),
            maximum_atoms=int(native.maximum_atoms),
            method_ir_identity=self.method_ir_identity,
            correction_identity=self.correction_identity,
            table_sha256=self.table_sha256,
            radii_sha256=self.radii_sha256,
        )

    def execute(
        self,
        geometries: Sequence[np.ndarray | None] | None = None,
        *,
        gradients: bool = True,
    ) -> tuple[D3CorrectionResult, ...]:
        self._require_open()
        if geometries is not None and len(geometries) != len(self._counts):
            raise ValueError(
                "changed geometry list must match the prepared system count"
            )

        geometry_owners: list[np.ndarray] = []
        native_inputs = None
        input_count = 0
        if geometries is not None:
            input_descriptors: list[_native.D3BatchInputDescriptor] = []
            for atoms, geometry in zip(self._counts, geometries, strict=True):
                if geometry is None:
                    input_descriptors.append(
                        _native.D3BatchInputDescriptor(
                            ctypes.sizeof(_native.D3BatchInputDescriptor),
                            _native.ABI_VERSION,
                            None,
                            0,
                        )
                    )
                    continue

                if np.iscomplexobj(geometry):
                    raise TypeError("changed D3 coordinates must be real")
                xyz = np.ascontiguousarray(geometry, dtype=np.float64)
                if xyz.shape != (atoms, 3):
                    raise ValueError("changed D3 geometry has the wrong shape")
                geometry_owners.append(xyz)
                input_descriptors.append(
                    _native.D3BatchInputDescriptor(
                        ctypes.sizeof(_native.D3BatchInputDescriptor),
                        _native.ABI_VERSION,
                        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        xyz.size,
                    )
                )

            input_array_type = _native.D3BatchInputDescriptor * len(input_descriptors)
            native_inputs = input_array_type(*input_descriptors)
            input_count = len(input_descriptors)

        gradient_owners: list[np.ndarray | None] = []
        output_descriptors: list[_native.D3BatchItemResultDescriptor] = []
        for atoms in self._counts:
            gradient = np.empty((atoms, 3), dtype=np.float64) if gradients else None
            gradient_owners.append(gradient)
            output_descriptors.append(
                _native.D3BatchItemResultDescriptor(
                    ctypes.sizeof(_native.D3BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    0,
                    0.0,
                    None
                    if gradient is None
                    else gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    0 if gradient is None else gradient.size,
                    0,
                )
            )

        output_array_type = _native.D3BatchItemResultDescriptor * len(
            output_descriptors
        )
        native_outputs = output_array_type(*output_descriptors)
        _native.check(
            self._library,
            self._library.vibeqc_d3_batch_execute(
                self._batch,
                native_inputs,
                input_count,
                native_outputs,
                len(output_descriptors),
            ),
            context=self._context,
        )

        results = []
        for native, gradient in zip(native_outputs, gradient_owners, strict=True):
            raw = self._library.vibeqc_status_message(native.status)
            message = raw.decode("utf-8") if raw else f"status {native.status}"
            results.append(
                D3CorrectionResult(
                    status=int(native.status),
                    energy=float(native.energy),
                    gradient=(
                        gradient.copy()
                        if gradient is not None
                        and native.status == _native.STATUS_SUCCESS
                        else None
                    ),
                    backend=_backend_name(native.executed_backend),
                    message=message,
                )
            )
        return tuple(results)


def evaluate_d3_correction(
    method: str | MethodSpec | MethodIR,
    atomic_numbers: typing.Any,
    coordinates: typing.Any,
    *,
    device: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> D3CorrectionResult:
    """Evaluate one additive D3 correction and raise on an item-level failure."""

    with D3CorrectionBatch(
        method,
        [(atomic_numbers, coordinates)],
        device=device,
        device_id=device_id,
        maximum_bytes=maximum_bytes,
    ) as batch:
        result = batch.execute()[0]
    if not result.ok:
        raise RuntimeError(f"VIBEQC D3 item failed ({result.status}): {result.message}")
    return result
