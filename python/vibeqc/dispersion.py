"""Production standalone D3(BJ) and D4(BJ)-EEQ correction runtimes.

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
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method import (
    D3Spec,
    D4Spec,
    DispersionCorrectionPrimitive,
    GeometricCounterpoisePrimitive,
    MethodIR,
    MethodSpec,
    r2scan3c_gcp,
    resolve_method,
)
from vibeqc_compiler.method.d4_derivative import PRODUCTION_D4_EEQ_DERIVATIVE

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
    correction_identity: str
    variant_identity: str
    provider_identity: str
    scheduler_identity: str

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
    variant_identity: str
    provider_identity: str
    scheduler_identity: str


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


def _d4_correction(graph: MethodIR) -> DispersionCorrectionPrimitive:
    correction = _correction(graph)
    if not isinstance(correction.specification, D4Spec):
        raise TypeError("D4 correction execution requires a D4Spec")
    return correction


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
        if not isinstance(spec, D3Spec):
            raise TypeError("D3 correction execution requires a D3Spec")
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
        self.provider_identity = self._library.vibeqc_d3_provider_identity().decode(
            "ascii"
        )
        self.scheduler_identity = self._library.vibeqc_d3_scheduler_identity().decode(
            "ascii"
        )

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
                _native.D3_DAMPING_ZERO
                if spec.damping == "zero"
                else _native.D3_DAMPING_BJ,
                spec.s6,
                spec.s8,
                spec.a1,
                spec.a2,
                spec.s9,
                0.0 if spec.cn_cutoff is None else spec.cn_cutoff,
                0.0 if spec.pair_cutoff is None else spec.pair_cutoff,
                spec.pair_switch_width,
                maximum_bytes,
                spec.rs6,
                spec.rs8,
                spec.alp,
                0.0 if spec.atm_cutoff is None else spec.atm_cutoff,
                spec.atm_switch_width,
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
            raw_variant = self._library.vibeqc_d3_batch_variant_identity(self._batch)
            if not raw_variant:
                raise RuntimeError(
                    "production D3 did not publish a prepared variant identity"
                )
            self.variant_identity = raw_variant.decode("ascii")
            expected_variant = (
                "d3.zero-two-body"
                if spec.damping == "zero"
                else ("d3.bj-atm" if spec.s9 else "d3.bj-two-body")
            )
            if self.variant_identity != expected_variant:
                raise NotImplementedError(
                    f"prepared D3 variant {self.variant_identity!r} does not match "
                    f"MethodIR request {expected_variant!r}"
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
            variant_identity=self.variant_identity,
            provider_identity=self.provider_identity,
            scheduler_identity=self.scheduler_identity,
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
                    correction_identity=self.correction_identity,
                    variant_identity=self.variant_identity,
                    provider_identity=self.provider_identity,
                    scheduler_identity=self.scheduler_identity,
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


@dataclass(frozen=True)
class D4CorrectionResult:
    """One complete D4(BJ)-EEQ correction result."""

    status: int
    energy: float
    two_body_energy: float
    atm_energy: float
    gradient: np.ndarray | None
    charges: np.ndarray | None
    backend: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class D4RuntimeDiagnostic:
    """Resource, scheduling, replay and scientific-identity evidence."""

    backend: str
    profile: str
    plan_host_bytes: int
    execution_host_bytes: int
    device_bytes: int
    table_bytes: int
    workspace_bytes: int
    maximum_bytes: int
    total_atoms: int
    system_count: int
    maximum_atoms: int
    worker_blocks: int
    workspace_slots: int
    execution_count: int
    unchanged_geometry_replays: int
    changed_geometry_replays: int
    coordinate_h2d_bytes: int
    kernel_launches: int
    atm_enabled: bool
    method_ir_identity: str
    correction_identity: str
    table_sha256: str
    charge_parameter_sha256: str
    derivative_identity: str
    provider_identity: str
    scheduler_identity: str
    cache_identity: str
    peak_host_bytes: int
    peak_device_bytes: int


def _normalize_d4_system(
    value: typing.Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        atomic_numbers, coordinates, total_charge = value
    except (TypeError, ValueError) as error:
        raise TypeError(
            "each D4 system must be (atomic_numbers, coordinates, total_charge)"
        ) from error
    z, xyz = _normalize_system((atomic_numbers, coordinates))
    if isinstance(total_charge, bool) or not isinstance(
        total_charge, (int, float, np.integer, np.floating)
    ):
        raise TypeError("D4 total charge must be a finite real scalar")
    charge = float(total_charge)
    if not np.isfinite(charge):
        raise ValueError("D4 total charge must be finite")
    return z, xyz, charge


class D4CorrectionBatch:
    """Persistent complete D4(BJ)-EEQ owner for ragged CPU/CUDA fleets."""

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
        correction = _d4_correction(graph)
        spec = typing.cast("D4Spec", correction.specification)
        normalized = tuple(_normalize_d4_system(system) for system in systems)
        if not normalized:
            raise ValueError("D4 correction batch requires at least one system")

        self.method_ir = graph
        self.method_ir_identity = graph.identity
        self.correction_identity = spec.identity
        self._counts = tuple(int(z.size) for z, _, _ in normalized)
        self._library = _native.load_library(
            device="cuda" if device == "cuda" else None,
            device_id=device_id,
        )
        if not hasattr(self._library, "vibeqc_d4_batch_prepare"):
            raise RuntimeError("loaded VIBEQC library does not expose production D4")

        table_sha256 = self._library.vibeqc_d4_table_sha256().decode("ascii")
        charge_sha256 = self._library.vibeqc_d4_charge_parameter_sha256().decode(
            "ascii"
        )
        derivative_identity = self._library.vibeqc_d4_derivative_identity().decode(
            "ascii"
        )
        provider_identity = self._library.vibeqc_d4_provider_identity().decode("ascii")
        scheduler_identity = self._library.vibeqc_d4_scheduler_identity().decode(
            "ascii"
        )
        if (
            table_sha256 != spec.table_sha256
            or charge_sha256 != spec.charge_parameter_sha256
        ):
            raise NotImplementedError(
                "MethodIR D4 data identity does not match compiled production tables"
            )
        if derivative_identity != PRODUCTION_D4_EEQ_DERIVATIVE.identity:
            raise RuntimeError(
                "compiler/native D4 derivative lowering identities disagree"
            )
        self.table_sha256 = table_sha256
        self.charge_parameter_sha256 = charge_sha256
        self.derivative_identity = derivative_identity
        self.provider_identity = provider_identity
        self.scheduler_identity = scheduler_identity
        self.cache_identity = canonical_hash(
            {
                "provider": provider_identity,
                "scheduler": scheduler_identity,
                "method_ir": self.method_ir_identity,
                "correction": self.correction_identity,
                "table_sha256": table_sha256,
                "charge_parameter_sha256": charge_sha256,
                "derivative": derivative_identity,
            }
        )
        profile = {
            "standard": _native.D4_PROFILE_STANDARD_EEQ,
            "r2scan3c": _native.D4_PROFILE_R2SCAN3C_EEQ,
        }[spec.profile]

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
            native_systems: list[_native.D4SystemDescriptor] = []
            for z, xyz, total_charge in normalized:
                z_owners.append(z)
                xyz_owners.append(xyz)
                native_systems.append(
                    _native.D4SystemDescriptor(
                        ctypes.sizeof(_native.D4SystemDescriptor),
                        _native.ABI_VERSION,
                        z.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        z.size,
                        total_charge,
                    )
                )
            system_array_type = _native.D4SystemDescriptor * len(native_systems)
            model = _native.D4BjEeqDescriptor(
                ctypes.sizeof(_native.D4BjEeqDescriptor),
                _native.ABI_VERSION,
                profile,
                spec.s6,
                spec.s8,
                spec.s9,
                spec.a1,
                spec.a2,
                spec.ga,
                spec.gc,
                spec.cn_cutoff,
                spec.pair_cutoff,
                spec.atm_cutoff,
                maximum_bytes,
            )
            _native.check(
                self._library,
                self._library.vibeqc_d4_batch_prepare(
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
            self._library.vibeqc_d4_batch_destroy(self._batch)
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
            raise RuntimeError("D4 correction batch is closed")

    def diagnostic(self) -> D4RuntimeDiagnostic:
        self._require_open()
        native = _native.D4RuntimeDiagnostic(
            ctypes.sizeof(_native.D4RuntimeDiagnostic), _native.ABI_VERSION
        )
        _native.check(
            self._library,
            self._library.vibeqc_d4_batch_get_diagnostic(
                self._batch, ctypes.byref(native)
            ),
            context=self._context,
        )
        profile = {
            _native.D4_PROFILE_STANDARD_EEQ: "standard",
            _native.D4_PROFILE_R2SCAN3C_EEQ: "r2scan3c",
        }.get(native.profile, f"profile-{native.profile}")
        return D4RuntimeDiagnostic(
            backend=_backend_name(native.backend),
            profile=profile,
            plan_host_bytes=int(native.plan_host_bytes),
            execution_host_bytes=int(native.execution_host_bytes),
            device_bytes=int(native.device_bytes),
            table_bytes=int(native.table_bytes),
            workspace_bytes=int(native.workspace_bytes),
            maximum_bytes=int(native.maximum_bytes),
            total_atoms=int(native.total_atoms),
            system_count=int(native.system_count),
            maximum_atoms=int(native.maximum_atoms),
            worker_blocks=int(native.worker_blocks),
            workspace_slots=int(native.workspace_slots),
            execution_count=int(native.execution_count),
            unchanged_geometry_replays=int(native.unchanged_geometry_replays),
            changed_geometry_replays=int(native.changed_geometry_replays),
            coordinate_h2d_bytes=int(native.coordinate_h2d_bytes),
            kernel_launches=int(native.kernel_launches),
            atm_enabled=bool(native.atm_enabled),
            method_ir_identity=self.method_ir_identity,
            correction_identity=self.correction_identity,
            table_sha256=self.table_sha256,
            charge_parameter_sha256=self.charge_parameter_sha256,
            derivative_identity=self.derivative_identity,
            provider_identity=self.provider_identity,
            scheduler_identity=self.scheduler_identity,
            cache_identity=self.cache_identity,
            peak_host_bytes=int(native.plan_host_bytes + native.execution_host_bytes),
            peak_device_bytes=int(native.device_bytes),
        )

    def execute(
        self,
        geometries: Sequence[np.ndarray | None] | None = None,
        *,
        gradients: bool = True,
        charges: bool = True,
    ) -> tuple[D4CorrectionResult, ...]:
        self._require_open()
        if geometries is not None and len(geometries) != len(self._counts):
            raise ValueError(
                "changed geometry list must match the prepared system count"
            )
        geometry_owners: list[np.ndarray] = []
        native_inputs = None
        input_count = 0
        if geometries is not None:
            input_descriptors: list[_native.D4BatchInputDescriptor] = []
            for atoms, geometry in zip(self._counts, geometries, strict=True):
                if geometry is None:
                    input_descriptors.append(
                        _native.D4BatchInputDescriptor(
                            ctypes.sizeof(_native.D4BatchInputDescriptor),
                            _native.ABI_VERSION,
                            None,
                            0,
                        )
                    )
                    continue
                if np.iscomplexobj(geometry):
                    raise TypeError("changed D4 coordinates must be real")
                xyz = np.ascontiguousarray(geometry, dtype=np.float64)
                if xyz.shape != (atoms, 3) or not np.isfinite(xyz).all():
                    raise ValueError(
                        "changed D4 geometry must be finite with the prepared shape"
                    )
                geometry_owners.append(xyz)
                input_descriptors.append(
                    _native.D4BatchInputDescriptor(
                        ctypes.sizeof(_native.D4BatchInputDescriptor),
                        _native.ABI_VERSION,
                        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        xyz.size,
                    )
                )
            input_array_type = _native.D4BatchInputDescriptor * len(input_descriptors)
            native_inputs = input_array_type(*input_descriptors)
            input_count = len(input_descriptors)

        gradient_owners: list[np.ndarray | None] = []
        charge_owners: list[np.ndarray | None] = []
        output_descriptors: list[_native.D4BatchItemResultDescriptor] = []
        for atoms in self._counts:
            gradient = np.empty((atoms, 3), dtype=np.float64) if gradients else None
            charge_values = np.empty(atoms, dtype=np.float64) if charges else None
            gradient_owners.append(gradient)
            charge_owners.append(charge_values)
            output_descriptors.append(
                _native.D4BatchItemResultDescriptor(
                    ctypes.sizeof(_native.D4BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    None
                    if gradient is None
                    else gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    0 if gradient is None else gradient.size,
                    None
                    if charge_values is None
                    else charge_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    0 if charge_values is None else charge_values.size,
                    0,
                )
            )
        output_array_type = _native.D4BatchItemResultDescriptor * len(
            output_descriptors
        )
        native_outputs = output_array_type(*output_descriptors)
        _native.check(
            self._library,
            self._library.vibeqc_d4_batch_execute(
                self._batch,
                native_inputs,
                input_count,
                native_outputs,
                len(output_descriptors),
            ),
            context=self._context,
        )
        results = []
        for native, gradient, charge_values in zip(
            native_outputs, gradient_owners, charge_owners, strict=True
        ):
            raw = self._library.vibeqc_status_message(native.status)
            message = raw.decode("utf-8") if raw else f"status {native.status}"
            success = native.status == _native.STATUS_SUCCESS
            results.append(
                D4CorrectionResult(
                    status=int(native.status),
                    energy=float(native.energy),
                    two_body_energy=float(native.two_body_energy),
                    atm_energy=float(native.atm_energy),
                    gradient=gradient.copy()
                    if gradient is not None and success
                    else None,
                    charges=(
                        charge_values.copy()
                        if charge_values is not None and success
                        else None
                    ),
                    backend=_backend_name(native.executed_backend),
                    message=message,
                )
            )
        return tuple(results)


@dataclass(frozen=True)
class GCPCorrectionResult:
    """One canonical r2SCAN-3c gCP correction result."""

    status: int
    energy: float
    gradient: np.ndarray | None
    backend: str
    message: str
    provider_identity: str
    correction_identity: str

    @property
    def ok(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class R2SCAN3CCorrectionResult:
    """Explicit D4 + gCP component sum for one canonical r2SCAN-3c item."""

    status: int
    energy: float
    gradient: np.ndarray | None
    backend: str
    message: str
    d4: D4CorrectionResult
    gcp: GCPCorrectionResult

    @property
    def ok(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class R2SCAN3CRuntimeDiagnostic:
    """Mixed-backend correction evidence for the canonical composite owner."""

    method_ir_identity: str
    d4: D4RuntimeDiagnostic
    gcp_backend: str
    gcp_provider_identity: str
    gcp_correction_identity: str


def _gcp_correction(graph: MethodIR) -> GeometricCounterpoisePrimitive:
    nodes = [
        primitive
        for primitive in graph.primitives
        if isinstance(primitive, GeometricCounterpoisePrimitive)
    ]
    if len(nodes) != 1:
        raise ValueError(
            "r2SCAN-3c execution requires exactly one GeometricCounterpoisePrimitive"
        )
    if nodes[0].specification != r2scan3c_gcp():
        raise NotImplementedError(
            "native gCP execution is qualified only for the canonical r2SCAN-3c profile"
        )
    return nodes[0]


def _evaluate_gcp_result(
    graph: MethodIR,
    atomic_numbers: typing.Any,
    coordinates: typing.Any,
    *,
    gradients: bool,
) -> GCPCorrectionResult:
    primitive = _gcp_correction(graph)
    z, xyz = _normalize_system((atomic_numbers, coordinates))
    library = _native.load_library()
    evaluator = getattr(library, "vibeqc_r2scan3c_gcp_evaluate", None)
    if evaluator is None:
        raise RuntimeError(
            "loaded VIBEQC library does not expose production r2SCAN-3c gCP"
        )
    provider = library.vibeqc_r2scan3c_gcp_provider_identity().decode("ascii")
    energy = ctypes.c_double()
    gradient = np.empty((z.size, 3), dtype=np.float64) if gradients else None
    status = int(
        evaluator(
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            z.size,
            xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            xyz.size,
            ctypes.byref(energy),
            None
            if gradient is None
            else gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            0 if gradient is None else gradient.size,
        )
    )
    raw = library.vibeqc_status_message(status)
    return GCPCorrectionResult(
        status=status,
        energy=float(energy.value)
        if status == _native.STATUS_SUCCESS
        else float("nan"),
        gradient=gradient.copy()
        if gradient is not None and status == _native.STATUS_SUCCESS
        else None,
        backend="cpu",
        message=raw.decode("utf-8") if raw else f"status {status}",
        provider_identity=provider,
        correction_identity=canonical_hash(primitive.semantic_payload()),
    )


def evaluate_r2scan3c_gcp(
    method: str | MethodSpec | MethodIR,
    atomic_numbers: typing.Any,
    coordinates: typing.Any,
    *,
    gradients: bool = True,
) -> GCPCorrectionResult:
    """Evaluate canonical r2SCAN-3c gCP using the native CPU production kernel."""

    graph = _method_ir(method)
    result = _evaluate_gcp_result(
        graph, atomic_numbers, coordinates, gradients=gradients
    )
    if not result.ok:
        raise RuntimeError(
            f"VIBEQC r2SCAN-3c gCP item failed ({result.status}): {result.message}"
        )
    return result


class R2SCAN3CCorrectionBatch:
    """Compose the canonical D4(BJ)-EEQ-ATM and gCP correction primitives once."""

    def __init__(
        self,
        method: str | MethodSpec | MethodIR,
        systems: Sequence,
        *,
        device: str = "cpu",
        device_id: int = 0,
        maximum_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        graph = _method_ir(method)
        expected = resolve_method("R2SCAN-3c", spin=graph.spin)
        if graph.manifest_identity != expected.manifest_identity:
            raise NotImplementedError(
                "composite correction execution requires the canonical r2SCAN-3c MethodIR"
            )
        _gcp_correction(graph)
        normalized = tuple(_normalize_d4_system(system) for system in systems)
        if not normalized:
            raise ValueError("r2SCAN-3c correction batch requires at least one system")
        for z, _, _ in normalized:
            graph.preflight_atomic_numbers(tuple(int(value) for value in z))
        self.method_ir = graph
        self.method_ir_identity = graph.identity
        self._prepared = tuple(
            (z.copy(), xyz.copy(), charge) for z, xyz, charge in normalized
        )
        library = _native.load_library()
        if (
            getattr(library, "vibeqc_r2scan3c_gcp_evaluate", None) is None
            or getattr(library, "vibeqc_r2scan3c_gcp_provider_identity", None) is None
        ):
            raise RuntimeError(
                "loaded VIBEQC library does not expose production r2SCAN-3c gCP"
            )
        self._gcp_provider_identity = (
            library.vibeqc_r2scan3c_gcp_provider_identity().decode("ascii")
        )
        self._gcp_correction_identity = canonical_hash(
            _gcp_correction(graph).semantic_payload()
        )
        self._d4 = D4CorrectionBatch(
            graph,
            normalized,
            device=device,
            device_id=device_id,
            maximum_bytes=maximum_bytes,
        )

    def close(self) -> None:
        self._d4.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def diagnostic(self) -> R2SCAN3CRuntimeDiagnostic:
        return R2SCAN3CRuntimeDiagnostic(
            method_ir_identity=self.method_ir_identity,
            d4=self._d4.diagnostic(),
            gcp_backend="cpu",
            gcp_provider_identity=self._gcp_provider_identity,
            gcp_correction_identity=self._gcp_correction_identity,
        )

    def execute(
        self,
        geometries: Sequence[np.ndarray | None] | None = None,
        *,
        gradients: bool = True,
    ) -> tuple[R2SCAN3CCorrectionResult, ...]:
        if geometries is not None and len(geometries) != len(self._prepared):
            raise ValueError(
                "changed geometry list must match the prepared system count"
            )
        effective: list[np.ndarray] = []
        if geometries is None:
            effective = [xyz for _, xyz, _ in self._prepared]
        else:
            for (z, prepared, _), geometry in zip(
                self._prepared, geometries, strict=True
            ):
                if geometry is None:
                    effective.append(prepared)
                    continue
                _, xyz = _normalize_system((z, geometry))
                effective.append(xyz)

        d4_results = self._d4.execute(
            None if geometries is None else effective,
            gradients=gradients,
            charges=True,
        )
        results: list[R2SCAN3CCorrectionResult] = []
        for (z, _, _), xyz, d4 in zip(
            self._prepared, effective, d4_results, strict=True
        ):
            gcp = _evaluate_gcp_result(self.method_ir, z, xyz, gradients=gradients)
            status = d4.status if not d4.ok else gcp.status
            success = d4.ok and gcp.ok
            gradient = None
            if success and gradients:
                if d4.gradient is None or gcp.gradient is None:
                    raise RuntimeError(
                        "successful r2SCAN-3c correction omitted a requested gradient"
                    )
                gradient = d4.gradient + gcp.gradient
            backend = d4.backend if d4.backend == "cpu" else f"{d4.backend}+cpu"
            message = (
                "success"
                if success
                else f"D4 correction failed: {d4.message}"
                if not d4.ok
                else f"gCP correction failed: {gcp.message}"
            )
            results.append(
                R2SCAN3CCorrectionResult(
                    status=status,
                    energy=d4.energy + gcp.energy if success else float("nan"),
                    gradient=gradient,
                    backend=backend,
                    message=message,
                    d4=d4,
                    gcp=gcp,
                )
            )
        return tuple(results)


def evaluate_r2scan3c_correction(
    method: str | MethodSpec | MethodIR,
    atomic_numbers: typing.Any,
    coordinates: typing.Any,
    *,
    total_charge: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
    gradients: bool = True,
) -> R2SCAN3CCorrectionResult:
    """Evaluate the complete canonical r2SCAN-3c D4 + gCP correction."""

    with R2SCAN3CCorrectionBatch(
        method,
        [(atomic_numbers, coordinates, total_charge)],
        device=device,
        device_id=device_id,
        maximum_bytes=maximum_bytes,
    ) as batch:
        result = batch.execute(gradients=gradients)[0]
    if not result.ok:
        raise RuntimeError(
            f"VIBEQC r2SCAN-3c correction failed ({result.status}): {result.message}"
        )
    return result


def evaluate_d4_correction(
    method: str | MethodSpec | MethodIR,
    atomic_numbers: typing.Any,
    coordinates: typing.Any,
    *,
    total_charge: float = 0.0,
    device: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> D4CorrectionResult:
    """Evaluate one complete D4(BJ)-EEQ correction and raise on item failure."""

    with D4CorrectionBatch(
        method,
        [(atomic_numbers, coordinates, total_charge)],
        device=device,
        device_id=device_id,
        maximum_bytes=maximum_bytes,
    ) as batch:
        result = batch.execute()[0]
    if not result.ok:
        raise RuntimeError(f"VIBEQC D4 item failed ({result.status}): {result.message}")
    return result
