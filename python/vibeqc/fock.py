"""Prepared method-neutral native J/K sources, fixed-density evaluation and SCF."""

from __future__ import annotations

import ctypes as ct
import math
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash

from . import _native

_OPERATORS = {"full_range": 0, "short_range": 1, "long_range": 2}
_APPROXIMATIONS = {"exact": 0, "density_fitted": 1}
_SPINS = {"restricted": 0, "unrestricted": 1}
_SCHEDULES = (
    "cpu_reference",
    "cuda_fused",
    "standard_df",
    "cpu_independent",
    "cuda_independent",
)
_DOUBLE = ct.POINTER(ct.c_double)


@dataclass(frozen=True)
class FockTerm:
    """One explicit operator/approximation and its signed Fock coefficient."""

    present: bool = True
    coefficient: float = 1.0
    operator: str = "full_range"
    omega: float = 0.0
    approximation: str = "exact"

    def __post_init__(self):
        if type(self.present) is not bool:
            raise TypeError("Fock presence must be bool")
        if self.operator not in _OPERATORS or self.approximation not in _APPROXIMATIONS:
            raise ValueError("unknown Fock operator/approximation")
        if (
            not math.isfinite(self.coefficient)
            or not math.isfinite(self.omega)
            or self.omega < 0
        ):
            raise ValueError("Fock coefficients and nonnegative omega must be finite")


@dataclass(frozen=True)
class FockBuildSpec:
    """Version-1 FP64 semantics. Use ``hf`` for spin-dependent HF defaults.

    Presence differs from a zero coefficient: a present term still requests
    its raw matrix. Reserved range operators fail during native preflight.
    """

    spin: str = "restricted"
    derivative_order: int = 1
    coulomb: FockTerm = field(default_factory=FockTerm)
    exchange: FockTerm = field(default_factory=lambda: FockTerm(coefficient=-0.5))

    def __post_init__(self):
        if (
            self.spin not in _SPINS
            or type(self.derivative_order) is not int
            or self.derivative_order not in (0, 1)
        ):
            raise ValueError("unsupported Fock spin/derivative order")
        if not isinstance(self.coulomb, FockTerm) or not isinstance(
            self.exchange, FockTerm
        ):
            raise TypeError("Fock terms must be FockTerm objects")

    @classmethod
    def hf(
        cls, spin="restricted", *, coulomb="exact", exchange="exact", derivative_order=1
    ):
        """Standard HF coefficients with independently requested approximations."""
        return cls(
            spin,
            derivative_order,
            FockTerm(approximation=coulomb),
            FockTerm(
                coefficient=-1.0 if spin == "unrestricted" else -0.5,
                approximation=exchange,
            ),
        )

    def to_dict(self):
        return {"version": 1, **asdict(self)}


class _Term(ct.Structure):
    _fields_ = [
        ("present", ct.c_int32),
        ("coefficient", ct.c_double),
        ("op", ct.c_int32),
        ("omega", ct.c_double),
        ("approximation", ct.c_int32),
    ]


class _Spec(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("spec_version", ct.c_uint32),
        ("spin", ct.c_int32),
        ("derivative_order", ct.c_uint32),
        ("coulomb", _Term),
        ("exchange", _Term),
    ]


class _Controls(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("screening_tolerance", ct.c_double),
        ("metric_relative_threshold", ct.c_double),
        ("device_budget_bytes", ct.c_uint64),
    ]


class _Result(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("matrix_count", ct.c_uint64),
        ("gradient_count", ct.c_uint64),
        *[
            (name, _DOUBLE)
            for name in (
                "coulomb",
                "exchange_alpha",
                "exchange_beta",
                "fock_alpha",
                "fock_beta",
                "gradient",
            )
        ],
        *[
            (name, ct.c_double)
            for name in (
                "energy_one_electron",
                "energy_two_electron",
                "nuclear_repulsion",
            )
        ],
    ]


class _Diagnostic(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("requested", _Spec),
        ("resolved", _Spec),
        ("backend", ct.c_int32),
        ("resolved_schedule", ct.c_int32),
        ("source_schedule", ct.c_int32),
        *[
            (name, ct.c_uint64)
            for name in (
                "nbf",
                "coordinate_count",
                "device_bytes",
                "device_budget_bytes",
                "auxiliary_rank",
                "auxiliary_tile",
            )
        ],
        ("screening_tolerance", ct.c_double),
        ("metric_relative_threshold", ct.c_double),
        ("df_streamed", ct.c_int32),
        ("direct_schedule", ct.c_char * 96),
        ("df_value_backend", ct.c_char * 48),
        ("df_value_mapping", ct.c_char * 32),
        ("df_response_mapping", ct.c_char * 32),
        ("one_electron_value_backend", ct.c_char * 32),
        ("one_electron_value_mapping", ct.c_char * 32),
        ("one_electron_response_mapping", ct.c_char * 32),
    ]


class _ScfControls(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("max_iterations", ct.c_uint32),
        ("diis_history", ct.c_uint32),
        ("energy_tolerance", ct.c_double),
        ("density_tolerance", ct.c_double),
    ]


class _ScfResult(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("density", _DOUBLE),
        ("forces", _DOUBLE),
        ("density_count", ct.c_uint64),
        ("force_count", ct.c_uint64),
        ("energy", ct.c_double),
        ("energy_change", ct.c_double),
        ("density_rms", ct.c_double),
        ("iterations", ct.c_uint32),
        ("initial_density_used", ct.c_int32),
        ("fock_builds", ct.c_uint64),
    ]


def _descriptor(kind, **values):
    return kind(struct_size=ct.sizeof(kind), abi_version=_native.ABI_VERSION, **values)


def _spec(spec):
    def term(t):
        return _Term(
            int(t.present),
            t.coefficient,
            _OPERATORS[t.operator],
            t.omega,
            _APPROXIMATIONS[t.approximation],
        )

    return _descriptor(
        _Spec,
        spec_version=1,
        spin=_SPINS[spec.spin],
        derivative_order=spec.derivative_order,
        coulomb=term(spec.coulomb),
        exchange=term(spec.exchange),
    )


def _spec_dict(spec):
    def term(t):
        return {
            "present": bool(t.present),
            "coefficient": t.coefficient,
            "operator": tuple(_OPERATORS)[t.op],
            "omega": t.omega,
            "approximation": tuple(_APPROXIMATIONS)[t.approximation],
        }

    return {
        "version": spec.spec_version,
        "spin": tuple(_SPINS)[spec.spin],
        "derivative_order": spec.derivative_order,
        "coulomb": term(spec.coulomb),
        "exchange": term(spec.exchange),
    }


def _bind(lib):
    lib.vibeqc_get_source_identity.argtypes = []
    lib.vibeqc_get_source_identity.restype = ct.c_char_p
    lib.vibeqc_fock_plan_create.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(_Spec),
        ct.POINTER(_Controls),
        ct.POINTER(ct.c_void_p),
    ]
    lib.vibeqc_fock_plan_create.restype = ct.c_int32
    lib.vibeqc_fock_plan_destroy.argtypes = [ct.c_void_p]
    lib.vibeqc_fock_plan_destroy.restype = None
    lib.vibeqc_fock_plan_last_error.argtypes = [ct.c_void_p]
    lib.vibeqc_fock_plan_last_error.restype = ct.c_char_p
    lib.vibeqc_fock_plan_evaluate.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_uint64,
        _DOUBLE,
        ct.c_uint64,
        ct.POINTER(_Result),
    ]
    lib.vibeqc_fock_plan_evaluate.restype = ct.c_int32
    lib.vibeqc_fock_plan_solve.argtypes = [
        ct.c_void_p,
        ct.POINTER(_ScfControls),
        _DOUBLE,
        ct.c_uint64,
        ct.POINTER(_ScfResult),
    ]
    lib.vibeqc_fock_plan_solve.restype = ct.c_int32
    lib.vibeqc_fock_plan_diagnostic.argtypes = [ct.c_void_p, ct.POINTER(_Diagnostic)]
    lib.vibeqc_fock_plan_diagnostic.restype = ct.c_int32


def _check(lib, status, detail):
    if status:
        message = (detail or lib.vibeqc_status_message(status)).decode("utf-8")
        exception = {1: ValueError, 3: NotImplementedError, 7: MemoryError}.get(
            status, RuntimeError
        )
        raise exception(message)


def _pointer(array):
    return array.ctypes.data_as(_DOUBLE) if array is not None else None


class _DiagnosticResult:
    """Keep a detached execution record available after native plan closure."""

    _diagnostic: dict

    @property
    def diagnostics(self):
        """Requested/resolved semantics and source provenance, copied on access."""
        return deepcopy(self._diagnostic)


@dataclass(frozen=True, eq=False)
class FockEvaluation(_DiagnosticResult):
    """Fixed-density energies in Hartree, Fock matrices, and optional dE2/dR.

    ``gradient`` contains only the two-electron geometric response. It is
    neither a complete force nor an XC geometric derivative. Arrays own
    immutable storage; no numerical result is reused for a changed density.
    """

    energy_one_electron: float
    energy_two_electron: float
    nuclear_repulsion: float
    fock: np.ndarray
    coulomb: np.ndarray | None
    exchange: np.ndarray | None
    gradient: np.ndarray | None
    identity: str
    _diagnostic: dict = field(repr=False)

    @property
    def energy(self):
        return (
            self.energy_one_electron + self.energy_two_electron + self.nuclear_repulsion
        )


@dataclass(frozen=True, eq=False)
class FockScfResult(_DiagnosticResult):
    """Converged declared J/K energy (Hartree), density and optional forces.

    Forces are complete negative geometric derivatives in Hartree/Bohr,
    shaped [atom,3]. No XC is included. The immutable density is suitable for
    an explicit warm replay of this plan; it is not a legacy HF checkpoint.
    """

    energy: float
    density: np.ndarray
    forces: np.ndarray | None
    iterations: int
    energy_change: float
    density_rms: float
    initial_density_used: bool
    fock_builds: int
    identity: str
    _diagnostic: dict = field(repr=False)


class FockPlan:
    """Own native J/K sources for an immutable ``NativeAO`` basis snapshot.

    The optional auxiliary NativeAO must describe the same atoms/geometry.
    Raw values and matching responses use the same declared approximation.
    CUDA J/K and two-electron derivatives execute on the selected device;
    returned matrices use host storage. This interface does not run DFT SCF.
    """

    def __init__(
        self,
        basis,
        spec=None,
        *,
        auxiliary=None,
        device="cpu",
        device_id=0,
        screening_tolerance=1e-12,
        metric_relative_threshold=1e-10,
        device_budget_bytes=0,
    ):
        from vibeqc_compiler.dft import NativeAO

        from .calculator import Calculator

        if not isinstance(basis, NativeAO) or (
            auxiliary is not None and not isinstance(auxiliary, NativeAO)
        ):
            raise TypeError("expected NativeAO orbital/auxiliary bases")
        spec = FockBuildSpec.hf() if spec is None else spec
        if not isinstance(spec, FockBuildSpec):
            raise TypeError("expected FockBuildSpec")
        if device not in ("cpu", "cuda"):
            raise ValueError("Fock device must be cpu or cuda")
        if not math.isfinite(screening_tolerance) or screening_tolerance < 0:
            raise ValueError("screening tolerance must be nonnegative and finite")
        if (
            not math.isfinite(metric_relative_threshold)
            or not 0 < metric_relative_threshold < 1
        ):
            raise ValueError("metric threshold must lie strictly between zero and one")
        if type(device_budget_bytes) is not int or not 0 <= device_budget_bytes < 2**64:
            raise ValueError("device budget must fit uint64")
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._basis, self._spec = basis, spec
        calc = Calculator(
            device=device,
            device_id=device_id,
            basis_representation="spherical"
            if basis.representation == "real_spherical"
            else "cartesian",
        )
        self._library = lib = calc._library
        _bind(lib)
        self._native_source = lib.vibeqc_get_source_identity().decode()
        self._device_id = calc._device_id if device == "cuda" else None
        context, system, aux_system = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        _native.check(
            lib,
            lib.vibeqc_context_create(
                ct.byref(calc._context_descriptor()), ct.byref(context)
            ),
        )
        try:
            system = calc._create_native_system(
                context, basis.atoms, basis.charge, basis.multiplicity, basis.shells
            )
            fitted = any(
                t.present and t.approximation == "density_fitted"
                for t in (spec.coulomb, spec.exchange)
            )
            if fitted and auxiliary is not None:
                aux_calc = Calculator(
                    device=device,
                    device_id=device_id,
                    basis_representation="spherical"
                    if auxiliary.representation == "real_spherical"
                    else "cartesian",
                )
                aux_system = aux_calc._create_native_system(
                    context,
                    auxiliary.atoms,
                    auxiliary.charge,
                    auxiliary.multiplicity,
                    auxiliary.shells,
                )
            controls = _descriptor(
                _Controls,
                screening_tolerance=screening_tolerance,
                metric_relative_threshold=metric_relative_threshold,
                device_budget_bytes=device_budget_bytes,
            )
            status = lib.vibeqc_fock_plan_create(
                context,
                system,
                aux_system,
                ct.byref(_spec(spec)),
                ct.byref(controls),
                ct.byref(self._handle),
            )
            _check(lib, status, lib.vibeqc_context_get_last_detail(context))
        finally:
            if aux_system:
                lib.vibeqc_system_destroy(aux_system)
            if system:
                lib.vibeqc_system_destroy(system)
            lib.vibeqc_context_destroy(context)
        self._auxiliary_identity = (auxiliary or basis).identity if fitted else None
        # Hash the values consumed by the native plan, including normalized
        # NumPy scalars, so provenance describes execution rather than input types.
        diagnostics = self.diagnostics
        self._execution_identity = canonical_hash(diagnostics)
        self._identity = canonical_hash(
            {
                "schema": "vibeqc.fock-plan/v1",
                "basis": basis.identity,
                "auxiliary": self._auxiliary_identity,
                "resolved": diagnostics["resolved"],
                "screening": diagnostics["screening_tolerance"],
                "metric_cutoff": diagnostics["metric_relative_threshold"]
                if fitted
                else None,
                "precision": "float64",
            }
        )

    @property
    def basis(self):
        return self._basis

    @property
    def spec(self):
        return self._spec

    @property
    def identity(self):
        return self._identity

    @property
    def execution_identity(self):
        """Prepared source/backend provenance, distinct from mathematical identity."""
        return self._execution_identity

    def _ensure_open(self):
        if not self._handle:
            raise RuntimeError("Fock plan is closed")

    @property
    def diagnostics(self):
        with self._lock:
            self._ensure_open()
            out = _descriptor(_Diagnostic)
            status = self._library.vibeqc_fock_plan_diagnostic(
                self._handle, ct.byref(out)
            )
            _check(
                self._library,
                status,
                self._library.vibeqc_fock_plan_last_error(self._handle),
            )
            return {
                "requested": _spec_dict(out.requested),
                "resolved": _spec_dict(out.resolved),
                "backend": "cuda" if out.backend == 1 else "cpu",
                "precision": "float64",
                "native_source_identity": self._native_source,
                "device_id": self._device_id,
                "resolved_schedule": _SCHEDULES[out.resolved_schedule],
                "source_schedule": _SCHEDULES[out.source_schedule],
                **{
                    name: getattr(out, name)
                    for name in (
                        "nbf",
                        "coordinate_count",
                        "device_bytes",
                        "device_budget_bytes",
                        "auxiliary_rank",
                        "auxiliary_tile",
                        "screening_tolerance",
                        "metric_relative_threshold",
                    )
                },
                "df_streamed": bool(out.df_streamed),
                **{
                    name: getattr(out, name).decode()
                    for name in (
                        "direct_schedule",
                        "df_value_backend",
                        "df_value_mapping",
                        "df_response_mapping",
                        "one_electron_value_backend",
                        "one_electron_value_mapping",
                        "one_electron_response_mapping",
                    )
                },
                "basis_identity": self.basis.identity,
                "auxiliary_identity": self._auxiliary_identity,
            }

    def _density_snapshot(self, density):
        """Own one validated input for native execution and provenance hashing."""
        n = self.basis.nao
        separate = self.spec.spin == "unrestricted"
        d = np.asarray(density)
        if np.iscomplexobj(d):
            raise TypeError("Fock densities must be real")
        # Own the evaluated density so ctypes execution and later identity
        # hashing cannot observe different caller-side mutations.
        d = np.array(d, dtype=np.float64, order="C", copy=True)
        if d.shape != ((2, n, n) if separate else (n, n)) or not np.isfinite(d).all():
            raise ValueError("Fock density shape/spin/finiteness mismatch")
        return d

    def evaluate(self, density, *, derivative=False):
        """Return native unscaled J/K, assembled Fock/energy and optional dE2/dR.

        D is [AO,AO] for restricted spin or [2,AO,AO] for unrestricted spin.
        Nonsymmetric raw inputs are supported; generated CUDA DF derivatives
        require symmetric densities and fail before any result is published.
        """
        if type(derivative) is not bool:
            raise TypeError("derivative must be bool")
        n = self.basis.nao
        separate = self.spec.spin == "unrestricted"
        d = self._density_snapshot(density)
        with self._lock:
            self._ensure_open()
            j = np.empty((n, n)) if self.spec.coulomb.present else None
            k = np.empty(d.shape) if self.spec.exchange.present else None
            f = np.empty_like(d)
            gradient = np.empty(self.basis.natom * 3) if derivative else None
            out = _descriptor(
                _Result,
                matrix_count=n * n,
                gradient_count=0 if gradient is None else len(gradient),
                coulomb=_pointer(j),
                exchange_alpha=_pointer(k[0] if separate and k is not None else k),
                exchange_beta=_pointer(k[1] if separate and k is not None else None),
                fock_alpha=_pointer(f[0] if separate else f),
                fock_beta=_pointer(f[1] if separate else None),
                gradient=_pointer(gradient),
            )
            status = self._library.vibeqc_fock_plan_evaluate(
                self._handle,
                _pointer(d[0] if separate else d),
                n * n,
                _pointer(d[1] if separate else None),
                n * n if separate else 0,
                ct.byref(out),
            )
            _check(
                self._library,
                status,
                self._library.vibeqc_fock_plan_last_error(self._handle),
            )
            identity = canonical_hash(
                {
                    "plan": self.identity,
                    "execution": self.execution_identity,
                    "density": sha256(d.tobytes()).hexdigest(),
                    "derivative": derivative,
                }
            )
            return FockEvaluation(
                out.energy_one_electron,
                out.energy_two_electron,
                out.nuclear_repulsion,
                immutable(f),
                immutable(j) if j is not None else None,
                immutable(k) if k is not None else None,
                immutable(gradient) if gradient is not None else None,
                identity,
                self.diagnostics,
            )

    def solve(
        self,
        *,
        initial_density=None,
        compute_forces=True,
        max_iterations=100,
        diis_history=8,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
    ):
        """Run shared SCF for the declared J/K energy, with no XC contribution.

        CUDA plans use host DIIS/eigensolves with CUDA integral consumers.
        Omit ``initial_density`` for a core guess or pass an explicit density
        with the same layout as ``evaluate``. The native ensemble guard rejects
        incorrect overlap-metric occupation or electron/spin trace; it does
        not silently normalize seeds. Sources are reused, while each solve
        starts fresh DIIS history. Failure raises and leaves the plan reusable.
        """
        if type(compute_forces) is not bool:
            raise TypeError("compute_forces must be bool")
        if any(
            type(x) is not int or not 0 < x < 2**32 - 1
            for x in (max_iterations, diis_history)
        ):
            raise ValueError("SCF counts must be positive uint32 values")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (energy_tolerance, density_tolerance)
        ):
            raise ValueError("SCF tolerances must be finite and positive")
        controls = _descriptor(
            _ScfControls,
            max_iterations=max_iterations,
            diis_history=diis_history,
            energy_tolerance=energy_tolerance,
            density_tolerance=density_tolerance,
        )
        seed = (
            None if initial_density is None else self._density_snapshot(initial_density)
        )
        with self._lock:
            self._ensure_open()
            n = self.basis.nao
            density = np.empty(
                (2, n, n) if self.spec.spin == "unrestricted" else (n, n)
            )
            forces = np.empty((self.basis.natom, 3)) if compute_forces else None
            out = _descriptor(
                _ScfResult,
                density=_pointer(density),
                density_count=density.size,
                forces=_pointer(forces),
                force_count=forces.size if compute_forces else 0,
            )
            status = self._library.vibeqc_fock_plan_solve(
                self._handle,
                ct.byref(controls),
                _pointer(seed),
                0 if seed is None else seed.size,
                ct.byref(out),
            )
            _check(
                self._library,
                status,
                self._library.vibeqc_fock_plan_last_error(self._handle),
            )
            identity = canonical_hash(
                {
                    "schema": "vibeqc.fock-scf/v1",
                    "plan": self.identity,
                    "execution": self.execution_identity,
                    "initial_density": None
                    if seed is None
                    else sha256(seed.tobytes()).hexdigest(),
                    "compute_forces": compute_forces,
                    "controls": {
                        name: getattr(controls, name)
                        for name in (
                            "max_iterations",
                            "diis_history",
                            "energy_tolerance",
                            "density_tolerance",
                        )
                    },
                    "density": sha256(density.tobytes()).hexdigest(),
                }
            )
            return FockScfResult(
                out.energy,
                immutable(density),
                immutable(forces) if compute_forces else None,
                out.iterations,
                out.energy_change,
                out.density_rms,
                bool(out.initial_density_used),
                out.fock_builds,
                identity,
                self.diagnostics,
            )

    def close(self):
        with self._lock:
            if self._handle:
                self._library.vibeqc_fock_plan_destroy(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
