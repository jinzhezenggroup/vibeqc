"""Converged-state CPU RCCSD Lambda consumer of generated equation actions.

This tooling adapter owns an immutable copy of an identified HF/CC result.
It reuses #465's checked transpose-solver contract and #179 GMRES. It is not
resident GPU execution, a native method registration, an RDM or a force API.
"""

from __future__ import annotations

import threading
import typing
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.evidence import canonical_hash
from vibeqc_compiler.common.solver_region import SolverRegion
from vibeqc_compiler.tensor import execute

from tools.vibeqc_posthf import ReferenceSnapshot
from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    ResponseGMRES,
    _array,
    _checked_bytes,
    _immutable,
    checked_transpose_solve,
    transpose_solver_contract,
)
from tools.vibeqc_response.krylov import GMRESOptions, _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .equations import amplitude_layouts
from .lambda_equations import build_lambda_programs
from .native_tensor_cpu import NativeCCTensorExecutor
from .solver import CCSDResult, _ccsd_programs


@dataclass(frozen=True)
class LambdaOptions:
    """Independent primal/adjoint gates and simultaneous logical host budget.

    ``max_bytes`` includes bound numerical state, generated interpreter buffers,
    weighted-coordinate scratch and declared Krylov storage. Python objects,
    the caller's replay lists and opaque NumPy/BLAS allocations are excluded;
    this is not a native/process peak-memory claim.
    """

    cc_tolerance: float = 1e-9
    lambda_tolerance: float = 1e-9
    max_bytes: int = 256 << 20
    gmres: GMRESOptions = field(
        default_factory=lambda: GMRESOptions(rtol=0.0, atol=1e-11)
    )

    def __post_init__(self) -> None:
        for name in ("cc_tolerance", "lambda_tolerance"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 0 < value <= 1e-9
            ):
                raise ValueError(f"{name} must be finite, positive and at most 1e-9")
        _checked_bytes(self.max_bytes, "Lambda host budget")
        if not isinstance(self.gmres, GMRESOptions):
            raise TypeError("Lambda solver options must be GMRESOptions")


@dataclass(frozen=True)
class CCSDLambdaResult:
    """Published only after fresh primal and two adjoint-equation checks.

    ``lambda1``/``lambda2`` use L = E_corr + <lambda,R> in the full dense
    Frobenius metric. They are not unconverted PySCF Lambda/RDM arrays.
    """

    lambda1: np.ndarray
    lambda2: np.ndarray
    reference_identity: str
    cc_state_identity: str
    scf_residual: float
    cc_r1_max: float
    cc_r2_max: float
    cc_residual_norm: float
    lambda_residual_norm: float
    independent_lambda_residual_norm: float
    independent_lambda_residual_max: float
    iterations: int
    operator_actions: int
    logical_reserved_host_bytes: int
    provenance: Mapping
    status: str = field(default="converged", init=False)

    @property
    def converged(self) -> typing.Any:
        return True


def _graph_bytes(program: typing.Any) -> typing.Any:
    return sum(n.spec.size * n.spec.itemsize for n in program.live_nodes) + sum(
        n.spec.size * n.spec.itemsize for n in program.outputs.values()
    )


def _feed_hash(feeds: typing.Any) -> typing.Any:
    return canonical_hash(
        {
            name: sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
            for name, value in feeds.items()
        }
    )


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDLambda:
    """Bind a successful conventional CPU CC result to its exact RHF snapshot.

    Replay feeds are checked against the CC integral hash and freshly evaluated
    equations; a convergence flag/history is not accepted as numerical proof.
    The copied state cannot alias mutable caller metadata or executor buffers.
    A live ``current_reference`` callback adds generation/lifetime checking.
    Without it this is explicitly a detached immutable tooling snapshot, not
    evidence that a native calculation/provider remains current or open.
    """

    reference: ReferenceSnapshot
    reference_identity: str
    cc_state_identity: str
    feeds: Mapping
    options: LambdaOptions
    logical_reserved_host_bytes: int
    primal_solver_region: SolverRegion | None

    def __init__(
        self,
        snapshot: typing.Any,
        cc_result: typing.Any,
        *,
        options: typing.Any = None,
        solver: typing.Any = None,
        current_reference: Callable[[], str] | None = None,
        backend: typing.Any = "cpu",
    ) -> None:
        if backend not in {"cpu", "native-cpu"}:
            raise NotImplementedError(
                "bound RCCSD Lambda supports interpreter or native CPU tooling only"
            )
        if not isinstance(snapshot, ReferenceSnapshot) or snapshot.algorithm != "RHF":
            raise TypeError("Lambda requires a validated RHF ReferenceSnapshot")
        if (
            not snapshot.converged
            or snapshot.scf_residual > snapshot.validation_tolerance
        ):
            raise ImplicitSolveError("Lambda requires a converged SCF reference")
        if (
            snapshot.hamiltonian_id != "conventional-unscreened"
            or snapshot.screening_tolerance != 0
        ):
            raise ValueError("Lambda requires the conventional unscreened Hamiltonian")
        if not isinstance(cc_result, CCSDResult):
            raise TypeError("Lambda requires an internal CCSDResult")
        if not cc_result.converged:
            raise ImplicitSolveError(
                f"Lambda requires converged CCSD: {cc_result.reason}"
            )
        if current_reference is not None and not callable(current_reference):
            raise TypeError("current_reference must be a callable live identity check")
        options = LambdaOptions() if options is None else options
        if not isinstance(options, LambdaOptions):
            raise TypeError("options must be LambdaOptions")
        provenance, replay = cc_result.provenance, cc_result.replay_inputs
        if not isinstance(provenance, Mapping) or not isinstance(replay, Mapping):
            raise TypeError("CC state is missing provenance/replay inputs")
        if (
            provenance.get("reference_id") != snapshot.identity
            or provenance.get("hamiltonian_id") != snapshot.hamiltonian_id
        ):
            raise ResponseCompatibilityError(
                "CC reference/Hamiltonian identity mismatch"
            )
        primal_solver_region = None
        solver_region_payload = provenance.get("solver_region_payload")
        if solver_region_payload is not None:
            try:
                primal_solver_region = SolverRegion.from_payload(solver_region_payload)
            except (TypeError, ValueError) as error:
                raise ResponseCompatibilityError(
                    "CC solver-region provenance is invalid"
                ) from error
            if (
                primal_solver_region.identity
                != provenance.get("solver_region_identity")
                or primal_solver_region.max_steps
                != provenance.get("solver_region_max_steps")
                or primal_solver_region.name != "rccsd-cpu"
            ):
                raise ResponseCompatibilityError(
                    "CC solver-region provenance does not match the executed primal"
                )
        o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
        programs = build_lambda_programs(o, v)
        independent = build_lambda_programs(o, v, form="expanded")
        cc_program, cc_reference_program = _ccsd_programs(o, v)
        if (
            provenance.get("equation_hash") != cc_program.logical_hash
            or provenance.get("independent_equation_hash")
            != cc_reference_program.logical_hash
        ):
            raise ResponseCompatibilityError(
                "CC equation identity is stale or unsupported"
            )
        layouts = amplitude_layouts(o, v)
        dimension = sum(layout.size for layout in layouts)
        solver = ResponseGMRES(dimension, options.gmres) if solver is None else solver
        contract = transpose_solver_contract(solver)
        specs = {
            n.attrs["name"]: n.spec
            for n in programs.primal.live_nodes
            if n.op == "input"
        }
        feed_bytes = sum(spec.size * spec.itemsize for spec in specs.values())
        amplitude_bytes = sum(layout.spec.size * 8 for layout in layouts)
        graphs = tuple(
            p
            for bundle in (programs, independent)
            for p in (
                bundle.primal,
                bundle.energy_vjp.program,
                bundle.residual_vjp.program,
            )
        )
        # All retained snapshots, publication copies and coordinate work overlap
        # conservatively; no amplitude-quadratic incidence matrix is constructed.
        required = (
            3 * snapshot.numeric_bytes
            + 3 * feed_bytes
            + 32 * amplitude_bytes
            + max(_graph_bytes(p) for p in graphs)
            + contract["workspace_bytes"]
        )
        _checked_bytes(required, "Lambda simultaneous logical host reservation")
        if required > options.max_bytes:
            raise ImplicitSolveError(
                f"Lambda simultaneous host workspace budget exceeded: {required} bytes"
            )
        put = lambda name, value: object.__setattr__(self, name, value)
        for name, value in (
            ("reference", snapshot),
            ("reference_identity", snapshot.identity),
            ("options", options),
            ("programs", programs),
            ("independent", independent),
            ("layouts", layouts),
            ("solver", solver),
            ("_solver_contract", MappingProxyType(contract)),
            ("_current_reference", current_reference),
            ("_lock", threading.RLock()),
            ("_tensor_backend", backend),
            (
                "tensor_executor",
                (
                    NativeCCTensorExecutor(max_bytes=options.max_bytes)
                    if backend == "native-cpu"
                    else None
                ),
            ),
            ("logical_reserved_host_bytes", required),
            ("primal_solver_region", primal_solver_region),
        ):
            put(name, value)
        self._assert_current(snapshot.identity)
        try:
            replay_snapshot = ReferenceSnapshot(**replay["snapshot"])
        except (KeyError, TypeError, ValueError) as error:
            raise ResponseCompatibilityError(
                "invalid CC replay reference snapshot"
            ) from error
        if replay_snapshot.identity != snapshot.identity:
            raise ResponseCompatibilityError("CC replay snapshot identity mismatch")
        arrays = {}
        for name, spec in specs.items():
            if name in ("t1", "t2"):
                value = getattr(cc_result, name)
            elif name not in replay:
                raise ValueError(f"CC replay is missing {name}")
            else:
                value = np.asarray(replay[name])
            arrays[name] = _immutable(_array(value, spec.shape, name))
        self._pack((arrays["t1"], arrays["t2"]))  # reject lossy T2 projection
        mathematical_inputs = {k: a for k, a in arrays.items() if k not in ("t1", "t2")}
        if _feed_hash(mathematical_inputs) != provenance.get("integral_hash"):
            raise ResponseCompatibilityError("CC integral/Fock identity mismatch")
        fock = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
        for name, value in (
            ("foo", fock[:o, :o]),
            ("fov", fock[:o, o:]),
            ("fvv", fock[o:, o:]),
        ):
            if not np.allclose(arrays[name], value, atol=1e-12, rtol=0):
                raise ResponseCompatibilityError(
                    "CC Fock feeds differ from bound reference"
                )
        put("feeds", MappingProxyType(arrays))
        put(
            "sqrt_weights",
            _immutable(
                np.sqrt(
                    np.concatenate(
                        [
                            np.asarray(layout.weights, dtype=np.float64)
                            for layout in layouts
                        ]
                    )
                )
            ),
        )
        put(
            "cc_state_identity",
            canonical_hash(
                {
                    "reference": snapshot.identity,
                    "inputs": _feed_hash(arrays),
                    "cc_equation": cc_program.logical_hash,
                    "independent_equation": cc_reference_program.logical_hash,
                }
            ),
        )
        if self.tensor_executor is not None:
            self.tensor_executor.prewarm(
                (
                    independent.primal,
                    programs.energy_vjp.program,
                    programs.residual_vjp.program,
                    independent.energy_vjp.program,
                    independent.residual_vjp.program,
                )
            )
        out = self._run(independent.primal)
        for value in (cc_result.correlation_energy, cc_result.total_energy):
            if value is None or not np.isfinite(value):
                raise ImplicitSolveError("CC result has no finite converged energy")
        energy = float(out["correlation_energy"])
        if (
            abs(energy - cc_result.correlation_energy) > 1e-10
            or abs(snapshot.reference_energy + energy - cc_result.total_energy) > 1e-10
        ):
            raise ResponseCompatibilityError(
                "CC energy does not match its bound amplitudes/reference"
            )
        r1, r2 = out["singles_residual"], out["doubles_residual"]
        put("cc_r1_max", float(np.max(np.abs(r1))))
        put("cc_r2_max", float(np.max(np.abs(r2))))
        put("cc_residual_norm", _vector_norm(self.sqrt_weights * self._pack((r1, r2))))
        if max(self.cc_r1_max, self.cc_r2_max) > options.cc_tolerance:
            raise ImplicitSolveError(
                "bound CC amplitudes fail fresh physical residual gate"
            )
        put(
            "equation_identity",
            canonical_hash(
                {
                    "actions": programs.provenance(),
                    "independent": independent.provenance(),
                    "layouts": [layout.to_payload() for layout in layouts],
                    "solver": contract,
                    "options": asdict(options),
                }
            ),
        )
        self._assert_current(snapshot.identity)

    def _assert_current(self, reference_identity: typing.Any) -> None:
        if reference_identity != self.reference_identity or (
            self._current_reference is not None
            and self._current_reference() != self.reference_identity
        ):
            raise ResponseCompatibilityError(
                "Lambda reference identity is stale or incompatible"
            )
        if transpose_solver_contract(self.solver) != self._solver_contract:
            raise ResponseCompatibilityError("Lambda solver contract changed")

    def _pack(self, arrays: typing.Any) -> typing.Any:
        return np.concatenate(
            [
                layout.pack(_array(a, layout.spec.shape, "CC amplitude/response"))
                for layout, a in zip(self.layouts, arrays)
            ]
        )

    def _unpack(self, vector: typing.Any) -> typing.Any:
        split = self.layouts[0].size
        return (
            self.layouts[0].unpack(vector[:split]),
            self.layouts[1].unpack(vector[split:]),
        )

    @property
    def tensor_backend(self) -> str:
        return (
            "native-cpu-tensorir"
            if self._tensor_backend == "native-cpu"
            else "numpy-cpu-interpreter"
        )

    def _tensor_execute(self, program: typing.Any, feeds: typing.Any) -> typing.Any:
        self._assert_current(self.reference_identity)
        if self.tensor_executor is not None:
            outputs = self.tensor_executor.execute(program, feeds)
        else:
            result = execute(program, feeds, max_bytes=self.options.max_bytes)
            if result.backend != "numpy-cpu-interpreter":
                raise ResponseCompatibilityError(
                    "Lambda tensor backend changed; no fallback allowed"
                )
            outputs = result.outputs
        self._assert_current(self.reference_identity)
        return outputs

    def _run(self, program: typing.Any, extra: typing.Any = None) -> typing.Any:
        return self._tensor_execute(
            program,
            {**self.feeds, **({} if extra is None else extra)},
        )

    def _rhs(self, programs: typing.Any) -> typing.Any:
        out = self._run(
            programs.energy_vjp.program, {"bar_correlation_energy": np.asarray(-1.0)}
        )
        return self.sqrt_weights * self._pack((out["bar_t1"], out["bar_t2"]))

    def _transpose(self, programs: typing.Any, vector: typing.Any) -> typing.Any:
        l1, l2 = self._unpack(vector / self.sqrt_weights)
        out = self._run(
            programs.residual_vjp.program,
            {"bar_singles_residual": l1, "bar_doubles_residual": l2},
        )
        return self.sqrt_weights * self._pack((out["bar_t1"], out["bar_t2"]))

    def solve(self, *, reference_identity: str) -> CCSDLambdaResult:
        """Solve without changing T, differentiating CC iterations, or emitting weights."""
        with self._lock:
            self._assert_current(reference_identity)
            owner = self

            class Operator:
                dimension = len(owner.sqrt_weights)

                def apply(self, vector: typing.Any) -> typing.Any:
                    return owner._transpose(owner.programs, vector)

            result = checked_transpose_solve(
                Operator(),
                self._rhs(self.programs),
                solver=self.solver,
                assert_current=lambda: self._assert_current(reference_identity),
            )
            # A second equation form checks the physical stationarity condition,
            # separately from the solver's status and shared-form true residual.
            residual = self._transpose(self.independent, result.solution) - self._rhs(
                self.independent
            )
            residual_norm = _vector_norm(residual)
            residual_max = float(np.max(np.abs(residual / self.sqrt_weights)))
            if (
                max(result.residual_norm, residual_norm, residual_max)
                > self.options.lambda_tolerance
            ):
                raise ImplicitSolveError(
                    "Lambda independent physical residual gate failed"
                )
            l1, l2 = self._unpack(result.solution / self.sqrt_weights)
            self._assert_current(reference_identity)
            return CCSDLambdaResult(
                _immutable(l1),
                _immutable(l2),
                self.reference_identity,
                self.cc_state_identity,
                self.reference.scf_residual,
                self.cc_r1_max,
                self.cc_r2_max,
                self.cc_residual_norm,
                result.residual_norm,
                residual_norm,
                residual_max,
                result.iterations,
                result.operator_actions,
                self.logical_reserved_host_bytes,
                MappingProxyType(
                    {
                        "equation_identity": self.equation_identity,
                        "lagrangian": "E_corr + <lambda, R>",
                        "inner_product": "dense Frobenius; sqrt-orbit-weighted independent solver coordinates",
                        "tensor_backend": self.tensor_backend,
                        "solver_backend": self._solver_contract["backend"],
                        "reference_binding": (
                            "live-reference-callback"
                            if self._current_reference is not None
                            else "detached-immutable-snapshot"
                        ),
                        "scope": "CPU amplitude response only; no parameter weights, RDMs or nuclear forces",
                    }
                ),
            )
