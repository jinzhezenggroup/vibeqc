"""Execute generated implicit VJPs through the existing #179 solver callback.

This is an explicit host-orchestrated development adapter, not a public force
endpoint or a native/device-resident Krylov loop. Tensor execution is injected
and identified; the default is the NumPy reference interpreter. CUDA executors
must be explicitly prepared by their resource owner. No backend fallback occurs.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method import ImplicitVJPPlan
from vibeqc_compiler.method.implicit import PREFIX
from vibeqc_compiler.tensor import execute

from .krylov import GMRESOptions, _single_workspace_bytes, _vector_norm, solve
from .problem import ResponseCompatibilityError


def _checked_bytes(value, name):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit byte count")
    return value


def _array(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.shape != shape:
        raise ValueError(f"{name} must have real FP64 dtype and shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _immutable(value):
    # A bytes-backed snapshot cannot be made writable by setflags(), and does
    # not alias caller arrays or an executor's reusable staging buffer.
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(
        value.shape
    )


class ImplicitSolveError(RuntimeError):
    """The implicit primitive rejected a primal/adjoint/result publication."""


class TransposeSolver(Protocol):
    """Opaque solve of the supplied, already-transposed Euclidean operator."""

    identity: str
    backend: str
    workspace_bytes: int
    rtol: float
    atol: float

    def solve(self, operator, rhs):
        """Return the #179 solution/status/residual/workspace result contract."""
        ...


@dataclass(frozen=True)
class ResponseGMRES:
    """Use #179 unchanged; do not introduce another Krylov implementation."""

    dimension: int
    options: GMRESOptions = field(default_factory=GMRESOptions)
    backend: str = field(default="python-response-gmres", init=False)

    def __post_init__(self):
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("response dimension must be a positive integer")
        if not isinstance(self.options, GMRESOptions):
            raise TypeError("response callback requires GMRESOptions")

    @property
    def workspace_bytes(self):
        return _single_workspace_bytes(self.dimension, self.options)

    @property
    def rtol(self):
        return self.options.rtol

    @property
    def atol(self):
        return self.options.atol

    @property
    def identity(self):
        return canonical_hash(
            {
                "callback": "vibeqc-response-gmres-v1",
                "dimension": self.dimension,
                "backend": self.backend,
                "operator": "caller-supplied-transpose-in-euclidean-coordinates",
                "options": asdict(self.options),
            }
        )

    def solve(self, operator, rhs):
        if operator.dimension != self.dimension:
            raise ValueError("response callback dimension mismatch")
        return solve(operator, rhs, options=self.options)


def transpose_solver_contract(solver: TransposeSolver) -> dict:
    """Freeze an opaque solver's identity, tolerances and declared storage."""
    for name in ("identity", "backend"):
        value = getattr(solver, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"implicit callback requires a nonempty {name}")
    for name in ("rtol", "atol"):
        value = getattr(solver, name, None)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"solver {name} must be finite and nonnegative")
    if solver.rtol >= 1 or max(solver.rtol, solver.atol) == 0:
        raise ValueError("solver requires rtol < 1 and a nonzero convergence tolerance")
    return {
        "identity": solver.identity,
        "backend": solver.backend,
        "workspace_bytes": _checked_bytes(solver.workspace_bytes, "solver workspace"),
        "rtol": float(solver.rtol),
        "atol": float(solver.atol),
    }


@dataclass(frozen=True)
class CheckedAdjointResult:
    """A solution rechecked against the supplied physical transpose action."""

    solution: np.ndarray
    residual_norm: float
    iterations: int
    operator_actions: int
    workspace_bytes: int


def checked_transpose_solve(operator, rhs, *, solver, assert_current=None):
    """Shared first-order adjoint acceptance for generated single/block states.

    The caller supplies the already-transposed Euclidean action and performs
    its own simultaneous state/operator/solver resource admission. No equation,
    coordinate map, preconditioner or Krylov algorithm is introduced here.
    A solver's success flag cannot bypass a fresh physical residual evaluation.
    """
    contract = transpose_solver_contract(solver)
    dimension = operator.dimension
    if type(dimension) is not int or dimension < 1:
        raise ValueError("adjoint dimension must be a positive integer")
    if assert_current is not None and not callable(assert_current):
        raise TypeError("assert_current must be callable")
    rhs = _immutable(_array(rhs, (dimension,), "adjoint RHS"))
    actions = 0

    def check():
        if assert_current is not None:
            assert_current()
        if transpose_solver_contract(solver) != contract:
            raise ResponseCompatibilityError("implicit solver contract changed")

    class CheckedOperator:
        def __init__(self):
            self.dimension = dimension

        def apply(self, vector):
            nonlocal actions
            check()
            vector = _immutable(_array(vector, (dimension,), "adjoint iterate"))
            actions += 1
            value = _immutable(
                _array(operator.apply(vector), (dimension,), "transpose action")
            )
            check()
            return value

    checked = CheckedOperator()
    check()
    result = solver.solve(checked, rhs)
    check()
    if result.converged is not True:
        raise ImplicitSolveError(f"implicit adjoint failed: {result.reason}")
    if (
        not math.isfinite(result.residual_norm)
        or result.residual_norm < 0
        or _checked_bytes(result.workspace_bytes, "reported solver workspace")
        > contract["workspace_bytes"]
    ):
        raise ImplicitSolveError(
            "implicit solver returned invalid residual/resource diagnostics"
        )
    if type(result.iterations) is not int or result.iterations < 0:
        raise ImplicitSolveError(
            "implicit solver returned invalid iteration diagnostics"
        )
    solution = _immutable(_array(result.solution, (dimension,), "adjoint solution"))
    residual = _vector_norm(checked.apply(solution) - rhs)
    target = max(contract["atol"], contract["rtol"] * _vector_norm(rhs))
    if not math.isfinite(target) or residual > target or result.residual_norm > target:
        raise ImplicitSolveError(
            f"implicit true adjoint residual failed: {residual:.6g} > {target:.6g}"
        )
    check()
    return CheckedAdjointResult(
        solution, residual, result.iterations, actions, result.workspace_bytes
    )


@dataclass(frozen=True)
class ReferenceTensorExecutor:
    """Identified CPU reference execution of the very same generated programs."""

    plan: ImplicitVJPPlan
    backend: str = field(default="numpy-cpu-interpreter", init=False)
    workspace_bytes: int = field(default=0, init=False)

    @property
    def plan_identity(self):
        return self.plan.identity

    @property
    def identity(self):
        return canonical_hash({"plan": self.plan_identity, "backend": self.backend})

    def execute(self, stage, feeds):
        return execute(
            self.plan.programs[stage],
            feeds,
            max_bytes=self.plan.reference_workspace_bytes,
        )


@dataclass(frozen=True)
class ImplicitVJPResult:
    """Publish only after primal, solver, true-residual and freshness gates."""

    parameter_cotangents: Mapping[str, np.ndarray]
    adjoint: np.ndarray
    primal_residual_norm: float
    adjoint_residual_norm: float
    iterations: int
    operator_actions: int
    plan_identity: str
    state_identity: str
    execution_identity: str
    solver_backend: str
    tensor_backend: str
    logical_reserved_host_bytes: int


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundImplicitState:
    """Immutable numerical snapshot plus declared solver/executor ownership.

    The default executor is CPU reference-only. An injected executor provides
    plan_identity, identity, backend, workspace_bytes (all simultaneously live
    host buffers), and execute(stage, feeds) -> outputs/backend. Its native
    device resource owner must perform device admission before preparation.

    ``current_reference`` is the runtime owner's live identity check. Without
    it this object represents only an immutable tooling snapshot, not a proof
    of a current native molecular state. Each VJP still requires its reference
    identity explicitly; snapshots from another geometry cannot be substituted.
    """

    plan: ImplicitVJPPlan
    reference_identity: str
    feeds: Mapping[str, np.ndarray]
    state_identity: str
    execution_identity: str
    primal_residual_norm: float
    logical_reserved_host_bytes: int
    solver: TransposeSolver
    executor: object

    def __init__(
        self,
        plan: ImplicitVJPPlan,
        feeds: Mapping,
        *,
        reference_identity: str,
        solver: TransposeSolver | None = None,
        executor=None,
        current_reference: Callable[[], str] | None = None,
        primal_atol: float = 1e-10,
        max_bytes: int = 256 << 20,
    ):
        if not isinstance(plan, ImplicitVJPPlan):
            raise TypeError("expected a generated ImplicitVJPPlan")
        if not isinstance(reference_identity, str) or not reference_identity.strip():
            raise ValueError("reference identity must be nonempty")
        if (
            type(primal_atol) not in (float, int)
            or not math.isfinite(primal_atol)
            or primal_atol <= 0
        ):
            raise ValueError("primal_atol must be finite and positive")
        if current_reference is not None and not callable(current_reference):
            raise TypeError("current_reference must be a callable live identity check")
        if not isinstance(feeds, Mapping) or set(feeds) != set(plan.spec.inputs):
            raise ValueError("implicit feeds must exactly match the residual inputs")
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "reference_identity", reference_identity)
        object.__setattr__(
            self,
            "solver",
            ResponseGMRES(plan.spec.dimension) if solver is None else solver,
        )
        object.__setattr__(
            self,
            "executor",
            ReferenceTensorExecutor(plan) if executor is None else executor,
        )
        object.__setattr__(self, "_current_reference", current_reference)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(
            self, "_contract", MappingProxyType(self._callback_contract())
        )
        if self.executor.plan_identity != plan.identity:
            raise ResponseCompatibilityError(
                "executor belongs to a different implicit plan"
            )
        _checked_bytes(max_bytes, "implicit host budget")
        required = (
            plan.reference_workspace_bytes
            + self._contract["solver_workspace_bytes"]
            + self._contract["executor_workspace_bytes"]
        )
        _checked_bytes(required, "combined implicit host reservation")
        if required > max_bytes:
            raise ImplicitSolveError(
                "implicit simultaneous host workspace budget exceeded"
            )
        object.__setattr__(self, "logical_reserved_host_bytes", required)
        self._assert_current(reference_identity)
        arrays = {
            name: _array(feeds[name], spec.shape, name)
            for name, spec in plan.spec.inputs.items()
        }
        object.__setattr__(
            self,
            "feeds",
            MappingProxyType(
                {name: _immutable(value) for name, value in arrays.items()}
            ),
        )
        object.__setattr__(
            self,
            "state_identity",
            canonical_hash(
                {
                    "reference": reference_identity,
                    "plan": plan.identity,
                    "inputs": {
                        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                        for name, value in self.feeds.items()
                    },
                }
            ),
        )
        object.__setattr__(
            self,
            "execution_identity",
            canonical_hash(
                {
                    "state": self.state_identity,
                    "callbacks": dict(self._contract),
                    "primal_atol": float(primal_atol),
                    "logical_host_reservation": required,
                }
            ),
        )
        primal = self._run("primal")["value"]
        object.__setattr__(self, "primal_residual_norm", _vector_norm(primal))
        if self.primal_residual_norm > primal_atol:
            raise ImplicitSolveError(
                f"implicit primal is not converged: residual={self.primal_residual_norm:.6g}"
            )
        self._assert_current(reference_identity)

    def _callback_contract(self):
        solver, executor = self.solver, self.executor
        for obj in (solver, executor):
            for name in ("identity", "backend"):
                if (
                    not isinstance(getattr(obj, name, None), str)
                    or not getattr(obj, name).strip()
                ):
                    raise ValueError(f"implicit callback requires a nonempty {name}")
        for name in ("rtol", "atol"):
            value = getattr(solver, name, None)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"solver {name} must be finite and nonnegative")
        if solver.rtol >= 1 or max(solver.rtol, solver.atol) == 0:
            raise ValueError(
                "solver requires rtol < 1 and a nonzero convergence tolerance"
            )
        return {
            "solver": solver.identity,
            "solver_backend": solver.backend,
            "solver_workspace_bytes": _checked_bytes(
                solver.workspace_bytes, "solver workspace"
            ),
            "solver_rtol": float(solver.rtol),
            "solver_atol": float(solver.atol),
            "executor": executor.identity,
            "executor_backend": executor.backend,
            "executor_workspace_bytes": _checked_bytes(
                executor.workspace_bytes, "executor workspace"
            ),
            "executor_plan": executor.plan_identity,
        }

    def _assert_current(self, reference_identity):
        if reference_identity != self.reference_identity or (
            self._current_reference is not None
            and self._current_reference() != self.reference_identity
        ):
            raise ResponseCompatibilityError(
                "implicit reference identity is stale or incompatible"
            )
        if self._callback_contract() != self._contract:
            raise ResponseCompatibilityError(
                "implicit solver/executor contract changed"
            )

    def _run(self, stage, extra=None):
        feeds = dict(self.feeds)
        if extra:
            feeds.update(extra)
        execution = self.executor.execute(stage, feeds)
        if execution.backend != self._contract["executor_backend"]:
            raise ResponseCompatibilityError(
                "implicit tensor backend changed; no fallback is allowed"
            )
        expected = self.plan.programs[stage].outputs
        if set(execution.outputs) != set(expected):
            raise ImplicitSolveError("implicit executor returned incomplete outputs")
        return {
            name: _immutable(
                _array(execution.outputs[name], node.spec.shape, f"{stage}/{name}")
            )
            for name, node in expected.items()
        }

    def vjp(
        self, state_cotangent, *, reference_identity: str, direct=None
    ) -> ImplicitVJPResult:
        """Apply the first-order implicit rule without any solver-history tape."""
        with self._lock:
            self._assert_current(reference_identity)
            spec = self.plan.spec
            seed = _array(state_cotangent, spec.state_spec.shape, "state cotangent")
            direct = {} if direct is None else direct
            if not isinstance(direct, Mapping) or set(direct) - set(
                spec.parameter_names
            ):
                raise ValueError("direct cotangents contain unknown parameter names")
            extra = {
                PREFIX + f"direct_{name}": _immutable(
                    _array(
                        direct.get(
                            name, np.zeros(spec.inputs[name].shape, dtype=np.float64)
                        ),
                        spec.inputs[name].shape,
                        f"direct/{name}",
                    )
                )
                for name in spec.parameter_names
            }
            rhs = _immutable(
                self._run("rhs", {PREFIX + "seed": seed})["value"].reshape(-1)
            )
            owner = self

            class Operator:
                dimension = spec.dimension

                def apply(self, vector):
                    return owner._run(
                        "transpose",
                        {PREFIX + "vector": vector.reshape(spec.residual_spec.shape)},
                    )["value"].reshape(-1)

            result = checked_transpose_solve(
                Operator(),
                rhs,
                solver=self.solver,
                assert_current=lambda: self._assert_current(reference_identity),
            )
            y, residual = result.solution, result.residual_norm
            extra[PREFIX + "vector"] = y.reshape(spec.residual_spec.shape)
            source = self._run("source", extra)
            adjoint = self._run(
                "adjoint", {PREFIX + "vector": extra[PREFIX + "vector"]}
            )["value"]
            self._assert_current(reference_identity)
            return ImplicitVJPResult(
                MappingProxyType(
                    {
                        name: _immutable(source[f"bar_{name}"])
                        for name in spec.parameter_names
                    }
                ),
                _immutable(adjoint),
                self.primal_residual_norm,
                residual,
                result.iterations,
                result.operator_actions,
                self.plan.identity,
                self.state_identity,
                self.execution_identity,
                self._contract["solver_backend"],
                self._contract["executor_backend"],
                self.logical_reserved_host_bytes,
            )
