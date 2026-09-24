"""Block-streamed, amplitude-relaxed correlation-energy input weights.

Scientific VJPs are backend-independent TensorIR. The default execution stays on
the bound CPU owner; callers may provide an explicitly validated external tensor
executor (for example the bounded CUDA CC response executor) without changing
the mathematical response identity.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.evidence import canonical_hash
from vibeqc_compiler.common.solver_region import RegionDerivative, SolverRegion

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _array,
    _checked_bytes,
    _immutable,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .lambda_equations import PARAMETERS, build_parameter_vjp
from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, _feed_hash, _graph_bytes

if typing.TYPE_CHECKING:
    from vibeqc_compiler.tensor import TensorSpec

_WEIGHT_ATOL = 1e-12
_WEIGHT_RTOL = 1e-10


def _tensor(value: typing.Any, spec: typing.Any, label: typing.Any) -> typing.Any:
    value = _array(value, spec.shape, label)
    for symmetry in spec.symmetries:
        if not np.allclose(
            value,
            symmetry.sign * value.transpose(symmetry.permutation),
            atol=1e-11,
            rtol=1e-10,
        ):
            raise ValueError(f"{label} violates the declared input symmetry")
    return value


@dataclass(frozen=True)
class CCSDParameterWeight:
    """One immutable, symmetry-projected dense-Frobenius block cotangent.

    ``contract`` accepts a direction in this same independent block convention.
    A physical ERI change may affect several overlapping/permuted input blocks;
    contract ALL of those directions once, without arbitrary factor-of-two fixes.
    No reference-energy or upstream normal-ordering term is included here.
    """

    parameter: str
    values: np.ndarray
    spec: TensorSpec
    reference_identity: str
    cc_state_identity: str
    lambda_identity: str
    response_identity: str
    independent_max_abs_error: float
    logical_reserved_host_bytes: int
    provenance: Mapping

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _immutable(_tensor(self.values, self.spec, "CC parameter weight")),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def contract(self, direction: np.ndarray) -> float:
        """Contract a symmetry-compatible perturbation; never silently project it."""
        direction = _tensor(direction, self.spec, "CC parameter direction")
        with np.errstate(over="raise", invalid="raise"):
            value = float(np.sum(self.values * direction))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite CC parameter-weight contraction")
        return value


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDResponse:
    """Own checked Lambda values for a single immutable converged CC state.

    The opaque Lambda result's identity and ACTUAL stationarity are rechecked;
    a convergence flag or residual diagnostic is not numerical evidence.
    The existing bound state and its lifetime checks remain authoritative.
    No solver is called and no iterates are differentiated during weight requests.

    ``max_bytes`` is a simultaneous logical-numeric budget including the bound
    Lambda consumer's reservation, owned multipliers, and the current generated
    block programs/copies. Caller-retained earlier blocks, Python objects and
    opaque NumPy/BLAS workspace are excluded. This is not a process-RSS cap.
    """

    bound: BoundCCSDLambda
    lambda1: np.ndarray
    lambda2: np.ndarray
    lambda_identity: str
    response_identity: str
    derivative_plan_identity: str
    solver_region: SolverRegion | None
    max_bytes: int
    logical_reserved_host_bytes: int
    shared_lambda_residual_norm: float
    independent_lambda_residual_norm: float
    independent_lambda_residual_max: float
    tensor_executor: typing.Any

    def __init__(
        self,
        bound: BoundCCSDLambda,
        result: CCSDLambdaResult,
        *,
        max_bytes: int | None = None,
        tensor_executor: typing.Any = None,
    ) -> None:
        if not isinstance(bound, BoundCCSDLambda) or not isinstance(
            result, CCSDLambdaResult
        ):
            raise TypeError(
                "CC response requires BoundCCSDLambda and its CCSDLambdaResult"
            )
        max_bytes = bound.options.max_bytes if max_bytes is None else max_bytes
        _checked_bytes(max_bytes, "CC response host budget")
        if tensor_executor is not None and (
            not callable(getattr(tensor_executor, "execute", None))
            or not isinstance(getattr(tensor_executor, "backend", None), str)
        ):
            raise TypeError("external CC tensor executor must expose execute() and backend")
        amplitude_bytes = sum(layout.spec.size * 8 for layout in bound.layouts)
        required = bound.logical_reserved_host_bytes + 8 * amplitude_bytes
        _checked_bytes(required, "CC response simultaneous logical host reservation")
        if required > max_bytes:
            raise ImplicitSolveError(
                "CC response state exceeds simultaneous host budget"
            )
        with bound._lock:
            bound._assert_current(bound.reference_identity)
            if (
                result.reference_identity != bound.reference_identity
                or result.cc_state_identity != bound.cc_state_identity
                or not isinstance(result.provenance, Mapping)
                or result.provenance.get("equation_identity") != bound.equation_identity
                or result.provenance.get("lagrangian") != "E_corr + <lambda, R>"
                or result.provenance.get("tensor_backend")
                not in {
                    "numpy-cpu-interpreter",
                    "native-cpu-tensorir",
                    "cuda-fp64-ordinary-stream",
                    "cuda-fp64-resident-actions",
                }
            ):
                raise ResponseCompatibilityError(
                    "Lambda result belongs to a different CC state or convention"
                )
            l1 = _immutable(
                _tensor(result.lambda1, bound.layouts[0].spec, "Lambda singles")
            )
            l2 = _immutable(
                _tensor(result.lambda2, bound.layouts[1].spec, "Lambda doubles")
            )
            vector = bound.sqrt_weights * bound._pack((l1, l2))
            shared = bound._transpose(bound.programs, vector) - bound._rhs(
                bound.programs
            )
            independent = bound._transpose(bound.independent, vector) - bound._rhs(
                bound.independent
            )
            norms = (
                _vector_norm(shared),
                _vector_norm(independent),
                float(np.max(np.abs(independent / bound.sqrt_weights))),
            )
            if (
                not all(np.isfinite(x) for x in norms)
                or max(norms) > bound.options.lambda_tolerance
            ):
                raise ImplicitSolveError(
                    "CC response requires a freshly verified physical Lambda solution"
                )
            bound._assert_current(bound.reference_identity)
        put = lambda name, value: object.__setattr__(self, name, value)
        for name, value in (
            ("bound", bound),
            ("lambda1", l1),
            ("lambda2", l2),
            ("max_bytes", max_bytes),
            ("logical_reserved_host_bytes", required),
            ("shared_lambda_residual_norm", norms[0]),
            ("independent_lambda_residual_norm", norms[1]),
            ("independent_lambda_residual_max", norms[2]),
            ("tensor_executor", tensor_executor),
        ):
            put(name, value)
        put(
            "lambda_identity",
            canonical_hash(
                {
                    "cc_state": bound.cc_state_identity,
                    "equations": bound.equation_identity,
                    "multipliers": _feed_hash({"lambda1": l1, "lambda2": l2}),
                }
            ),
        )
        put(
            "response_identity",
            canonical_hash(
                {
                    "lambda": self.lambda_identity,
                    "scope": "correlation-only fixed-orbital input blocks",
                    "shared_primal": bound.programs.primal.logical_hash,
                    "independent_primal": bound.independent.primal.logical_hash,
                    "weight_atol": _WEIGHT_ATOL,
                    "weight_rtol": _WEIGHT_RTOL,
                }
            ),
        )
        put(
            "derivative_plan_identity",
            canonical_hash(
                {
                    "schema": "vibeqc.ccsd.implicit-vjp-region-rule-v1",
                    "equations": bound.equation_identity,
                    "shared_primal": bound.programs.primal.logical_hash,
                    "independent_primal": bound.independent.primal.logical_hash,
                    "solver_contract": dict(bound._solver_contract),
                    "inner_product": (
                        "dense Frobenius; sqrt-orbit-weighted independent solver coordinates"
                    ),
                    "scope": "amplitude-relaxed correlation-only fixed-orbital input blocks",
                }
            ),
        )
        solver_region = None
        if bound.primal_solver_region is not None:
            solver_region = replace(
                bound.primal_solver_region,
                derivatives=(
                    RegionDerivative("implicit_vjp", self.derivative_plan_identity),
                ),
                derivative_policy="custom",
            )
        put("solver_region", solver_region)

    @property
    def parameters(self) -> tuple[str, ...]:
        return PARAMETERS

    @property
    def tensor_backend(self) -> str:
        return (
            self.bound.tensor_backend
            if self.tensor_executor is None
            else self.tensor_executor.backend
        )

    def _execute_tensor(self, program: typing.Any, feeds: typing.Any) -> typing.Any:
        if self.tensor_executor is None:
            return self.bound._tensor_execute(program, feeds)
        return self.tensor_executor.execute(program, feeds)

    def _prepare(self, parameter: typing.Any) -> typing.Any:
        if self.solver_region is not None:
            rule = self.solver_region.derivative_rule("implicit_vjp")
            if rule.identity != self.derivative_plan_identity:
                raise ResponseCompatibilityError(
                    "CC solver-region derivative registration is stale"
                )
        shared = build_parameter_vjp(self.bound.programs.primal, parameter)
        independent = build_parameter_vjp(self.bound.independent.primal, parameter)
        spec = next(
            n.spec
            for n in self.bound.programs.primal.live_nodes
            if n.op == "input" and n.attrs["name"] == parameter
        )
        required = (
            self.logical_reserved_host_bytes
            + max(_graph_bytes(p.program) for p in (shared, independent))
            + 6 * spec.size * spec.itemsize
        )
        _checked_bytes(required, "CC parameter-weight simultaneous logical reservation")
        if required > self.max_bytes:
            raise ImplicitSolveError(
                f"CC parameter-weight host budget exceeded for {parameter}: {required} bytes"
            )
        return shared, independent, spec, required

    def required_bytes(self, parameter: str, *, reference_identity: str) -> int:
        """Plan one block's conservative logical storage, without numeric execution.

        A request above this instance's budget still raises; no partial weight is
        generated. Start with a sufficiently large budget to inspect its cost.
        """
        with self.bound._lock:
            self.bound._assert_current(reference_identity)
            *_, required = self._prepare(parameter)
            self.bound._assert_current(reference_identity)
            return required

    def _weight_from_prepared(
        self,
        parameter: str,
        prepared: tuple[typing.Any, typing.Any, typing.Any, int],
        *,
        reference_identity: str,
    ) -> CCSDParameterWeight:
        bound = self.bound
        with bound._lock:
            bound._assert_current(reference_identity)
            shared, independent, spec, required = prepared
            extra = {
                "bar_correlation_energy": np.asarray(1.0),
                "bar_singles_residual": self.lambda1,
                "bar_doubles_residual": self.lambda2,
            }
            values = []
            for program in (shared.program, independent.program):
                bound._assert_current(reference_identity)
                outputs = self._execute_tensor(program, {**bound.feeds, **extra})
                # Retain independent immutable evidence before executing the
                # other graph; shared executor buffers must not alias this check.
                values.append(
                    _immutable(
                        _tensor(
                            outputs[f"bar_{parameter}"], spec, "CC parameter weight"
                        )
                    )
                )
                bound._assert_current(reference_identity)
            if not np.allclose(
                values[0], values[1], atol=_WEIGHT_ATOL, rtol=_WEIGHT_RTOL
            ):
                raise ImplicitSolveError("independent CC parameter-weight check failed")
            error = float(np.max(np.abs(values[0] - values[1])))
            result = CCSDParameterWeight(
                parameter,
                values[0],
                spec,
                bound.reference_identity,
                bound.cc_state_identity,
                self.lambda_identity,
                self.response_identity,
                error,
                required,
                {
                    "parameter_vjp": shared.program.logical_hash,
                    "independent_parameter_vjp": independent.program.logical_hash,
                    "tensor_backend": self.tensor_backend,
                    "inner_product": "dense Frobenius; declared input symmetry projector",
                    "scope": "amplitude-relaxed correlation-only fixed-orbital mathematical input weight",
                    "hf_reference_energy": "excluded",
                    "normal_ordering_pullback": "excluded",
                    "orbital_response": "excluded",
                    "physical_rdm": False,
                    "solver_region_primal_identity": (
                        None
                        if bound.primal_solver_region is None
                        else bound.primal_solver_region.identity
                    ),
                    "solver_region_bound_identity": (
                        None
                        if self.solver_region is None
                        else self.solver_region.identity
                    ),
                    "solver_region_derivative_mode": (
                        None if self.solver_region is None else "implicit_vjp"
                    ),
                    "solver_region_derivative_identity": (
                        None
                        if self.solver_region is None
                        else self.derivative_plan_identity
                    ),
                    "reference_binding": (
                        "live-reference-callback"
                        if bound._current_reference is not None
                        else "detached-immutable-snapshot"
                    ),
                },
            )
            bound._assert_current(reference_identity)
            return result

    def weight(self, parameter: str, *, reference_identity: str) -> CCSDParameterWeight:
        """Generate/check one block, publishing it only after the final lifetime gate."""
        bound = self.bound
        with bound._lock:
            bound._assert_current(reference_identity)
            prepared = self._prepare(parameter)
        return self._weight_from_prepared(
            parameter, prepared, reference_identity=reference_identity
        )

    def iter_weights(
        self, parameters: Iterable[str] = PARAMETERS, *, reference_identity: str
    ) -> Iterator[CCSDParameterWeight]:
        """Stream individually qualified blocks; this is NOT an atomic force result.

        Release/contract each block before requesting the next to retain the
        reported bound. Earlier blocks retained by the caller are outside it.
        A late lifetime/numerical failure raises instead of publishing that
        block; already yielded immutable blocks retain their original identity.
        """
        if isinstance(parameters, str):
            raise TypeError("parameter blocks must be an iterable, not a string")
        names = tuple(parameters)
        if (
            not names
            or any(not isinstance(n, str) or n not in PARAMETERS for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("parameter blocks must be nonempty, supported and unique")
        bound = self.bound
        with bound._lock:
            bound._assert_current(reference_identity)
            prepared = tuple((name, self._prepare(name)) for name in names)
            executor = (
                self.tensor_executor
                if self.tensor_executor is not None
                else bound.tensor_executor
            )
            if executor is not None and callable(getattr(executor, "prewarm", None)):
                executor.prewarm(
                    program
                    for _, (shared, independent, _, _) in prepared
                    for program in (shared.program, independent.program)
                )
            bound._assert_current(reference_identity)
        for name, block in prepared:
            yield self._weight_from_prepared(
                name, block, reference_identity=reference_identity
            )
