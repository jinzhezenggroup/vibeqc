"""Complete conventional CPU RCCSD analytic gradients for small validation cases.

The source, HF export, dense MO weights and dense native derivative arrays are
explicit <=12-AO validation boundaries. Generated derivative mathematics and
shared solvers are reused; this does not enable the native/public force API,
resident GPU execution, frozen-core/open-shell/ECP, DF or (T) gradients.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.tensor import execute

from tools.vibeqc_posthf import MOBlock
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import GMRESOptions, NativeJKBackend, RHFResponseOperator
from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    ResponseGMRES,
    _array,
    _checked_bytes,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError
from tools.vibeqc_validation.schema import canonical_hash

from .gradient_equations import build_ao_weight_program, build_hamiltonian_programs
from .lambda_equations import PARAMETERS
from .lambda_response import BoundCCSDResponse
from .lambda_solver import BoundCCSDLambda, LambdaOptions, _graph_bytes
from .solver import SolverOptions, solve


@dataclass(frozen=True)
class CCSDGradientOptions:
    """Strict scientific gates and disclosed stage-level logical budgets."""

    max_bytes: int = 256 << 20
    provider_budget_bytes: int = 64 << 20
    scf_tolerance: float = 1e-12
    scf_max_iterations: int = 150
    orbital_residual_tolerance: float = 1e-10
    stationarity_tolerance: float = 1e-8
    minimum_orbital_curvature: float = 1e-8
    cc_options: SolverOptions = field(
        default_factory=lambda: SolverOptions(
            residual_tolerance=1e-12,
            energy_tolerance=1e-14,
            max_iterations=150,
        )
    )
    lambda_options: LambdaOptions = field(default_factory=LambdaOptions)
    z_options: GMRESOptions = field(
        default_factory=lambda: GMRESOptions(rtol=0, atol=1e-12)
    )

    def __post_init__(self):
        for name in ("max_bytes", "provider_budget_bytes"):
            _checked_bytes(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if type(self.scf_max_iterations) is not int or self.scf_max_iterations < 1:
            raise ValueError("scf_max_iterations must be a positive integer")
        for name, ceiling in (
            ("scf_tolerance", 1e-10),
            ("orbital_residual_tolerance", 1e-9),
            ("stationarity_tolerance", 1e-8),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 0 < value <= ceiling
            ):
                raise ValueError(
                    f"{name} must be finite, positive and at most {ceiling}"
                )
        if (
            type(self.minimum_orbital_curvature) not in (int, float)
            or not np.isfinite(self.minimum_orbital_curvature)
            or self.minimum_orbital_curvature < 1e-10
        ):
            raise ValueError(
                "minimum orbital curvature must be finite and at least 1e-10"
            )
        for name, cls in (
            ("cc_options", SolverOptions),
            ("lambda_options", LambdaOptions),
            ("z_options", GMRESOptions),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")


@dataclass(frozen=True)
class CCSDGradientResult:
    """Complete detached energy gradient; failure never constructs this result.

    Gradients are dE/dR in Eh/bohr. ``forces`` has the opposite sign. Component
    arrays are unprojected: no removal of net force/torque hides missing terms.
    """

    total_energy: float
    correlation_energy: float
    gradient: np.ndarray
    physical_components: Mapping
    integral_components: Mapping
    scf_residual: float
    cc_residual: float
    lambda_residual: float
    z_residual: float
    orbital_stationarity: float
    minimum_orbital_curvature: float
    reference_identity: str
    cc_state_identity: str
    response_identity: str
    operator_identity: str
    source_identity: str
    diagnostics: Mapping

    def __post_init__(self):
        object.__setattr__(self, "gradient", _immutable(self.gradient))
        for name in ("physical_components", "integral_components"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        k: _immutable(
                            _array(v, self.gradient.shape, "gradient component")
                        )
                        for k, v in getattr(self, name).items()
                    }
                ),
            )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def forces(self):
        return _immutable(-self.gradient)


def _validate_source(source):
    if (
        not isinstance(source, NativeSource)
        or source.backend != "cpu-reference-native-shell-tiles"
    ):
        raise TypeError("complete CCSD gradient requires an explicit native CPU source")
    source._check_open()
    if source.nbf > 12:
        raise ValueError(
            "complete CPU CCSD gradient validation supports at most 12 AOs"
        )
    if source.naux:
        raise ValueError(
            "complete CCSD gradient validation does not support DF/auxiliary bases"
        )
    if (
        source.multiplicity != 1
        or source.electron_count % 2
        or not 0 < source.electron_count < 2 * source.nbf
    ):
        raise ValueError(
            "complete CCSD gradient requires closed-shell occupied and virtual spaces"
        )


def _derivative_bytes(source):
    return 3 * len(source.atoms) * (2 * source.nbf**2 + source.nbf**4 + 1) * 8


def _nuclear_energy(source):
    result = 0.0
    for i, a in enumerate(source.atoms):
        for b in source.atoms[:i]:
            distance = np.linalg.norm(np.asarray(a.position) - b.position)
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError("invalid coincident/nonfinite nuclear centers")
            result += a.atomic_number * b.atomic_number / distance
    return result


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDGradient:
    """Bind raw integral/orbital response to the exact solved CC/Lambda snapshot.

    The Z solve uses the shared native-streamed RHF operator and GMRES. A tiny
    explicit orbital matrix generated by an independent Fock JVP is used for
    curvature, operator and final-residual checks, not as a new CC solver.
    No T2-by-T2 Jacobian is formed. Dense h/g/weight/derivative storage is exposed
    as a small-system validation boundary; it is not a production peak-RSS cap.
    """

    def __init__(self, response, provider, *, options=None):
        started = time.perf_counter()
        options = CCSDGradientOptions() if options is None else options
        if not isinstance(options, CCSDGradientOptions):
            raise TypeError("gradient options must be CCSDGradientOptions")
        if not isinstance(response, BoundCCSDResponse) or not isinstance(
            provider, ConventionalProvider
        ):
            raise TypeError(
                "CC gradient requires a bound CC response and conventional provider"
            )
        source = provider.source
        _validate_source(source)
        reference = response.bound.reference
        if (
            provider.backend != "cpu"
            or provider.snapshot.identity != reference.identity
            or reference.algorithm != "RHF"
            or reference.hamiltonian_id != "conventional-unscreened"
            or reference.screening_tolerance != 0
            or reference.hf_backend != "native-cpu"
            or reference.scf_residual > 1e-9
            or reference.nmo != source.nbf
        ):
            raise ResponseCompatibilityError(
                "CC gradient reference/provider/CPU Hamiltonian mismatch"
            )
        for name in ("geometry_hash", "basis_hash", "representation"):
            if getattr(source, name) != getattr(reference, name):
                raise ResponseCompatibilityError(
                    f"CC gradient source/reference {name} mismatch"
                )
        put = lambda name, value: object.__setattr__(self, name, value)
        for name, value in (
            ("response", response),
            ("provider", provider),
            ("source", source),
            ("reference", reference),
            ("reference_identity", reference.identity),
            ("source_identity", source.identity),
            ("options", options),
        ):
            put(name, value)
        self._assert_current()
        n, o = reference.nmo, reference.nocc
        programs = build_hamiltonian_programs(o, n - o)
        ao_program = build_ao_weight_program(n)
        z_solver = ResponseGMRES(o * (n - o), options.z_options)
        # Simultaneous logical numeric reservation, including buffers retained
        # through the later native dense-derivative call and independent checks.
        # Provider/HF internals and Python/opaque BLAS allocation remain separate.
        block_reservation = max(
            response.required_bytes(name, reference_identity=reference.identity)
            for name in PARAMETERS
        )
        required = (
            block_reservation
            + 2
            * max(
                _graph_bytes(p)
                for p in (
                    programs.primal,
                    programs.weights,
                    programs.orbital_jvp.program,
                    ao_program,
                )
            )
            + 8 * (24 * n**4 + 64 * n**2 + 8 * z_solver.dimension**2)
            + 3 * _derivative_bytes(source)
            + z_solver.workspace_bytes
        )
        _checked_bytes(required, "CC complete-gradient logical reservation")
        if required > options.max_bytes:
            raise ImplicitSolveError(
                f"CC complete-gradient host budget exceeded before integral reads: {required} bytes"
            )
        put("logical_reserved_host_bytes", required)
        put("programs", programs)
        put("ao_program", ao_program)
        raw_g = provider.get(MOBlock((tuple(range(n)),) * 4)).to_host()
        raw_h = reference.coefficients.T @ reference.hcore @ reference.coefficients
        put(
            "raw_inputs",
            MappingProxyType(
                {
                    "h": _immutable(raw_h),
                    "g": _immutable(raw_g),
                    "rotation": _immutable(np.eye(n)),
                }
            ),
        )
        values = self._run(programs.primal, self.raw_inputs)
        for name in PARAMETERS:
            if not np.allclose(
                values[name], response.bound.feeds[name], atol=1e-10, rtol=1e-12
            ):
                raise ResponseCompatibilityError(
                    f"raw Hamiltonian differs from bound CC input {name}"
                )
        hf_energy = float(values["reference_electronic_energy"]) + _nuclear_energy(
            source
        )
        if abs(hf_energy - reference.reference_energy) > 1e-8:
            raise ResponseCompatibilityError(
                "HF reference energy does not match raw h/g plus nuclear energy"
            )
        zero_seeds = {
            "bar_" + name: _immutable(np.zeros_like(response.bound.feeds[name]))
            for name in PARAMETERS
        }
        zero_seeds["bar_reference_electronic_energy"] = _immutable(np.array(0.0))
        correlation_seeds = {
            "bar_" + w.parameter: w.values
            for w in response.iter_weights(reference_identity=reference.identity)
        }
        correlation_seeds["bar_reference_electronic_energy"] = np.array(0.0)
        corr = self._pullback(correlation_seeds)
        hf = self._pullback(
            {**zero_seeds, "bar_reference_electronic_energy": np.array(1.0)}
        )
        same_space = max(
            float(np.max(abs(corr["stationarity"][:o, :o]))),
            float(np.max(abs(corr["stationarity"][o:, o:]))),
        )
        if same_space > options.stationarity_tolerance:
            raise ImplicitSolveError(
                "CC same-space orbital stationarity failed; no canonical-gap patch is applied"
            )
        backend = NativeJKBackend(
            source,
            axis_tile=max(source.shell_sizes),
            budget_bytes=options.provider_budget_bytes,
        )
        problem = RHFResponseOperator.build_problem(
            reference,
            backend,
            perturbation_labels=("ccsd-correlation-orbital-lagrangian",),
        )
        operator = RHFResponseOperator(problem, backend)
        put("operator", operator)
        put("operator_identity", operator.identity)
        basis = np.eye(operator.dimension)
        matrix = np.column_stack([self._generated_orbital_action(x) for x in basis])
        if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
            raise ImplicitSolveError("generated RHF response matrix is not symmetric")
        curvature = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0])
        if curvature <= options.minimum_orbital_curvature:
            raise ImplicitSolveError(
                "RHF orbital response is unstable or near-singular"
            )
        for direction in (
            np.ones(operator.dimension),
            np.arange(1, operator.dimension + 1, dtype=float),
        ):
            direction /= np.linalg.norm(direction)
            if not np.allclose(
                operator.apply(direction), matrix @ direction, atol=1e-10, rtol=1e-9
            ):
                raise ImplicitSolveError(
                    "native RHF response action differs from generated Fock JVP"
                )
        rhs = corr["orbital_rhs"].reshape(-1)
        owner = self

        class Transpose:
            dimension = operator.dimension

            def apply(self, vector):
                return owner.operator.apply_transpose(vector)

        z_started = time.perf_counter()
        z = checked_transpose_solve(
            Transpose(), rhs, solver=z_solver, assert_current=self._assert_current
        )
        independent_z_residual = _vector_norm(matrix.T @ z.solution - rhs)
        if independent_z_residual > options.orbital_residual_tolerance:
            raise ImplicitSolveError("independent physical Z-vector residual failed")
        z_seeds = {**zero_seeds, "bar_fov": -z.solution.reshape(o, n - o)}
        orbital = self._pullback(z_seeds)
        total_seeds = {key: correlation_seeds[key] + z_seeds[key] for key in zero_seeds}
        total_seeds["bar_reference_electronic_energy"] = np.array(1.0)
        total = self._pullback(total_seeds)
        for name in ("hcore", "eri", "overlap", "rotation_gradient"):
            if not np.allclose(
                total[name],
                hf[name] + corr[name] + orbital[name],
                atol=1e-11,
                rtol=1e-11,
            ):
                raise ImplicitSolveError(
                    "CC gradient component weights fail independent sum check"
                )
        stationarity = float(np.max(abs(total["stationarity"])))
        if stationarity > options.stationarity_tolerance:
            raise ImplicitSolveError("complete HF+CC+Z orbital stationarity failed")
        for name, value in (
            ("weights", total),
            (
                "component_weights",
                MappingProxyType(
                    {"hf": hf, "correlation": corr, "orbital_response": orbital}
                ),
            ),
            ("z_result", z),
            ("orbital_matrix", _immutable(matrix)),
            ("orbital_rhs", _immutable(rhs)),
            ("minimum_orbital_curvature", curvature),
            ("orbital_stationarity", stationarity),
            ("independent_z_residual", independent_z_residual),
            ("same_space_stationarity", same_space),
        ):
            put(name, value)
        put(
            "weight_identity",
            canonical_hash(
                {
                    "response": response.response_identity,
                    "source": source.identity,
                    "hamiltonian": programs.primal.logical_hash,
                    "weights": programs.weights.logical_hash,
                    "ao_transform": ao_program.logical_hash,
                    "operator": operator.identity,
                    "options": asdict(options),
                }
            ),
        )
        put(
            "timings",
            MappingProxyType(
                {
                    "prepare_weights_seconds": time.perf_counter() - started,
                    "z_solve_and_weight_seconds": time.perf_counter() - z_started,
                }
            ),
        )
        self._assert_current()

    def _assert_current(self):
        self.source._check_open()
        if (
            self.provider._closed
            or self.provider.source is not self.source
            or self.source.backend != "cpu-reference-native-shell-tiles"
            or self.source.identity != self.source_identity
            or self.provider.snapshot.identity != self.reference_identity
        ):
            raise ResponseCompatibilityError(
                "CC gradient source/provider/reference is stale or closed"
            )
        self.response.bound._assert_current(self.reference_identity)
        if (
            hasattr(self, "operator_identity")
            and self.operator.identity != self.operator_identity
        ):
            raise ResponseCompatibilityError(
                "CC orbital-response operator identity changed"
            )

    def _run(self, program, feeds):
        self._assert_current()
        result = execute(program, feeds, max_bytes=self.options.max_bytes)
        if result.backend != "numpy-cpu-interpreter":
            raise ResponseCompatibilityError(
                "CC gradient tensor backend changed; no silent fallback"
            )
        if set(result.outputs) != set(program.outputs):
            raise ResponseCompatibilityError(
                "CC gradient program returned an incomplete output set"
            )
        out = MappingProxyType(
            {
                name: _immutable(_array(result.outputs[name], node.spec.shape, name))
                for name, node in program.outputs.items()
            }
        )
        self._assert_current()
        return out

    def _pullback(self, seeds):
        return self._run(self.programs.weights, {**self.raw_inputs, **seeds})

    def _generated_orbital_action(self, vector):
        generator = self.operator.problem.layout.generator_matrix(vector)
        out = self._run(
            self.programs.orbital_jvp.program,
            {**self.raw_inputs, "d_rotation": generator},
        )
        return -out["d_fov"].reshape(-1)

    def ao_weights(self, weights=None):
        weights = self.weights if weights is None else weights
        return self._run(
            self.ao_program,
            {
                "coefficients": self.reference.coefficients,
                **{name: weights[name] for name in ("hcore", "eri", "overlap")},
            },
        )

    def gradient(self) -> CCSDGradientResult:
        """Contract all analytic native derivatives; never finite-difference energy."""
        started = time.perf_counter()
        self._assert_current()
        raw = self.source.integral_derivatives(
            output_budget_bytes=self.options.max_bytes
        )
        n, ncoord = self.source.nbf, 3 * len(self.source.atoms)
        shapes = {
            "hcore": (ncoord, n, n),
            "overlap": (ncoord, n, n),
            "eri": (ncoord, n, n, n, n),
            "nuclear": (ncoord,),
        }
        if set(raw) != set(shapes):
            raise ResponseCompatibilityError(
                "native CC gradient derivative fields are incomplete"
            )
        derivatives = {
            name: _array(raw[name], shape, "native " + name + " derivative")
            for name, shape in shapes.items()
        }
        self._assert_current()

        def contract(weights):
            ao = self.ao_weights(weights)
            with np.errstate(over="raise", invalid="raise"):
                terms = {
                    name: np.einsum(
                        "qi,i->q",
                        derivatives[name].reshape(ncoord, -1),
                        ao[name].reshape(-1),
                        optimize=False,
                    )
                    for name in ("hcore", "eri", "overlap")
                }
            if any(not np.isfinite(v).all() for v in terms.values()):
                raise ImplicitSolveError("nonfinite analytic CC gradient contraction")
            return terms

        integral = contract(self.weights)
        integral["nuclear"] = derivatives["nuclear"]
        gradient = sum(integral.values())
        physical = {
            name: sum(contract(weights).values())
            for name, weights in self.component_weights.items()
        }
        physical["nuclear"] = derivatives["nuclear"]
        if not np.allclose(gradient, sum(physical.values()), atol=1e-10, rtol=1e-11):
            raise ImplicitSolveError("complete CC gradient component sum failed")
        correlation = float(
            self.response.bound._run(self.response.bound.independent.primal)[
                "correlation_energy"
            ]
        )
        self._assert_current()
        result = CCSDGradientResult(
            self.reference.reference_energy + correlation,
            correlation,
            gradient.reshape(-1, 3),
            {k: v.reshape(-1, 3) for k, v in physical.items()},
            {k: v.reshape(-1, 3) for k, v in integral.items()},
            self.reference.scf_residual,
            max(self.response.bound.cc_r1_max, self.response.bound.cc_r2_max),
            max(
                self.response.shared_lambda_residual_norm,
                self.response.independent_lambda_residual_norm,
            ),
            max(self.z_result.residual_norm, self.independent_z_residual),
            self.orbital_stationarity,
            self.minimum_orbital_curvature,
            self.reference_identity,
            self.response.bound.cc_state_identity,
            self.response.response_identity,
            self.operator_identity,
            self.source_identity,
            {
                "weight_identity": self.weight_identity,
                "hamiltonian_id": self.reference.hamiltonian_id,
                "logical_reserved_host_bytes": self.logical_reserved_host_bytes,
                "native_derivative_output_bytes": _derivative_bytes(self.source),
                "provider_budget_bytes": self.provider.budget_bytes,
                "native_hf_backend": self.reference.hf_backend,
                "tensor_backend": "numpy-cpu-interpreter",
                "orbital_backend": "native-cpu-shell-tile-jk",
                "orbital_solver": "shared-response-gmres",
                "dense_orbital_curvature_check": True,
                "dense_cc_jacobian": False,
                "dense_mo_eri_and_weights": True,
                "dense_ao_derivative_oracle": True,
                "native_public_force_capability": False,
                "triples_gradient": False,
                "same_space_stationarity": self.same_space_stationarity,
                "z_iterations": self.z_result.iterations,
                "z_operator_actions": self.z_result.operator_actions,
                "memory_boundary": "logical owned numeric state; separate HF/provider/opaque BLAS/Python allocations",
                **dict(self.timings),
                "derivative_contraction_seconds": time.perf_counter() - started,
            },
        )
        self._assert_current()
        return result


def complete_gradient_validation(source, *, options=None) -> CCSDGradientResult:
    """Run fresh native HF -> CC -> Lambda -> Z -> complete analytic gradient.

    Source is borrowed and remains open. All four solver states have independent
    physical gates. Only conventional real all-electron restricted CCSD is
    qualified here; no partial forces or unvalidated backend fallbacks exist.
    """
    started = time.perf_counter()
    options = CCSDGradientOptions() if options is None else options
    if not isinstance(options, CCSDGradientOptions):
        raise TypeError("gradient options must be CCSDGradientOptions")
    _validate_source(source)
    if 3 * _derivative_bytes(source) > options.max_bytes:
        raise ImplicitSolveError(
            "CC derivative state budget exceeded before HF execution"
        )
    reference, _ = export_rhf(
        source,
        tolerance=options.scf_tolerance,
        max_iterations=options.scf_max_iterations,
    )
    if reference.scf_residual > 1e-9:
        raise ImplicitSolveError(
            "CC gradient requires a strictly converged physical RHF reference"
        )
    with ConventionalProvider(
        reference, source, budget_bytes=options.provider_budget_bytes
    ) as provider:
        cc = solve(reference, provider, options=options.cc_options)
        if not cc.converged:
            raise ImplicitSolveError(f"CC gradient primal failed: {cc.reason}")

        def current_reference():
            source._check_open()
            if provider._closed:
                raise ResponseCompatibilityError("CC gradient provider closed")
            return provider.snapshot.identity

        bound = BoundCCSDLambda(
            reference,
            cc,
            options=options.lambda_options,
            current_reference=current_reference,
        )
        multipliers = bound.solve(reference_identity=reference.identity)
        response = BoundCCSDResponse(bound, multipliers, max_bytes=options.max_bytes)
        gradient = BoundCCSDGradient(response, provider, options=options).gradient()
    from dataclasses import replace

    return replace(
        gradient,
        diagnostics={
            **gradient.diagnostics,
            "total_endpoint_seconds": time.perf_counter() - started,
        },
    )
