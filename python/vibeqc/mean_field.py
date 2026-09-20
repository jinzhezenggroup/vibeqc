"""Fixed-density MethodIR mean-field consumer of native common J/K sources."""

import typing
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method import (
    ExactExchangePrimitive,
    MethodIR,
    NonlocalCorrelationPrimitive,
    RangeSeparatedExchangePrimitive,
    SemilocalXCPrimitive,
    UnsupportedMethod,
)
from vibeqc_compiler.xc.spec import FunctionalSpec

from .fock import FockBuildSpec, FockPlan, FockTerm


@dataclass(frozen=True, eq=False)
class MeanFieldEvaluation:
    """Full fixed-density energy and AO Fock including semilocal XC.

    The energy includes nuclear repulsion. This object does not imply SCF
    stationarity and contains no geometric XC gradient or complete force.
    """

    energy: float
    fock: np.ndarray
    xc_energy: float
    identity: str
    fock_identity: str
    xc_identity: str
    method_identity: str | None = None
    method_plan_identity: str | None = None
    nonlocal_energy: float = 0.0
    nonlocal_identity: str | None = None


@dataclass(frozen=True, eq=False)
class FixedDensityExchangeEvaluation:
    """Exchange-only fixed-density energy and AO potential from raw K matrices."""

    energy: float
    potential: np.ndarray
    identity: str
    method_identity: str
    operator_keys: tuple[tuple[str, Fraction], ...]


def exchange_operator_key(primitive: typing.Any) -> typing.Any:
    """Return the exact operator/omega key consumed by fixed-density exchange."""

    if isinstance(primitive, ExactExchangePrimitive):
        return primitive.operator, Fraction(0)
    if isinstance(primitive, RangeSeparatedExchangePrimitive):
        return primitive.operator, primitive.omega
    raise TypeError("expected an exact-exchange primitive")


def assemble_fixed_density_exchange(
    method: typing.Any, density: typing.Any, raw_exchange: typing.Any
) -> typing.Any:
    """Apply MethodIR exchange coefficients to provider-produced raw K matrices.

    Restricted total-density K contributes Vx=-a*K/2; unrestricted same-spin K
    contributes Vx_s=-a*K_s. In both cases Ex=1/2 Tr(D Vx), so one resolved
    coefficient graph drives energy and potential. This function does not build K.
    """
    if not isinstance(method, MethodIR):
        raise TypeError("fixed-density exchange assembly requires MethodIR")
    d = np.asarray(density)
    if method.reference == "restricted":
        valid_density = d.ndim == 2 and d.shape[0] == d.shape[1]
        density_factor = -0.5
    else:
        valid_density = d.ndim == 3 and d.shape[0] == 2 and d.shape[1] == d.shape[2]
        density_factor = -1.0
    if not valid_density or np.iscomplexobj(d) or not np.isfinite(d).all():
        raise ValueError(
            "density shape/reference mismatch or nonfinite/complex density"
        )
    expected_shape = d.shape

    primitives = tuple(
        p
        for p in method.primitives
        if isinstance(p, (ExactExchangePrimitive, RangeSeparatedExchangePrimitive))
    )
    keys = tuple(exchange_operator_key(p) for p in primitives)
    if set(raw_exchange) != set(keys):
        raise ValueError("raw exchange operator set does not match MethodIR")
    potential = np.zeros(expected_shape, dtype=np.float64)
    raw_hashes = []
    for primitive, key in zip(primitives, keys, strict=True):
        k = np.asarray(raw_exchange[key])
        if k.shape != expected_shape or np.iscomplexobj(k) or not np.isfinite(k).all():
            raise ValueError(f"invalid raw K for operator {key!r}")
        k = np.asarray(k, dtype=np.float64)
        potential += density_factor * float(primitive.coefficient) * k
        raw_hashes.append(sha256(np.ascontiguousarray(k).tobytes()).hexdigest())
    energy = 0.5 * float(np.sum(np.asarray(d, dtype=np.float64) * potential))
    if not np.isfinite(energy):
        raise ArithmeticError("nonfinite fixed-density exchange energy")
    density_hash = sha256(np.ascontiguousarray(d).tobytes()).hexdigest()
    identity = canonical_hash(
        {
            "schema": "vibeqc.fixed-density-exchange/v1",
            "method": method.identity,
            "density_sha256": density_hash,
            "operators": [
                [op, str(omega), raw_hash]
                for (op, omega), raw_hash in zip(keys, raw_hashes, strict=True)
            ],
        }
    )
    return FixedDensityExchangeEvaluation(
        energy,
        immutable(potential),
        identity,
        method.identity,
        keys,
    )


@dataclass(frozen=True)
class FixedDensityMethodPlan:
    """Executable energy/Fock projection of one supported MethodIR graph.

    MethodIR remains the scientific source of truth. This projection records
    the existing J/K provider request needed to execute it; it adds neither a
    method-name branch nor SCF/geometric-gradient capability.
    """

    method: MethodIR
    functional: FunctionalSpec
    fock_spec: FockBuildSpec

    def __post_init__(self) -> None:
        if not isinstance(self.fock_spec, FockBuildSpec):
            raise TypeError("executable MethodIR plan requires FockBuildSpec")
        functional, spec = _compile_fixed_density_components(
            self.method,
            coulomb_approximation=self.fock_spec.coulomb.approximation,
            exchange_approximation=self.fock_spec.exchange.approximation,
            provider_derivative_order=self.fock_spec.derivative_order,
        )
        if self.functional != functional or self.fock_spec != spec:
            raise ValueError(
                "executable MethodIR plan differs from its declared physics"
            )

    @property
    def capabilities(self) -> typing.Any:
        return ("energy", "fock")

    def semantic_payload(self) -> typing.Any:
        return {
            "schema": "vibeqc.fixed-density-method-plan/v1",
            "method_identity": self.method.identity,
            "spin": self.method.spin,
            "reference": self.method.reference,
            "fock_spec": self.fock_spec.to_dict(),
            "capabilities": self.capabilities,
        }

    def to_payload(self) -> typing.Any:
        return {
            **self.semantic_payload(),
            "method": self.method.to_payload(),
            "method_manifest_identity": self.method.manifest_identity,
            "functional_identity": self.functional.identity,
        }

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.semantic_payload())


def _compile_fixed_density_components(
    method: typing.Any,
    *,
    coulomb_approximation: typing.Any = "exact",
    exchange_approximation: typing.Any = "exact",
    provider_derivative_order: typing.Any = 0,
) -> typing.Any:
    """Compile semilocal plus full-range exchange to the common J/K request.

    Exact exchange supplies a physical fraction a_x. The native provider uses
    raw same-spin K with its density convention, so the Fock coefficient is
    -a_x/2 for restricted total density and -a_x for unrestricted spin density.
    Energy and Fock therefore consume one resolved coefficient.
    """
    if not isinstance(method, MethodIR):
        raise TypeError("fixed-density method compilation requires MethodIR")
    semilocal = None
    exchange = None
    for primitive in method.primitives:
        if isinstance(primitive, SemilocalXCPrimitive):
            semilocal = primitive
        elif isinstance(primitive, ExactExchangePrimitive):
            exchange = primitive
        elif isinstance(primitive, NonlocalCorrelationPrimitive):
            pass
        else:
            raise UnsupportedMethod(
                f"fixed-density execution does not support primitive {primitive.kind!r}"
            )
    if semilocal is None:
        raise UnsupportedMethod(
            "fixed-density execution requires one semilocal XC primitive"
        )

    exchange_term = FockTerm(present=False, coefficient=0.0)
    if exchange is not None:
        operators = {"full-range": "full_range"}
        try:
            operator = operators[exchange.operator]
        except KeyError as error:
            raise UnsupportedMethod(
                "fixed-density execution cannot lower exchange operator "
                f"{exchange.operator!r}"
            ) from error
        exchange_term = FockTerm(
            coefficient=float(exchange.fock_coefficient(method.spin)),
            operator=operator,
            approximation=exchange_approximation,
        )

    fock_spec = FockBuildSpec(
        spin=method.reference,
        derivative_order=provider_derivative_order,
        coulomb=FockTerm(coefficient=1.0, approximation=coulomb_approximation),
        exchange=exchange_term,
    )
    return semilocal.functional, fock_spec


def compile_fixed_density_method(
    method: typing.Any,
    *,
    coulomb_approximation: typing.Any = "exact",
    exchange_approximation: typing.Any = "exact",
    provider_derivative_order: typing.Any = 0,
) -> typing.Any:
    """Compile a MethodIR graph into a self-consistent executable energy/Fock plan."""
    functional, spec = _compile_fixed_density_components(
        method,
        coulomb_approximation=coulomb_approximation,
        exchange_approximation=exchange_approximation,
        provider_derivative_order=provider_derivative_order,
    )
    return FixedDensityMethodPlan(method, functional, spec)


class FixedDensityMeanField:
    """Combine executable semilocal XC with the existing common J/K provider.

    Legacy direct construction retains the unit-J/absent-K semilocal contract.
    from_method instead binds an executable MethodIR plan, including full-range
    exact exchange. No HF J/K contraction is duplicated here and no complete
    DFT SCF or geometric-gradient capability is registered.
    """

    def __init__(
        self,
        fock: typing.Any,
        xc: typing.Any,
        *,
        method_plan: typing.Any = None,
        nonlocal_correlation: typing.Any = None,
    ) -> None:
        from vibeqc_compiler.dft import FixedDensityNonlocalCorrelation
        from vibeqc_compiler.xc.integration import FixedDensityXC

        if not isinstance(fock, FockPlan) or not isinstance(xc, FixedDensityXC):
            raise TypeError("expected FockPlan and FixedDensityXC")
        if nonlocal_correlation is not None and not isinstance(
            nonlocal_correlation, FixedDensityNonlocalCorrelation
        ):
            raise TypeError(
                "nonlocal_correlation must be FixedDensityNonlocalCorrelation"
            )
        spec = fock.spec
        if method_plan is None:
            if (
                not spec.coulomb.present
                or spec.coulomb.coefficient != 1
                or spec.exchange.present
            ):
                raise ValueError(
                    "semilocal mean field requires unit Coulomb and absent exchange"
                )
            if nonlocal_correlation is not None:
                raise ValueError(
                    "nonlocal execution requires an explicit MethodIR plan"
                )
        else:
            if not isinstance(method_plan, FixedDensityMethodPlan):
                raise TypeError("method_plan must be a FixedDensityMethodPlan")
            if (
                fock.diagnostics["resolved"] != method_plan.fock_spec.to_dict()
                or xc.spec != method_plan.functional
            ):
                raise ValueError(
                    "Fock/XC providers do not match the executable MethodIR plan"
                )
            expected_nonlocal = next(
                (
                    primitive
                    for primitive in method_plan.method.primitives
                    if isinstance(primitive, NonlocalCorrelationPrimitive)
                ),
                None,
            )
            if (expected_nonlocal is None) != (nonlocal_correlation is None):
                raise ValueError(
                    "nonlocal provider does not match the executable MethodIR plan"
                )
            if expected_nonlocal is not None and (
                nonlocal_correlation.spec != expected_nonlocal.spec
                or nonlocal_correlation.coefficient != expected_nonlocal.coefficient
            ):
                raise ValueError(
                    "nonlocal provider does not match the executable MethodIR plan"
                )
        self._fock = fock
        self._xc = xc
        self._method_plan = method_plan
        self._nonlocal = nonlocal_correlation

    @classmethod
    def from_method(cls, fock: typing.Any, method: typing.Any) -> typing.Any:
        """Bind a MethodIR graph to an already prepared common J/K provider."""
        from vibeqc_compiler.dft import FixedDensityNonlocalCorrelation
        from vibeqc_compiler.xc.integration import FixedDensityXC

        if not isinstance(fock, FockPlan):
            raise TypeError("expected FockPlan")
        spec = fock.spec
        plan = compile_fixed_density_method(
            method,
            coulomb_approximation=spec.coulomb.approximation,
            exchange_approximation=spec.exchange.approximation,
            provider_derivative_order=spec.derivative_order,
        )
        nonlocal_primitive = next(
            (
                primitive
                for primitive in plan.method.primitives
                if isinstance(primitive, NonlocalCorrelationPrimitive)
            ),
            None,
        )
        nonlocal_correlation = None
        if nonlocal_primitive is not None:
            nonlocal_correlation = FixedDensityNonlocalCorrelation(
                nonlocal_primitive.spec,
                coefficient=nonlocal_primitive.coefficient,
            )
        return cls(
            fock,
            FixedDensityXC(plan.functional),
            method_plan=plan,
            nonlocal_correlation=nonlocal_correlation,
        )

    @property
    def method_plan(self) -> typing.Any:
        return self._method_plan

    def integrate(
        self, grid: typing.Any, density: typing.Any, *, tile_points: typing.Any = 256
    ) -> typing.Any:
        """Evaluate the same density and immutable basis in both consumers.

        XC validates molecular grid/basis compatibility and functional domain
        before native J executes. Both component identities enter the result.
        """
        if np.iscomplexobj(density):
            raise TypeError("mean-field densities must be real")
        snapshot = np.array(density, dtype=np.float64, order="C", copy=True)
        xc = self._xc.integrate(
            self._fock.basis, grid, snapshot, tile_points=tile_points
        )
        nonlocal_result = None
        if self._nonlocal is not None:
            nonlocal_result = self._nonlocal.integrate(
                self._fock.basis, grid, snapshot, tile_points=tile_points
            )
        native = self._fock.evaluate(snapshot)
        fock = native.fock + xc.potential
        energy = native.energy + xc.energy
        if nonlocal_result is not None:
            fock = fock + nonlocal_result.potential
            energy += nonlocal_result.energy
        if not np.isfinite(energy) or not np.isfinite(fock).all():
            raise ArithmeticError("nonfinite combined mean-field result")
        method_plan_identity = (
            None if self._method_plan is None else self._method_plan.identity
        )
        method_identity = (
            None if self._method_plan is None else self._method_plan.method.identity
        )
        identity_payload = {
            "schema": "vibeqc.fixed-density-mean-field/v1",
            "fock": native.identity,
            "xc": xc.identity,
        }
        if method_plan_identity is not None:
            identity_payload = {
                **identity_payload,
                "schema": "vibeqc.fixed-density-mean-field/v2",
                "method_plan": method_plan_identity,
            }
        nonlocal_identity = None
        nonlocal_energy = 0.0
        if nonlocal_result is not None:
            nonlocal_identity = nonlocal_result.identity
            nonlocal_energy = nonlocal_result.energy
            identity_payload = {
                **identity_payload,
                "schema": "vibeqc.fixed-density-mean-field/v3",
                "nonlocal": nonlocal_identity,
            }
        return MeanFieldEvaluation(
            energy,
            immutable(fock),
            xc.energy,
            canonical_hash(identity_payload),
            native.identity,
            xc.identity,
            method_identity,
            method_plan_identity,
            nonlocal_energy,
            nonlocal_identity,
        )
