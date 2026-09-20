"""Generated stationary mean-field source contractions (#163 B2.1).

This is a compiler/diagnostic boundary, not a molecular force implementation.
The stationary envelope supplies the Hamiltonian and overlap constraint that
an XC-only MethodIR cannot infer. Existing TensorIR AD generates integral
source weights; neither SCF iterations nor the supplied D/W are differentiated.
Native state/provider binding and complete CPU/CUDA endpoints remain separate.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    constant,
    einsum,
    execute,
    input_tensor,
    reduce_sum,
    transpose_program,
)

from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod
from .typecheck import BackendCapability, verify_method_ir

VERSION = "stationary-gradient-plan-v1"
SCF_POINT_MODEL = "semilocal-scaled-v1/pbe-spin-c2-1e-18"

_STATIONARY_GRADIENT_CAPABILITY = BackendCapability(
    "stationary-gradient-plan",
    ("float64",),
    ("unpolarized", "polarized"),
    (1,),
    ("rho", "sigma"),
    ("semilocal-xc",),
)


@dataclass(frozen=True)
class StationaryMeanField:
    """Explicit mathematical envelope; no native-owner or solve-epoch tokens.

    Restricted D and W already include occupation two. Unrestricted inputs
    contain separately occupation-weighted alpha/beta blocks. This contract
    does not certify that any supplied arrays are a converged physical state.
    """

    point_model: str
    hamiltonian: str = "all-electron"
    coulomb: str = "direct-full-range"
    occupations: str = "fixed-integer"
    topology_policy: str = "stable-explicit-grid-v1"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.point_model not in ("interior-v1", SCF_POINT_MODEL):
            raise UnsupportedMethod("unsupported XC point-model contract")
        if self.hamiltonian not in ("all-electron", "scalar-semilocal-ecp"):
            raise UnsupportedMethod("unsupported stationary hamiltonian contract")
        supported = {
            "coulomb": "direct-full-range",
            "occupations": "fixed-integer",
            "topology_policy": "stable-explicit-grid-v1",
            "dtype": "float64",
        }
        for name, value in supported.items():
            if getattr(self, name) != value:
                raise UnsupportedMethod(f"unsupported stationary {name} contract")


@dataclass(frozen=True)
class GradientSource:
    """Required derivative source and the part of its chain owned upstream."""

    name: str
    primitive: str
    geometric_sources: tuple[str, ...]


_SOURCES = (
    GradientSource("one_electron", "one_electron", ("ao_center", "nuclear_center")),
    GradientSource("coulomb", "coulomb", ("all_eri_centers",)),
    GradientSource("xc_ao", "semilocal_xc", ("ao_center",)),
    GradientSource("xc_grid", "semilocal_xc", ("grid_point",)),
    GradientSource("xc_weight", "semilocal_xc", ("partition_weight",)),
    GradientSource("overlap_pulay", "overlap_constraint", ("ao_center",)),
    GradientSource("nuclear", "nuclear_repulsion", ("nuclear_center",)),
)
_INTEGRAL_SOURCES = ("one_electron", "coulomb", "overlap_pulay")
_ECP_SOURCES = (
    GradientSource("ecp_local", "ecp_local_residual", ("ao_center", "ecp_center")),
    GradientSource(
        "ecp_nonlocal", "ecp_projector_residual", ("ao_center", "ecp_center")
    ),
)


def _positive(value: typing.Any, name: typing.Any) -> typing.Any:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _input(
    name: typing.Any, indices: typing.Any, *, differentiable: typing.Any = False
) -> typing.Any:
    return input_tensor(
        name, TensorSpec(tuple(indices), role="input", differentiable=differentiable)
    )


def _unit_seeded_weight(primal: typing.Any) -> typing.Any:
    """Specialize the generated scalar-objective VJP at its exact unit seed.

    This is SSA substitution, not another AD rule. Rebuilding through Node's
    validated constructor preserves the existing TensorIR semantics. Dead
    integral values and the external cotangent seed disappear from the program.
    """
    reverse = transpose_program(primal, ("energy",), inputs=("integrals",))
    mapped = {}
    for node in reverse.program.live_nodes:
        if node.op == "input" and node.attrs["name"] == "bar_energy":
            mapped[node] = constant(
                (1,), replace(node.spec, role="constant", differentiable=False)
            )
        else:
            mapped[node] = replace(node, inputs=tuple(mapped[n] for n in node.inputs))
    return mapped[reverse.program.outputs["bar_integrals"]]


@dataclass(frozen=True)
class IntegralGradientBlock:
    """Executable, bounded ordered-element contraction, before atom scatter.

    ``integral_derivatives[t,q]`` holds only one provider tile and one coordinate
    block. The caller supplies full ordered AO-pair/quartet tuples, not packed
    tuples with implicit symmetry multiplicities. No global AO-rank-four
    cotangent or coordinate-by-integral Jacobian is requested by this interface.
    """

    source: str
    plan_identity: str
    objective: Program
    weights: Program
    contraction: Program

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(
            {
                "plan": self.plan_identity,
                "source": self.source,
                "objective": self.objective.logical_hash,
                "weights": self.weights.logical_hash,
                "contraction": self.contraction.logical_hash,
            }
        )


@dataclass(frozen=True)
class StationaryGradientPlan:
    """One semilocal mathematical plan shared by future native lowerings.

    Construction represents a complete inventory, not complete executable
    geometric providers. Integral contractions and the final reduction are
    generated here; XC/grid derivatives, atom scatter and native state binding
    still come from independently qualified consumers. Public forces stay off.
    """

    method: MethodIR
    mean_field: StationaryMeanField

    def __post_init__(self) -> None:
        if not isinstance(self.method, MethodIR):
            raise TypeError("stationary gradient requires resolved MethodIR")
        if not isinstance(self.mean_field, StationaryMeanField):
            raise TypeError(
                "stationary gradient requires an explicit mean-field envelope"
            )
        if (
            len(self.method.primitives) != 1
            or type(self.method.primitives[0]) is not SemilocalXCPrimitive
        ):
            raise UnsupportedMethod(
                "required primitive has no stationary-gradient rule"
            )
        verify_method_ir(
            self.method,
            capability=_STATIONARY_GRADIENT_CAPABILITY,
            dtype=self.mean_field.dtype,
            derivative_order=1,
        )
        required = {"energy-density", "feature-gradient"}
        if not required <= set(self.method.primitives[0].derivative_capabilities):
            raise UnsupportedMethod("required XC feature derivative is unavailable")

    @property
    def sources(self) -> typing.Any:
        if self.mean_field.hamiltonian == "scalar-semilocal-ecp":
            return (
                replace(_SOURCES[0], primitive="kinetic_effective_charge_attraction"),
                *_ECP_SOURCES,
                *_SOURCES[1:-1],
                replace(_SOURCES[-1], primitive="effective_charge_nuclear_repulsion"),
            )
        return _SOURCES

    @property
    def source_names(self) -> typing.Any:
        return tuple(source.name for source in self.sources)

    @property
    def spin_blocks(self) -> typing.Any:
        return 2 if self.method.spin == "polarized" else 1

    def to_payload(self) -> typing.Any:
        return {
            "schema": VERSION,
            "method": self.method.semantic_payload(),
            "mean_field": asdict(self.mean_field),
            "sources": [asdict(source) for source in self.sources],
            "convention": "gradient in Eh/bohr; force is its negative",
            "density_convention": "occupation-weighted; sum alpha/beta for Coulomb",
            "integral_layout": "full ordered tuples; no implicit symmetry factors",
            "xc_coefficients": "already applied inside the resolved semilocal primitive",
        }

    @property
    def identity(self) -> typing.Any:
        """Mathematics only; backend artifacts and live leases have other owners."""
        return canonical_hash(self.to_payload())

    def require_native_endpoint(self, backend: typing.Any) -> None:
        """A generated plan alone never grants a complete molecular capability."""
        if backend not in ("cpu", "cuda"):
            raise ValueError("unknown stationary-gradient backend")
        raise NotImplementedError(
            f"complete {backend} stationary gradients require native state/provider "
            "binding and independent XC/grid/endpoint qualification (#163 B2.2/C)"
        )

    def integral_block(
        self,
        source: typing.Any,
        *,
        terms: typing.Any,
        coordinates: typing.Any = 3,
        max_elements: typing.Any = 65536,
    ) -> typing.Any:
        """Generate dL/dI and its contraction with a bounded derivative tile.

        L_h = sum_t D_total[t] h[t]
        L_J = 1/2 sum_t D_total_left[t] D_total_right[t] (ab|cd)[t]
        L_S = -sum_t W_total[t] S[t]

        In the J block each t denotes an ordered quartet: left/right densities
        are the corresponding (ab)/(cd) entries. In the h/S blocks t denotes an
        ordered pair. Providers remain responsible for correct center mapping.
        Only I is differentiated. D/W must come from a validated stationary
        owner when a later native endpoint binds the plan.
        """
        if (
            source not in (*_INTEGRAL_SOURCES, *(s.name for s in _ECP_SOURCES))
            or source not in self.source_names
        ):
            raise ValueError("source is not an integral-gradient primitive")
        _positive(terms, "terms")
        _positive(coordinates, "coordinates")
        _positive(max_elements, "max_elements")
        # Gate before generating any shape-sized AD constants or executing data.
        if terms * (coordinates + 2 * self.spin_blocks + 4) > max_elements:
            raise ValueError("integral-gradient block exceeds the element budget")
        t = Index("t", IndexSpace("ordered_terms", "batch", terms))
        s = Index("s", IndexSpace("density_spin", "spin", self.spin_blocks))
        q = Index("q", IndexSpace("coordinate_block", "batch", coordinates))
        integrals = _input("integrals", (t,), differentiable=True)
        left_name = "weighted_density" if source == "overlap_pulay" else "density_left"
        left = reduce_sum(_input(left_name, (s, t)), (0,))
        if source == "coulomb":
            right = reduce_sum(_input("density_right", (s, t)), (0,))
            energy = einsum(
                "t,t,t->", left, right, integrals, coefficient=Fraction(1, 2)
            )
        else:
            factor = -1 if source == "overlap_pulay" else 1
            energy = einsum("t,t->", left, integrals, coefficient=factor)
        provenance = {"stationary_plan": self.identity, "source": source}
        objective = Program({"energy": energy}, provenance=provenance)
        weight = _unit_seeded_weight(objective)
        weights = Program({"weights": weight}, provenance=provenance)
        gradient = einsum("t,tq->q", weight, _input("integral_derivatives", (t, q)))
        contraction = Program({"gradient": gradient}, provenance=provenance)
        return IntegralGradientBlock(
            source, self.identity, objective, weights, contraction
        )

    def reduction_program(
        self, *, atoms: typing.Any, sources: typing.Any = None
    ) -> typing.Any:
        """Generate one complete component sum with an explicit coverage gate.

        Inputs are already atom-scattered gradients from each named source.
        XC includes the MethodIR coefficients upstream: reweighting it here
        would double count. No partial sum is silently padded with zero sources.
        """
        _positive(atoms, "atoms")
        sources = self.source_names if sources is None else tuple(sources)
        if any(not isinstance(name, str) for name in sources):
            raise TypeError("gradient source names must be strings")
        if len(sources) != len(set(sources)):
            raise ValueError("duplicate gradient source")
        if set(sources) != set(self.source_names):
            raise ValueError("incomplete or unknown gradient source coverage")
        a = Index("a", IndexSpace("atoms", "batch", atoms))
        x = Index("x", IndexSpace("cartesian", "batch", 3))
        # Canonical source order is independent of provider completion order.
        nodes = [_input(name, (a, x)) for name in self.source_names]
        return Program(
            {"gradient": add(*nodes, coefficients=(1,) * len(nodes))},
            provenance={
                "stationary_plan": self.identity,
                "role": "component-reduction",
            },
        )

    def reduce_diagnostic(
        self,
        components: typing.Any,
        *,
        atoms: typing.Any,
        max_bytes: typing.Any = 8 * 1024 * 1024,
    ) -> typing.Any:
        """Strict CPU-interpreter diagnostic; never certifies public forces.

        The existing interpreter checks shape, FP64 dtype, finite values and
        logical retained-byte budget. It returns detached arrays transactionally.
        Live owner/provider checks belong to the native consumer, not this API.
        """
        if not isinstance(components, Mapping):
            raise TypeError("gradient components must be a mapping")
        program = self.reduction_program(atoms=atoms, sources=components.keys())
        return execute(program, components, max_bytes=max_bytes).outputs["gradient"]
