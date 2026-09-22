"""Generic MethodIR-derived stationary DFT HVP planning (#180/#181).

This compiler boundary decides whether a resolved MethodIR and stationary
mean-field envelope have the primitive families needed by the first LDA/GGA
Hessian/HVP slice and generates bounded integral-source contractions. It does
not solve CPKS, execute grid motion, or publish a molecular Hessian capability.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    linearize,
)

from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod
from .stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from .typecheck import BackendCapability, verify_method_ir

VERSION = "stationary-hvp-plan-v1"

_STATIONARY_HVP_CAPABILITY = BackendCapability(
    "stationary-hvp-plan",
    ("float64",),
    ("unpolarized", "polarized"),
    (2,),
    ("rho", "sigma"),
    ("semilocal-xc",),
)


@dataclass(frozen=True)
class HVPSource:
    """One required directional source owned by a shared primitive family."""

    name: str
    primitive: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class HVPIntegralBlock:
    """Executable source-level contraction for one integral HVP tile.

    ``weights`` is the fixed-density first-derivative weight generated from
    the stationary energy. ``response_weights`` is its exact TensorIR JVP for
    the supplied density/weighted-density direction. ``contraction`` combines
    that electronic-response term with the provider's fixed-weight second
    integral HVP vector. Native providers still own shell/center recovery and
    live state validation; this block only defines the bounded algebra they
    consume.
    """

    source: str
    plan_identity: str
    objective: Program
    weights: Program
    response_weights: Program
    contraction: Program
    response_inputs: tuple[str, ...]

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(
            {
                "plan": self.plan_identity,
                "source": self.source,
                "objective": self.objective.logical_hash,
                "weights": self.weights.logical_hash,
                "response_weights": self.response_weights.logical_hash,
                "contraction": self.contraction.logical_hash,
                "response_inputs": self.response_inputs,
            }
        )


_HVP_SOURCES = (
    HVPSource(
        "one_electron",
        "one_electron",
        ("first-integral-direction", "second-integral-hvp"),
    ),
    HVPSource(
        "coulomb",
        "coulomb",
        ("density-response", "first-integral-direction", "second-integral-hvp"),
    ),
    HVPSource(
        "xc_ao",
        "semilocal_xc",
        ("feature-hessian", "density-response", "ao-geometry-direction"),
    ),
    HVPSource(
        "xc_grid",
        "semilocal_xc",
        ("feature-hessian", "density-response", "grid-geometry-direction"),
    ),
    HVPSource(
        "xc_weight",
        "semilocal_xc",
        ("feature-hessian", "density-response", "partition-weight-direction"),
    ),
    HVPSource(
        "overlap_pulay",
        "overlap_constraint",
        ("metric-response", "first-integral-direction", "second-integral-hvp"),
    ),
    HVPSource(
        "nuclear",
        "nuclear_repulsion",
        ("nuclear-hvp",),
    ),
)

_INTEGRAL_SOURCES = ("one_electron", "coulomb", "overlap_pulay")


def _positive(value: typing.Any, name: str) -> int:
    """Validate bounded tensor extents before constructing any large DAG."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _input(
    name: str, indices: tuple[Index, ...], *, differentiable: bool = False
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(tuple(indices), role="input", differentiable=differentiable),
    )


@dataclass(frozen=True)
class StationaryHVPPlan:
    """Inspect a MethodIR and derive the first common LDA/GGA HVP topology.

    The first admitted compiler slice is intentionally narrow: direct,
    all-electron, real FP64, fixed-integer RKS/UKS with rho/sigma semilocal XC.
    New functionals inside that primitive family inherit this plan automatically.
    New physics remains fail-closed until its primitive second-order rule exists.
    """

    method: MethodIR
    mean_field: StationaryMeanField

    def __post_init__(self) -> None:
        if not isinstance(self.method, MethodIR):
            raise TypeError("stationary HVP requires resolved MethodIR")
        if not isinstance(self.mean_field, StationaryMeanField):
            raise TypeError("stationary HVP requires an explicit mean-field envelope")
        if self.mean_field.point_model != SCF_POINT_MODEL:
            raise UnsupportedMethod(
                "stationary HVP requires the qualified native SCF point model"
            )
        if self.mean_field.hamiltonian != "all-electron":
            raise UnsupportedMethod(
                "stationary HVP first slice requires an all-electron Hamiltonian"
            )

        semilocal = tuple(
            primitive
            for primitive in self.method.primitives
            if type(primitive) is SemilocalXCPrimitive
        )
        if len(semilocal) != 1 or len(self.method.primitives) != 1:
            raise UnsupportedMethod(
                "required primitive has no stationary-HVP second-order rule"
            )
        verify_method_ir(
            self.method,
            capability=_STATIONARY_HVP_CAPABILITY,
            dtype=self.mean_field.dtype,
            derivative_order=2,
        )
        required = {"energy-density", "feature-gradient", "feature-hessian"}
        if not required <= set(semilocal[0].derivative_capabilities):
            raise UnsupportedMethod(
                "required semilocal XC second-order feature derivative is unavailable"
            )

    @property
    def semilocal(self) -> typing.Any:
        return next(
            primitive
            for primitive in self.method.primitives
            if type(primitive) is SemilocalXCPrimitive
        )

    @property
    def active_ingredients(self) -> typing.Any:
        return tuple(self.method.requirements["ingredients"])

    @property
    def sources(self) -> typing.Any:
        return _HVP_SOURCES

    @property
    def source_names(self) -> typing.Any:
        return tuple(source.name for source in self.sources)

    @property
    def response_contract(self) -> typing.Any:
        return {
            "solver": "shared-native-cpks-issue-179",
            "xc_action": "feature-hessian-times-direction",
            "integral_second_order": "issue-178-weighted-hvp",
            "assembly": "matrix-free-directional",
        }

    def integral_block(
        self,
        source: str,
        *,
        terms: int,
        coordinates: int = 3,
        max_elements: int = 65536,
    ) -> HVPIntegralBlock:
        """Build one bounded integral contribution to a stationary HVP.

        The returned programs are deliberately provider-facing rather than a
        molecular endpoint. ``weights`` differentiates the stationary energy
        with respect to an ordered integral tile. ``response_weights`` is the
        demand-driven JVP of that weight, so it carries the density/weighted
        density response once. ``contraction`` then adds

        ``d(weight) * dI + weighted_d2I(direction)``.

        The second term is consumed as the already-weighted output vector from
        #178's ``weighted_hvp`` provider. No shell tensor or full molecular
        Hessian is materialized here, and the block does not create or own a
        CPKS solver. XC/grid/partition sources remain outside this method until
        their native geometric directional providers are qualified.
        """
        if source not in _INTEGRAL_SOURCES or source not in self.source_names:
            raise ValueError("source is not a stationary-HVP integral primitive")
        _positive(terms, "terms")
        _positive(coordinates, "coordinates")
        _positive(max_elements, "max_elements")
        # Two bounded vectors plus the spin-resolved response fields are the
        # largest symbolic feeds. Keep generation fail-closed before creating
        # any dense constants or identity projections.
        spin_blocks = 2 if self.method.spin == "polarized" else 1
        if terms * (coordinates + 2 * spin_blocks + 4) > max_elements:
            raise ValueError("stationary-HVP integral block exceeds element budget")

        t = Index("t", IndexSpace("ordered_terms", "batch", terms))
        q = Index("q", IndexSpace("coordinate_block", "batch", coordinates))

        # Reuse the stationary-gradient source energy/weight DAG. This keeps
        # first- and second-order source coefficients in one owner; the only
        # new algebra here is the demand-driven JVP of the existing weights.
        gradient_block = StationaryGradientPlan(
            self.method, self.mean_field
        ).integral_block(
            source,
            terms=terms,
            coordinates=coordinates,
            max_elements=max_elements,
            differentiate_densities=True,
        )
        response_inputs = (
            ("density_left",)
            if source == "one_electron"
            else ("density_left", "density_right")
            if source == "coulomb"
            else ("weighted_density",)
        )

        provenance = {
            "stationary_hvp_plan": self.identity,
            "source": source,
            "role": "integral-source-hvp",
        }
        objective = gradient_block.objective
        weights = gradient_block.weights
        tangent = linearize(
            weights,
            response_inputs,
            outputs=("weights",),
        )
        response_weights = Program(
            {"response_weights": tangent.program.outputs["d_weights"]},
            tangent.program.definitions,
            provenance={**provenance, "role": "integral-source-response-weight"},
        )

        response_weight = _input("response_weights", (t,))
        first = _input("integral_derivatives", (t, q))
        second = _input("weighted_second_hvp", (q,))
        response_term = einsum("t,tq->q", response_weight, first)
        second_term = second
        contraction = Program(
            {
                "response": response_term,
                "second": second_term,
                "hvp": add(response_term, second_term),
            },
            provenance={**provenance, "role": "integral-source-hvp-contraction"},
        )
        return HVPIntegralBlock(
            source,
            self.identity,
            objective,
            weights,
            response_weights,
            contraction,
            tuple(response_inputs),
        )

    def to_payload(self) -> typing.Any:
        return {
            "schema": VERSION,
            "method": self.method.semantic_payload(),
            "mean_field": asdict(self.mean_field),
            "active_ingredients": list(self.active_ingredients),
            "sources": [asdict(source) for source in self.sources],
            "response": self.response_contract,
            "convention": "energy HVP in Eh/bohr^2; force derivative is its negative",
            "capability": (
                "compiler plan only; native molecular HVP/Hessian remains separately gated"
            ),
        }

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(self.to_payload())

    def require_native_endpoint(self, backend: typing.Any) -> None:
        if backend not in ("cpu", "cuda"):
            raise ValueError("unknown stationary-HVP backend")
        raise NotImplementedError(
            f"complete {backend} stationary DFT HVP requires native geometric "
            "directional consumers and independent #180 qualification"
        )
