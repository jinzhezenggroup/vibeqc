"""Generic MethodIR-derived stationary DFT HVP planning (#180/#181).

This module is deliberately data-only.  It decides whether a resolved MethodIR
and stationary mean-field envelope have the primitive families needed by the
first LDA/GGA Hessian/HVP slice.  It does not solve CPKS, execute grid motion,
or publish a molecular Hessian capability.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .spec import MethodIR, SemilocalXCPrimitive, UnsupportedMethod
from .stationary_gradient import SCF_POINT_MODEL, StationaryMeanField
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
