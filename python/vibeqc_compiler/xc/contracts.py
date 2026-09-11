"""Typed contracts for differentiation of a fixed discrete semilocal energy.

The energy is sum_g w_g e(z_g), where e is per volume. AO-center, grid-point,
quadrature-weight and density axes are independent sources. Neither electronic
stationarity nor physical grid motion can be inferred from this scalar alone.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement

from vibeqc_compiler.common.provenance import canonical_hash

from .spec import FunctionalSpec, UnsupportedXC


@dataclass(frozen=True)
class IngredientContract:
    """Real symmetric AO ingredients with explicit spin and trace conventions.

    rho_s = phi D_s phi; grad rho_s contains both differentiated AO legs;
    sigma_ab = grad rho_a dot grad rho_b has no extra factor two; and
    tau_s = one-half sum_k d_k phi D_s d_k phi. LDA/GGA requests prune tau.
    """

    spin: str = "polarized"
    family: str = "gga"

    def __post_init__(self):
        if self.spin not in ("polarized", "unpolarized") or self.family not in (
            "lda",
            "gga",
        ):
            raise UnsupportedXC("contractions support real LDA/GGA spin ingredients")

    @property
    def feature_indices(self):
        """Required slots of the audited scalar functional's feature ABI."""
        return tuple(
            range(
                (2 if self.family == "lda" else 5)
                if self.spin == "polarized"
                else (1 if self.family == "lda" else 2)
            )
        )

    @property
    def ao_order(self):
        return 0 if self.family == "lda" else 1

    def to_payload(self):
        return {
            "spin": self.spin,
            "family": self.family,
            "features": self.feature_indices,
            "density": "real symmetric full matrices; total D means Da=Db=D/2",
            "trace": "sum_s,mu,nu V_s[mu,nu] deltaD_s[mu,nu]; both offdiagonal entries",
            "sigma_ab_factor": 1,
            "tau_factor": "1/2",
            "jets": "ordinary spatial derivatives without factorials",
        }


@dataclass(frozen=True)
class DerivativeRequest:
    """One observable on a fixed quadrature/mask/regularization branch.

    Geometry means explicit first-order partials at fixed D, including separately
    supplied AO-center, point and weight motion. It is not a molecular force API.
    Response is a linear action on a symmetric density direction, never AO^4 data.
    """

    observable: str = "potential"

    def __post_init__(self):
        if self.observable not in ("energy", "potential", "response", "geometry"):
            raise UnsupportedXC("unsupported XC contraction observable/order")

    @property
    def scalar_order(self):
        return (
            0
            if self.observable == "energy"
            else 2
            if self.observable == "response"
            else 1
        )

    @property
    def held_fixed(self):
        if self.observable == "geometry":
            return (
                "density",
                "basis exponents/coefficients",
                "functional parameters",
                "mask membership",
                "regularization",
            )
        return (
            "basis",
            "grid points",
            "quadrature weights",
            "functional parameters",
            "mask membership",
            "regularization",
        )


@dataclass(frozen=True)
class DiscreteEnergyContract:
    """Traceable semilocal discrete-energy and derivative-output description."""

    functional: FunctionalSpec
    request: DerivativeRequest = DerivativeRequest()

    def __post_init__(self):
        if not isinstance(self.functional, FunctionalSpec) or not isinstance(
            self.request, DerivativeRequest
        ):
            raise TypeError("expected typed functional and derivative request")
        if any(
            (
                self.functional.exact_exchange,
                self.functional.range_omega,
                self.functional.long_range_exchange,
            )
        ):
            raise UnsupportedXC(
                "semilocal contractions do not include exact exchange/RSH"
            )

    @property
    def ingredients(self):
        return IngredientContract(
            self.functional.spin,
            "gga" if "sigma" in self.functional.ingredients else "lda",
        )

    @property
    def ao_order(self):
        """Geometry needs one extra ordinary spatial AO derivative."""
        return self.ingredients.ao_order + (self.request.observable == "geometry")

    @property
    def scalar_outputs(self):
        """Request only active functional derivatives; unused tau slots disappear."""
        indices = self.ingredients.feature_indices
        outputs = [()]
        if self.request.scalar_order >= 1:
            outputs.extend((i,) for i in indices)
        if self.request.scalar_order >= 2:
            outputs.extend(combinations_with_replacement(indices, 2))
        return tuple(outputs)

    def to_payload(self):
        return {
            "schema": "vibeqc.xc-discrete-energy.v1",
            "functional": self.functional.identity,
            "ingredients": self.ingredients.to_payload(),
            "observable": self.request.observable,
            "held_fixed": self.request.held_fixed,
            "ao_order": self.ao_order,
            "scalar_outputs": self.scalar_outputs,
            "energy": "sum_g w_g e_xc(z_g); e_xc per volume, never epsilon per particle",
            "geometry_sources": ("AO centers", "grid points", "quadrature weights"),
            "force_convention": "force=-gradient; returned explicit XC partials are gradients",
            "method_terms": "stationarity, orbital response, Pulay and other energy terms belong to method owners",
        }

    @property
    def identity(self):
        return canonical_hash(self.to_payload())
