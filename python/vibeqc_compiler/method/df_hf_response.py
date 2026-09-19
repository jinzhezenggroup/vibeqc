"""DF-HF stationary source-weight composition (#358).

This module owns method equations only. Integral derivative generation stays in
#143, the metric spectral derivative is a shared matrix-function custom rule,
and native DF code retains tiling, BLAS/eigensolver calls, streams and lifetime
management. The first production-shaped slice is restricted RHF at a supplied
stationary density/energy-weighted density; no SCF iteration history is
differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)

from .df_hf_response_contract import (
    CONTRACT_IDENTITY,
    RESPONSE_SLICE,
    RHF_COULOMB_COEFFICIENT,
    RHF_EXCHANGE_COEFFICIENT,
)
from .matrix_function import SymmetricMatrixFunctionSpec
from .stationary import ParameterSource, StationaryProblem, StationaryState


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coefficient(value, name: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be an exact Fraction")
    return value


@dataclass(frozen=True)
class DensityFittingRHFResponsePlan:
    """Compiler-owned RHF source algebra around one full-rank fitted state.

    ``fitted[p,i,j]`` is defined by M[p,q] fitted[q,i,j] = A[p,i,j].
    At a stationary RHF density D and energy-weighted density W,

      E = D:h - W:S
        + cJ/2 (D:A_p)(D:fitted_p)
        - cK A_p,ij D_ki fitted_p,kl D_lj.

    Differentiating the explicit stationary problem yields h/S/A/M weights.
    The full-rank state is a concrete first integration slice; production
    truncated-metric response uses :meth:`metric_rule` with explicit branch
    semantics rather than differentiating an eigensolver gauge.
    """

    nbf: int
    naux: int
    coulomb_coefficient: Fraction = RHF_COULOMB_COEFFICIENT
    exchange_coefficient: Fraction = RHF_EXCHANGE_COEFFICIENT

    def __post_init__(self):
        _positive(self.nbf, "nbf")
        _positive(self.naux, "naux")
        _coefficient(self.coulomb_coefficient, "coulomb_coefficient")
        _coefficient(self.exchange_coefficient, "exchange_coefficient")

    def problem(self) -> StationaryProblem:
        ao = IndexSpace("df_rhf_ao", "ao", self.nbf)
        aux = IndexSpace("df_rhf_aux", "auxiliary", self.naux)
        i, j = (Index(name, ao) for name in ("i", "j"))
        p, q = (Index(name, aux) for name in ("p", "q"))

        matrix = TensorSpec((i, j), role="input", differentiable=False)
        density = input_tensor("density", matrix)
        weighted_density = input_tensor("weighted_density", matrix)
        hamiltonian = input_tensor(
            "one_electron",
            TensorSpec((i, j), role="parameter", differentiable=True),
        )
        overlap = input_tensor(
            "overlap", TensorSpec((i, j), role="parameter", differentiable=True)
        )
        three_center = input_tensor(
            "three_center",
            TensorSpec((p, i, j), role="parameter", differentiable=True),
        )
        metric = input_tensor(
            "metric", TensorSpec((p, q), role="parameter", differentiable=True)
        )
        fitted = input_tensor(
            "fitted",
            TensorSpec((p, i, j), role="parameter", differentiable=True),
        )

        one_electron = einsum("ij,ij->", density, hamiltonian)
        pulay = einsum("ij,ij->", weighted_density, overlap, coefficient=-1)
        raw_charge = einsum("ij,pij->p", density, three_center)
        fitted_charge = einsum("ij,pij->p", density, fitted)
        coulomb = einsum(
            "p,p->",
            raw_charge,
            fitted_charge,
            coefficient=self.coulomb_coefficient / 2,
        )
        exchange = einsum(
            "pij,ki,pkl,lj->",
            three_center,
            density,
            fitted,
            density,
            coefficient=-self.exchange_coefficient,
        )
        residual = add(
            einsum("pq,qij->pij", metric, fitted),
            three_center,
            coefficients=(1, -1),
        )
        equations = Program(
            {
                "energy": add(one_electron, pulay, coulomb, exchange),
                "fitted_residual": residual,
            },
            provenance={
                "method": "RHF",
                "response_slice": RESPONSE_SLICE,
                "contract_identity": CONTRACT_IDENTITY,
                "issue": 358,
            },
        )
        sources = (
            ParameterSource("geometry", "nuclear-cartesian-v1"),
            ParameterSource("density", "rhf-stationary-density-v1"),
            ParameterSource(
                "metric",
                "df-coulomb-metric-v1",
                dependencies=("geometry",),
                pullback_identity="generated-df-metric-derivative-143-v1",
            ),
            ParameterSource(
                "one_electron",
                "one-electron-hamiltonian-v1",
                dependencies=("geometry",),
                pullback_identity="generated-one-electron-derivative-v1",
            ),
            ParameterSource(
                "overlap",
                "ao-overlap-v1",
                dependencies=("geometry",),
                pullback_identity="generated-overlap-derivative-v1",
            ),
            ParameterSource(
                "three_center",
                "df-three-center-A-v1",
                dependencies=("geometry",),
                pullback_identity="generated-df-three-center-derivative-143-v1",
            ),
            ParameterSource("weighted_density", "rhf-energy-weighted-density-v1"),
        )
        return StationaryProblem(
            equations=equations,
            objective="energy",
            states=(
                StationaryState(
                    "fitted",
                    "fitted_residual",
                    "auxiliary-major-dense-ao-v1",
                    "full-rank-df-metric-v1",
                ),
            ),
            sources=sources,
            model_identity=(
                f"rhf-df-response-v1/{CONTRACT_IDENTITY}/n={self.nbf}/a={self.naux}/"
                f"cj={self.coulomb_coefficient}/ck={self.exchange_coefficient}"
            ),
            solver_contract="symmetric-df-metric-linear-solve-v1",
        )

    def compile(self, *, max_elements: int = 1_000_000):
        """Generate h/S/A/M weights through the common StationaryProblem AD."""
        return self.problem().compile(max_elements=max_elements)

    def metric_rule(self, relative_threshold: float) -> SymmetricMatrixFunctionSpec:
        """Explicit fixed-rank production rule for M+ response.

        The full-rank stationary fitted-state plan above never differentiates
        eigenvectors. Native truncated-rank execution instead binds this shared
        pseudoinverse Fréchet rule to the owner-validated metric eigensystem.
        """
        return SymmetricMatrixFunctionSpec(
            self.naux,
            "df-coulomb-metric-v1",
            relative_threshold,
            function="pseudoinverse",
        )
