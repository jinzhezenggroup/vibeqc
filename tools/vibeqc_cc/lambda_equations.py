"""Generated fixed-amplitude RCCSD Lambda equation actions (issue #152 A).

These are mathematical programs, not a response solve or a nuclear-force API.
They reuse the unpreconditioned #148 energy/residual DAG and #151 AD. No CC
iteration, DIIS update, denominator preconditioner or Jacobian is differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.tensor import (
    JVPProgram,
    Program,
    VJPProgram,
    linearize,
    transpose_program,
)

from .doubles import build_ccsd_program

AMPLITUDES = ("t1", "t2")
RESIDUALS = ("singles_residual", "doubles_residual")


@dataclass(frozen=True)
class CCSDLambdaPrograms:
    """Backend-independent amplitude derivatives at fixed Fock/MO integrals.

    Feed dense FP64 ``t1`` and pair-symmetric ``t2`` plus the same mathematical
    inputs as ``primal``. ``energy_vjp`` takes ``bar_correlation_energy=-1`` for
    the Lambda RHS. ``residual_vjp`` takes ``bar_singles_residual`` and
    ``bar_doubles_residual`` and returns the transpose action. Both return
    ``bar_t1``/``bar_t2``. ``residual_jvp`` takes ``d_t1``/``d_t2`` and returns
    ``d_singles_residual``/``d_doubles_residual``.

    All inner products are dense Frobenius products. The t2 adjoint is
    projected onto simultaneous (i,j,a,b)->(j,i,b,a) symmetry, not separate
    occupied/virtual antisymmetry. Independent packed callers must retain
    orbit multiplicities (or use sqrt-weighted Euclidean coordinates).

    The Lambda equation is J* lambda = -grad(E_corr), corresponding to
    L = E_corr + <lambda, R>. These multipliers are not PySCF Lambda arrays or
    physical RDMs without an explicit convention conversion.
    """

    primal: Program
    energy_vjp: VJPProgram
    residual_vjp: VJPProgram
    residual_jvp: JVPProgram

    def provenance(self) -> dict:
        """Record the physical-equation identity and every generated action."""
        return {
            "schema": "vibeqc.cc.lambda_equations",
            "schema_version": 1,
            "scope": "fixed-amplitude conventional real RCCSD; no response solve",
            "inner_product": "dense Frobenius; t2 simultaneous pair exchange",
            "lagrangian": "E_corr + <lambda, R>",
            "rhs_energy_cotangent": -1,
            "primal_logical_hash": self.primal.logical_hash,
            "energy_vjp": self.energy_vjp.provenance(),
            "residual_vjp": self.residual_vjp.provenance(),
            "residual_jvp": self.residual_jvp.provenance(),
        }


def build_lambda_programs(
    nocc: int, nvir: int, *, form: str = "shared"
) -> CCSDLambdaPrograms:
    """Generate energy RHS, J* and J actions from the existing RCCSD equations.

    The dense-symmetry AD boundary uses permutation/add primitives: no dense
    T2-by-T2 Jacobian or packed-incidence matrix is generated. The returned
    programs use the existing interpreter budgets and CPU/CUDA lowering;
    building a program alone does not establish device execution support.

    No convergence status is implied: arbitrary fixed amplitudes are useful
    for dot/finite-difference tests. A future solved-state consumer must bind
    the converged SCF/CC state and validate identity, residuals and resources
    before accepting a Lambda result.
    """
    primal = build_ccsd_program(nocc, nvir, form=form, diagnostics=False)
    return CCSDLambdaPrograms(
        primal,
        transpose_program(primal, ("correlation_energy",), inputs=AMPLITUDES),
        transpose_program(primal, RESIDUALS, inputs=AMPLITUDES),
        linearize(primal, AMPLITUDES, outputs=RESIDUALS),
    )
