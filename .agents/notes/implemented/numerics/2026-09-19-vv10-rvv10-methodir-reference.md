# VV10/rVV10 MethodIR energy primitive and reference oracle

> The initial rVV10 kernel and its PySCF comparison below are superseded by
> [the revised-kernel correction](2026-09-19-rvv10-kernel-correction.md).
> They are retained as historical evidence of the wrong reparameterized-VV10
> path, not as current rVV10 qualification. Original VV10 is unchanged.

## Scope

Issue #491 slice A introduces a first-class nonlocal-correlation family without
claiming self-consistent potential, molecular gradients, response, or production
CUDA support. The public MethodIR representation fails closed for all later
capabilities: NonlocalCorrelationPrimitive.derivative_capabilities contains
only energy.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Scientific identity

The immutable shared definition records the VV10-family variant, exact rational
parameters, provenance, density/gradient convention, atomic units, quadrature
convention, positive-density reference domain, and full double-integral
one-half convention. Original audited parameterizations are:

- VV10: b = 5.9, C = 0.0093; Vydrov and Van Voorhis, JCP 133, 244103 (2010).
- rVV10: b = 6.3, C = 0.0093; Sabatini, Gorni, de Gironcoli, PRB 87, 041108(R) (2013).

The CPU oracle evaluates the finite-system real-space kernel in FP64:

- omega = sqrt(C |grad rho|^4 / rho^4 + 4 pi rho / 3)
- kappa = 1.5 b pi (rho / (9 pi))^(1/6)
- beta = (3 / b^2)^(3/4) / 32
- phi(i,j) = -3 / [2 g(i,j) g(j,i) (g(i,j) + g(j,i))]
- g(i,j) = omega_i |r_i-r_j|^2 + kappa_i
- epsilon_i = beta + 1/2 sum_j w_j rho_j phi(i,j)
- E_nlc = sum_i w_i rho_i epsilon_i

The oracle tiles the j dimension and never needs a full grid-pair matrix.
A matrix helper exists only for bounded small-grid symmetry diagnostics and
rejects requests beyond its explicit point limit.

## Ownership decision

An initial implementation placed NonlocalCorrelationSpec under method, but the
compiler structure gate correctly rejected the resulting dft -> method
dependency. The scientific parameter identity now lives in
vibeqc_compiler.common.nonlocal_correlation; both the DFT reference oracle and
the MethodIR primitive depend downward on that neutral definition. No structure
exception was added.

## Independent numerical evidence

A four-point fixed-grid fixture was evaluated independently with PySCF 2.14.0
pyscf.dft.numint._vv10nlc, supplied only in a temporary validation environment
and not imported by VibeQC production/compiler code.

| Variant | VibeQC oracle (Eh) | PySCF 2.14.0 (Eh) | abs diff |
| --- | ---: | ---: | ---: |
| VV10 | 0.0020004754428724408 | 0.0020004754428724408 | 0.0 |
| rVV10 | 0.0018147083462731767 | 0.0018147083462731767 | 0.0 |

The fixture also checks kernel symmetry, negative pair-kernel sign, grid
permutation invariance, tile-size invariance, the final weighted-density
contraction, and fail-closed behavior outside the reference domain.

## Verification

- ruff check on all changed Python files: pass.
- Compiler structure check: Checked 207 compiler modules; 0 dependency errors.
- Target pytest suite: 22 passed.
- Existing MethodSpec call sites were AST-scanned; none pass five or more
  positional arguments, and the new field was appended after the existing
  dispersion field to preserve current positional meaning.

## Explicit non-goals

This slice does not advertise a KS potential, analytic nuclear force, response
or Hessian-vector support, production screening, or native CUDA execution.
Those remain #491 B-D. Full omegaB97M-V composition remains gated by the
separate meta-GGA and range-separated-hybrid work in #164/#167.

An expanded regression invocation including KS/exact-exchange execution reached
54 passed and 8 skipped before 5 tests requiring a built native libvibeqc failed
at library discovery. No assertion in those tests was reached. The clean slice-A
worktree intentionally did not borrow an unrelated native shared library.
