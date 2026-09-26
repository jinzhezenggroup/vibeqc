# Decision: first-order symmetric matrix-function custom rule

Status: implemented (CPU reference and generated linear-response graph only)
Date: 2026-09-19

## Problem

Issue #466 needs a reusable inverse-square-root derivative for RI/DF and other
matrix-function consumers. The existing #293 metric response is method-local
validation code. Copying it into another method would multiply ownership, while
ordinary AD through eigenvector gauges fails at degeneracy. The original issue's
phrase "subspaces are frozen" is also ambiguous: fixed rank must retain the
response of the spectral projector, including retained/discarded cross terms.

## Decision

Put a typed, versioned custom rule under the method compiler, separate from the
DFT `MethodIR` energy-component union. Permit the one-way `method -> tensor`
compiler dependency so ordinary projection, elementwise weighting and
back-projection reuse TensorIR rather than a second contraction algebra.
No reverse dependency or compiler import of the public runtime is allowed.

Prepare spectral coefficients with a NumPy CPU reference. Use rationalized
retained/retained divided differences, exact cross-subspace divided differences,
and a scale-relative separation guard. Truncated PSD nullspaces are legal;
materially indefinite matrices and unresolved branches fail. A rebind preserves
rank while recomputing projectors. It does not freeze eigenvectors or assert
continuity along arbitrary finite geometry jumps.

The first-order map is self-adjoint for symmetric input matrices with the full
Frobenius metric. Packed metrics and higher derivative orders need distinct
rules. The serialized derivative graph treats spectral state as fixed
coefficients; differentiating that graph with respect to its seed is not a
matrix-function Hessian.

## Rejected alternatives

- Differentiating eigensolver iterations or eigenvector orientations: unnecessary
  gauge singularities and no scientific advantage.
- Dropping retained/discarded blocks: omits projector response even at fixed rank.
- Subtracting nearly equal inverse square roots: avoidable cancellation; use the
  rationalized formula with its repeated-eigenvalue limit.
- Registering this as a DFT energy component or silently promoting GPU support:
  this is a generic custom derivative contract, with CPU spectral preparation.
- Retiring the #293 oracle now: it remains a separate implementation for checks;
  production consumer integration has not been delivered in this slice.

## Invariants

Preserve strict FP64/symmetry checks, current-state identity, full-Frobenius
transpose factors, cutoff membership with projector response, and explicit
first-order/backend boundaries. Logical CPU byte admission is not native peak
memory evidence. Native eigensolver/workspace/stream ownership stays future work.

## Evidence

`tests/python/test_matrix_function.py` checks closed forms, an independent
Sylvester linear solve, three finite-difference steps, dot tests, gauge/rotation
and scale covariance, PSD nullspace cross response, rank changes and invalid
inputs, immutable snapshots, generated-program replay, #151 objective-VJP
composition, CUDA planning without a device, and the unchanged #293 oracle.
`test_compiler_structure.py` protects the new dependency direction and rejects
runtime/reference imports. No molecular calculation or GPU allocation is needed.

## Consequences and revisit conditions

This is a first compiler-level contract, not completion of #466/#193. Native
spectral-state integration, bounded CPU/CUDA execution, automatic graph dispatch,
and real public force evidence remain required. Revisit the boundary when #465
and #181 supply complete custom-rule composition or native backends supply
validated spectral-state/workspace contracts; do not add method-specific copies.

## References

- Issues #466, #181, #151, #193; existing #293 validation facade.
- `tools/vibeqc_mp2/gradient.py::_inverse_sqrt_metric_response` (independent oracle).
- N. J. Higham, "What Is a Frechet Derivative?", 2020:
  https://nhigham.com/2020/06/23/what-is-a-frechet-derivative/

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Subsequent native integration

The [native runtime/range decision](2026-09-20-native-matrix-function-weighted-range.md)
extends the original CPU-reference-only boundary. The original rationale above
is retained as historical scope, not a claim that native integration is still absent.
