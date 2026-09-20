# Decision: native matrix-function responses scale the seed with the spectral rule

Status: implemented
Date: 2026-09-20

## Boundary

This extends the [original compiler/CPU-reference decision](2026-09-19-symmetric-matrix-function-rule.md)
with the shared native CPU response and runtime-sized, compiler-emitted CUDA
inverse-square-root and pseudoinverse VJPs. Native callers retain eigensystem,
rank/threshold validation, workspace and stream ownership. The old pseudoinverse
launcher remains an ABI-compatible forwarding entry point. This is not a public
RI-MP2 force or automatic higher-order AD capability; the independent #293 oracle
remains available, and the precomputed TensorIR reference has its original domain.

## Range defect and repair

A representable final VJP does not imply its isolated divided-difference factor
is representable. The former coefficient-first native and CUDA paths could
return zero or nonfinite results for finite power-of-two eigenvalues and seeds.
For inverse square root, lambda=2^1000 and seed=2^800 require -2^-701, not zero;
lambda=2^-700 and seed=2^-500 require -2^549, not a nonfinite failure.
The corresponding pseudoinverse examples have the same issue.

Keep the ordinary coefficient-first arithmetic when that coefficient is normal.
Otherwise divide normalized mantissas and combine their binary exponents before
one final scaling. Include retained/discarded projector terms through the same
weighted quotient; do not freeze the projector or clip spectral values.

## Invariants and alternatives

No threshold, rank-crossing rule, eigenvector layout or full-Frobenius convention
changes. CPU and generated CUDA retain their established row/column layouts.
Do not simply replace an underflowed coefficient with zero, relax acceptance or
use wider host arithmetic as an implicit CUDA fallback. This local range repair
does not claim arbitrary-precision sums: the surrounding matrix products still
require representable FP64 intermediates and are checked by their owners.

## Evidence

Eight independent exact-dyadic scalar/projector cases fail against the original
native source. They cover both matrix functions, coefficient underflow and
overflow, and retained/discarded coupling. The native regression compiles the
actual response source; the CUDA regression launches both generated functions.
Existing finite-difference, Sylvester, repeated-eigenvalue, rank-crossing and
FP64 subnormal checks remain unchanged. Execution counts and device/source
bindings are recorded in the PR review; this note does not turn an unexecuted
optional device test into a pass or invent a performance measurement.

## Revisit when

Extend matrix-level range admission or replace the reference coefficient array
only with independent whole-operation tests and explicit resource contracts.
Complete public RI-force integration and full device ownership remain separate.

Refs #466, #181, #193; PR #657.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Integration with concurrent repair

A parallel native/CUDA repair reached master while this PR was under review.
The final integration reuses that canonical seeded-exponent implementation,
rather than retaining two equivalent helpers. All eight independent dyadic
scalar/projector cases added here remain regression gates on the combined tree.
Neither discarded-mode motion nor ordinary-scale arithmetic is removed.
