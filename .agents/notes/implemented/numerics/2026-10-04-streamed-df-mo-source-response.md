# Decision: stream the staged DF MO-source reverse map through BLAS

Status: implemented
Date: 2026-10-04

## Problem

The retained-factor response in #1795 supplies complete fixed-Fock CCSD
factor cotangents, but a force chain also needs the derivative of
`B[p,q,Q] = C[mu,p] C[nu,q] A[mu,nu,P] W[P,Q]`. Expanding both orbital
contractions repeats N^4 Q work. Differentiating only one appearance of C,
assuming symmetric inputs during lowering, or discarding metric-root response
would omit independent derivative terms.

## Decision

Apply shared TensorIR AD to the existing staged `df_mo_source` program.
Lower the resulting dependency graph to packed BLAS calls, deriving operand
orders, transposes, dimensions and row reductions from its GEMM contracts.
Reject changed unsupported dependency topologies rather than infer scientific
equations from coincident AO/MO shapes.

The generated traversal reads the same immutable raw source twice. First it
forms the shared forward projection and transformed A, then computes bar_W.
It reuses transformed-A storage for its cotangent, consumes the first C
derivative, and reuses the first projection for its cotangent. The second
pass accumulates the other C derivative while emitting one raw-A cotangent
row at a time. This needs two N*N*Q scratch tensors and two N*Q row buffers,
plus N*N and Q*Q small output arrays. It retains no complete raw-A copy.

Exact semantic work is `6 N^3 Q + 2 N^2 Q^2` summands in `3 N + 5` GEMMs;
the source supplies `2 N^2 Q` values and the raw cotangent consumer receives
`N^2 Q` values. Read/consume callbacks preserve every row and reduction.
The beta=1 GEMM realizes the AD's second coefficient term, not a handwritten
alternative derivative. No N^4 tensor or numerical Jacobian is produced.

## CUDA ownership and boundaries

`posthf::pullback_df_mo_source_cuda` borrows immutable device C/W/bar_B and
the caller's stream/BLAS handle, checks stream and scalar-pointer ownership,
admits complete local scratch plus borrowed inputs/caller storage, and uses
one sticky finite audit across every GEMM. Read, consume and finish callbacks
enqueue only on that stream. Their outputs stay provisional until the call
passes its complete error audit. Owned buffers drain on every exception,
including callbacks that throw after queuing downloads.

This primitive differentiates independent full, unit-weight A and W
coordinates. A molecular owner must bind the immutable native integral source
and physical frame, map the pair-projected factor cotangent to full B, and
apply the shared fixed-rank spectral rule to bar_W. It must reject unresolved
rank crossings and retain discarded-subspace motion. Those operations, DF
triples response, conventional-reference orbital/Pulay response and nuclear
contractions remain necessary before public DF forces can be enabled.

## Validation

Tests use nonsymmetric A/W/seeds as well as physical pair symmetry to expose
transposition errors. Shared AD and native BLAS execution are compared with
independent complete four-operand expressions at `atol=3e-11, rtol=3e-13`.
Simultaneous A/C/W directional differences exercise both occurrences of C.
Large shape queries cover N=230/Q=488 and N=264/Q=666 without allocating
their arrays. Runtime tests cover exact and one-byte-short budgets, stream
and pointer-mode mismatches, both source passes, queued-download exceptions
and earlier overflow followed by a zero seed.

Local host qualification passed 49 source/sector cases (14 real-device cases
skip outside an explicit allocation). On n1's Slurm RTX 5090, all 14 native
numeric/resource/failure scenarios passed against independently prepared dense
references; the largest absolute difference was `7.105427357601002e-14`.
The same scenarios linked against the complete rebuilt library passed CUDA
memcheck with zero errors. Its SHA256 is
`54d0a85b22f663a7aad047712f23e76d9a254f2e8b178a81bf18336b72aac41b`.
Compiler/ownership hooks and focused type checks passed. Detailed logs and
the copied-node validation artifacts are ignored under `.artifacts/source-response/`;
no new tracked benchmark evidence payload is added.

## Rejected alternatives and revisiting

Do not hand-code a second derivative algebra in the CUDA owner, retain an
unbounded raw-A source, or expand the orbital chain to N^4 Q. A future owner
may optionally cache raw rows only when it charges their complete lifetime
and retains this two-pass bounded fallback. Dense raw cotangent publication
is a validation choice; production callbacks can contract and release rows.

See `tests/python/test_df_cc_source_program.py` for reproducible gates and
[the factor response decision](2026-10-04-df-retained-factor-response.md)
for upstream cotangent conventions.
