# Decision: stage the shared DF orbital source before whitening

Status: implemented
Date: 2026-10-03

## Problem

`DensityFittedBlockProvider` expanded both orbital reductions inside every MO
pair, requiring N^4 Q scalar contraction summands before metric whitening.
This CPU RI-MP2 response owner is also a precedent for the native molecular
DF-CC source: carrying the expanded transform forward would replace the
removed CC auxiliary work with another avoidable source bottleneck.

## Decision

Define two orbital projections and metric whitening in shared post-HF TensorIR.
Derive packed matrix dimensions, transposes, operand order and work queries
from its binary GEMM contracts. Fix only the leading free AO index for source
row traversal; do not remove a reduction. Generate this traversal through CMake
for native consumers, with the generator and algebra in source identity.

The existing CPU provider executes the traversal using its CPU linear algebra
backend. The retained whitened array temporarily holds the first projection;
that projection is dead before whitening overwrites it. The second array keeps
the unwhitened MO source required by response. No additional full tensor,
four-index intermediate, resource-policy change or precision change is needed.

## Invariants

- Source and output pairs are full, unit-weight, auxiliary-contiguous arrays.
  Packed-triangle producers must satisfy this contract before execution.
- Coefficients and inverse root are row-major. The callback contract is packed
  column-major GEMM with alpha=1, beta=0; the CPU adapter reverses the product.
- The supplied reference/Hamiltonian is unchanged. Metric eigensolver, cutoff,
  rank branch and native finite-state publication gates retain ownership.
- Native owners preflight allocation, keep buffers disjoint, and discard partial
  state on failure. The CPU constructor publishes only after finite validation.
- A future asynchronous owner must order row reuse on its stream and drain all
  dependent work before publication. No asynchronous overlap is implemented here.

## Evidence

Independent direct three-/four-operand NumPy contractions qualify every A/B
entry at absolute tolerance 3e-12 (relative 2e-14), including non-symmetric raw
and metric inputs that expose hidden transposition mistakes. A standalone probe
links the real scalar CPU GEMM implementation and checks actual source-row,
GEMM and summand counts, zero/overflow rejection, and source-failure propagation.
Generation is deterministic and runs under Python `-S` without the runtime.

The CMake-built native MP2 gradient suite passes its independent RI reconstruction,
orbital-response and relaxed-weight gates. Public CPU RI-MP2 tests pass four
fixture energy/component gates and H2/water analytic-force finite differences
(2e-6 Eh/Bohr), including force translation invariance. These tests exercise the
changed production provider; no CUDA molecular-source endpoint is claimed.

The complete transform (including whitening) changes from N^4 Q + N^2 Q^2 to
2 N^3 Q + N^2 Q^2 contraction summands:

| N/Q | Old summands | New summands | Retained A/B bytes, unchanged |
| --- | ---: | ---: | ---: |
| 230/488 | 1,378,221,897,600 | 24,472,809,600 | 413,043,200 |
| 264/666 | 3,266,030,668,032 | 55,422,537,984 | 742,680,576 |

These are dense algebraic source-stage work counts, not complete CCSD(T)/force
timings or claims of formal CCSD scaling reduction. The old scalar path skipped
exactly zero orbital-coefficient products; that data-dependent shortcut is not
subtracted from the model. Raw integral generation and metric
factorization are separate work and memory consumers.

## Rejected alternatives and consequences

Keeping the expanded orbital loops behind a memory budget bounds storage but
retains repeated work. Adding a third full-sized intermediate needlessly raises
peak memory. Whitening each AO row before all orbital projections is not needed
for this consumer and does not preserve its existing two-array response state
as directly. No mixed-precision or double-buffer default is justified by this
change alone.

## Revisit when

A native CUDA molecular CC source can consume the same generated traversal with
streamed integral rows and the shared cuSOLVER metric owner. Qualify its full
source/solve endpoint, ownership, memory admission and fallback independently;
consider #1763 fusion, #1764 precision and #1765 overlap only with measured
component and complete-endpoint benefit.

## References

- Issues #1763, #1764, #1765.
- PR #1769: sum auxiliary intermediates before expensive CC T2 consumers.
- `tests/python/test_df_cc_source_program.py`
- `tests/native/test_mp2_gradient.cpp`
- `tests/python/test_mp2_public.py`
