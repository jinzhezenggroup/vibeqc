# Decision: exploit symmetry in occupied DF exchange and response

Status: implemented
Date: 2026-10-03

## Problem

At master `b9c626d15b2c19c21a848cb3625932d57f754a71`, the complete
96-atom/768-AO RHF DF energy-and-force warm call takes about 5.57 seconds on
an RTX 5090. Two occupied exchange Grams take about 0.96 seconds together;
force response also roots all `r*r` occupied pairs and computes both triangles
of their auxiliary metric Gram. Capacity alone does not make these operations
work efficient.

## Decision

Use compact symmetric response products and a bounded parallel Gram schedule:

1. For an exact physical packed final-K lease, retain symmetric occupied pairs
   through the response root and metric Gram. Diagonals contribute once,
   off-diagonals twice. Preserve the Coulomb adjoint with accumulating SYRK,
   then mirror the lower triangle. Restore full occupied staging only when the
   compact Gram has finished, through a disjoint dead buffer.
2. Lower large packed occupied K Grams to a generated 32-by-32 triangular tiled
   FP64 product, split into at most 64 deterministic reduction slices. Borrow
   only the existing exchange intermediate. The final U lease is untouched.
   A second kernel sums partials in fixed order and writes both triangles.

The compiler owns the schedule, pair map and emitted contractions. Native code
owns existing buffer lifetimes, capture-compatible launches, streams and error
propagation. No new permanent control, allocation, precision relaxation or
scientific acceptance rule is introduced.

## Invariants

- The response shortcut requires singleton RHF, the exact validated final
  determinant/owner generation, full-rank retained metric root and a physical
  packed AO source. Symmetry is never inferred from dimensions alone.
- The first `r` compact entries are diagonals. Averaging off-diagonal entries
  removes only FP64 reduction-order asymmetry of physical `C^T B C`.
- Dense/nonsymmetric, spectral, rank-truncated, unqualified final-state and UHF
  response paths keep their established algebra. Providers without SYRK use
  checked full GEMM products for the weighted metric Gram, with actual FLOPs
  reported. Discarded metric directions are never reconstructed from whitened
  factors.
- Gram admission requires `n >= 384`, `a*r >= 32768`, integer-safe dimensions
  and `ceil(a*r/32768)*n*n` doubles in the existing intermediate. All other
  shapes/layouts retain SYRK/GEMM. Native runtime failure is not a fallback.
- Generated tiles zero-pad both tails. Only lower partials are initialized;
  both output triangles read that same initialized lower sum. No atomics or
  data-dependent summation order are allowed.
- Work counters count actual padded tile arithmetic and reduction elements.
  Endpoint timing includes setup, SCF, strict final validation and full forces.

## Rejected alternatives

Changing packed B*C tile sizes did not improve its approximately 359 ms target
projection. Unpacking bounded panels for cuBLAS was slower, about 434–494 ms.
These routes are not promoted.

The [earlier split-Gram note](../../rejected/2026-09-17-split-occupied-gram.md)
rejected batched full GEMM: computing both triangles added FLOPs and changed
SCF iterations enough to lose the kernel saving. Repeating that provider probe
still gives only about 419 ms versus 478 ms SYRK. The new triangular kernel
instead measures about 189 ms, retains nearly triangular arithmetic, and was
qualified with a new complete-endpoint protocol. This is not a revival of the
rejected full-GEMM implementation.

A response-only candidate saved about 8.5% (5.557 to 5.085 seconds). Its component
win alone was not used to claim the final combined endpoint improvement.

## Evidence

The retained [qualification bundle](../../../../benchmarks/results/hf-df-96-20261003/README.md)
contains all complete timing samples, strict independent energy/force gates,
SCF iteration counts, work counters, build/source identities and reproduction
commands. Same-card n5 measurements put combined warm time at 4.488 seconds
versus 5.557–5.574 seconds baseline; moved warm is 4.500 versus 5.576 seconds.
Cold and moved reconvergence retain 24 and 13 iterations respectively; every
warm replay retains one iteration. No sample is filtered by work count.

Host tests independently check emitted pair/BLAS algebra against raw-energy
finite differences and occupied rotations. Capacity tests cross-check Python
and emitted C++ admission. GPU tests compare ragged Gram shapes, both occupation
weights and signed weights against full BLAS and scalar long-double entries;
CUDA graph replay replaces factors and must overwrite all partials. Molecular
checks cover independent PySCF forces, exact/corrected leases, spectral/dense
controls, RHF/UHF and geometry changes. Sanitizers and final campaign receipts
are retained in the bundle rather than copied as raw logs.

## Consequences and revisit conditions

The deterministic reduction order differs from SYRK, so future changes must
requalify actual cold/warm/moved SCF work and all independent force samples.
The shape gate describes measured large packed workloads, not a universal
speed guarantee. Broaden it only with complete endpoint and resource evidence.
Larger or memory-limited shapes must demonstrate explicit fallback or bounded
partial storage before any new performance claim.
