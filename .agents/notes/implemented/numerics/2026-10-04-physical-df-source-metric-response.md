# Decision: retain the physical DF source frame for its spectral reverse map

Status: implemented
Date: 2026-10-04

## Problem

The streamed MO-source reverse map in #1796 differentiates independent A, C
and W. Molecular response must use the exact geometry, MO coefficients,
metric eigensystem and cutoff that produced the correlation factors. Rebuilding
a nominally equivalent source can change that frame or its retained subspace.
Compressed factor cotangents also need the forward pair projection's adjoint:
copying Bov into both cross sectors would double its derivative.

## Decision

`build_df_source_cuda(..., retain_response_state=true)` optionally retains an
opaque shared state owning its existing streamed J/K source plan and immutable
device coefficients. The coefficient allocation moves into this state; there
is no second upload. Ordinary energy calls retain their original destruction
policy. The source reports the retained numeric bytes, and its packing/block
phase admission includes coefficients that now survive the transform.

Each plan already has a process-unique factor-basis token. Because this owner
creates a new plan for each immutable C frame, that token binds both the source
and its orbitals. Propagate it through the source result, CC problem and factor
response. A molecular pullback rejects zero/mismatched tokens and missing state;
a generic supplied-factor pullback may still use token zero. This is an internal
provenance contract, not proof that arbitrary supplied seeds are converged.

Shared TensorIR AD of the actual pair projection/sector selection generates
`factor_embedding_vjp`, including half the compressed Bov cotangent in each
cross sector and symmetric Frobenius projection of Boo/Bvv. No primal factors
or four-index intermediates are required. Existing dynamic scatter lowering
and symbolic last-use arenas execute this map on CPU/CUDA.

`pullback_df_source_cuda` serializes calls sharing the owner. It binds both raw
source passes to that immutable source, invokes the #1796 BLAS traversal, and
applies the shared generated inverse-square-root VJP using the retained forward
eigensystem. The plan's unresolved-rank-crossing flag is checked first. Retained/
discarded subspace motion is preserved, including rank-deficient metrics; an
inverse-only/full-rank formula is not substituted.

The raw cotangent callback uses full, unit-weight [mu,nu,P] coordinates. Its
consumer contracts every pair once, without triangular doubling. The finish
callback receives bar_C and the symmetric full Frobenius bar_metric. Every
callback enqueues on the owner's stream and receives provisional results until
the entire call succeeds. Buffers drain before destruction on exceptions,
including callbacks that throw after queued host downloads. Reentry on the same
state from its callback is prohibited; ordinary repeated calls are supported.

## Work and storage

The embedding has 17 generated elementwise operations and an arena of
`Q (5 N^2 - o v)` doubles under the current shared liveness allocator. All
intermediates have rank at most three; it performs no contractions. The streamed
MO VJP reads `2 N^2 Q` raw values and emits `N^2 Q` cotangent values, performing
`6 N^3 Q + 2 N^2 Q^2` summands in `3 N + 5` BLAS calls. The shared spectral
adapter additionally performs four Q-by-Q-by-Q basis contractions and its
pointwise divided differences/symmetrization. No complete nuclear-force timing
or large-response performance qualification is claimed here.

Admission includes the retained source/metric/C owner, seed vector capacities,
uploads, embedding arena, all MO reverse scratch, two spectral scratch matrices
and bar_metric, plus finite flags. Borrowed C/W/BMO overlap is counted once when
calling the inner primitive. Callback destinations and other owners are the
caller's explicit additional charge. The raw source remains streamed rather
than retaining a complete raw-A tensor. Spectral outputs receive a finite audit
before successful publication. This adapter does not retain a new eigensystem
or recompute a metric factorization.

## Evidence

Independent committed PySCF AO/metric data cover H2, water, LiH and an f-shell
system, each with and without a duplicated auxiliary s shell. Arbitrary seeds
exercise all sector weights. Dense independent contractions check bar_A/bar_C
at `atol=rtol=3e-10`. Fourth-order directional finite differences check A/C and
metric eigenvalue/subspace motion at `atol=rtol=2e-8`. Isospectral rotations move
the null and retained spaces without raising a discarded eigenvalue through
the cutoff, testing the full fixed-rank rule rather than only its retained block.

A separate cold native source -> CCSD -> Lambda -> retained-factor -> physical
source test uses the duplicated H2 auxiliary basis. Simultaneous A/C/metric
response matches independent two-electron determinant-energy differences at
`atol=3e-9, rtol=3e-8`. Fock is held fixed in this gate: it isolates correlation
source derivatives and does not claim orbital stationarity or nuclear forces.

RTX 5090 validation under finite local Slurm allocations passed 62 tests across
this module, upstream factor/source cases (including auxiliary g values), and
conventional public CCSD/CCSD(T) force regressions. The complete 23-case new
module passed CUDA memcheck with zero errors. Failure tests cover absent state,
another equal-shape MO frame, zero token, malformed/nonfinite seeds, cutoff
crossings, one-byte-short/exact budgets, both callback failure points and reuse
after a queued-download exception. The original host C/result are destroyed
before response, and repeated response is checked against the first traversal.

The complete rebuilt library SHA256 is
`3920d7dd1815f498c01d9ea1b699c47cbfbf077dc6980118d1c40e5d12a28bb7`.
Builds used explicit C++/CUDA ccache launchers and the shared `/data/jzzeng/ccache`.
Logs remain ignored under `.artifacts/metric-response/`; no tracked benchmark
evidence payload was added. Reproduce using
`tests/python/test_df_source_metric_response.py` with
`GENERATIVEQC_DF_SOURCE_RESPONSE_CUDA_TEST=1` and `GENERATIVEQC_LIBRARY` pointing
to the complete rebuilt CUDA library, inside the required Slurm allocation.

## Remaining force boundary and revisiting

This internal source response is not a public DF force endpoint. Native DF
triples response/corrected Lambda, conventional-reference orbital/Z and Pulay
response, contracted nuclear derivatives (including auxiliary g), and complete
hundreds-AO force timing/accuracy qualification remain necessary. Do not change
the conventional-RHF/correlation-only DF Hamiltonian to DF-RHF implicitly.

Future compiler elementwise fusion can reduce the embedding arena/launch count;
it must preserve the same independent-sector metric and full source projection.
The shared spectral basis contractions can separately gain a bounded BLAS
lowering without changing the custom rule or admitting unresolved rank changes.

See [the streamed reverse map decision](2026-10-04-streamed-df-mo-source-response.md)
and [the retained-factor decision](2026-10-04-df-retained-factor-response.md).
