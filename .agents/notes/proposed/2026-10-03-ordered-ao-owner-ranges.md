# Proposal: bound cooperative geometry's AO-owner reduction searches

Status: qualified prototype; small warm-only gain, dependency/review pending
Date: 2026-10-03

## Problem

The shared AO point panel evaluates each spin/AO pullback once, but its atom
writers subsequently scan every selected AO to find their contributions. This
performs `G * A * M` owner-label probes, with grid points `G`, atoms `A` and
active AOs `M`. Resident local AO maps reduce `M`, not the repeated atom scan.
The original [point-panel note](../implemented/performance/2026-10-03-cooperative-ao-point-panel.md)
explicitly identified that metadata scan as an unresolved possible cost.

## Decision and invariants

During the existing AO validation/pullback traversal, check whether selected
AO **owner labels**, rather than global AO IDs, are nondecreasing. A monotone
sequence permits one lower-bound search per atom followed by only that atom's
contiguous AO interval. The resulting reduction uses at most
`A * (floor(log2(M)) + 1) + M + A` owner-label probes per point for nonempty
maps. The preceding validation adds `M - 1` predecessor-label checks. Arbitrary
nonmonotone ownership retains the original complete scan.

Each atom retains one writer, and additions retain their original selected-AO
order. Thread zero's ordered moving-grid accumulation is unchanged. Neither
floating-point atomics nor new scientific algebra is introduced. Permuting AO
IDs within an atom does not disable the fast path. Missing atoms, sparse active
subsets, repeated owner labels and empty maps remain valid.

The existing shared `collective_valid` word encodes validity and monotonicity:
initialize to three, atomically clear bit one for an owner inversion, and
atomically write zero for invalid input. No inversion vote can resurrect an
invalid-input vote. All threads consume the result after the existing barrier.
The valid post-vote states are one or three, both compatible with the later
Becke team's nonzero-valid contract. A predecessor AO ID is checked before its
owner label is read. The existing setup flag is not reused for this vote:
doing so without another barrier could race another thread's setup read.

The shared control remains 16 bytes. No ABI, retained metadata, resource
allowance, geometry lane count or tile size changes. Insufficient shared AO
panel space retains the scalar AO fallback. The point-panel alias barrier and
all Becke point/pair work remain unchanged; this does not remove `G * A^2`
partition work or solve the complete-endpoint reference gap by itself.

## Evidence so far

The actual generated kernel is executed by the existing C++ thread harness
with generated Becke math. Expanded cases include grouped owners at 96/768
and 128/1024 atoms/AOs, holes, one owner, within-owner AO permutations, sparse
global IDs, empty maps, arbitrary reversed maps, shared-panel fallback, tails,
external seeds and late invalid AO IDs/owners. AO and moving-grid channels
remain bitwise equal to the scalar implementation under the harness's explicit
noncontracted arithmetic. Guard bytes and sticky failure behavior are checked.

The focused host/ABI/qualification suite passes 29 tests. A subsequent
18-test kernel run instruments **actual reduction owner-label reads**:

| Atoms / active AOs / points | Grouped owner probes | Previous full scan |
| --- | ---: | ---: |
| 12 / 96 / 17 | 3,196 | 19,584 |
| 96 / 768 / 5 | 8,960 | 368,640 |
| 128 / 1,024 / 5 | 12,160 | 655,360 |

Nonmonotone cases retain exactly `G * A * M` reduction probes, empty maps
perform zero, and scalar-panel fallback performs none in this cooperative
reduction. These are host semantic counts, not GPU timings, not AO evaluation
counts, and not total geometry operation counts.

Native production build completed on n5 with explicit CXX/CUDA ccache launchers
and checkout-root `CCACHE_BASEDIR`. Ccache statistics are shared/cumulative,
not an isolated build hit-rate claim. The matching library is transferred
directly n5 to n1. Candidate identities:

- Source: `0beb3d81d31dfefe257b6baa4c8b5d2d21d3b2df1e96ef37f11915d17a6207f3`.
- Library: `c071235850716c36fc0b1f682a8237967a00bf4ef715d8e1e19eb66177c1f462`.

Real-device independent force oracles, schedule comparisons, sanitizers and
matched complete endpoints must finish before a performance or readiness claim.
Local ignored receipts are under `.artifacts/ordered-owner-host.log`,
`.artifacts/build-candidate-n5-retry.log` and `.artifacts/qualify-owner-n1.log`.

First device job 5579 passes 14 checks, including monotone subset/empty-map
geometry and 96/128-atom schedules, but fails five checks. Four fail before
their force oracle because the deployment contains only the main library,
not the packaged UKS/r2SCAN stationary artifacts. The fifth tries a reversed
active-ID map that the public grid producer intentionally rejects as unsorted.
The corrected test explicitly verifies that refusal; arbitrary owner order is
still exercised directly by the generated-kernel host harness. The missing
artifacts are built on n5 and deployed into a separate qualification directory,
not added to the library directory of an already running endpoint campaign.
The original failed log is retained, and these failures are not counted as
passes or used to weaken any numerical gate.

After that deployment correction, job 5589 passes all four formerly missing
UKS/r2SCAN independent analytic oracles. Job 5588 passes six real-device routing
tests and eight complete PBE0 RKS/UKS FP64/AUTO/alias tests with their independent
analytic and reconverged finite-difference checks. The routing matrix includes
synthetic reversed **owner labels** installed before native topology creation:
this exercises the actual nonmonotone fallback without bypassing the public
grid's sorted-ID requirement. It proves schedule equivalence, not physical
forces for that synthetic labeling. Four sanitizer modes are still running.

Paired same-allocation job 5581 completes all 24 original/moved endpoints at
24 atoms, both using local force AO maps, dense SCF and default tile/budgets.
The independent verifier passes every same-geometry oracle-repeat comparison,
checks equal resources and confirms all map-work counts except elapsed
discovery time are identical. Warm median is 4.734867→4.685815 seconds (1.04%);
the five-sample ranges are 4.718856–4.745606 and 4.677431–4.706459 seconds.
This small result is not a broad speedup claim. Cold compiler caches are shared;
the 96-atom paired campaign remains running in job 5582. Receipts are in
`.artifacts/owner-24-summary.json` and `.artifacts/owner-fetched/`.

## Rejected alternatives

- Assuming sorted AO IDs: sorted IDs need not imply sorted atom ownership.
- Sorting arbitrary maps or atomically reducing: changes floating-point order.
- An atom/AO CSR cache: adds storage and lifetime/budget obligations for a
  range query that can be proved directly from the current validated map.
- Larger tiles alone: does not remove the repeated atom/AO scan.

## Final qualification and priority

The pending work described above completes in jobs 5588/5582. All four sanitizer
modes pass six routing checks each with zero errors/hazards. The strengthened
host harness checks every owner read and includes an invalid predecessor ID;
removing that bounds guard is rejected by an abort, not just a source-string
check. The 18 host kernel cases pass again.

Both sizes now have complete paired endpoint evidence: 48 native calls pass
every same-geometry reference-repeat comparison. At 96 atoms warm median is
64.503557→63.449513 seconds (1.63%), but moved grows 253.660078→268.924869
seconds with 12→13 SCF iterations. Cold changes 750.176532→584.307066 seconds
with 29→27 iterations and shared compiler caches; no causal cold win follows.
All phase arrays, bounds, work counts, source/library identities and original
qualification failures are losslessly retained with a standalone verifier in
`benchmarks/results/pbe0-ordered-ao-owners-20261003/`.

The user asks to prioritize large hotspots. Stop extending this metadata-search
slice: its measured gain is only 1–1.6%. Reprofile the current complete 96-atom
endpoint before pursuing more full-range derivative or Becke work. An earlier
real-grid stratified sample in the separate Becke-zero investigation found
only 2.21% of reverse pairs removable by the conservative two-zero rule at
96 atoms (0.041% at 24 atoms). That is not a promising large-hotspot fix;
do not repeat the prototype merely because the local zero-JVP proof is simple.

## Revisit when

Complete-endpoint evidence shows the extra monotonicity checks or searches
outweigh the savings. Retain the bounded arbitrary-map and shared-capacity
fallbacks unless an independently enforced producer contract replaces them.
