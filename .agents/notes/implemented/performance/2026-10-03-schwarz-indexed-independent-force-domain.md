# Decision: enumerate the Schwarz-live independent force domain

Status: implemented, experimental opt-in; not a default promotion
Date: 2026-10-03

## Corrected-binary qualification and historical retention

This section supersedes earlier pending qualification statements, not their
historical measurements. PR1767's reviewed shared-claim repair is compiled as
source `ffc146230c1b44103fba01a8b14a061deadbb29d17e5160c79d198fca6e7b10b`,
library `c43436f7e39901e01434e39d46e5a81e4d71ca3af3489c1d92a0791fe681d280`.
The unconditional CTA barrier before each leader claim write protects prior
readers across empty pages and inactive/screened skips. The old tail barrier
or caching a first read cannot establish that invariant.

Native job5550 passes the four extracted-protocol host/plain/synccheck/racecheck
gates, baseline/OFF/ON through-f independent CPU oracles, batch/exact-prefix
budgets, and memcheck/initcheck (zero errors). Fresh corrected-binary jobs5552
and5551 complete all144 native endpoints; both modes share this source/binary
and allocation. Independent all-reference-repeat comparisons pass the unchanged
`1e-8 Eh` / `1e-7 Eh/Bohr` gates (maxima`1.060e-10` / `3.623e-11`).

Corrected96 warm is76.949750→63.354224s (17.67% less), moved-warm
76.872031→63.543085s. Negative corrected48 moved is70.550929→76.673851s
(12→14iterations). Every cold/moved and variable-iteration observation remains;
ordered compilation/artifact caches preclude isolated cold-speedup claims.
The schedule remains default OFF and the larger reference gap is unresolved.

Current acceptance is the compact lossless corrected campaign in
`benchmarks/results/pbe0-screened-pages-20261003/`. To avoid keeping superseded
raw campaigns alongside corrected acceptance under a nearly full aggregate
storage cap, pre-fix/coarse bytes remain in already-published immutable Git
commit `0b99c6ce298f2726373f1909ad10a37b5acffc43` at the same paths. The new
bundle preserves the original README, verifier and every old member hash;
ignored local copies are also retained. No external publication or Release,
aggregate-cap increase, unrelated-evidence deletion, or relabeling of old
timings is used. The earlier source/endpoint sections below are historical.

The later merge of master `86c422bdc` preserves both the indexed-budget fixture
and master's new `eri_tiles_only` native-test argument. It is a new checkout
identity, not a new measurement of the corrected campaign. Combined storage
with the tile-planner PR then exceeded the unchanged 64 MiB cap by 23,981 bytes.
Eleven earlier reports owned by this PBE0 task were therefore losslessly gzipped
(725,408 to 110,869 payload bytes). Exact decompressed hashes, original Git
revision and all complete/control/failed outcomes remain in the storage manifest
and regression test; unrelated evidence and the main README are untouched.

## Problem

The independent J/K fallback groups shell pairs into 32-entry blocks and uses
the maximum Schwarz bound to reject whole block products. Its prepared owner
nevertheless initializes the block order with identity, mixing large and small
bounds. It also enumerates every block product even when its first geometric
gate rejects nearly all of them. A bounded queue alone does not bound the work.

On unmodified master `59eee77f45468c5bee79fb077ba2082a46436c5b`, an actual
96-atom README def2-SVP provider has 73,920 shell pairs and 2,310 blocks. A
diagnostic exports the native bounds, populates its current density bounds
through the actual generated-Coulomb enqueue, and counts the existing block
predicate without evaluating new reference ERIs. It uses the earlier retained
converged native density; this is a workload census, not a new SCF endpoint.

| Work capacity | Identity order | Descending per-system order |
| --- | ---: | ---: |
| Complete block triangle | 2,669,205 | 2,669,205 |
| Surviving block products | 832,515 | 139,884 |
| Candidates expanded before exact shell screening | 852,014,240 | 143,025,952 |

The individual-pair upper-bound predicate admits 142,757,104 pairs. Sorting
therefore eliminates most block-bound inflation without changing a cutoff.
These counts are neither surviving exact shell quartets nor primitive/root
evaluations, and the 83.2% candidate reduction is NOT an endpoint speedup.

## Candidate

Diagnostic `GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=1` enables per-system descending
Schwarz order, with physical-pair ID tie breaking. Preparation reuses the
generated J owner's existing bounds download. It adds no force-time download.

Each sorted block row has one geometry-admitted ket prefix. Retain exclusive
row counts, not block-pair records, and binary-search the row for each claimed
live block product. The density-dependent block gate and exact shell/AO force
gates remain downstream and unchanged. Preserve the original physical pair
orientation when constructing each queue task; sorting is not a new symmetry
or density-contraction convention.

Preparation takes O(P log P + B log B), with P shell pairs and B blocks. The
compressed outer scheduler takes O(S log B) for S geometry-live block products,
rather than O(B squared) unconditional block visits. Worst-case dense work is
still quadratic in shell pairs; this is output-sensitive screening, not a
universal linear-scaling ERI claim. The geometry-only prefix may admit more
blocks than the density-weighted counts above; do not relabel those counts as
the actual prefix length.

## Storage and fallback

The additional device index contains B+1 uint64 words, charged to the retained
owner and its admission bound (18,488 bytes for the 96-atom example). Its host
staging vectors are charged to preparation and outlive the stream-draining
owner on failure. The per-system pair permutation reuses the existing device
order allocation. A tight lease keeps sorted blocks with the ordinary triangle
instead of discarding an otherwise admitted generated owner. Pure J and the
default route remain unchanged; no full quartet list or new oracle dependency
is introduced.

## Initial coarse evidence

The host test compiles the actual planner and device row/system decoder, comparing
their physical pair set with exhaustive enumeration on 480 randomized domains
plus empty systems/rows, exact block boundaries, ties, zero screening, threshold
neighbors, overflow/underflow and invalid bounds.

The candidate on base `59eee77f` has source identity
`5f4f839c3ba6cd6282eb1bbfe9b47fb43c5ab58d2f383db63960f379ed935fc3`
and library SHA256
`520282d2834d98b3dd17335c90b4c73812084615bda2929ec919ae5ef32dc7a9`.
n1 job5514 passes the baseline and candidate OFF/ON native through-f CPU-ERI
source gates, then candidate ON memcheck/initcheck with zero reported errors.
Job5516 independently compares a two-system multirow fixture against CPU ERI
derivatives, including zero and nonzero screening. Exact owner budget,
one byte less, and the budget without prefix storage respectively select
indexed, sorted-triangular, sorted-triangular. Every source agrees within
4.5e-16; both sanitizers report zero errors, including the budget fallback.

Job5515 brackets candidate OFF/ON with baseline-before/after on one allocated
GPU. Every one of the 24 fixed-density source arrays passes the 3e-10 gate;
the maximum all-pair difference is 1.15e-11. Actual cursor claims match the
domain exactly. At96, visits drop from 2,669,205 to139,884 (94.8%); the retained
prefix adds exactly18,488 bytes. Pooled baseline warm source time is27.0146s,
versus16.1588s indexed (40.2% reduction). These are NOT SCF/force endpoint timings.

The24 identity-density diagnostic regresses from0.04350s to0.29506s despite
reducing block visits from10,731 to7,436. Reduced candidate work alone therefore
does not qualify a default. The initial hypothesis was concentrated surviving
work within coarse blocks or altered class locality; the later census below
confirms load concentration. Complete cold/warm/changed-geometry endpoints and
investigation of any production regression remain required. Do not hide this
counterexample with an arbitrary molecule-size threshold.

The initial diagnostic job5508 did not populate the generated density bound:
the generic host J/K entry is not that resident seam. It failed explicitly and
was corrected to call the actual generated-Coulomb enqueue. n1 job5509 then
completed the census using finite Slurm; n5 job1411 was canceled while pending.
The baseline library hash is
`5689ed28152f62857923eb82cf63a207a3cdb3c700c775d4da046666b1c0928a`.
Exact inputs, compiler/cache receipts and counts are retained under ignored
`.artifacts/` in `qc-pbe0-screened-blocks-20261003`. No PR is submitted yet.

## Rejected alternatives

- Enlarging a bounded queue leaves both sources of redundant enumeration.
- Materializing all surviving quartets loses the existing memory contract.
- A density-dependent persistent prefix would need rebuilding on every density
  and a new invalidation contract; the geometry-only index avoids that hazard.
- Replacing screening with an approximate AO mask or reusing old-geometry
  metadata changes the scientific domain and is outside this exact-schedule fix.

## Follow-up: restore scheduling parallelism

The 24-atom identity-density census now confirms equal exact shell work but
different load distribution. Both orders admit 7,904 shell quartets, including
2,040 high-order quartets. Identity distributes the latter over 1,154 nonempty
block products (maximum 25 per block); Schwarz order puts them into only 88
(maximum 384). The high-order AO candidate capacity remains 127,236, while its
maximum per block rises from 3,165 to 26,124. These are exact shell-gate counts
and AO capacities, not executed primitive/root counters.

A separate candidate on master `a20b8801f` keeps the same geometry prefix and
splits each indexed block product into 16 independent 64-candidate pages.
Workers no longer serially own all 1,024 candidates in a dense sorted product.
Scientific predicates, physical orientation and prefix storage are unchanged.
Unindexed / insufficient-prefix-budget owners retain the ordinary traversal.
Empty diagonal/tail pages are explicitly allowed and tested; no quartet list,
density-dependent retained state, new cutoff, or molecule-size heuristic appears.

Work accounting must now distinguish products from scheduling claims: an indexed
source claims 16 times the prefix product count plus the worker termination
claims. At96 that is 2,238,144 page claims, NOT 139,884 claims; the block domain
still has 139,884 products and the same 143,025,952 candidate capacity. At24
page claims exceed the original triangular block count. The expected benefit
is balancing the unchanged surviving scientific work after removing candidate
amplification, not treating more claims as less work.

The page candidate remains default OFF. Its completed source/resource and
endpoint gates are recorded below. Its measured source identity is
`d062e73d8595e1f753e16d543c8fe951fa7d38fed7c883c47978eb2ec7c2dfbb`,
library SHA256
`ae00a76b40a2a19e72b4f3bc7d2da63c1b8fc65f400a9ea97ad0f0f2ac8c70c2`.
The matched clean a20 baseline has identity
`f46f35ea7d273ffe1b534e3e93082c28c6ef6508a0a89e6702575f9c75fb4046`
and library SHA256
`bd00a793de8809a136d1f15e692db029308dc25bcd5c8a0d0e1a76a241caa4fb`.

Job5529 passes baseline/OFF/ON native through-f and independent CPU source,
batch and exact prefix-budget gates; both candidate sanitizers report zero
errors. The page partition host test compiles the actual production claim/page
arithmetic for zero, diagonal and non-multiple tails. Focused host selection:
141 passed. Job5530 validates all24 source arrays and actual cursor/page claims;
maximum all-pair source difference is1.13e-11.96 warm source time is27.23250s
pooled baseline versus13.59034s indexed pages (50.10% less).24 identity-density
time is0.04329s versus0.07288s: much less load imbalance than the coarse candidate,
but still a counterexample to universal speedup. No complete endpoint or default
claim follows from these diagnostic times.

## Coarse candidate complete endpoint control

Jobs5522/5523 completed all144 native endpoints (12 per variant and size),
independently checked against retained GPU4PySCF references with the existing
1e-8 Eh / 1e-7 Eh/Bohr gates. Each baseline/candidate pair used the same Slurm
allocation and device, matching source/library identities and reference bytes.
Complete warm medians in seconds:

| Atoms | Baseline59 | Coarse candidate |
| ---: | ---: | ---: |
| 3 | 0.344457 | 0.362197 |
| 6 | 0.740636 | 0.726254 |
| 12 | 1.660586 | 1.577039 |
| 24 | 5.019934 | 5.057231 |
| 48 | 17.904941 | 15.632263 |
| 96 | 76.526261 | 66.108613 |

The 96-atom complete warm reduction is13.61%, not the source-only40.2%.
At24 the artificial identity-density regression does not translate
proportionally to the converged physical endpoint. The small physical
regression and measured load concentration motivate paging rather than a
molecule-size cutoff. These coarse measurements are not evidence for the
separate paged binary. Cold includes required JIT/cache setup, without purging
caches, and is not an isolated scheduling speedup.

## Final integration boundary

The PR tree is based on master `e47058a51`, whose changes after measured
`a20b8801f` affect unrelated xTB comparison and accepted CC evidence storage.
Clang-format23.1.1 changes only whitespace in the paged production/test C++;
the final production source identity is
`bf9dbcb826577dc4eaef0891017c98729cc474558723fa239273cee6b4c176d7`.
The original deployed campaign trees remain frozen. A comparison of all1,349
canonical identity paths finds only `direct_bounded_fallback.cu` changed; its
bytes match after removing ASCII whitespace. The final rebuilt library is
`100be12f9b691ec655b8044a2fa5566e09176e829493647419c0de04c791f180`.
n1 job5536 verifies that compiled identity against the final deployed source,
then passes baseline/OFF/ON through-f native oracles and both candidate
memcheck/initcheck runs with zero errors. The two-system/exact-prefix-budget
fixture passes in all three ON runs. Expanded focused host selection:221 passed.
The experiment does not qualify automatic selection merely because its measured
water-cluster warm endpoints improve.

## Paged complete endpoint decision

Jobs5532/5533 complete all144 native endpoints against the unchanged independent
references. Maximum all-repeat errors are1.037e-10 Eh and3.113e-11 Eh/Bohr,
within the unchanged1e-8/1e-7 gates. All original and moved warm replays take
one SCF iteration; no sample is filtered or normalized by iteration count.

| Atoms | Baseline a20 warm (s) | Paged warm (s) |
| ---: | ---: | ---: |
| 3 | 0.346606 | 0.268082 |
| 6 | 0.742711 | 0.617528 |
| 12 | 1.661318 | 1.393128 |
| 24 | 5.017627 | 4.494019 |
| 48 | 18.019789 | 15.169900 |
| 96 | 78.806055 | 65.513278 |

At96 the complete warm reduction is16.87%, not the source-only50.10%.
Changed-geometry warm medians are78.970583→65.678894s. But96 cold is
577.476754→581.458329s (28→29 iterations), and moved is263.215809→267.119967s
(12→13 iterations). The48 cold also takes an extra iteration. These small
complete-endpoint regressions and the artificial24 sparse-density slowdown
remain visible. Retain opt-in status; do not add a size heuristic or promote
automatically from the warm water-cluster wins.

The [retained endpoint records](../../../../benchmarks/results/pbe0-screened-pages-20261003/README.md)
include exact force arrays, all phase timings, source/library identities,
reconstruction patch and a no-GPU independent rechecker. Full endpoint records
are losslessly compressed; coarse and paged evidence remain separate.

## Review correction: shared claim consumption

PR #1767 review identified a real race not covered by the earlier scalar page
arithmetic, memcheck or initcheck gates. The publication barrier did not prevent
the next leader write while a delayed warp still read the previous shared
`block_quartet`. Empty diagonal/tail pages have no candidate-loop barrier;
inactive-system and block-screen continues also bypass it. Caching the claim
in a register alone would not protect its first read.

An unconditional CTA barrier now precedes the leader's atomic claim on every
loop iteration. All prior readers must finish before overwrite, regardless of
which no-work path they took. The publication barrier still precedes new reads.
The regression extracts the actual preparation, claim and page fragments,
delays a nonleader warp before its first read, and checks every lane's work and
termination claims over indexed/triangular, empty diagonal/tail and two skip
routes. Real-GPU plain, synccheck and racecheck variants pass, alongside the host
barrier invariant. Full native-oracle and complete-endpoint requalification is
pending. Earlier measured records and their identities are retained as
historical, not silently relabeled as corrected-binary evidence.
