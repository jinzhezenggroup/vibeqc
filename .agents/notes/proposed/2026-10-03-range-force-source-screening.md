# Proposal: screen range-exchange force tasks by their requested source

Status: proposed; independent numerical gates and full-grid 24/48-atom comparisons pass;
96-atom qualification remains pending
Date: 2026-10-03

## Problem

The 96-atom WB97M-V complete force endpoint spends about 222 seconds in integral
derivatives. The bounded SR/LR derivative launcher publishes only exchange,
but passes the mixed J/K demand to its queue. Even when a caller explicitly
requests K alone, the force-product screen still admits a quartet from its
Coulomb density product. Distant Coulomb blocks can therefore retain tasks whose
exchange products fall below the existing force cutoff.

## Candidate and boundaries

Honor the existing source masks in the force-product gate, and set the SR/LR
exchange-only derivative launcher to K demand for both RKS and UKS. Keep each
same-spin exchange product independent; do not use cancellation between signed
J/K coefficients or between spin channels. The mixed J/K default and unrelated
full-range consumers retain their existing selection.

The same demand also selects the existing raw-K linear bound. In restricted
calculations it has unit weight, rather than the mixed Fock's one-half weight,
so this first gate can retain some exchange tasks the old mixed bound rejected.
The new overall task set is not claimed to be a strict subset. Screening
thresholds remain unchanged, including `min(screening, 1e-14)` for force density
products. Recurrences, radial operators, FP64 precision, coefficient/scatter
contracts, buffers and bounded/generic fallback availability do not change.

## Evidence and promotion requirements

The isolated branch starts at master
`cd08953d5765d389baba62935f2025816be57075`. Its clean Release CUDA 12.9/sm120
baseline library SHA-256 is
`4e392ae19aa94656f031ef8104a4f7d1c4b4fd5a9417a41ea2bcf5111ae942f9`.
Explicit ccache launchers are verified in all 441 compiler commands; exact build
and cache receipts are retained in `.artifacts/exchange-force-screen/`.

A host harness executes the actual scalar screening function. The baseline
fails the K-only force-product case; the candidate passes 42 checks covering
Coulomb-only, exchange-only, beta-only UKS, independent source admission and
neighbors of the force cutoff. This demonstrates source selection, not the
accuracy of a complete force or a performance improvement.

Before promotion require independent real-device full/SR/LR derivative gates,
RKS/UKS complete energy/force and displaced-geometry gates, and same-allocation
baseline/candidate complete 24/48-atom timings followed by a 96-atom check.
Retain every sample, convergence/work metadata and 1e-8 Eh / 1e-7 Eh/Bohr gates.
Do not infer the number of executed shell quartets from logical capacity or
from canonical-source counters, which do not observe this bounded route.

## Completed numerical qualification and 24-atom endpoint

Candidate `63e762b46130a0d52202f513d4a06293191ee7b8`, library
`0340ce35e4253ed8dfd5b10a9b8ef1b123506acf88284967e9c658555561dc4f`, passes
native s/p/d/f SR/LR derivatives and Cartesian order-two CPU finite-difference
checks in n1 Slurm 5497. All seven independent complete WB97M-V force/rebuild
tests pass, including RKS/UKS and stale-state/failure isolation. Generic provider
tests complement the molecular tests: production omega=0.3 uses the modified
bounded LR route after its separate full-range J/K traversal.

n5 Slurm 1408 runs both builds sequentially on one RTX 5090, water24/192
spherical def2-SVP AOs, full 48 x 16 x 32 grid and three fixed-density warm
repeats. Each build has a fresh GPU4PySCF full-density Fock control; all five
complete energy-plus-host-force pairs per build pass 1e-8 Eh / 1e-7 Eh/Bohr,
and all five reference endpoints pass internal consistency gates.

| Complete endpoint | Base / s | Candidate / s |
| --- | --- | --- |
| Native warm median | 39.517558426 | 38.686160628 |
| Fresh reference warm median | 28.173318023 | 28.192626931 |
| Native cold | 284.732771 | 283.218920 |
| Native priming | 39.505115 | 38.695058 |

The native warm improvement is 2.104%; native remains slower than the independent
reference. This modest result does not close the large-system performance gap.
The initial n5 attempt (1407) failed at library load and supplies no accepted
timings. Its separate receipt records an older preloaded CUDART. The completed
run verifies CUDA 12.9 runtime identity before either engine executes.

n1 Slurm 5498 additionally checks water12/def2-TZVP on the diagnostic 24 x 8 x
16 grid against retained independent CPU original/moved geometry oracles. All
five observations per build pass, with maximum energy error 1.60e-12 Eh and
force error 8.97e-10 Eh/Bohr. Changed-geometry endpoints are 92.708560 and
91.329185 seconds (11 iterations each). Its single warm sample and reduced
grid are correctness evidence, not the full-grid performance comparison.

Raw journals, exact library hashes, terminal receipts and the all-observation
verifier remain in `.artifacts/exchange-force-screen/`. The verifier checks
the molecular grid, convergence, all force entries, comparator policy and
reported medians. n1 Slurm 5499 has completed the full-grid 48-atom run (exit 0).
Later master `d35ae539f` contains separate low-order full-range force reuse;
these frozen measurements remain attributed to base `cd08953d5`.

## Completed controlled 48-atom endpoint

n1 Slurm 5499 runs both frozen builds sequentially in one RTX 5090 allocation,
water48/384 spherical def2-SVP AOs, full 48 x 16 x 32 grid, cold, priming and
three engine-local warm repeats. All five pairs per build and all independent
full-Fock reference consistency checks pass. Maximum errors across both builds
are 1.16e-11 Eh and 6.33e-10 Eh/Bohr.

| Complete endpoint | Base / s | Candidate / s |
| --- | --- | --- |
| Native cold | 1175.920684069 | 1153.960843235 |
| Native priming | 155.593584530 | 131.872031592 |
| Native warm median | 155.284395609 | 131.989203334 |
| Fresh reference warm median | 105.602346297 | 105.418984771 |

Native cold uses 21 iterations for both builds; every priming/warm solve uses
one. The reference uses 16 cold iterations/17 J-K builds and one warm
iteration/two J-K builds. The native warm improvement is 15.002%, while native
still takes 1.252x the candidate's independent reference median. The first
warm integral-derivative component falls from 57.885080 to 34.679230 seconds;
grid/pair drain stays near 46 seconds. Grid, geometry, collocation and allocation
work fields remain identical. Actual executed bounded shell-quartet counts are
not exported and remain unavailable; the timings do not establish a strict
subset of screened work.

Candidate-only full-grid 96-atom qualification now runs in n1 Slurm 5507 with a
fresh full-Fock reference and three warm repeats. Its 4-hour finite allocation
and the latest-master composition remain pending. This larger run is numerical
and endpoint qualification, not an isolated base/candidate speed comparison.
