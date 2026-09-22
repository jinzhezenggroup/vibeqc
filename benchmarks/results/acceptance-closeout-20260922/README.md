# Acceptance closeout audit — 2026-09-22

This audit records completed checks and explicit failures for issues whose
implementation was reported complete. It does not qualify the full production
tree. Test repairs are reviewed separately in [#1046](https://github.com/jinzhezenggroup/vibeqc/pull/1046).

## Source and execution identity

- Production source: `0d89ab6fb6219d9af6641d5cfafe0b43f8387206`, clean checkout.
- Library SHA-256: `7bb9c7ccd6381dbdaefb3cff752b8ede66445f3578b36ee79253874fa131c364`.
- Empty Release build, CUDA 12.9.86, `sm_120`, fast compile OFF, compiler cache OFF.
- RTX 5090, driver 580.95.05. Every real-device run used a finite Slurm
  `main --gres=gpu:5090:1` allocation and preserved assigned device visibility.
- Test-only repairs were compiled or imported from a separate checkout against
  the immutable production library. They do not imply a production algorithm fix.
- The compiler compatibility follow-up uses PR #1032 source
  `4a56f4d5b92c2a011ba6525c1d2f68da88bbe1aa`; the production build above does not
  include that later change. That PR was merged independently while it was being
  verified; this audit did not merge it.

`manifest.json` hashes retained files and local raw receipts. JSON gzip members
retain exact measurement arrays and sample order. Raw logs, XML, binaries and
profiler archives are not published. `test-results.json.gz` and
`native-results.json` retain semantic outcomes, including failed checks.

## Issue dispositions

| Issue | Disposition and remaining boundary |
| --- | --- |
| #487 | Closed after #1032 restored omitted historical facade aliases; expanded compatibility tests 60 passed, compiler audit 314 modules / zero errors. An initial incomplete closure was corrected by reopening before this follow-up. |
| #971 | Closed for mandatory provider abstraction A/B/C: 153 host tests passed and retained H200 qualification verified. Optional D/E/F remain follow-up scope. CUB stays opt-in. |
| #240 | Open: build/dependency evidence passes at the pinned source; runtime gates fail, and later native/build changes need reconciliation. |
| #669 | Open: both implementation children #670/#671 are closed, but the combined historical/current acceptance campaign failed and was stopped. |
| #172 | Open: three bounded hydrogen cases pass; H3+ and water requests fail, and full scaling/resource/domain qualification is incomplete. |
| #662 / #668 | Open: the timeline runner does not exercise the integrated prepared/AOT production owner; final performance/scaling/capability evidence remains incomplete. |
| #949 | Open: 384/768 BLAS counters and independent numerical endpoints pass; the complete matched 768 scalar comparison is unfinished. Generic scalar-promotion diagnostics already landed in #1015. |
| #439 / #206 | Open: this audit does not supply the full causal ledger, fresh stock GPU4PySCF matrix, changed-geometry coverage or all parent dependencies. |

## Build and runtime checks

The cold build completed in 1477.90 s with maximum resident host memory
4,018,652 KiB: 620 commands, 216 CUDA compilations, two device links and a
384,660,784-byte shared library. Content-neutral owner touches rebuilt
`rhf_graph.cpp` in 3.13681 s, `rhf_bucket.cpp` in 3.90727 s and `cuda_rhf.cpp` in
8.73854 s. Each also rebuilt the two generated CUDA-driver Implib glue objects
and relinked, with zero CUDA compilations/device links and an unchanged library
hash. These are rebuild-scope measurements, not a claim that the build ran only
one command. SCF dependency audit: 217 modules, 770 edges, zero errors;
106 structure tests passed. Exact commands and objects are in `build-times.json`
and `build-audit.json.gz`.

Slurm job 11195 ran the original native suite: **81/86 passed**. Failures:

- MP2 negative test assumed a valid CUDA request must fail on a GPU host.
  A deterministic one-byte staging-budget check passes against the same library.
- DF response tests used raw-value traffic expectations predating resident fitted
  values and scratch borrowing. The attempted repair failed and is excluded from
  #1046; `df-counter-repair-unqualified.patch` retains that rejected candidate.
  The diagnostic probe additionally reaches the retained-host UHF nonfinite-input
  rejection assertion, which remains active and fails. This is an open contract
  question; no successful native DF qualification is claimed.
- Fully polarized XC tail failure remains tracked by #1028.
- OH LDA-UKS final closure fails after 14 iterations, rather than exhausting its
  200-iteration cap. Stabilized iteration 10 has tiny residual; the final physical
  closure disables stabilization and alternates densities for four correction
  attempts. Diagnosis is recorded in #1002. No physical check was waived.
- Native DFT API expected AUTO precision for r2SCAN, whose public contract
  explicitly requires FP64. After correcting that assumption, strict r2SCAN-UKS
  still fails (`method=11`, CPU/CUDA status 5, CUDA KS physical evaluation failed).

The first HF/DF/MP2 Python run timed out at 1800 s because the 14-AO energy/tile-tail
test implicitly requested the now-supported MP2 gradient. After requesting energy
explicitly, the complete 251-case replay produced **247 passed / 4 failed** in
754.02 s (job 11204). Two failures were stale Fock diagnostic / exact HF iteration
assertions. The other two are charged UHF water DF final-state failures at
automatic and 8 MiB budgets, reproduced and tracked in #1041. Matching direct-UHF
and RHF controls pass. Focused repair results are in `qualification-summary.json`;
they are not relabeled as a fresh all-green full-suite run.

Other completed runtime evidence:

- Independent weighted ERI: 3349 records / 141 tiles / four runs, passed.
- Existing r2SCAN-3c suites: 31 passed.
- Six public CUDA LDA/PBE/r2SCAN RKS/UKS gradients plus prepared replay/recovery
  and changed-geometry batch isolation: eight passed after inspecting the current
  shared lease fields (job 11206).
- 14-AO MP2 CPU/CUDA independent energy/provider-tail gates: two passed (11202).

## Direct-HF campaign: not accepted

`direct-384-historical-{1,2}.json.gz` retain two timing-only processes. Both fresh
independent GPU4PySCF solves failed convergence at `conv_tol=1e-12`,
`conv_tol_grad=1e-10`, `direct_scf_tol=1e-14`, max 200. Timing protocol v1 also
primed each arm only initially; observed control-transition order dependence
invalidates clean factorial speed claims. Protocol v2 primes every transition,
but its only partial 768-AO run was cancelled (11210). There is no qualified v2
result. See `direct-disposition.json` and the retained versioned runners.

Historical 3.161731 s, controlled baseline 5.605620 s and current forced-rebuild
controls remain distinct. The archived source was dirty; the present fallback
does not reconstruct an exact pre-#698 revision. Existing 48-atom README reference
geometry differs from this 384-AO fixture. The matching 96-atom reference uses a
looser gradient convergence threshold and cannot replace the failed strict solve.

## DF response evidence

At 384 AO, seven interleaved clean pairs with a frozen density and per-transition
priming pass independent PySCF 2.14.0 DF gates (`1e-9 Eh`, `1e-8 Eh/Bohr`):

| Response policy | E+F median / s | Max energy error / Eh | Max force error / Eh/Bohr |
| --- | ---: | ---: | ---: |
| Forced panel, BLAS | 0.470987 | 5.457e-12 | 2.212e-12 |
| Forced panel, scalar diagnostic | 17.699395 | 5.457e-12 | 5.354e-12 |
| Automatic production, separate run | 0.293659 | 5.457e-12 | 2.132e-12 |

Every measured native sample uses two iterations. Intrusive counters were
collected afterwards: panel BLAS has 384 charge BLAS dots and zero scalar charge
dots; scalar mode has 384 scalar charge dots. The diagnostic switch also changes
density/metric products, so the complete endpoint ratio is **not a charge-only
causal speedup**. Automatic production uses the occupied-factor route and reuses
the retained final projection; it is a separate ownership policy, not another
sample of the panel arm. No fresh GPU4PySCF timing comparison is claimed here.

At 768 AO, the independent CPU reference completed at unchanged strict settings,
but the scalar campaign reached its 1200 s command limit after one complete BLAS
sample and zero scalar samples. The repeated automatic oracle task was cancelled.
Job 11215 then reused the exact retained independent reference, after checking
geometry, bases, charge/spin, library and oracle thresholds, for two bounded runs:

| Response policy | Samples | E+F median / s | Max energy / force error |
| --- | ---: | ---: | ---: |
| Forced panel, BLAS | 7 | 5.122863 | 3.865e-11 / 2.188e-12 |
| Automatic production | 7 | 2.796958 | 3.865e-11 / 2.325e-12 |

All these native samples use six iterations. The separate panel diagnostic has
768 charge BLAS dots and zero scalar charge dots. The exact reference and adapter
hashes are retained in each record. These checks establish numerical and runtime
lowering evidence. They do not complete the original paired scalar campaign or
establish a same-branch causal comparison with historical two-update endpoints;
#949's remaining matched performance conclusion stays open with #439/#206.

## Canonical r2SCAN-3c qualification

Job 11205 used the defining basis, production default grid and strict
`E=1e-12 / D=1e-10` controls. Independent totals combine PySCF/Libxc electronic
gradients on the identical explicit grid, DFT-D4 4.2.0 and independent gCP.

| Case | AOs | Result | Max energy / force error |
| --- | ---: | --- | --- |
| H2 | 10 | Cold, three warm, changed/fresh and multistep total-energy FD pass | 1.825e-13 / 1.954e-12 |
| H2 dimer | 20 | Cold, three warm, changed/fresh pass | 2.194e-13 / 1.582e-11 |
| H3 doublet | 15 | Cold, three warm, changed/fresh pass | 2.727e-13 / 1.374e-10 |
| H3+ singlet | — | Initial public batch execution fails numerically | No qualified phases |

Ragged failure isolation is blocked by that initial H3+ failure. Water also
returns a numerical failure, but this audit did not localize it to the explicit
higher-angular-momentum capability check. The current stationary-force layout
allows s/p only; canonical heavier-element basis support does not imply full
H–Ar force qualification. Reported public resource diagnostics are null in these
bounded cases, so a complete resource acceptance claim is also unsupported.

## Reproduction and diagnostic boundaries

The retained runners contain exact historical paths and commands. Reproduction
requires mapping those paths to equivalent isolated source/build/oracle locations,
checking source and library identities, and submitting device work via Slurm.
`run_pipeline.sh` is the original campaign plan, **not** an instruction to rerun
stopped Direct-HF arms or repeat the production build. Use the individual runners
and preserve failure dispositions. Diagnostic C++ patches print intermediate
state/counters; the DF probes deliberately bypass traffic assertions and are not
passing native-test evidence. The separately repaired tests retain numerical,
resource-budget, transactional and replay gates.
