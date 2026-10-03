# Experiment: GPU semilocal preliminary density for WB97M-V

Status: private complete-endpoint experiment; no public API/default promotion
Date: 2026-10-04

## Motivation and boundary

The [cold-work investigation](2026-10-03-wb97mv-cold-scf-work.md) found that
same-functional coarse-grid preparation still costs about 69 seconds at 24
atoms. Loosening its energy/density thresholds leaves the strict physical
residual gate and its 19 iterations unchanged. The next experiment changes
the preliminary operator instead: GPU LDA or PBE has neither exact exchange
nor VV10 pair work. The final target remains full-grid FP64 WB97M-V.

This is a private driver using the existing native KS density restore
contract, not an extension of CPU-only `InitialGuessSpec` or the public HF
`initialize_from` API. `KsPreparedBatch::restore_warm_states` checks density
dimensions, coordinates, diagnostics and the source AO metric/spin convention.
Only the converged same-basis density is imported; target Fock, DIIS,
functional, grid, convergence and energy baseline are not imported. The
target constructs fresh physical work and retains its ordinary bounded retry.

## Fixed comparison

Run no preliminary solve, `lda-rks`, and `pbe-rks` sequentially on one RTX 5090.
Both sources use spherical def2-SVP, the target nuclei/charge/spin, grid
16×8×16, FP64, DIIS 8, at most 64 iterations and energy/density controls
1e-6/1e-4. The source must actually converge before its density can be used.
An unfinished source leaves the core guess intact; non-convergence is the only
source failure status allowed to continue. Resource/runtime errors propagate.

The target keeps grid 48×16×32, energy/density/screening controls
1e-11/1e-9/1e-12, DIIS 8 and maximum 180 iterations. SCF/force AO maps and
indexed forces are enabled for every target variant. Native experimental SCF
AO maps currently admit WB97M-V only: the private single-threaded driver
temporarily disables that opt-in for the separate LDA/PBE owner, restores it
on every exit and retains the already-prepared target's map. This is not a
general mechanism for selecting per-owner policies concurrently.

Complete cold includes source construction, preparation, solve, synchronized
density export/import and destruction as well as all target preparation and
the first complete energy/analytic-force call. Actual source XC submissions,
source iterations and available Fock-build diagnostics are retained; an absent
Fock-build counter remains absent. The driver has a 16 MiB density/coordinate
transfer bound. It does not establish joint public host/device-budget admission.

After a 3-atom pilot passes all 15 cold/priming/warm pairs, the same allocation
continues the 24-atom comparison. Every pair must pass independent 1e-8 Eh /
1e-7 Eh/Bohr gates, finite/shape/convergence checks, one-iteration warm replay,
actual target AO selection and on-GPU reference XC. Every accepted preliminary
variant must also prove successful source convergence and inclusion of its
complete cost in the cold timer. No source result is compared to the reference
as though it were a WB97M-V answer.

## Provenance and attempts

Source 49f0f9bc078a3f714ccc4fa1b394f638c784bacf includes master dc6ea9940.
That master update changes evidence consumers only; all 1355 native build inputs
remain identical to the GPU-qualified master-9c/VWN composition:

- Source identity: `ebdf07921464440e085b2925a1bd061ba9090788485d1cab01e3cada7122814c`
- Library SHA-256: `5a1b86cef1c0e978118d3024dc36861ebe8cd326dee8a6fe944a6b34000a5461`

Each job verifies source/library and measurement-script hashes before execution.
No old timings are relabeled to this composition. All source and build receipts,
scripts, raw results and unsuccessful attempts remain in ignored
`.artifacts/semilocal-seed-20261004/`.

Slurm 5634 stops because LDA preparation inherited the WB97M-V-only map flag.
Slurm 5637 then completes the no-seed/LDA pilot endpoints but stops on the
invalid shorthand `pbe`; the supported selector is `pbe-rks`. An attempted
retry, 5639, is cancelled after local setup failed due to an omitted Python
search path. These are driver/setup failures, not numerical method rejections;
their outputs are preserved separately and do not qualify the full campaign.
The corrected finite n1 allocation is Slurm 5640. It restores the source map
policy explicitly, validates provider names before deployment and propagates
unexpected source failure statuses. The completed comparisons are below.

## Completed small pilot

Slurm 5640 completes all three 3-atom variants and passes the independent
15-pair verifier before continuing to 24 atoms. Maximum errors are below
8.669e-13 Eh / 1.177e-9 Eh/Bohr; every reference XC component stays on GPU
and every priming/warm replay takes one iteration. The target gates and full
grid are unchanged.

| Preliminary provider | Native complete cold | Paired reference complete cold | Target iterations / XC submissions | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 9.404071 s | 9.744874 s | 15 / 15 | none |
| LDA | 8.536679 s | 9.598415 s | 12 / 12 | 15 / 0.656032 s |
| PBE | 8.156632 s | 9.645183 s | 11 / 11 | 15 / 0.681369 s |

Both sources actually converge and are imported. Their XC submission counts
are 15 each, their Fock-build counters remain unavailable, and the complete
source cost is included in the respective cold preparation totals. The small
pilot supports testing the mechanism at 24 atoms; it does not establish a
large-system benefit or justify a default/API promotion.

## Completed 24-atom control

Slurm 5640 finishes successfully, including all three 24-atom controls and
their independent 15-pair E/F verifier. Maximum errors are below 3.070e-12 Eh /
7.540e-10 Eh/Bohr. Every priming/warm replay takes one iteration, every
reference XC component reports GPU execution, and source/target work remains
explicit. Complete cold includes all source preparation and transfer:

| Preliminary provider | Native complete cold | Paired reference complete cold | Target iterations / XC submissions | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 240.379900 s | 116.000208 s | 18 / 18 | none |
| LDA | 191.435694 s | 115.936782 s | 13 / 13 | 18 / 13.088323 s |
| PBE | 192.097899 s | 116.104328 s | 13 / 13 | 18 / 13.345290 s |

The sources each converge with 18 actual XC submissions. Source Fock-build
counts remain unavailable, not inferred. Both native seeded targets use their
imported density without a cold retry. LDA/PBE reduce complete cold by
20.36% / 20.09% relative to the same-allocation unseeded control. Unlike the
same-functional coarse-grid source, these inexpensive operators leave a
material net gain after charging their whole cost. Native cold still takes
about 1.65 times the paired reference; this is not cold superiority. The
subsecond LDA/PBE difference is not established beyond single-run variability.

Warm medians are 26.122775 / 25.843800 / 25.843492 s for none/LDA/PBE, with
paired references 27.374984 / 27.372269 / 27.381791 s. All warm calls replay
their own frozen final density. Do not attribute their small differences to
the preliminary operator, which performs no work in those calls.

## Larger and displaced qualification

Finite n1 Slurm 5645 starts the 48-atom three-way comparison only after 5640
finishes, its complete verifier passes and at least one source reduces
24-atom complete cold by 5%. This guard prevents expanding a failed or
unprofitable small control. The larger run retains both source choices and
the unseeded control, with the same native inputs, binary, target controls and
measurement scripts. It remains in progress; no 48-atom seed benefit is claimed.
The first launcher used an unsupported Slurm command-line option and created
no job; that failure is retained separately. The corrected finite allocation
uses `afterany:5640` plus explicit success/scientific checks inside the job.

Slurm 5644 separately moves hydrogen 1 by (0.02, -0.01, 0.015) Bohr and creates
both source and target at those identical displaced coordinates. This tests a
fresh displaced endpoint, not an in-place warm geometry update. All 15 3-atom
pairs pass independent E/F and coordinate checks (maximum errors below
1.024e-12 Eh / 1.065e-9 Eh/Bohr), with on-GPU reference XC and one-iteration
priming/warm calls. Complete cold is 10.336747 / 8.675310 / 8.130738 s for
none/LDA/PBE; target iterations are 15/12/11. Each source takes 15 iterations
and costs 0.654103 / 0.673232 s in full.

The 24-atom displaced comparison also completes in Slurm 5644. All 15 pairs
pass the independent E/F, coordinate, source-cost and backend checks, with
maximum errors below 3.070e-12 Eh / 5.350e-10 Eh/Bohr. Every priming/warm
replay takes one iteration. Complete cold, including all source work, is:

| Preliminary provider | Native complete cold | Paired reference complete cold | Target iterations | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 281.416608 s | 130.752835 s | 21 | none |
| LDA | 222.939598 s | 130.674971 s | 15 | 24 / 17.473217 s |
| PBE | 235.506555 s | 130.489469 s | 16 | 23 / 17.127575 s |

LDA retains a 20.78% complete-cold reduction after this perturbation; PBE
retains 16.31%. Native remains slower than the corresponding reference.
These are fresh displaced solves, not same-plan geometry-rebuild timings,
and do not qualify the still-running larger-size or public resource contracts.

The separate scripts, geometry records, raw pairs, independent verifiers and
Slurm/source/library receipts remain in ignored
`.artifacts/semilocal-seed48-20261004/` and
`.artifacts/semilocal-seed-displaced-20261004/`. They do not replace any of the
frozen README timing reports.

## Completed 48-atom LDA/control subset and 96-atom expansion

Slurm 5645 completes the unseeded and LDA variants at 48 atoms. A separate
read-only subset verifier checks all ten cold/priming/warm E/F pairs, source
costs, actual SCF/force AO selection and reference XC on GPU. Maximum errors
are below 1.251e-11 Eh / 6.305e-10 Eh/Bohr; every priming/warm call takes one
iteration. PBE remains in progress and is not represented as completed.

| Preliminary provider | Native complete cold | Paired reference complete cold | Target iterations | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 960.174251 s | 475.260614 s | 21 | none |
| LDA | 752.846451 s | 475.987336 s | 15 | 25 / 49.247219 s |

The full 49.247219 s source cost includes 36.502280 s preparation/solve and
12.739289 s export/import, plus the remaining measured source lifecycle work.
All 25 source XC evaluations are recorded. The 21.59% complete-cold reduction
survives charging that whole cost; native cold remains about 1.58 times its
reference. Warm medians are 93.390685 / 93.425450 s for none/LDA, versus paired
reference 103.141942 / 103.421357 s. The source performs no work in warm calls.

`verify-completed-modes.py` preserves the original verifier's scientific gates
while explicitly selecting the two completed variants and writing a separate
`matched48-completed-subset-verified.json`; it does not alter the original
three-variant runner/verifier or any raw record. No pending PBE sample is
included or dropped from a completed comparison.

The next 96-atom unseeded/LDA comparison uses newly qualified integration
678f7eb88/master 837c2a51c, source identity d198acd7 and library fe826ad7 as
fully recorded in the resident integration note. It runs on n1 RTX 5090 with
a finite six-hour allocation and fresh identical source/target coordinates.
Startup requires the ten-pair 48-atom receipt, at least 5% complete-cold benefit,
and both seven-case numerical/fallback qualifications of the new composition.
All source costs remain charged and all endpoint gates remain unchanged.
This larger comparison is pending; the private source bridge is still not a
public CUDA preliminary-SCF or public resource-budget contract. Receipts remain
under ignored `.artifacts/seed96-master837-20261004/`.

## Promotion gates

Require larger and displaced-system complete endpoints before claiming a
useful target domain. A production extension would need an explicit CUDA
preliminary provider contract, total resource admission, diagnostics and
failure/warm-priority tests; a profitable private density bridge alone does
not satisfy those requirements. Keep the unseeded path and all target gates.
