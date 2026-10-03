# Investigation: reduce complete WB97M-V cold SCF work

Status: measured diagnosis; public scheduling/convergence controls remain experimental
Date: 2026-10-03

## Complete cold boundary

A cold endpoint includes synchronized preparation and the first converged energy
plus analytic forces, including calculator/engine construction and returned host
force arrays. Imports and separate library probes are outside this boundary;
it is not an empty compiler/filesystem-cache experiment. The retained warm
advantage of joint SCF/force AO maps does not establish cold superiority.
At 48 atoms the qualified joint composition still takes 962.211204 s versus
474.695895 s for its paired GPU4PySCF reference. Neither engine receives the
other's density. All target tolerances and independent E/F gates remain fixed.

## Actual cold profile

n1 RTX 5090 finite Slurm 5601 profiles preparation and first complete E/F
execution of composition 0132d7584/library b82e468613c5c90e076aa10082b2389c8dda74335f625a7b77641c1c085ef3f9.
This is the same joint AO-map composition as README PR #1786, with 24 atoms,
192 spherical def2-SVP AOs and 589824 matched unpruned grid points. The
instrumented solve takes 18 iterations; its energy/force errors against the
independent retained reference are 2.161e-12 Eh / 4.925e-10 Eh/Bohr or less.

Nsight reports 236.420841 s summed GPU kernel time. The 18 SCF-feature VV10
pair launches total 127.025807 s (53.7%); the separate geometry VV10 pair
launch is 9.643805 s. Canonical/generated J/K value kernels total 57.575371 s;
XC potential assembly 16.913895 s, semilocal evaluation 13.624040 s, bounded
force derivatives 3.157142 s, AO collocation 2.752049 s and density panels
2.514039 s. Kernel grouping follows retained names. These instrumented times
are not clean endpoint latencies or percentages of another benchmark run.

The profile supports reducing repeated SCF work as a cold-start priority.
Removing AO columns alone cannot remove the VV10 pair domain. The two existing
potential-assembly traversals also have real cost, but fusing them would require
retaining semilocal coefficients with explicit resource/lifetime accounting.
No fusion or new pair screening is implemented by this investigation.

## Public SCF tile experiment

n1 Slurm 5596 passes seven complete independent molecular/displaced-energy/
stale-state cases with tile_points=1024 (66 successful calls, 467 XC submissions).
Slurm 5597 then compares public SCF tile sizes 256 and 1024 sequentially on one
GPU, holding joint AO selection and the force planner fixed. Every cold, priming
and three warm E/F pair passes 1e-8 Eh / 1e-7 Eh/Bohr; all reference XC backends
report on-GPU. Both cold variants take 18 iterations; every warm takes one.

| SCF tile points | Native complete cold | Paired reference complete cold | Native warm median | Paired reference warm median |
| --- | ---: | ---: | ---: | ---: |
| 256 | 243.279439 s | 118.246257 s | 26.795078 s | 27.934585 s |
| 1024 | 232.195245 s | 118.284526 s | 26.160322 s | 27.870576 s |

Cold decreases 4.56%, but remains slower than reference. SCF tiles decrease
2304 to 576, numeric host peak decreases 3558152 to 890120 bytes and reserved
XC device bytes increase 7034208 to 13897056. Sampled point/AO-square work
increases from 9896403968 to 11154944000 because larger tiles retain more AOs.
The force map and force scheduling remain unchanged. This is a measured
scheduling tradeoff, not a fourfold speedup or default-promotion justification.
The same complete 48-atom comparison finishes in finite Slurm 5600. All ten
cold/priming/warm E/F pairs pass the independent unchanged gates (maximum
9.778e-12 Eh / 6.307e-10 Eh/Bohr), every warm call takes one iteration, and
reference XC remains on GPU. Both native cold solves take 21 iterations:

| SCF tile points | Native complete cold | Paired reference complete cold | Native warm median | Paired reference warm median |
| --- | ---: | ---: | ---: | ---: |
| 256 | 963.522488 s | 475.194824 s | 97.760966 s | 103.112731 s |
| 1024 | 941.008620 s | 475.510140 s | 96.158374 s | 103.201642 s |

Cold decreases 2.34% and warm 1.64%. SCF tiles decrease 4608 to 1152, numeric
host peak decreases 14194184 to 3549704 bytes and reserved device bytes increase
21709472 to 30046880. Point/AO-square work increases from 25257336832 to
30993444864. Force maps and their work remain unchanged. The 96-atom planner
boundary still needs its own measurement; these two sizes do not promote a
universal 1024-point default. Measurements remain attributed to 0132d7584 and
its original b82 library, not the newer master integration.

## Next controlled experiment and provenance

Frozen composition ebb2892e3 integrates actual master 1a4acc519 and both claim
barriers. Its 1351 manifest inputs reproduce source identity
5ec094b1be8c296fb12b0d8cca080ad23ab004c9f4df98536897e7e02e198930;
complete library SHA-256 is
fc614342e2897fb57366bb97e76124d59bbb7efb8ba82b91367d19c9fc6a5937.
CXX/CUDA compiler commands use ccache, with before/after statistics retained.
The unchanged native discovery test executable retains its earlier sanitizer
qualification; the new complete library receives fresh independent full E/F
and changed-geometry tests, plus both real-device claim-barrier regressions.

After qualification, finite Slurm 5606 compares public DIIS history lengths
8, 12 and 16, sequentially in one GPU allocation. Grids, FP64, density cutoff,
energy/density/screening tolerances, full-density J/K and all-pair gates stay
fixed. A larger history changes proposal construction only; every accepted
result must reconverge the same physical target. All attempts, including any
regression, remain in the output set. There is no changed production default.
Larger and displaced-geometry evidence is required before promoting any
profitable control.

The new composition passes seven independent complete E/F/displaced-energy/
stale-state tests with joint maps and seven with force cache allowance zero;
each group checks 66 successful native calls and 467 XC submissions. Native
discovery/WB97M-V targets pass, as do ten real-device claim-barrier regression
cases. These are qualification receipts, not clean performance measurements.

The completed 24-atom comparison passes the independent verifier for all 15
cold/priming/warm pairs. Reference XC remains on GPU, every warm/priming call
takes one iteration, and maximum energy/force errors are below 2.388e-12 Eh /
5.514e-10 Eh/Bohr. The original target tolerances remain unchanged.

| DIIS history | Native complete cold | Paired reference complete cold | Native cold iterations / XC submissions | Native warm median |
| --- | ---: | ---: | ---: | ---: |
| 8 | 243.605405 s | 117.903633 s | 18 / 18 | 26.777580 s |
| 12 | 231.518032 s | 117.534887 s | 17 / 17 | 26.920691 s |
| 16 | 230.928286 s | 117.187306 s | 17 / 17 | 26.620800 s |

One fewer expensive SCF submission reduces complete cold by about 5%; it does
not close the reference gap. The 12/16 difference is not established beyond
single-cold-run variability.

The same-binary 48-atom comparison completes in finite n1 Slurm 5614. All 15
cold/priming/warm pairs pass the independent verifier, with maximum errors
below 1.206e-11 Eh / 6.307e-10 Eh/Bohr. Reference XC remains on GPU and every
warm/priming call takes one iteration. Every variant is retained:

| DIIS history | Native complete cold | Paired reference complete cold | Native cold iterations / XC submissions | Native warm median |
| --- | ---: | ---: | ---: | ---: |
| 8 | 961.199066 s | 472.069554 s | 21 / 21 | 97.112920 s |
| 12 | 964.064881 s | 471.756488 s | 21 / 21 | 96.571170 s |
| 16 | 916.073826 s | 471.785992 s | 20 / 20 | 97.400964 s |

History 12 does not reduce iteration work at 48 atoms; history 16 saves one
submission and 4.69% of complete cold. This small gain still leaves the native
endpoint 1.94 times the paired reference. Do not promote a universal history
default or spend another 96-atom campaign on this knob alone: first investigate
a cheaper GPU preliminary operator, retaining DIIS as a possible later combined
control. The original raw reports and `matched48-verified.json` remain under
`.artifacts/cold-scf-20261003/` in the SCF composition worktree.

The public preliminary-SCF and cross-method seed APIs are currently CPU-only
or HF-only respectively. This investigation does not bypass those contracts,
import a reference density, or present an uncharged preliminary solve as cold.

## Same-method coarse-grid seed probe

The native KS prepared-batch owner explicitly implements `warm_state` and
`restore_warm_states`, despite the density-buffer C API's historical HF name.
A private measurement driver tests that existing native contract with identical
WB97M-V functional, basis, nuclei, core, spin and exact-exchange provider. Only
the preliminary quadrature and preliminary stopping controls differ. This is
not an extension of the public CPU-only preliminary-SCF API, cross-method
initialization or proof of public constrained-budget support.

The source uses 8×6×12 or 16×8×16 grids, 1e-6 Eh / 1e-4 density stopping controls,
DIIS history 8 and at most 64 iterations. The target keeps the full 48×16×32
grid and original 1e-11 Eh / 1e-9 density tolerances. The native import validates
the density state; the target rebuilds its Fock operator and fresh DIIS without
adopting the source convergence or energy baseline. The driver checks scientific
identity before transfer. A nonconverged source leaves the target's core guess
intact; unrelated exceptions propagate. No reference density is imported.

All source preparation, solve, density export/import and destruction belong to
the native cold preparation timer. Source iterations and actual XC submissions
are reported separately. The native source `fock_builds` field is unavailable
(`None`), and is not replaced by an iteration-derived estimate. The first pilot
terminated while serializing that unavailable field, after its source solve;
the failed attempt is retained separately and is not reported as a numerical
failure or a completed endpoint. The corrected pilot uses a nullable field.

Finite Slurm 5611 completes all three 3-atom controls and their independent
15-pair E/F gate before continuing to 24 atoms. Complete cold is 9.839958 s
without a seed, 10.073681 s with grid 8 and 9.420318 s with grid 16. Target
iterations are 15/12/10, respectively, while both sources require 16 iterations;
their complete costs are 1.869509 / 2.152752 s. The small gain and regression are
both retained. These pilot results do not establish a large-system improvement;
the 24-atom comparison below supplies its own complete gate.

The completed 24-atom experiment passes all 15 independent E/F pairs with
maximum errors below 3.525e-12 Eh / 5.804e-10 Eh/Bohr. All warm/priming calls
take one iteration and reference XC remains on GPU. Complete cold, including
all preliminary work, is:

| Preliminary grid | Native complete cold | Paired reference complete cold | Target iterations | Source iterations / complete cost |
| --- | ---: | ---: | ---: | ---: |
| none | 249.801891 s | 120.544875 s | 18 | none |
| 8×6×12 | 274.592088 s | 120.206042 s | 15 | 18 / 63.252052 s |
| 16×8×16 | 243.123315 s | 120.009562 s | 12 | 19 / 70.353333 s |

Grid 8 regresses 9.92%; grid 16 reduces complete cold only 2.67%, despite
removing six expensive target iterations. Source preparation and convergence
consume most of the saved time. The next private experiment keeps grid 16
and compares looser preliminary stopping controls with the same final target
and unchanged all-pair gates. It does not relax the scientific endpoint or
claim public preliminary-SCF support. No default is changed.

Finite n1 Slurm 5620 runs four 24-atom controls sequentially: no seed, then
grid-16 preliminary energy/density tolerances 1e-3/1e-2, 1e-4/1e-3 and the
original 1e-6/1e-4. Every preliminary solve must converge under its recorded
controls before installing the density; the final full-grid target remains
1e-11/1e-9 with DIIS history 8. The same frozen ebb/fc614 source and library
are verified inside Slurm. All source work remains in complete-cold preparation.
All four controls complete and pass the independent 20-pair E/F verifier;
maximum errors are 3.866e-12 Eh / 5.805e-10 Eh/Bohr, reference XC remains on
GPU and every warm/priming call takes one iteration. Complete cold is
244.881151 s without a seed, then 238.627422 / 238.556393 / 238.649670 s for
loose / medium / tight source controls. Paired reference cold is
118.438526 / 118.489155 / 118.328431 / 118.224691 s. All three source solves
still take 19 iterations and cost 69.060948 / 69.157789 / 69.145649 s; their
full-grid targets each take 12 iterations rather than 18.

The unchanged iteration count has a specific source explanation:
`CudaKsPlan::Impl` requires both physical residual norms below
`min(1e-9, options.density_tolerance)` in addition to energy and density
change. Loosening the latter controls does not loosen those residual gates.
This is therefore a negative result for the proposed stopping-only shortcut,
not evidence that a genuinely bounded preliminary solve is unprofitable.
Do not weaken the final-state validator or label an unfinished source as
converged to obtain a cheaper seed. The existing warm-state contract only
publishes converged states; accepting an unfinished preliminary density
would need an explicit separate initial-guess contract. No source-control
default is promoted and this ineffective knob sweep is not expanded to 96
atoms without a distinct, justified intervention.

Ignored artifacts in the SCF composition worktree retain:
- `.artifacts/scf-active-ao/profiles/cold24-complete.*` and the independent
  `results/cold24-complete-trace.json` scientific/work record;
- `.artifacts/scf-tiles/results/matched{24,48}-verified.json`, every raw sample and
  the independent all-pair verifier;
- `.artifacts/cold-scf-20261003/` for latest-master source/binary identities,
  qualification, exact scripts, the 15-pair verifier and all cold-control attempts;
- `.artifacts/coarse-grid-seed-20261003/` for the native warm-state experiment,
  source/target work and timing records, exact scripts and failed-attempt receipt.
- `.artifacts/coarse-grid-stopping-20261003/` for the four preliminary stopping
  controls, independent all-pair verifier and immutable final-target settings.
