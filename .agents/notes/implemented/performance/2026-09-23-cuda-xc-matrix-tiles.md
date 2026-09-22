# Decision: compiler-owned bounded XC matrix tiles

Status: implemented; endpoint qualification in progress
Date: 2026-09-23

## Problem

Issue #1102: a bounded two-iteration PBE48 profile (Slurm 11350) attributes
48% of captured GPU kernel time to scalar density-product and potential
assembly contractions. The original density kernel independently reloads an
entire AO row for each output. Potential assembly repeats point coefficient
and gradient products for every AO pair. This is separate from preparation
(#1101) and source-integral work (#1078).

## Decision

Keep allocation-free resident execution. Compiler-owned 16-by-16 shared-memory
matrix tiles reuse density/AO operands; compact coefficient panels replace
per-pair gradient coefficient multiplication. The canonical AO-pair bilinear
moves from the XC jet-pullback builder into `dft.xc_bilinear`, which both
geometric pullbacks and matrix-panel lowering consume. The scalar Graph derives
panel coefficients. Only the shared neutral Graph/CUDA emitters are admitted
as IntegralIR dependencies for these two DFT owners; no integral recurrence or
method policy dependency is allowed.

GGA combines its three spatial coefficients into one panel. Symmetric cross
assembly uses half the scalar contribution in that panel. Meta-GGA retains
three additional half-kinetic panels; tau's scientific half is already in the
input coefficient and is unchanged. Signed weights and coefficients are legal.
The compiler preserves all quadrature points and AO pairs, applying weights
exactly once. FP64 numerical gates remain unchanged.

The work panels are reused only after reference/directional density features
and point coefficients are complete on the same stream. Their existing
spins*work_jets*tile*nao allocation is sufficient. Shared scratch is 4,352 bytes
for density and 8,704 bytes for potential per block (padded double arrays).
No extra global storage or device/host staging is introduced. Small shapes and
shapes exceeding the two-dimensional launch domain retain the original bounded
scalar kernels. Partial tiles load zeros without leaving barriers early; one
triangle is authoritative and mirrored exactly.

## Work accounting

The AO/point/spin census and density reduction count do not change. Each matrix
tile reuses each loaded density/AO value across up to 16 outputs. Potential
prepacking costs O(spins*points*nao*jets), instead of multiplying coefficient
and gradient factors independently at O(spins*points*nao^2). Symmetric matrix
assembly visits two legs per work jet (one jet for LDA/GGA, four for meta-GGA),
with no source/grid rebuilds or screening. Zero padding can add up to 15 lanes
per partial reduction block. Complete cold/warm/geometry and tile-tail evidence
must be checked before claiming an endpoint benefit.

## Rejected alternatives

A fresh cuBLAS handle retains library-internal allocations even with caller
workspace (the shared tensor runtime conservatively allows 96 MiB). Adding
that owner solely for these contractions would change global resource planning.
A future borrowed-handle schedule may be worthwhile but must qualify lifetime,
workspace, full endpoints and a bounded fallback. Increasing point tiles alone
changes resource pressure without removing repeated coefficient work. Private
native mathematical loops would split compiler scientific ownership.

## Evidence

49 host compact-factor/contraction/response/geometric derivative tests pass.
The new factor test uses independent dense matrix algebra, arbitrary AO jets,
signed weights and signed coefficients for LDA/GGA/meta-GGA. Native tests add
Cartesian and spherical f-shell shapes above the 16-AO threshold, partial AO
and point blocks, all three semilocal families and both spin modes, exact arena
canaries and CUDA Graph execution. Slurm 11380 passes the allocated dedicated native matrix schedule suite,
including variational finite differences and failure isolation. Slurm 11373
passes 26 composed direct/DF KS energy/resource cases (two expected skips).

Slurm 11376, bounded two unconverged PBE48 iterations, retains 5,184 grid tiles
per iteration / 10,368 contraction calls. Compared with Slurm 11350, density
contraction aggregate is 1.701→0.646 s, potential contraction 4.132→1.636 s;
new panel formation remains additional work. Two SCF iterations take
11.754→8.101 s. Both profiles include instrumentation overhead and preparation
kernels in aggregate percentages; no converged endpoint claim follows from
these diagnostics. Preparation is 0.974 s with the separate #1104 overlay.

An older composed integration fails the pre-existing zero-beta r2SCAN tail
case at n=2 (Slurm 11372), below this schedule's admission threshold. That base
predates merged #1054. The independent candidate is based on master 959b5612
and includes #1054; its full original native suite reaches a finite but unequal
zero-beta r2SCAN V element, 2.410941123885864 versus 2.4109061845583866
(Slurm 11374/11377). The new matrix cases pass before that unchanged n=2
fallback case. Keep the original full suite and gate intact; the dedicated
matrix CTest also exposes new schedule coverage independently. The baseline with no tiling changes reproduces the identical discrepancy
(Slurm 11382, head 086acada), so this numerical boundary is separately tracked.
Do not misattribute it to tiling.

## Revisit when

A reusable shared BLAS owner can preserve exact budgets, or endpoint profiling
shows that register blocking, tile shape or another contraction family wins
without changing the admitted mathematical work.

## Composed complete endpoint qualification

Slurm 11381, water24 / 192 spherical def2-SVP AOs / identical grid and FP64
settings, two warm repeats, all four native/reference energy pairs admitted:

| Path | Prior cold execute s | Candidate cold execute s | Candidate preparation s | Candidate warm s | Max error Eh |
| --- | ---: | ---: | ---: | --- | ---: |
| Direct PBE | 27.1023 | 21.9398 | 0.3845 | 2.7771 / 2.7904 | 1.103e-11 |
| DF PBE | 17.6167 | 12.3651 | 0.5953 | 1.5922 / 1.6056 | 3.752e-11 |

Prior baseline is the preserved v9 integration (Slurm 11363/11366); candidate
v10 adds only the XC contraction overlay to that integration. Both retain
preceding setup/grid/source repairs and are not standalone PR-head results.
Neither grid, convergence tolerances nor density-fitting settings were changed.
Both cold solves retain 16 iterations. These improvements qualify small full
energy endpoints, not full 48/96 endpoints or force performance.

Slurm 11385 additionally passes signed LDA/PBE density-response actions against
two independently rebuilt CPU potential differences (1e-7 absolute gate), with
nontrivial spin coupling and partial point/AO blocks. This directly checks that
potential prepacking waits until directional features have consumed work.

Full water48 / 384-AO PBE qualification, all four reference pairs admitted:

| Path / Slurm | Preparation s | Cold execute s / iterations | Warm samples s | Max error Eh |
| --- | ---: | --- | --- | ---: |
| Direct / 11384 | 0.952830119 | 83.546013499 / 21 | 11.924645068 / 11.970010264 | 7.1623e-11 |
| DF / 11386 | 3.165835626 | 58.986216210 / 21 | 8.428240424 / 8.431483759 | 4.3656e-11 |

Every preparation/solve retained the 120 s watchdog. The former direct water48
cold solve exceeded that limit; these candidates finish and pass accuracy. No
full 96-atom SCF claim is made. Preserved v10 binary SHA-256:
`36b15fc1e9aba3f76c710132195978da1d8eaefc2c48829c782303b440a32873`;
source-overlay archive SHA-256:
`2baf846f1a06e658f7406e86965e5a6526e9e7d99e8e37485d720c140e92b1eb`.
The independently reproduced n=2 r2SCAN tail discrepancy is issue #1105.

## Integration with updated master

Merged master 41834864, preserving the new FP32-compute AO fixture option
and the tiled response fixture's explicit layout constructor. The response
layout forwards AO precision so the existing unsupported-mixed-response gate
still applies. Slurm 11395 passes the dedicated matrix schedule suite on this
combined tree. The old n=2 r²SCAN tail is independently repaired by #1108;
full native acceptance still depends on that source-level repair.
