# Decision: compiler-generated CUDA molecular quadrature

Status: implemented; native and small endpoint qualification complete
Date: 2026-09-23

## Problem

Issue #1099: CUDA KS preparation unconditionally invoked the scalar reference
`MolecularGrid` constructor. Per grid point, every atom pair recomputed its
separation and both point-center distances. On the benchmark water geometry,
12 atoms / 331,776 points took 2.960611203 seconds and 24 atoms / 663,552 points
took 25.375294054 seconds in this constructor alone. PBE48 was stopped after
roughly 160 seconds during preparation, before SCF timing began. These host
component diagnostics are not CPU benchmark results or SCF convergence failures.

## Decision

The compiler's XC package owns coordinates, distance reuse, Becke scalar lowering,
log normalization, tile width and allocation layout. Native code owns shared
Gauss-Legendre rule input, allocation, launches, synchronization and host export.
The Becke primal comes from the existing `grid_response_program` Graph, not a
second polynomial implementation. The ordinary CPU constructor stays independent.

For A atoms and P points, geometry computes A(A-1)/2 nontrivial distances once;
point-center distance evaluations fall from P*A*(A-1) to P*A. Each point/atom
worker visits other atoms in original contribution order. The original pair
orientation preserves `log(pair)` versus `log1p(-pair)` tails. This evaluates
P*A*(A-1) switches instead of the reference's P*A*(A-1)/2: the deliberate doubling
avoids a point-by-pair tensor, atomics, and large per-thread arrays. Log work is
unchanged at P*A*(A-1). Overall unscreened partition work remains cubic when P
scales with A; this change removes repeated expensive geometry and serial work,
not the underlying scientific pair census.

Scratch uses at most T=4096 points, with double arrays of 4*A + 1536 + A*A +
2*T*A + 4*T elements, plus one integer error flag. The compiler emits the shape
query, including overflow gates before allocation. The public resource planner
adds the Fock provider's retained bytes to this setup phase; KS/XC state does
not yet coexist. All actual allocations use the common tracked allocator.

## Invariants and retained compatibility

- Preserve atom/radial/polar/azimuth order, both grid versions, resolved radii,
  coincidence tolerance, iteration count, normalized ownership and exports.
- No CPU partition fallback in the production CUDA factory.
- No unsafe algebraic reversal of pair orientation or relaxed fast math.
- Retain host grid arrays for current snapshot/derivative APIs. GPU-to-host
  tile downloads and subsequent XC upload remain visible endpoint work.
- Keep the KS v1 three-output allocation bridge unchanged; quadrature adds its
  own shape-only bridge and old libraries fail closed for the new inventory.

## Rejected alternatives

Caching only distances in the serial constructor would retain a production
reference execution dependency and serialized cubic partition work. A dense
point-by-pair log tensor uses quadratic per-tile atom storage. A private duplicate
Becke polynomial would split scientific ownership from the derivative compiler.
A device-resident grid ABI can remove staging later but is not required to fix
this preparation cliff and requires separate ownership/derivative export work.

## Evidence

Slurm 11343: standalone allocated native CUDA test passed independent scalar
weights/coordinates, versions 1/2, iterations 1–5, partial tiles, radii/fallback,
single/near-coincident/reordered/translated atoms, contracted derivative exports,
96-atom small grids, analytic equal-coincident ownership, exact allocation peak,
one-byte-short OOM cleanup and shape overflow. The CUDA path is compiled with
FMA contraction disabled. Full library/endpoint gates are still pending.

Do not infer complete SCF improvement from constructor timing. The existing
host orthogonalizer, Fock build and other preparation work remain separate costs.

## Revisit when

Device-resident grid ownership can be shared by XC and derivative snapshots;
larger systems justify symmetry-shared pair evaluation without unbounded storage;
or independently qualified screening can reduce the actual partition census.

Slurm 11344: standalone GPU factory shape probes (synthetic O/H center lattice,
v1 one-Bohr radii, 54×16×32 points/atom) took 0.118630423 / 0.150758124 /
0.412826299 / 2.067071713 seconds at 12/24/48/96 atoms. Device scratch was
931,332 / 1,721,604 / 3,309,060 / 6,511,620 bytes. These are component diagnostics,
with different geometry/radii from the historical water baseline; no matched
speedup or complete SCF performance is inferred.

The CMake-only CI environment exposed an import dependency in the existing
response consumer: importing its scalar IR also imported NumPy. Scalar program
construction now lives in `grid_response_ir.py`, with lazy array evaluation and
canonical re-exports from the existing consumer. Bare `python -S` generation is
byte-identical (SHA-256 `7ea4e4803d62918553f556b4c96db8014947ac3debccc27b0c9b6b4f2c9a08e5`).
39 host IR/response/native-contraction tests pass. The endpoint harness now arms
its existing independent-process watchdog around `native/prepare` as well as
SCF solves, retaining a preparation-stage journal on timeout.

## Composed endpoint qualification and remaining blockers

The combined integration (not an exact PR head) has native binary SHA-256
`5b2d2189af086f4a001795e1e7c2b946e9d5329f3af6e887218adb107ebd7475` and retained
source-overlay archive SHA-256
`6975ceaa04c9df83552ec8dae6915fb9ae55d9c60a0fb2c2d3be46f0580f86a5`.
Slurm 11346 passes the allocated native quadrature suite plus 9 Python
RKS/UKS ordinary-solver/resource cases (cold/warm/geometry/final-state, exact
budget, failed preparation and release).

Slurm 11347, same water24 / 192-AO direct PBE benchmark: preparation takes
2.431443798 s, cold execute 27.261958854 s (16 iterations), warm samples
3.435724267 / 3.455501552 s. Maximum energy error over all four recorded
native/reference pairs is 1.728039933e-11 Eh. The earlier integration baseline
has preparation 27.836176385 s and cold execute 27.248856407 s. Thus the
observed prepare-plus-cold total changes from 55.0850 s to 29.6934 s, while SCF
execution time is essentially unchanged. This comparison is composed evidence;
other repair overlays are present, and it is not a standalone-head ablation.

Slurm 11348, water48: preparation completes in 26.609609028 s, then the cold SCF
solve hits the 120 s watchdog and is stopped. The progress wrapper's 26.906 s
includes watchdog/event overhead; use the inner timer for preparation. No
converged 48-atom endpoint or 96-atom full endpoint is claimed. Full README
benchmark publication remains paused.

The remaining setup cost is independently identified by the existing reference
observer (Slurm 11349): overlap orthogonalization 14.0008 s and core guess
11.6858 s within 26.5435 s preparation. Track their default CPU Jacobi calls in
#1101. A bounded two-step Nsight diagnostic (Slurm 11350) attributes 48% of
captured GPU kernel time to dense XC density/potential scalar contractions;
track compiler Tensor/BLAS lowering in #1102. These distinct costs must not be
misattributed to quadrature or relaxed by changing convergence/grid thresholds.

Slurm 11365 qualifies exact code head 91b22a87: allocated native quadrature
scalar/analytic/resource/derivative suite passes, together with 36 Python
energy-diagnostic/resource checks (one expected skip). The diagnostics select
energy explicitly; implicit default forces exercise a separate consumer and
are not claimed here. This head does not contain the composed setup or XC fixes.
All PR CI jobs pass, including NVIDIA compilation and CuMetal GPU tests.

## Integrated review qualification

After integrating reviewed DF scheduling and master `41834864`, Slurm 11400
passes the independent native quadrature suite, full native LDA/PBE/r2SCAN
RKS/UKS E/V and state suite, and full native KS suite. An additional 25 ordinary
solver/resource cases and eight public complete-force/replay/changed-geometry
cases pass without skips. These retain independent PySCF energy/gradient
references and the existing exact-budget, failure isolation and cleanup gates.
The 39 host quadrature/response/native-contraction cases and six watchdog tests
also pass, as do the full pre-commit and compiler ownership checks.

The tested library embeds source identity
`334dd4cf28a8c0727c1a9e101732c1ca56e88e918f9e62a7f785ccdfcebb6c04`
and has SHA-256
`3039622e137d0fbbc0e44d2b99f3ae9ecc91678f32f7c828e5ccda4e6bb4a10c`.
This extends integration correctness and resource evidence. It does not add a
large-system timing claim or change the earlier composed-benchmark limitations.
