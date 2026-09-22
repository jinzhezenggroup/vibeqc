# Decision: compiler-generated CUDA molecular quadrature

Status: implemented; complete endpoint qualification in progress
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
