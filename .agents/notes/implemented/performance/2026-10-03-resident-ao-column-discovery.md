# Decision: expose bounded AO-only discovery on resident point tiles

Status: implemented (explicit producer; no automatic mask/cache policy)
Date: 2026-10-03

## Problem

After admitting explicit AO maps on resident grid/geometry leases, a caller
still needs to discover and retain a scientifically qualified map. The initial
complete-endpoint experiment discovered masks by borrowing a full feature tile;
that unnecessarily includes dense density contractions in cold discovery.
CPU axis-aligned region envelopes can also be loose for complete outer radial
shells whose boxes contain the molecule.

## Decision and invariants

Add optional `grid_cuda_select_ao_device_v1` and
`CudaGrid.select_ao_device_points`. Evaluate all configured AO jets on one
resident immutable point tile, then retain a column if any magnitude exceeds
the caller's explicit finite positive threshold. This computes a sampled-jet
mask; it is neither a region envelope nor an energy/force error certificate.
No threshold, automatic cache, or production selection policy is enabled.

The native routine requires a full-capacity local grid owner, invalidates the
previous task generation, and borrows its charged AO-map buffer for integer
flags. Its deterministic integer OR reduction uses no floating point sum and
publishes at most one flag per AO per point block. Adjacent CUDA lanes read
adjacent AO columns. Every configured jet and actual point participates.
Nonfinite AO output fails; it is never hidden by omission. A valid later call
clears the old device error. Density state remains untouched.

Discovery performs no density contraction and retains no new device allocation.
It downloads one flag per AO plus the error scalar. Python returns immutable,
sorted indices. The caller owns mask retention, budget, geometry and derivative
identity, point readiness/lifetime, and complete scientific qualification.
Discovery synchronizes before returning and cannot run during a borrowed lease.
Old native artifact libraries keep their explicit-map/identity methods and
report a capability miss if this optional producer is requested.

## Evidence

On n1 RTX 5090 under finite Slurm, job 5547 passed 21 independent tests in
7.47 s, then the same 21 under memcheck in 8.64 s with zero errors. CPU fixture
AO jets cover water, Cartesian/spherical f shells, tight/diffuse exponents,
orders 0 through 3, short/partial point tiles, four thresholds, immutable output,
stale-view invalidation, failed-input recovery, and rejection during a lease.
The first attempt, job 5546, passed all 20 numerical cases but expected the word
"borrowed" instead of the existing "leased" error text; its failed log remains.

Job 5549 scanned every order-2 AO on full 48 x 16 x 32 def2-SVP spherical water
grids, at 1024 points/tile and cutoff `1e-16`. Every selected ID matched the
retained previous full-jet scan. Native producer host-wall totals were 0.342 s
(24 atoms), 1.166 s (48), and 4.129 s (96). These are single diagnostic component
observations, not complete endpoint speedups. Actual AO-jet values visited were
1,132,462,080; 4,529,848,320; and 18,119,393,280 respectively. Discovery density
contractions were zero. Returned mask numeric bytes were 594,016; 1,408,624;
and 3,490,752. Device allocation stayed exactly at the existing owner plan.

The isolated diagnostic uploads constructed point tiles to supply resident
inputs; those 3 x G x 8 host-to-device bytes are reported separately. A 1 GiB
diagnostic grid allowance does not change production memory defaults. The
producer was JIT compiled from current native source; the frozen library
`0c5b5f67...` supplies unchanged basis support. No fresh whole-library or
complete-endpoint qualification is implied by this component gate.

Exact source archives, tests, raw results, compiler metadata and failed attempts
are retained in ignored `.artifacts/resident-ao-masks/` and its same-named remote
bundle. The scan's JIT library SHA-256 is
`0fdb24d4e5649673fb9b411c1bdca4b03f43663cd10fdae9276d566bda962ae5`.

## Next boundary

Charge retained maps and discovery costs, bind cache identity to immutable basis,
geometry, grid order and derivative domain, retain a dense fallback for budget
or capability misses, and independently qualify complete energy/forces and moved
geometries before any automatic screening policy is admitted. The SCF grid
consumer requires its own local potential/scatter integration.

Review follow-up: the initial fixture loop prepared capacity 129 but its actual
stored grids were smaller than 128 points. Job 5563 adds twelve independent
cases with actual 129/257-point inputs and >32, nonmultiple-of-32 AO columns.
Three copies of the stored Cartesian/spherical f-shell fixtures repeat their
independently recorded columns exactly. The first 128/256 points are a quiet
fixture row; only the final point exceeds the chosen column's threshold. The
third AO copy crosses the AO-block boundary, so both the late point block and
AO tail must contribute. Removing that final point removes the designated AO.
Orders 0, 2 and 3 pass exact ID comparisons with the stored oracle. All 33
producer tests pass normally (7.73 s) and under memcheck (8.70 s, zero errors).
This repairs the independent cross-block/tail coverage claim without changing
producer code, cutoffs or endpoint policy.
