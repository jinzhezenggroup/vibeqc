# Decision: bind bounded AO maps to geometry and derivative domains

Status: implemented (private explicit owner; no automatic caller/default policy)
Date: 2026-10-03

## Decision

`ResidentAoMapCache` retains immutable selected AO maps for one exact
basis/geometry/grid/device/point-allocation/tile/order domain. Density changes
are deliberately absent from the key: discovery depends on AO jets alone.
Callers must validate the native molecular-grid token before entering this
owner and must explicitly choose and scientifically qualify a positive cutoff.
An order-1 map cannot be relabelled for an order-2 force consumer.

Pointer equality supplements scientific identities; it never substitutes for
them. `CudaGrid._rebind_centers` previously left basis identity unchanged. A new
read-only `geometry_generation` therefore increments before every validated
native center update, including one that fails. Both cache hits and discovery
recheck that epoch under the grid's lease lock. Changed geometry requires a new
cache owner; a failed/malformed binding is not a dense availability fallback.

The additional numeric host budget covers retained indices plus a conservative
20 bytes/global AO transient reserve for flag/index/immutable-output staging.
Admission also reserves the next worst-case returned map before discovery, so
retained capacity plus the transient reserve stays within the stated budget.
Full identity maps and empty maps retain no numeric indices. Python container
headers remain an explicit exclusion, as with the existing grid resource plan.
Full AO device scratch is separately charged to the existing CudaGrid; neither
mean retained AO counts nor a GM-squared work ratio can reduce its capacity.

Budget or optional capability misses return the full AO map. Numerical/device
failures propagate. The owner reports discovery work/time, zero discovery
D contractions, selected and dense point/AO-square work, tile/empty counts,
minimum/maximum/summed active AOs, and exact retained numeric bytes. Counters
reset per endpoint while immutable geometry maps survive warm calls.

## Validation

Twenty host protocol cases cover exact-key rejection, center/basis epochs,
closed owners, partial tails, immutable masks and policy fields, empty/full
maps, bounded retention, explicit capability/budget fallbacks, and error recovery.
On n1 RTX 5090 through finite Slurm, job 5559 passed all 42 combined cases
(22 GPU AO/cache and 20 host protocol) in 7.39 s, then all 42 under memcheck in
8.78 s with zero errors. The added GPU case checks the original mask against
independent stored AO fixtures, moves centers while preserving the point
pointer, rejects the old cache, and compares a newly discovered map with the
independent CPU AO evaluator at the moved geometry. Orders 0--3 and the existing
Cartesian/spherical f, tight/diffuse, error and lifetime gates also pass.

The first test attempt omitted the required explicit full-capacity local-grid
allocation; admission correctly rejected it. The second launcher used an
incorrect remote directory and did not execute tests. These failed receipts
are retained beside the successful source-v3 archive in ignored
`.artifacts/resident-map-cache/` and the same remote bundle. Qualification uses
the JIT AO producer with the previously qualified native library for unchanged
basis support; no new complete-endpoint claim follows from this component gate.

## Remaining qualification boundary

Callers must charge this entire cache allowance within their enclosing host
budget, rebuild on domain changes, preserve dense fallback, and retain the
existing energy/force acceptance gates. Production default promotion still
requires complete cold/warm/moved-geometry and constrained-budget evidence.
