# Spatial AO tasks

`vibeqc_compiler.dft.spatial` separates deterministic geometric regions from
hardware tile sizes. A task retains source point IDs and bounds, sorted active
shell/AO maps, requested ordinary spatial derivatives, discarded-AO envelope
diagnostics, and a generation identity binding basis, geometry, quadrature and
policy. Every source point occurs exactly once, with its original weight and
owner. Regions use a median split of the widest coordinate; `region_points`
does not select a warp width or matrix-library shape.

`SpatialPolicy(screening="off")` changes only the evaluation order. The
`absolute_ao_jet` policy omits a column only when conservative region bounds on
**every** requested derivative are at most `cutoff`. Bounds include normalized
primitive coefficients, negative contraction coefficients and all public
Cartesian-to-spherical terms without relying on cancellation. Overflow retains
the AO; a zero value cannot hide a nonzero derivative. These bounds describe
individual AO derivatives, not density, XC energy or molecular-force error.

A screened task defines fixed approximated collocation: omitted AO jets are
zero on that task. The same mask is used for features, energy and matrix
derivatives. Changing geometry or region policy can change membership; this
does not define a smooth nuclear-motion branch. No complete SCF, nuclear force
or adaptive error-allocation capability follows from the AO cutoff.

## Explicit execution candidate

```python
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid

with PreparedSpatialGrid(
    basis, grid,
    policy=SpatialPolicy(region_points=128),
    tile_points=32,
    resource_budget=ResourceBudget(host_bytes=128 << 20),
) as prepared:
    for tile in prepared.iter_features(density):
        consume(tile.point_ids, tile.weights, tile.features)
```

The basis is an existing immutable `NativeAO`, owned by the caller. The grid
can be `ExplicitGrid` or a compatible `MolecularGrid`. Molecular preparation
materializes only O(points) coordinates/weights/owners after a shared-budget
preflight; it never creates a global point-by-AO table. `tile_points` changes
execution packing without changing task membership or the mathematical mask.
The first candidate consumes complete through-order jet domains, including all
first derivatives for density features. Descriptors also support subsets for
future consumers; execution rejects unsupported subsets explicitly.

CPU and CUDA evaluate selected AO records directly. Density gathering takes
the complete `D[I,I]` for each spin, including cross-shell terms. Total-density
input is split equally between spins using the existing density convention.
CPU fixed-density XC accepts `FixedDensityXC.integrate(..., spatial=prepared)`
and scatters each complete local matrix into the global AO domain. Functional
domain validation remains unchanged, including unsupported vacuum derivatives.

CUDA uses the same generated AO polynomial/feature policy for dense and local
schedules. `CudaGrid(..., active_ao_capacity=...)` reserves local jets, density,
work panels and local matrix buffers. Full global density **and** potential
matrices remain resident O(NAO²) costs. Native map/gather/scatter and library
operations are reusable runtime; remaining normalized AO traversal/layout glue
is still counted conservatively as scientific code in the ownership ledger.
The independent CPU evaluator and external numerical fixtures remain oracles.

## Native consumers and lifetime

`prepared.device_tasks(density)` is a context manager yielding serial
`(task, point_ids, lease)` entries. A CUDA consumer uses `lease.view`, whose ABI
is defined in `src/dft/grid_task_view.cuh`. AO jets and features remain on the
device. Consumers enqueue on the borrowed stream, write both spin local
potential matrices, then call `lease.scatter()`. The ordinary path downloads
only the scalar error status. Host local matrices and global matrix downloads
are explicit diagnostic options.

A lease expires before the next task, when its context exits, or when the
outer iterator closes. The owner rejects density replacement, reconfiguration,
nested tasks and closure during a lease; native scatter also checks the exact
generation. Raw pointers must not escape the lease. New density executions
invalidate older diagnostic iterators. Geometry/basis/quadrature replacements
construct and validate new tasks before replacing usable old state.

## Resource scope and selection

Construction metadata, quadrature, local host/device panels, global matrices,
gather/scatter buffers and the declared cuBLAS allowance compose through
`ResourceRequest` and `ResourceBudget`. Replacement checks include old and new
capacities concurrently; failed replacement leaves the old owner usable.
The implementation streams and recomputes tiles, retaining no molecular AO
cache. Object headers, allocator rounding, externally retained output tiles
and CUDA context/modules/stacks beyond declared allowances remain explicit
scope exclusions. Numeric capacity estimates are not measured process peaks.

Dense `PreparedGrid` remains an available baseline. Local packing is an
explicit candidate, with no automatic promotion or universal speedup claim.
Construction, AO evaluation, density gathering/contraction, scatter, transfers
and complete available endpoints all contribute to a selection decision.
