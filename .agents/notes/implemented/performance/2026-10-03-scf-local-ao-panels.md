# Decision: admit explicit local AO domains in native SCF grid contraction

Status: implemented (explicit internal API; no automatic screening policy)
Date: 2026-10-03

## Decision

`CudaXcAoTiles` supplies immutable CSR maps for the exact point-tile sequence.
`cuda_xc_local_ao_layout` validates sorted unique global indices, empty tiles,
and complete offsets before charging device indices and retained host offsets.
`CudaXcPlan` copies metadata during setup and retains no caller host pointers.
Full AO scratch capacity remains unchanged; no mean-count capacity assumption
or additional density/potential matrix is introduced.

The compiler-owned scalar and tiled density kernels gather D[I,I] directly
from the global density into their existing contractions. AO collocation and
feature work use local column strides. Potential kernels scatter the unique
local triangle into the global matrix on the owner's serialized stream. Local
execution clears the complete output once because the first map need not cover
all matrix entries; dense execution retains its previous first-tile
initialization. Empty tiles publish zero density features and point totals.

Nonlocal AO assembly uses the same local maps and compact bilinear panels,
without consuming semilocal point totals. The nonlocal energy is still added
once. Full-density validation remains global, so invalid values outside a map
are not hidden. Local maps admit physical FP64 execution only; mixed arithmetic
and response retain their existing dense paths. There is no cutoff, discovery,
cache or selection default in this API.

## Validation and boundaries

The component executable tests Cartesian/spherical through-f bases, RKS/UKS,
7- and 19-point tiles with partial tails, full/sparse/empty maps, changing
independent density matrices, exact canary-bounded allocations, malformed maps,
nonlocal assembly and graph replay. The CPU oracle evaluates full AO jets and
zeros omitted columns point by point before independent full AO-pair feature
and potential contraction; it does not share the local/tiled GPU schedule.

n1 Slurm job 5554 passed the local component gates. Job 5555 passed those same
gates under both memcheck and synccheck, with zero errors. Subsequent normal
CMake-built component executables also passed the local gates. There is no
complete endpoint speedup claim or automatic map policy in this slice.

Running the previously unlinked full native test target exposed three older
test issues: its missing CUDA quadrature object/dependency; an allocation
assertion that counted the shared owner's extra atomic weights as XC-owned;
and a B3LYP mixed-density success case despite its explicit capability rejection.
The build target and those two assertions now match the existing contracts.
The complete regression then fails its original B3LYP empty-spin 49,152-point
potential gate: -0.06745407700233734 versus -0.067454077066442977, a
6.4105637465061704e-11 difference. A separately built pre-local-AO executor
(c3389e701 sources/objects, job 5561) reproduces exactly the same values and
failure. That gate remains unchanged and the complete regression is not
reported as passing. Targeted WB97M-V qualification is recorded separately.

Ignored `.artifacts/local-ao/` retains source patches, original link failure,
all failed full regressions, baseline reproduction, binaries, ccache commands
and counts. The first component link explicitly supplied the missing unchanged
quadrature object; later builds use the corrected CMake target. The composition
also applies the separately diagnosed bounded-integral claim barrier before any
new complete endpoint qualification. Component AO tests do not use that integral
path.

## Next boundary

The SCF preparation owner must budget and supply scientifically qualified maps,
then expose their actual work counts and lifetime. Force reuse additionally
requires order-2 jet coverage and identical point ordering. Complete cold, warm,
changed-geometry and constrained-budget energy/force checks are still required
before promoting a native selection policy.

Follow-up: job 5562 exited zero after the normal CMake-built executable passed
`--wb97mv` (local-AO cases, existing matrix schedules, and dense WB97M-V
RKS/UKS E/V/tail/state gates), then `--local-ao` under memcheck with zero errors.
Sixty-seven focused host compiler/admission tests passed. The loaded native
library SHA-256 is `3db5e3d9ed879529d30f8a05876a0d3703e48f765298b9aa7ada9fb1b486fce6`;
its `873e1bfe...` source identity was independently recomputed from all 1,349
manifest inputs and matched the loaded library. This does not supersede the
preserved unrelated full-regression failure or establish endpoint speed.
