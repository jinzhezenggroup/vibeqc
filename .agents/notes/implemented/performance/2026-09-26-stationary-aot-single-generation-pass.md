# Decision: Generate the stationary component AOT inventory once per build

Status: implemented
Date: 2026-09-26

## Problem

The component AOT build launched 23 independent Python processes, each requesting
one shard. Each process called `derivative_cuda_sources` for all 362 requests and
discarded 22 of the 23 resulting translation units. In-process caches could not
share that work across CMake commands: a clean build emitted 529 translation units
and lowered 8,326 requests to retain just 23 units covering 362 requests.

## Decision

Give all 23 primitive source outputs to one CMake custom command, and add
`--all-shards` to the generator. That invocation lowers the canonical inventory
once and writes every shard with the existing `write_if_changed` helper. The
generated-source runner records dependencies for every output in its depfile.

Keep the canonical full-inventory emitter, its per-unit and aggregate source
budgets, and its shard-count/width contract checks. Keep the existing single-file
CLI modes compatible, but do not use `--shard-index` for CMake generation. Reject
invalid shard indices before starting expensive lowering.

## Alternatives

Emitting only the selected requests in each process could also eliminate redundant
lowering. The shared invocation reuses the existing inventory implementation and
aggregate size validation, with no additional compiler API or scheduling logic.
An in-process cache alone cannot fix independent generator processes.

## Invariants

- Preserve every shard filename, symbol, request order, and generated source byte.
- Preserve the 23-shard, 16-request-width artifact contract, with 10 requests in
  the final shard.
- Preserve generation without a GPU, runtime library, or reference oracle.
- Leave unchanged output timestamps intact, and regenerate if any output is
  missing.

## Evidence

`tests/python/test_stationary_aot_generation.py` executes the actual CMake
registration and generator with only the primitive CUDA emitter stubbed. Both
Ninja and Unix Makefiles produce 23 emissions totaling 362 requests on a clean
parallel build, zero additional emissions on an unchanged rebuild, and one
complete inventory pass when a secondary output is deleted. CLI checks cover
output preservation and rejection of conflicting arguments before lowering.
Running the same build-graph regression against the original CMake registration
failed as expected, recording 529 emissions and 8,326 request lowerings.

A fresh-process comparison against the generator at `644d6f9f` found all 23
source files byte-identical (22,993,773 bytes total). The old shard-0 invocation,
including exporting the cached reference inventory, took 62.10 seconds; the new
`--component-domain spd --all-shards --output <directory>` invocation wrote all
23 files in 62.17 seconds. These are generator-process timings on this host,
not complete CUDA build timings; compilation, linking, and GPU execution were
not rerun for this source-preserving build-rule change.

The focused generator, AOT dependency/admission/package/precision, derivative
schedule, and stationary CUDA lowering suite passed all 79 tests. The compiler
structure audit reported 355 modules and zero dependency errors.

## Consequences and revisit conditions

A missing individual shard regenerates the full inventory once, preserving
unchanged files for object-cache reuse. Revisit selected-request emission if
partial-inventory regeneration becomes a measured bottleneck and an independent
aggregate-size check is retained. The change concerns build-time generation;
it makes no GPU execution or complete build wall-time speedup claim.

## References

- PR #1387
- `cmake/VibeQCCuda.cmake`
- `tools/generate_stationary_force_aot.py`
- `.agents/notes/proposed/2026-09-25-stationary-component-aot-packaging.md`
