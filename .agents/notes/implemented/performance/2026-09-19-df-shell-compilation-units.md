# Decision: independent DF shell compilation units

Status: implemented
Date: 2026-09-19

## Problem

Issue #522 identified a long final CUDA compilation step despite 154/157
ccache hits on master `5a7fdeb2689553c0a304dad3338ba184d850ef60`.
The monolithic DF derivative owner instantiated all 64 angular classes and
three schedules, both supported mathematical lowerings and panel/packet
consumers. Editing a production policy invalidated the same expensive object
as editing numerical code. A high object-count hit rate hid that cost.

## Decision

The existing scientific emitters can emit a canonical class subset without
changing their default output. The compiler emits 64 stable CUDA launch units,
class-local mathematical headers, and a small registry. Native shell execution
and diagnostic templates remain native runtime code; they are moved, not
rewritten as another scientific implementation. All numerical class units keep
the normal optimizer, including in compile-fast CI.

Production-manifest selection stays in the light dispatcher. The registry and
class units do not contain policy bytes. New Rys capability for a class changes
its local math/capability headers rather than all numerical objects. Shared
primitive math, ABI or execution-template changes still invalidate every
consumer that really depends on them.

Subset capability constants have internal linkage. A subset does not redefine
the full external `for_each_class` template. This avoids inconsistent inline
objects/templates across independently compiled translation units.

## Rejected alternatives

- Increasing build parallelism alone cannot parallelize the former final nvcc
  process. Stable independent units expose real work to the existing bounded
  two-job pool instead of assuming more CPU or memory.
- Fast compilation remains unsuitable for these numerical kernels: NVCC 12.9
  can over-allocate shared memory. Neither a lower optimizer level nor removed
  schedules/fallbacks is an acceptable build-time shortcut.
- Compiling every class against the complete generated shell header would
  retain broad cache invalidation when just one mathematical class changes.
- Policy-based pruning would couple class binaries to promotion metadata and
  remove explicit comparison/fallback paths. All supported variants remain.

## Invariants and validation

The three public launch APIs, class order, full/prototype domains, error
propagation, symmetry/packing, screening, diagnostics and numerical templates
are preserved. Tests compile the actual thin dispatcher with fake device
queries, check emitted class/schedule definitions against the full emitter,
and prove a policy-only edit preserves class-file bytes and modification times.
An edit local to auxiliary-f class 003 changes only that class's math header.

The original full polynomial, Rys, and capability emitter outputs were checked
byte-for-byte against the pinned base before compiling. Independent existing
DF/Rys mathematical and CUDA execution tests remain mandatory. Build timings
must distinguish all-source cold builds, class-specific misses, policy-only
changes, and warm no-op builds. The fast CI library is never runtime performance
qualification evidence.

## Consequences and revisit conditions

More small compiler invocations increase fixed parse/startup overhead and can
increase fully cold CPU work; do not promise a wall-time reduction without a
matched measurement. The purpose is to remove the serial tail and make the
common incremental cases local. Revisit grouping only with matched cold and
incremental timing, memory/resource reports, and preserved independent kernels.
Do not claim all CUDA build latency is fixed: packaging and unrelated kernels
have separate costs.

Refs #522, #349, #353, #444, #435.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Local qualification observations

A CUDA 12.9.86 / GCC 11.4 / sm_120 real+PTX build on an AMD EPYC 7K62
measured the original full-optimization monolith at 752.705 seconds. The 64
independent class objects took a 382.326-second span with the existing two-job
pool, with a maximum individual class job of 17.470 seconds. These are DF-owner
compilation measurements, not whole-CI durations, repeated statistical estimates
or a promised time on GitHub runners. Other CPU validation work overlapped on
the 48-core host. The initial complete library and two native-test targets took
635.84 seconds and included unrelated objects.

The 534 panel/packet kernel identities are preserved. Cuobjdump reports identical
static shared, stack and local-memory usage for every corresponding kernel;
maximum static shared storage remains 40,464 bytes. NVCC nevertheless makes
other code-generation decisions after partitioning: 330 register counts decrease,
and none increase. Identical source mathematics is not proof of identical SASS.

The controlled same-rest-of-library old-object substitution has a raw DSO size
of 183,557,096 bytes versus 319,000,360 for the class-unit library. Conversely,
gzip level 6 payload size changes from 85,870,743 to 73,463,199 bytes. Neither
raw size nor register count alone establishes runtime performance. Keep this
tradeoff visible when qualifying cold and prepared execution; the substituted
comparison library is a local test artifact, not a source-identity-qualified
release. No release or distribution was published.

The original native shell-pair test assumed small AO domains could not select
Rys. The old monolith reproduces that assertion failure after dimension-only
admission was retired. The updated test runs both explicit legacy and automatic
policies, genuinely exercises all three explicit schedules, and conserves Boys
plus Rys primitive work and the independent Rys root count. Both native GPU
targets pass under Slurm on the 5090. The obsolete zero-Rys assertion was not
fixed by changing production mathematics or restoring the retired AO whitelist.

A final-tree incremental check disables ccache for the mutation arms. Changing
only the production manifest's qualification flag rebuilds exactly the light
DF dispatcher and the source-identity owner in 4.801 seconds, with no numerical
class objects changed. A class-003-only generated-header edit rebuilds exactly
that numerical object in 13.021 seconds. Restoring the original inputs and
rebuilding leaves a 0.183-second no-op build with no changed objects. Both
mutation tests restore their original inputs in `finally` blocks and finish
with a clean tracked worktree. This measures actual object invalidation, not
just unchanged generated-source hashes or a reported cache hit percentage.
