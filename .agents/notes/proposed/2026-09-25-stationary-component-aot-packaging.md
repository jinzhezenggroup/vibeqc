# Decision: Package component-expanded stationary CUDA derivatives as sharded AOT

Status: proposed
Date: 2026-09-25

## Problem

The qualified public stationary CUDA force path can use packaged AOT for the existing
s/p Cartesian domain, but spherical d-shell expansion requires the full s/p/d
first-derivative component inventory. On the current production path this component
mode falls back to runtime scientific CUDA compilation even when an installed AOT
directory is available. That violates the #1186 production requirement that admitted
stationary execution not require runtime scientific compilation.

The full s/p/d derivative inventory is too large to treat as one translation unit.
Its existing compiler-owned first-derivative schedule already defines a bounded,
canonical sharding of the exact derivative mathematics.

## Decision

Package the s/p/d primitive derivative inventory as a fixed set of shared CUDA
relocatable-object shards and link those objects into small functional/spin-specific
stationary wrappers. Keep the original s/p AOT v2 artifacts unchanged. Introduce a
separate v3 manifest/contract for the component-expanded artifacts.

Build-time tooling owns composition of the integral derivative schedule with the
stationary method wrapper. The method compiler module records only the qualified
component domain and bounded shard contract; it does not import the integral
schedule. This preserves the repository compiler ownership rule that method modules
must not depend on integral modules while still binding the installed artifact to
the generated source inventory at package time.

Runtime loading verifies the plan, component domain, shard count/width, target code
objects, source/contract identity, binary hash, and FP64 compile contract before the
artifact is admitted. Missing or mismatched component AOT fails closed rather than
silently invoking a runtime compiler when the public installed all-electron path has
selected AOT execution.

## Rejected alternatives

- Keep compiling component-expanded derivative CUDA at runtime. This preserves the
  current behavior but does not satisfy the no-runtime-scientific-compilation
  production requirement.
- Import the integral derivative schedule from
  `vibeqc_compiler.method.stationary_cuda`. The compiler structure audit rejects
  the method-to-integral dependency and it couples method composition to integral
  schedule ownership.
- Emit one monolithic s/p/d primitive translation unit. The existing schedule is
  intentionally sharded to bound generated-source and compiler resource costs.
- Replace the existing s/p v2 package contract. That would widen compatibility risk
  for already-qualified installed artifacts without being needed for d-shell
  support.

## Invariants

- Existing s/p v2 artifact names and loader semantics remain compatible.
- Primitive scientific mathematics continues to come from the canonical
  first-derivative compiler; the new packaging layer does not reimplement it.
- Only the full qualified s/p/d component domain is admitted by the new v3 path.
- ECP and other paths that still require runtime generation retain explicit
  fallback behavior; this change does not silently widen their capability.
- Scientific tolerances, force acceptance, work/resource caps, and public method
  admission are unchanged.
- Installed artifact identity remains deterministic and contains no checkout path,
  timestamp, or transient compiler state.

## Evidence

Current source-level evidence on the implementation worktree:

- `python tools/check_compiler_structure.py`: 355 modules, 0 dependency errors.
- `tests/python/test_stationary_aot_no_compiler.py`: 13 passed, including a
  spherical d-shell public-force negative control that fails if NVCC discovery is
  attempted before component AOT loading.
- `git diff --check`: clean.

CUDA build/device and complete endpoint evidence remain required before this
proposal can be considered implemented or production-qualified.

## Consequences

Installed CUDA builds gain additional generated primitive shards and six
functional/spin component-expanded wrapper artifacts, increasing build and package
size. Runtime execution gains a deterministic installed-artifact path for admitted
d-shell stationary forces and avoids runtime scientific compilation for that path.

## Revisit when

Revisit if the canonical derivative schedule changes its bounded shard inventory,
if a compiler-owned multi-entry artifact format can reduce package/build cost
without weakening identity, or if a different generated representation proves
materially cheaper under complete endpoint resource measurements.

## References

- #1186
- #1191
- merged #1146
- `python/vibeqc_compiler/integral/first_derivative_schedule.py`
- `python/vibeqc_compiler/method/stationary_cuda.py`
