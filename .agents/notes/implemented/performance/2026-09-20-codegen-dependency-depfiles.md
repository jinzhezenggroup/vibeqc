# Decision: track codegen invalidation with import depfiles

Status: implemented
Date: 2026-09-20

## Problem

Build-time generators previously depended on the complete recursive
`python/vibeqc_compiler/*.py` set. Any compiler edit therefore invalidated every
generated family, even when the changed module was not loaded by that generator.
The complete source identity must remain broad for artifact compatibility, but
using that identity inventory as a rebuild dependency made incremental builds
unnecessarily expensive.

## Decision

Run CMake-registered Python generators through `tools/run_codegen.py`. After a
successful invocation, the runner records repository-local Python modules that
were actually loaded and publishes them as a Make/Ninja depfile. CMake keeps the
generator script, runner, manifests, parameter files and other explicit inputs as
ordinary dependencies.

The complete `VibeQCSourceIdentity.json` inventory remains unchanged in scope.
Rebuild dependency discovery and compatibility identity are deliberately
separate.

## Rejected alternatives

Maintaining handwritten per-generator Python dependency lists was rejected
because transitive compiler imports evolve frequently and such lists would
silently become stale. Keeping the recursive compiler glob was safe but preserved
the unnecessary fan-out this change is intended to remove.

## Invariants

- A loaded repository-local Python module invalidates the generator output when it changes.
- An unrelated compiler module does not rerun the generator merely because the
  complete source identity changed.
- Non-imported data inputs remain explicit CMake dependencies.
- Failed generators do not publish a new depfile.
- The compatibility/source identity remains complete and independent of depfile
  narrowing.

## Evidence

- `python -m pytest tests/python/test_cmake_ownership.py -q`: 6 passed in the
  implementation worktree.
- Real `vibeqc_xc_cpu_codegen` generation recorded 9 repository-local Python
  dependencies in its depfile on node3.
- Touching the loaded `python/vibeqc_compiler/xc/spec.py` reran the XC generator.
- `ruff check tools/run_codegen.py tests/python/test_cmake_ownership.py` passed.

## Consequences

Incremental compiler edits now invalidate only generated families whose loaded
Python dependency graph changed, while complete compatibility identity remains
conservative. The first successful generator invocation creates its depfile; the
generator script and runner remain explicit bootstrap dependencies.

## Superseded in part

#1382 removes configure-time content hashing from source identity while preserving
the complete compatibility inventory and CMake file-membership discovery. Existing
identity inputs now regenerate the build-identity header through the build graph.
See `2026-09-26-build-time-source-identity.md`.

## References

- `cmake/VibeQCGenerated.cmake`
- `tests/python/test_cmake_ownership.py`
- #1382

---

Agent: ChatGPT
Model: GPT-5.6 Sol
