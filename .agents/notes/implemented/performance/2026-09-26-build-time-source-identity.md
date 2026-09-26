# Decision: hash source identity at build time

Status: implemented
Date: 2026-09-26

## Problem

The compatibility source identity intentionally covers a broad runtime/compiler
inventory. Hashing every inventory file during CMake configure made ordinary
source edits trigger a full reconfigure before compilation could resume, even
though file membership—not file contents—is the part CMake must discover while
constructing the build graph.

## Decision

Keep source-identity inventory discovery in CMake, including
`GLOB_RECURSE CONFIGURE_DEPENDS` for additions/removals, but move content hashing
and `build_identity.hpp` rendering to the normal generated-source build graph.
`tools/generate_build_identity.py` consumes the same repository-owned manifest
and template and is itself an identity input. The `vibeqc` target depends on the
generated header, so a content edit rebuilds identity before compilation without
requiring a configure pass.

This supersedes only the content-invalidation part of
`2026-09-20-source-identity-manifest.md` and
`2026-09-20-codegen-dependency-depfiles.md`. Their single-inventory and
import-depfile ownership decisions remain in force.

## Rejected alternatives

- Continue hashing all inventory bytes during configure: correct but forces
  avoidable configure work on ordinary edits.
- Remove CMake inventory discovery entirely: additions/removals matching recursive
  groups could then miss the required reconfigure edge.
- Narrow the compatibility inventory to generator depfiles: imported-module
  rebuild dependencies and compatibility identity have different safety goals.

## Invariants

- Native and Python source identity continue to use the same manifest and sorted
  repository-relative inventory.
- Adding or removing a recursively covered file still triggers CMake reconfigure.
- Editing an existing identity input regenerates `build_identity.hpp` before
  compiling `vibeqc`, without requiring reconfigure.
- Paths, timestamps, Git metadata and checkout location remain excluded.
- Missing or unsafe manifest entries and unsupported schema remain fail-closed.
- The generated file is rewritten only when its bytes change.

## Evidence

- `tests/python/test_cmake_ownership.py::test_source_identity_hashing_runs_at_build_time`
  exercises the generator and output macros.
- Exact-head CI, CuMetal CUDA, Pre-commit and PR-overlap checks pass for #1382.

## Consequences

Incremental source edits avoid an unnecessary configure pass. The build graph has
one additional generated-header target, while compatibility identity remains
conservative and file-membership changes retain CMake's configure-time discovery.

## Revisit when

Revisit if CMake gains a cheaper native content-hash dependency primitive that
preserves membership invalidation, or if native/Python identity inventories are
redesigned together.

## References

- #1382
- `.agents/notes/implemented/architecture/2026-09-20-source-identity-manifest.md`
- `.agents/notes/implemented/performance/2026-09-20-codegen-dependency-depfiles.md`

---

Agent: ChatGPT
Model: GPT-5.6 Sol
