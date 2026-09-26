# Decision: one manifest owns native/autotune source identity inputs

Status: implemented
Date: 2026-09-20

## Problem

The native CMake build identity and `vibeqc.autotune.source_identity` were intended
to hash the same scientific/runtime source set, but each maintained its own
handwritten list of recursive groups and explicit files. Adding a generator,
runtime source family or compatibility input to only one list could make a
changed checkout appear compatible with an older tuned library or artifact.

## Decision

`cmake/VibeQCSourceIdentity.json` is the single inventory for:

- recursive source groups and filename patterns;
- explicit build/generator inputs that do not belong to those groups.

`cmake/VibeQCSourceIdentity.cmake` expands that manifest for the native build
and retains CMake `CONFIGURE_DEPENDS` semantics. `vibeqc.autotune` expands the
same manifest before hashing a source checkout.

The manifest itself and the CMake expansion helper are identity inputs. Paths
remain repository-relative; Git metadata and checkout location remain excluded.

## Invariants

- Both consumers reject unsupported manifest schema, missing explicit files and
  unsafe absolute/parent-traversal paths.
- Recursive source roots must exist and have at least one declared pattern.
- Inputs are deduplicated and sorted by repository-relative path before hashing.
- Adding/removing a file matching a CMake recursive group triggers reconfigure.
- The existing native-vs-Python source identity test remains the end-to-end
  equivalence gate.

## Rejected alternatives

- Keep the two lists synchronized by review: this is the drift being removed.
- Let Python be the only collector and have CMake shell out for the hash: CMake
  would lose its reliable file-add/remove reconfigure dependency semantics.
- Broadly hash the whole repository: benchmark evidence, Git metadata and other
  non-runtime files would cause unnecessary tuning cache invalidation.

## Evidence

- `tests/python/test_local_profiles.py::test_native_source_identity_and_probe_abi_match_checkout`
- normal CPU CMake configure/build path
- pre-commit JSON/CMake/Python formatting and structure checks

## Consequences

Adding a scientifically relevant source family now requires one manifest edit
rather than coordinated CMake and Python changes. The two consumers still own
their environment-specific mechanics, but not separate inventory policy.

## Superseded in part

#1382 keeps this single-inventory and CMake membership-discovery contract but
moves identity byte hashing and header rendering from configure time into the
normal build graph. See
`../performance/2026-09-26-build-time-source-identity.md`.

## References

- #349
- #353
- #1382
