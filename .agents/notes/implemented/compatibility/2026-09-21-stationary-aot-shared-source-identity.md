# Stationary AOT shared-source invalidation

Status: implemented
Date: 2026-09-21

## Decision

Include the shared `common` compiler family in the stationary AOT compatibility
source fingerprint and the corresponding CMake post-link manifest dependencies.
The previous explicit family list omitted these implementation bytes, so a shared
source revision could retain an old contract identity. Compatibility forwarding
modules in other families do not substitute for their implementation hashes.

The existing plan, spin, source, binary, architecture and strict-FP64 checks remain
mandatory. No emitted CUDA algebra, public force domain or device qualification
is changed. Existing packaged artifacts must be rebuilt to carry the new contract.
The broader per-family fingerprint intentionally favors conservative invalidation
over accepting a potentially stale scientific artifact.

## Evidence

Failure-first tests simulate changed hashes for real shared source paths without
editing checkout files. All six functional/spin identities were unchanged before
the repair. The CMake dependency evaluation also omitted the new expected inputs.
The same tests pass after the fingerprint and dependency filter are repaired.

Agent: ChatGPT (Even-PR Review)
Model: GPT-6 Astra Pro
