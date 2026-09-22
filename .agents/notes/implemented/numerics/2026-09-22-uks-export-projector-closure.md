# Certify warm UKS physical frames before successful publication

Status: implemented
Date: 2026-09-22

## Problem

The I/PBE-UKS LANL2DZ recovery fixture could publish a successful energy after
an invalid-coordinate call, then fail the unchanged native force-snapshot gate.
The preceding SCF projection changed density by RMS 1.50248e-11, but projecting
the newly rebuilt physical F[D] differed from the retained density by RMS
1.07978e-10, above the requested 1e-10 gate. Commutator, electron trace,
idempotency and canonicality checks passed. The finalizer had checked the
previous projection rather than the new physical operator used by export.

## Decision

The CPU prepared method owner validates a successful retained warm UKS frame
with the existing final-state validator before publishing success or replacing
its last-good seed. A rejected frame is reported as nonconverged, allowing the
existing batch owner to perform its already bounded single cold retry. No new
solve is inserted into snapshot/force export. Cold solve semantics, the UKS
iteration controller and occupation stabilization remain unchanged.

## Rejected alternative

Requiring an ordinary unshifted Aufbau projector inside every retained-state
UKS solve broke the native OH stabilized-occupation regression. Simply excluding
stabilized solves also failed to repair the I/PBE recovery case. The correction
therefore belongs at the prepared warm-result publication boundary, where an
exportable frame is required and the bounded cold-retry lifecycle already exists.

## Invariants

No energy, density, residual, force, canonicality or finite-difference threshold
is relaxed. No reference density is supplied. The existing validator still
checks the actual physical Fock and density; neither a shifted proposal nor a
stale orbital frame is relabeled as physical. No unbounded iteration is added.
Extra frame validation/cold retry is real work, not a speedup claim.

## Evidence and remaining scope

Two energy-only Cartesian/spherical recovery regressions fail before repair.
The existing PBE force and replay/recovery assertions remain unchanged. Both
those regressions and all native tests, including retained-state OH, must pass.
Br/I LDA-UKS cold-SCF nonconvergence is separate and remains blocking in #872.

Agent: ChatGPT (Even-PR Review R3)
Model: GPT-6 Astra Pro
