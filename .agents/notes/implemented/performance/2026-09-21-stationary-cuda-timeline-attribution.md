# Decision: stationary CUDA timeline attribution

Status: implemented
Date: 2026-09-21

## Boundary

Stationary CUDA performance evidence reports an additive exclusive host-wall
timeline separately from device-event attribution. Host phases are switched at
existing orchestration boundaries and must reconcile to the measured endpoint
wall time. CUDA events attribute setup, primitive transfer/kernel/reduction,
geometry transfer/kernel/reduction, synchronization wait, and final D2H work.
Device-event intervals are diagnostic attribution only and are never added to
exclusive wall time.

## Synchronization and ownership

Profiling is opt-in. Event creation is transactional, retained by the stationary
source owner, and destroyed with that owner. Event recording is placed around
existing stream work; timing is read only after synchronization that the source
path already requires. Profiling must not add a new synchronization point or
change stream ownership, scientific formulas, tolerances, admission caps, or
public force capability.

## Evidence admission

Published evidence requires finite, nonnegative phase durations, finite endpoint
energy, a finite Cartesian `[natom, 3]` gradient, and explicit endpoint/timeline
reconciliation. Cold, artifact-warm, same-state-warm, and changed-geometry runs
must use isolated cache/evidence ownership so a benchmark never deletes caller
state. Host/stub tests validate instrumentation contracts but do not substitute
for real-device timing evidence.

Agent: ChatGPT
Model: GPT-5.6 Sol
