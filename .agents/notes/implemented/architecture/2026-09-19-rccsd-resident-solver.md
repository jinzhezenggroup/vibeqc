# Decision: run RCCSD current/trial/DIIS state inside one resident TensorIR owner

Status: implemented
Date: 2026-09-19

## Problem

The #376 GPU solver evaluated the correct generated equations but uploaded all
integral/amplitude inputs and downloaded large residual tensors on every current
and trial evaluation. #420 provided a generic resident span ABI, while
`src/cc/cuda_state.cuh` already reserved DIIS/reduction storage, but the pieces
were not connected into a complete solver. Consequently #149 B remained open
and a future Lambda/gradient consumer could not inherit a solved resident state.

## Decision

Keep the existing #148 physical equation and #149 generated Jacobi TensorIR as
the sole scientific owner. Compile the primary plan with the #420 resident ABI
and a plan-specific post-run/control extension. Upload static F/g, D and initial
T once. Pin current T1/T2 and all outputs for the owner lifetime. Use the exact
#146 DIIS reservation for current/last amplitudes, dense amplitude/error
histories, Gram/augmented-system scratch, reduction partials and scalar status.

After each current run, reduce physical R1/R2 maxima on device and transfer only
energy/R1max/R2max to host control. On an unconverged iteration, save current T
and D2D-copy generated next-T outputs into the input spans. Evaluate that exact
trial without a residual download, store the trial T/R pair in resident DIIS,
solve the bounded augmented system on GPU, and combine T1/T2 slices back into
the pinned inputs. An ill-conditioned DIIS solve drops the oldest resident pair
and retries, matching the CPU policy. Zero-DIIS leaves the trial resident.

Convergence is not certified by resident scalars alone: download final T once
and run the separately retained expanded physical TensorIR GPU replay. Only that
fresh replay may publish a converged `CCSDResult`. The solved owner itself stays
alive until explicitly closed. Its immutable owner hash binds reference,
integral, equation, artifact and device identities; a separate solved-state
hash binds that owner to replay-qualified final T1/T2. Starting another solve
invalidates the solved-state hash until a fresh expanded replay passes.

## Failure and lifetime rules

Preparation validates reference/provider and reads MO blocks before any solve.
After upload, numerical iteration and final replay use owned feeds; releasing the
provider cannot alter the resident solve. Current T is copied to a dedicated
last-finite reservation before trial mutation. Nonfinite current/trial/DIIS
execution never falls back to CPU and returns/downloads the appropriate finite
state for replayable failure reporting.

The ordinary host-staged GPU solver is retained as an independent backend. The
internal energy facade selects resident execution only with the explicit
`cuda-resident` name. Native registry, public force support and homogeneous
prepared batching are not implied.

## Resource/transfer invariant

There is one initial large H2D upload. During iteration, primary TensorIR runs
return only their 4-byte resident ABI status; current-state control additionally
reads three FP64 scalars and tiny DIIS status integers. `per_iteration_large_*`
transfer counters are zero by construction. Final amplitudes are downloaded once
for expanded replay/result publication. Reservations are part of the same #146
plan budget; no hidden second amplitude/DIIS allocation is created.

## Evidence

Real RTX 5090 / CUDA 12.9 / sm_120 tests cover H2, H2O and CH4 convergence,
reference amplitude/energy agreement, reusable solved-owner amplitudes, provider
release after preparation, explicit nonconvergence, zero-DIIS resident Jacobi,
repeated solve without another large upload, and simultaneous independent H2/H2O
owners. Identity-bearing `AmplitudeSnapshot` warm starts reject changed
geometry/generation/orbitals before compilation/upload.
H2O uses 15 current iterations / 29 primary runs with 11,920 B initial large H2D,
zero per-iteration large H2D/D2H, 524 B scalar-control D2H, and one 880 B final
amplitude download. The same ordinary-stream control law reports 356,720 B
primary+replay H2D and 51,272 B complete-plan D2H upper bound on that fixture.
The observed cached 1.09 s vs 0.73 s wall times are not a performance-promotion
claim.

## Revisit when

#149 C introduces a native method owner/fleet. Move this same resident state
contract behind the native registry and add isolated homogeneous batch owners;
do not replace the generated equation or weaken final expanded replay.

## References

#149, #193, #420, #376, #148; `docs/rccsd_gpu.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol
