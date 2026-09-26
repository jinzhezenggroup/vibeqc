# Decision: retain repeated Lambda action inputs on CUDA

Status: implemented
Date: 2026-09-20

## Scope and ownership

This extends the [ordinary-stream Lambda owner](2026-09-20-generated-cuda-lambda-owner.md).
The shared and expanded TensorIR equations remain unchanged. Phase zero performs
both primal checks; phase one releases those owners and prepares the RHS and
independent final check alongside one resident shared transpose action.
The existing host GMRES still owns Krylov vectors, iteration and convergence.
This is not device-resident Krylov, public nuclear forces or GPU parameter response.

## Invariants

One global host/device resource plan covers simultaneously live providers and
reserves the bound host response state. The resident provider must match its
named global-plan selection. Static CC/Fock/integral inputs are uploaded once;
each action transfers only the changing Lambda seeds, error status and requested
T cotangents. Report cumulative transfers and synchronizations, not a residency
claim derived merely from a backend name. Reject unexpected static re-uploads.

Retain live-reference/solver checks, shared and independently expanded primal
replay, true solve residual and the independent final stationarity gate. Failure
must not publish a Lambda result, and phase/constructor failures close native
owners. There is no CPU substitution for any GPU action.

## Evidence and alternatives

The review pass executed the complete six-test Lambda suite under a finite
RTX 5090 allocation with CUDA 12.9: five host-contract tests and one real water
Lambda test passed. The latter compiled the resident provider, compared both
amplitude blocks to the independent CPU solve and checked final stationarity,
resource admission and per-action transfer totals. The existing qualification
JSON remains evidence for its original source, not a rewritten record of this run.
No endpoint speedup or broad molecular qualification is claimed here.

Keeping all six providers live was simpler but retained unnecessary primal
storage. Re-uploading immutable scientific inputs on every action was correct
but did not meet the resident-action goal. Moving the host Krylov controller
onto the device would require separate ownership, convergence and resource work;
this change does not imply that transition.

Revisit on explicit qualification of device-resident Krylov, wider CC states,
concurrent owners or parameter-response generation. Preserve the independent
replay and fail-closed resource contract when widening the scope.

Agent: ChatGPT
Model: GPT-6 Astra Pro
