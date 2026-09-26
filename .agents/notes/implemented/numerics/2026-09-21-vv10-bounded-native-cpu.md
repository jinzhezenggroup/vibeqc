# Decision: bounded native CPU pair execution for VV10/rVV10

Status: implementation slice for #491D
Date: 2026-09-21

## Problem

The existing #548/#722 implementation owns the audited VV10/rVV10 mathematics,
fixed-density KS potential, and analytic geometric source chain, but its pair
work still executes through the small-grid NumPy reference evaluator. That path
has a hard `max_points` oracle gate and cannot be used as the production
primitive lowerer required by the self-consistent MethodIR/KS execution seam.

## Decision

Add a native fixed-grid pair plan with a deliberately narrow first production
boundary:

- CPU FP64 only; CUDA stays explicitly unavailable rather than falling back;
- the plan accepts a versioned VV10/rVV10 kernel, exact MethodIR coefficient,
  fixed grid coordinates/weights, total density, and density gradient;
- one streamed/tiled pair traversal produces energy, `vrho`/`vsigma`, and the
  explicit point/weight geometric derivatives from the same kernel definition;
- scratch is O(N_grid): six retained FP64 vectors for local scales and their
  derivatives, with an exact pre-execution `maximum_bytes` admission gate;
- the full O(N_grid^2) kernel matrix is never materialized;
- diagnostics expose point count, tile size, owned workspace, maximum budget,
  and deterministic pair-evaluation count.

The low-level C/Python primitive is intentionally not wired into public
`Calculator` capability yet. PR #781 owns the MethodIR -> KS execution seam;
a future #491 slice can attach this primitive lowerer there without adding a
`wb97m-v` method-name branch. CUDA remains a separate qualification step.

## Validation contract

The native plan is compared directly against the retained independent Python
VV10/rVV10 oracle for both kernel variants and a non-unit exact coefficient.
Energy, feature derivatives, and explicit geometric derivatives must all agree.
Changing pair tile size must not change scientific results, and an undersized
workspace budget must fail before execution.

Agent: ChatGPT
Model: GPT-5.6 Sol
