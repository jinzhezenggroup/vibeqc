# Decision: assemble full RHF Hessians from bounded multi-RHS HVP blocks

Status: implemented
Date: 2026-09-19

## Problem

After B3 delivered one complete conventional RHF HVP, B4 still needed bounded
multiple-RHS execution and a full Cartesian Hessian without reverting to the
tiny dense second-integral algorithm. The response layer already owned
sequential, blocked and recycled solve_many; Hessian code needed to reuse
that owner rather than loop over independent scalar CPHF solvers.

## Decision

The single nuclear-response implementation is factored into preparation,
linear solve and reconstruction. The scalar API still calls the existing
single-RHS solve, preserving B3 behavior. The batch API prepares each
direction with the same metric/Pulay convention, stacks only the nonredundant
rotation RHSs, calls one shared solve_many, then reconstructs each D1/W1 in
the original order.

rhf_hvp_many owns one RHF response operator per block. Directional H1/S1
sources are generated independently, the orbital solve is shared/recycled, and
the already-qualified B3 second-integral, relaxation and nuclear HVP consumers
assemble each complete result.

rhf_hessian supplies canonical atom/xyz unit directions in bounded blocks and
places each Hv into its corresponding Hessian column. It does not symmetrize
the result and does not substitute a diagonal/partial output when memory is
insufficient.

## Resource contract

Each block receives a total numeric budget. Before provider work it reserves a
conservative bound for directions, H1/S1 plus validation/prepared MO matrices,
prepared RHS storage, published response arrays and all retained HVP
components. The remaining bytes become the maximum solve_many workspace.
The reported complete numeric peak bound is the retained bound plus the
solver-reported peak workspace.

A full-Hessian request first reserves the complete output matrix; the remaining
budget is passed to each block. Compiler metadata, native call stacks, CUDA
context and loaded library code remain explicit exclusions rather than hidden
inside the numeric budget.

## Rejected alternatives

- Calling rhf_hvp independently for every column: correct but discards the
  shared operator and #179 multi-RHS/recycling contract.
- Building the dense #449 reference Hessian: useful as an oracle but not the B4
  execution algorithm.
- Symmetrizing columns after assembly: would hide a missing directional term.
- Returning only a diagonal when the full matrix does not fit: changes the
  requested property and therefore fails closed instead.

## Invariants

- Scalar B3 response still uses exactly one scalar solve.
- Batch response uses one solve_many call and preserves RHS order.
- Every published block contains complete nuclear/core/Pulay/two-electron and
  relaxation contributions.
- Full Hessian columns follow canonical flattened (atom, xyz) order.
- Raw symmetry is measured, never enforced by averaging.
- Budget failure occurs before publishing partial output.

## Evidence

tests/python/test_hessian_block.py checks recycled block HVPs against the
independent #449 dense Hessian, forbids the scalar solver from the batch path,
checks a block-size-two full Hessian against the same independent reference,
and verifies block/full-output budget rejection before provider work.

The existing B3 HVP suite is rerun with B4 to protect the scalar path and
scientific component decomposition. Existing directional-response regression
continues to validate metric, source and failure/replay semantics.

## Consequences

B4 provides a bounded tools-level complete Cartesian RHF Hessian without
claiming production-size support. Multiple directions amortize the response
operator/solver but second-integral HVP and relaxation contractions remain
directional CPU consumers. B2 may later move the shared transforms/Krylov state
onto device without changing B4 scientific assembly.

## Revisit when

Revisit block scheduling when large-system state export is available, when B2
adds resident multi-RHS response, or when measured work/memory evidence
supports a different default block size.

## References

- #180
- #179
- #564
- tools/vibeqc_hessian/block.py
- tools/vibeqc_hessian/perturbation.py
- tests/python/test_hessian_block.py

Agent: ChatGPT
Model: GPT-5.6 Sol
