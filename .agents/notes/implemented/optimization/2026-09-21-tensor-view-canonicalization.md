# Decision: canonicalize only lossless TensorIR view chains first

Status: implemented
Date: 2026-09-21

## Decision

Add a `view_canonicalization` middle-end pass after identity-transpose cleanup and before CSE. The first slice removes only operations whose TensorIR semantics prove they are representation-only: same-dtype casts, full slices, identity broadcasts, nested reshapes, and composed transposes.

The pass deliberately does not reassociate arithmetic or apply neutral/absorbing-element rewrites. TensorIR preserves IEEE-sensitive execution order; later algebraic simplification must carry an explicit numerical proof/contract rather than inheriting fast-math assumptions.

Nested reshape and transpose chains are collapsed directly to their source or to one equivalent operation. Reconstructed nodes still pass the ordinary TensorIR verifier, so index-space, symmetry, dtype, representation, and differentiability contracts remain fail-closed.

## Evidence

- Focused pass/optimizer/execution set: 22 passed.
- Full `tests/python/test_tensor_*.py`: 609 passed, 117 skipped.
- Tests retain signed-zero bit patterns through lossless view elimination.
- Float32 -> float64 round trips remain explicit and are not collapsed.
- Signed-zero arithmetic demonstrates that IEEE-sensitive arithmetic is outside this pass.

## Follow-up

Continue #682 Slice B with separately reviewable exact algebraic/index canonicalizations. CSE/GVN remains a later slice, and profitability/register-pressure policy must not be folded into this backend-neutral pass.

---
Agent: ChatGPT
Model: GPT-5.6 Sol
