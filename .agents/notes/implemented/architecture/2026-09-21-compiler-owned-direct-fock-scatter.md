# Decision: compiler-owned Direct-Fock scatter contraction

Status: implemented
Date: 2026-09-21

## Problem
Retained Direct-HF Fock fallbacks and generated shell-class Fock kernels each encoded
the same RHF/UHF Coulomb/exchange scatter coefficients. The native copy in
direct_fock_accumulation.cuh remained a handwritten scientific owner even when a
shell class itself had moved to IntegralIR/AOT lowering.

## Decision
The scientific compiler now owns one parameterized Fock scatter emitter. Generated
shell kernels and the retained native fallback adapter are rendered from that same
source. The source-tree direct_fock_accumulation.cuh remains only as a compatibility
include so existing native task/runtime code need not change include topology.

## Rejected alternatives
Deleting all native Fock fallbacks in this slice was rejected because portable and
unpromoted/resource-rejected shell classes still require bounded native execution.
Keeping two literal CUDA implementations was rejected because it preserves duplicate
scientific ownership and lets RHF/UHF signs or factors drift.

## Invariants
RHF and UHF density conventions, eightfold ERI symmetry scatter, atomic update order,
and the RHF exchange factor of -0.5 are unchanged. Native fallback selection and
generated shell-class promotion policy are unchanged.

## Evidence
tests/python/test_codegen_fock_ownership.py verifies that the scientific coefficients
occur only in the compiler emitter, that both generated adapters contain identical spin
semantics, and that build-time header generation is deterministic. Existing compiler,
ownership, and CUDA build checks remain the integration gates.

## Consequences
One handwritten Direct-HF scientific file leaves the #356 retirement ledger without
removing any fallback capability. Remaining order-2/quartet/native ERI bodies can now
retire independently while continuing to call the same compiler-owned scatter adapter.

## Revisit when
Remove the compatibility include entirely after the final retained native Fock consumer
has been promoted or replaced by a compiler-owned fallback lowering.

## References
Issue #356.

Agent: ChatGPT
Model: GPT-5.6 Sol
