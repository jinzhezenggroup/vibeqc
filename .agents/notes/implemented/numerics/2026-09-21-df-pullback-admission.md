# Decision: preserve representable DF cotangents and bound their live arrays

Status: implemented
Date: 2026-09-21

## Problem

The new B-to-A/M pullback formed a symmetric average by adding two full-sized
values before halving. Finite metrics and raw-source cotangents near the FP64
limit therefore overflowed despite having a representable result. Its logical
budget also omitted simultaneous raw-A buffers and contraction intermediates.

## Decision

Apply half weights before addition, as in the shared spectral matrix rule.
Bound accumulator, projection and immutable-publication overlap. Fix the two
three-operand contraction paths so their intermediate dimensions are explicit;
do not allow a cost optimizer to introduce an unaccounted outer product.
Check the logical budget before metric copying/hashing and spectral preparation.
Borrowed input storage, Python objects and opaque BLAS workspaces remain outside
this logical-array contract; this is not an RSS or production allocator bound.

## Evidence and invariants

Two finite extreme-value cases and a simultaneous-buffer budget case fail before
repair. A genuinely nonfinite-result control remains rejected. Same-Hamiltonian
finite differences, multiple-block accumulation and fixed-rank cutoff tests must
remain unchanged. No threshold, spectral VJP rule or complete-force claim changes.

Agent: ChatGPT
Model: GPT-6 Astra Pro
