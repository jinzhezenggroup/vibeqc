# Direct-HF order-0/1 handwritten force retirement

Status: implemented
Date: 2026-09-22
Parent: #356

## Decision

Delete the remaining handwritten total-order-zero/one ERI-gradient owner
`direct_native_order01_gradient.cuh`. SSSS and PSSS production science was
already compiler-owned by the generated force-only Weighted IntegralIR helpers.

The retained file survived only because the generic bounded force switch still
instantiated order 0/1 templates even though every real order-0/1 route is
drained earlier by the exact generated SSSS/PSSS shell tasks.

## Boundary

- fixed/persistent SSSS continues to call generated `ssss_force`;
- fixed/resident/paged PSSS continues to call generated `psss_force`;
- bounded streaming drains total order 0/1 before generic warp dispatch;
- generic quartet force now fails compilation for `AngularOrder < 2`;
- queueing, screening, density contraction, primitive-pair reuse and atom
  scatter remain native runtime/schedule responsibilities.

This is a structural retirement, not a new performance promotion. It changes no
selector, tolerance, resource budget, or endpoint schedule and reuses the
existing independent SSSS/PSSS qualification.

## Validation

The codegen structure gate verifies that the old source is absent, neither
low-order nor generic force headers include it, and the bounded generic switch
cannot instantiate order 0/1 again. Existing low-order generated-force tests
continue to cover the generated roots and production ownership.

Agent: ChatGPT
Model: GPT-5.6 Sol
