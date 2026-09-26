# Decision: retain one authority for native stationary reduction

Status: implemented within the unpromoted candidate
Date: 2026-09-21

## Problem

Removing the final TensorIR round trip must not introduce a second scientific
sum. The initial candidate repeated a fixed seven-term loop independently of
StationaryGradientPlan.reduction_program and allowed the reduced ABI to be called
on an artifact whose complete source inventory included external contributions.

## Decision

Specialize the existing one-atom TensorIR reduction into the existing native
source arena. Read ordered input names and exact coefficients from its validated
add node, and emit its logical hash alongside the code. The native owner still
owns allocation, launches, synchronization, scratch and transactional publication.
An unsupported mathematical topology fails during generation. Inventories with
external ECP/exchange/nonlocal contributions retain their complete TensorIR path;
the seven-source reduced ABI explicitly rejects them before device work.

## Invariants and evidence

Three new guards fail on the original candidate and pass after the repair: plan
hash binding, changed coefficients reaching emitted arithmetic, and rejection of
an incomplete external-source inventory. Four host and four real allocated-CUDA
cases execute the exact emitted reducer, preserving ordered cancellation, signed
zeros, signed inputs, tail bounds, and rejection before host output publication.
These reducer tests are not a full molecular force or endpoint performance run.

## Revisit when

Generalize the storage and reduction together only after a complete source layout
and its independent endpoint/resource qualification exist. Do not silently pad
missing physical sources or infer completed qualification from a source hash.

Agent: ChatGPT
Model: GPT-6 Astra Pro
