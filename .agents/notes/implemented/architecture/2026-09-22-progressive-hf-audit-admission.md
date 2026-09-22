# Progressive HF audit admission precedes native work

Status: implemented
Date: 2026-09-22

## Problem

The verification budget reached only cross_overlap, after NativeAO and FockPlan
had already allocated and executed. It did not reserve the simultaneous density,
Fock, overlap, commutator, immutable-result and hashing buffers.

## Decision

Reserve a conservative 12 matrices per spin plus eight common matrices, and ask
the existing HF resource planner to admit its full provider envelope with that
additional host reserve before creating native audit owners. Reusing the full HF
inventory deliberately overestimates a single fixed-density Fock audit; do not
introduce a second untracked formula for ERI/DF/native provider storage here.
Like the shared resource contract, this is a numerical-buffer admission model,
not a measured bound on process, allocator or driver overhead. Unsupported
provider inventories do not authorize an unbounded audit.

Budget rejection remains distinct from physical failure: return budget_exhausted
and no verified target while preserving the successfully solved target result.
Source/target mathematics, transfer, solver tolerances and reference accuracy
classification are unchanged. Native audit contexts still use scoped cleanup.

## Evidence

Four new admission-order tests fail before the repair, because native ownership
is reached before either an insufficient workspace cap or a rejected provider
envelope. Additional actual RHF/DF endpoints exercise tiny capacities and a cap
that fits the provider alone but not the combined audit. Ordinary RHF/UHF direct
and DF target audits remain successful under the default capacity.

Agent: ChatGPT (Even-PR Review R9 hjhmhw3o)
Model: GPT-6 Astra Pro
