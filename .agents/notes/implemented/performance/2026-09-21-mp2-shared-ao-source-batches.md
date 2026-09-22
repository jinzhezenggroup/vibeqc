# Decision: share bounded AO source scans across conventional MP2 blocks

Status: implemented
Date: 2026-09-21

## Problem

Direct and exchanged conventional MP2 blocks consumed identical AO source tiles
in separate complete scans. Reusing only their transformed outputs would retain
unbounded MO intermediates; faster isolated transforms would not remove source
work. A source-driven multi-request boundary avoids both errors.

## Decision

`NativeBlockProvider::get_many` reads each AO tile once and applies the existing
cyclic four-stage transform independently to each requested MO block. It retains
per-request coefficients, output and CPU scratch or CUDA transform owners, but
shares immutable reference/source storage and the bounded AO tile. The common
host allowance is the existing reference/source capacity, 8 MiB runtime allowance
and four AO-tile-sized numeric buffers; request capacity is computed by removing
that common term from the existing numeric block plan and adding every request's
remaining host and device reservation. CUDA retained/library allowances remain
per owner: one source scan does not imply one upload or one transform owner.

Conventional energy reserves its generated kernel and consume-phase staging
before asking the provider for capacity. It batches direct/exchanged requests in
pairs, limits the group by actual jobs, and keeps the original tile order and
compensated energy fold. If only one provider request fits, the sequential path
remains explicit. Checked arithmetic and pre-allocation admission are preserved.

## Rejected alternatives and invariants

Do not materialize the complete AO tensor, copy a second MP2 equation, reorder
energy accumulation, or admit requests by output size alone. Batch output is
returned only after every request succeeds; scoped CUDA owners clean up partial
construction. The optional work counters describe attempted work on a failed
call and must not be interpreted as successful publication. Arbitrary ordered
slots and zero-padded tail columns retain the existing semantics.

## Evidence boundary

`tests/native/test_mp2_contract.cpp` checks an independent dense AO-to-MO oracle,
request parity, source reuse, unchanged transform work, exact budget fallback and
complete H2 MP2 request accounting. `test_mp2_gradient.cpp` protects the existing
energy/gradient conventions. The PR's recorded 12/24/36-AO source-work experiment
is historical author evidence, not a new end-to-end force or GPU speed claim.
Any future default-capacity or scheduling change must recheck numeric peak
storage and complete energy/force behavior, including CUDA owners and tails.

References: #366; #731; `src/posthf/native_provider.cpp`;
`src/posthf/block_capacity_generated.hpp`; `src/posthf/mp2_energy.cpp`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
