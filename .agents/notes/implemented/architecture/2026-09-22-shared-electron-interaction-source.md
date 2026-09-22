# Decision: Method-neutral AO electron-interaction source boundary

Status: implemented
Date: 2026-09-22

## Problem

Mean-field and post-HF code already share integral mathematics and generated kernels, but the
post-HF MO block consumer was hard-wired to `posthf::RawSource`. That concrete dependency made
it difficult to reuse the same MO transformation consumer with generated exact, density-fitted,
or future prepared interaction sources without changing MP2/CC code.

## Decision

Introduce `integrals::ElectronInteractionSource` as a method-neutral, read-only AO tile contract.
The contract owns no SCF, DFT, MP2, or CC semantics. It exposes the normalized orbital system,
orbital/auxiliary dimensions, explicit operator capability queries, and bounded caller-owned AO
tile reads.

`posthf::RawSource` remains the compatibility implementation and now implements this contract.
`posthf::NativeBlockProvider` consumes the abstract interaction source rather than the concrete
`RawSource`. Existing RawSource call sites remain source-compatible. The contract also reports
retained numeric bytes so a generic consumer cannot undercount a resident dense/DF source.

`scf::PreparedFockInteractionSourceView` is the first mean-field adapter. It borrows an existing
CPU `PreparedFockPlan` without copying tensors: exact owners expose resident four-center ERIs,
while DF owners expose resident metric and three-center values. CUDA owners deliberately expose
no host interaction tensors through this view. This slice does not rewrite numerical sources or
make J/K assembly part of the integral contract.

## Rejected alternatives

- Moving Fock J/K types into `integrals/`: rejected because J/K assembly is mean-field semantics,
  not a primitive integral-source responsibility.
- Making one large provider interface contain J/K, MO blocks, DF factors, response, and device
  execution: rejected because it would couple otherwise independent method layers and encourage
  capability checks for unrelated operations.
- Rewriting RawSource or the cyclic AO-to-MO transform: rejected because this change is an
  ownership/interface refactor and must not change scientific recurrence or transformation work.

## Invariants

- AO operator definitions and ordering are unchanged.
- Existing RawSource validation, recurrence, normalization and transactionality are unchanged.
- NativeBlockProvider still performs the same cyclic four-stage transform and memory accounting.
- Consumers must query optional operator capability before relying on it.
- Method layers retain ownership of Fock assembly, orbital transforms, correlation equations and
  response.

## Evidence

`tests/native/test_mp2_contract.cpp` constructs a forwarding implementation of
`ElectronInteractionSource` that is not a RawSource and verifies that NativeBlockProvider
produces the same MO block as the existing RawSource-backed path. It also routes a CPU exact
`PreparedFockPlan` through the same MO-block consumer and compares against the independent raw
source, checks resident DF metric/three-center reads, capability separation, source-residency
accounting, and transactional rejection of unsupported ERI reads. Existing MP2 contract tests
continue to cover AO-to-MO values, batching, memory bounds and source-work accounting.

## Consequences

The post-HF MO-block path now has an adapter seam for generated exact and DF sources. The current
RawSource remains available, so the migration does not require a flag day or duplicate numerical
implementation.

## Revisit when

Add concrete adapters when PreparedFockPlan/generated DF sources can expose AO tiles or
factorized blocks without violating their current lifetime, memory-budget and stream-ownership
contracts. At that point, prefer extending focused source interfaces over adding method-specific
operations to ElectronInteractionSource.

## References

- `src/integrals/electron_interaction_source.hpp`
- `src/posthf/raw_source.hpp`
- `src/scf/interaction_source_view.hpp`
- `src/posthf/native_provider.hpp`
- `tests/native/test_mp2_contract.cpp`
