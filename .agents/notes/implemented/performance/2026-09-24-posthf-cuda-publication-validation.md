# Decision: validate accumulated CUDA MO blocks at publication

Status: implemented
Date: 2026-09-24

## Problem

The conventional CUDA AO-to-MO transform validated the full accumulated MO block after every AO source tile. `NativeBlockProvider` and the Python `ConventionalProvider` traverse a four-dimensional AO tile domain, so the validation scan, 4-byte status download, and explicit stream synchronization were amplified by the number of source tiles even though a completed block is published only once.

With an AO axis tile of 2, a 12-AO block traverses 1,296 source tiles and a 24-AO block traverses 20,736. The prior schedule therefore scanned the complete MO output and polled the same finite-status boundary that many times per block.

## Decision

Keep cuBLAS transformation and DAXPY accumulation unchanged, but defer the full-output `check_scale(..., 1)` finite scan to an idempotent completed-block validation boundary. Each successful tile accumulation marks the resident block unvalidated. The Python provider validates once after the complete AO source traversal before returning a resident result; `device_pointer` validates before lending the resident pointer; host download validates before copying result bytes. Repeated publication operations reuse the validated state until another tile is added.

The CUDA error slot remains initialized at plan creation. cuBLAS API failures remain synchronous host errors, while accumulated numeric nonfinites remain observable before either resident or host publication.

## Rejected alternatives

Removing the finite scan entirely was rejected because independent numerical failure detection is part of the runtime contract. Validating only in `posthf_cuda_download_v1` was rejected after auditing the existing borrowed `device_pointer` contract: resident consumers may publish the device block without a host download. Reworking all post-HF CUDA sections into an asynchronous double-buffered pipeline was also rejected for this slice because `Context::section` currently owns timing-event synchronization; that broader scheduling change needs separate endpoint evidence and lifetime analysis.

## Invariants

- Preserve all four FP64 cuBLAS GEMMs and their order.
- Preserve the final FP64 DAXPY accumulation order and coefficient/source layouts.
- Preserve the existing nonfinite error message and fail before resident or host publication.
- Preserve cache hits without rescanning an already validated unchanged block.
- Do not alter AO source reads, transform FMA counts, memory accounting, method tolerances, or public capability.
- Keep complete endpoint and independent numerical gates as the promotion criterion.

## Evidence

For axis tile 2, validation launches/status polls per completed MO block change deterministically from 1,296 to 1 at 12 AO and from 20,736 to 1 at 24 AO. Corresponding 4-byte status-copy payload changes from 5,184 B to 4 B and 82,944 B to 4 B. Full-output element checks fall by the same source-tile factors. These are semantic work counts, not wall-time speedup claims.

`tests/python/test_posthf_cuda_transform_validation.py` locks per-tile scan removal, idempotent native validation ownership, provider validation before resident publication, validated device-pointer exposure, host-download ordering, and the representative work census. Existing conventional CUDA post-HF numerical/endpoint tests remain the device correctness gate.

## Consequences

The transform no longer identifies the exact source tile that first produces a numeric nonfinite. It detects the same persistent nonfinite once the block is complete and before that block can be returned or borrowed. This removes repeated whole-output work and validation-specific host/device synchronization without changing scientific arithmetic.

## Revisit when

Revisit if an intermediate consumer needs a validated partial block, if a later accumulation can legitimately recover a nonfinite accumulator to a finite value, or if post-HF timing ownership is redesigned to support overlap across AO source tiles.

## References

- `src/posthf/cuda_transform.cu`
- `src/posthf/cuda_transform.hpp`
- `src/posthf/native_provider.cpp`
- `tools/vibeqc_posthf/cuda.py`
- `tools/vibeqc_posthf/providers.py`
- `docs/maintainer/performance_engineering.md`
- Related: #153, #682

Agent: ChatGPT
Model: GPT-5.6 Sol
