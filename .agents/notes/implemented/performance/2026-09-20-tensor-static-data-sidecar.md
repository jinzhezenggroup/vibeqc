# Decision: externalize large immutable TensorIR CUDA data from generated source

Status: implemented
Date: 2026-09-20

## Problem

The #504 GFN2 pairwise consumer remained expensive to compile after the
O(target*source) scatter regression was removed. Pair-sized TensorIR constants
and ragged index tables were emitted as C++ array literals, so source size and
NVCC parsing work scaled with immutable data volume rather than executable code.

## Decision

Keep TensorIR scientific identity and device arena ownership unchanged, but
split production CUDA artifacts into executable source/binary plus an exact
immutable static.bin sidecar.

The production compile_cuda path emits no large constant/index literal arrays.
It serializes exact FP32/FP64 constant bytes and int64 index tables into a
compact sidecar with no device-alignment padding. The generated library exports
a preparation-only tensor_static_initialize entry point whose source contains
only bounded offset/size descriptors. PreparedCuda verifies the sidecar hash
and byte count, uploads each slice to its planned arena location, and
synchronizes before execution.

The legacy self-contained emit_cuda representation remains the default for
source-only consumers and prefixed/native bundle generation. Production
compilation and tuner source-cost estimates explicitly select external static
data, so existing MP2/native bundle ABI is unchanged by this slice.

## Identity and failure behavior

Artifact identity includes the static payload SHA-256 and byte count in addition
to the full TensorPlan identity. Cache hits verify both program.so and static.bin.
Native preparation rejects missing, corrupt, wrong-sized or null static payloads
before any execution.

The mathematical Program, plan identity, accumulation order, runtime input ABI,
and execution kernels are unchanged.

## Evidence boundary

Host/source checks plus a CUDA 12.9 compile-only CI gate cover both ordinary
and resident external-static generated artifacts without requiring a GPU.
Allocated-GPU before/after timing must be added separately; no runtime speedup
is claimed from source-size reduction alone.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Initialization lifetime repair

The preparation-only initializer must not rewrite immutable data after success:
resident output leases and captured execution identities do not carry an epoch
for such a mutation. A second initialization is therefore rejected before any
copy, retaining the valid original state. A partial upload failure drains the
stream before returning so queued transfers cannot outlive their borrowed host
payload. The unready state permits an explicit clean retry. Host-compiled tests
of the actual emitted ABI inject a second-copy failure and verify this ordering;
they use explicit transfer stubs and are not device-numerical evidence.

Review update: Agent ChatGPT; Model GPT-6 Astra Pro.
