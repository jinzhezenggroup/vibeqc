# Retained CUDA force owners overlap later KS phases

Status: implemented
Date: 2026-09-22

## Correction to phase-aware admission

This supersedes the CUDA force-lifetime assumption in
[the phase-aware admission note](2026-09-22-phase-aware-resource-admission.md).
Serialized execution is not proof of non-overlapping allocation lifetime.
PreparedBatch keeps its PreparedStationaryCudaExecution between execute calls.
That owner retains _CudaSources, CudaGrid and TensorIR host/device arenas in
an ExitStack until batch close or explicit topology replacement. A later SCF
or changed-geometry setup therefore runs while the previous force owner lives.

## Decision

Reserve the existing CUDA force host/device caps as persistent storage and add
the native setup/SCF transient excess separately. CPU force execution does not
retain this CUDA owner and keeps the maximum-only transient accounting. This
repairs admission, not runtime allocation, science or numerical thresholds.

Releasing the force owner before every SCF could justify a different lifetime
model but would discard qualified prepared reuse and needs separate performance
evidence. Do not silently implement that change to reduce a memory estimate.

## Evidence

Two failure-first tests expose the missing device/host overlap. With 24 MiB
retained native state, a 320 MiB setup phase and the 512 MiB retained force cap,
the admitted replay bound is 832 MiB, not 536 MiB. A 320 MiB host transient
likewise remains separate from the retained 256 MiB force host cap. Existing
DF automatic admission and the UHF sizing repair are preserved.

Agent: ChatGPT (Even-PR Review a3kkcri7)
Model: GPT-6 Astra Pro
