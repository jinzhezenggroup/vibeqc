# D3 ragged workspace is an aggregate extent

Status: implemented
Date: 2026-09-19

## Problem

The CUDA fleet planner and allocator passed total atoms to the single-system
workspace helper. That helper deliberately returns zero above the 4096-atom
per-system cap. A fleet of many individually valid small systems therefore
received a zero workspace even though its kernel still used `16 * begin` slices.

## Decision

Keep the per-system physics cap unchanged. Use one checked aggregate workspace
helper in both planning and allocation, including overflow of the byte extent.
Zero gradient publication storage before each replay because inactive,
energy-only and failed rows are downloaded together with successful peers.
Do not turn the single-system bound into an arbitrary fleet-size restriction.

## Evidence

The actual native CUDA regression uses 4100 independent one-atom systems. On
the original implementation it rejects the zero workspace before any unsafe
kernel access. The repair passes the complete execution and masked/nonfinite
replay, plus Compute Sanitizer memcheck and initcheck with zero errors.
The same source is a routine CPU CTest; its optional `cuda` argument exercises
the device owner under an explicitly allocated GPU. No timing, large-molecule
numerical qualification or complete process-memory claim follows from this test.

Agent: ChatGPT
Model: GPT-6 Astra Pro
