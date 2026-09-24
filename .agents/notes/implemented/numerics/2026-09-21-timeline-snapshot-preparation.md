# Decision: record stricter timeline fixture preparation

Status: implemented
Date: 2026-09-21

## Problem

The timeline benchmark prepares an energy-only SCF state before exporting a
strict physical snapshot. At density tolerance 1e-10, changed-geometry LDA-RKS
water passes ordinary SCF convergence but fails the unchanged native final-state
validator before entering the measured derivative consumer. Ordinary DIIS
proposal convergence is not a proof of that stricter physical-state contract.

## Decision

Use density tolerance 1e-12 for this benchmark's SCF preparation, retaining
energy tolerance 1e-12 and the 200-iteration bound. Serialize those exact settings
in the evidence, including failed/partial campaigns. SCF preparation is outside
the force endpoint; explicit state export remains inside the reported timeline.
This changes benchmark fixture preparation, not production SCF or snapshot policy.

## Rejected alternatives and invariants

Do not relax snapshot validation, reconstruct a Python-only reference, perform
unmeasured force calls, or silently retry failed exports. A general native
convergence/export-policy repair is a separate change. A blindly tighter 1e-13
preparation runs into FP64 roundoff on these fixtures and was not selected.
Keep failed campaigns and their original settings distinct from repaired runs.

## Evidence

On RTX 5090 / CUDA 12.9.1, the original LDA-water changed-geometry snapshot
regression fails and passes with the benchmark-local repair. The complete
six-method water snapshot experiment has 18/18 successful cold/warm/changed
exports at 1e-12, versus 17/18 at 1e-10 and 8/18 at 1e-13. These are state-export
checks, not a complete force/performance qualification. Two failure-first tests
protect the preparation settings and actual changed-geometry export. The
partial-campaign test also verifies that preparation settings are preserved.

Refs #675, #662.

Agent: ChatGPT
Model: GPT-6 Astra Pro
