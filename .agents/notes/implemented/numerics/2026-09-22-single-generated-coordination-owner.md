# One generated coordination pair owner for dense and neighbor-list CUDA

Status: implemented
Date: 2026-09-22

## Problem

The CUDA H3+ independent energy/force fixture failed before SCC with public
aggregate error 3. Temporary diagnostics traced it through an ineligible numerical
refresh to geometry error 13: the dense/sparse bitwise coordination-number gate.
The dense geometry owner used generated TensorIR/AD pair arithmetic, while the
neighbor-list owner still evaluated an independently handwritten logistic pair.
Reverting only the S/D/Q integral cutover did not fix this shared failure.

## Decision and invariants

Make both geometry consumers call the existing generated coordination primal and
its distance adjoint. Keep the exact bitwise admission check, canonical ascending
neighbor sums, cutoff and coordinate/radius validation, error isolation, VJP
scatter, stream ordering and failure-atomic publication unchanged. The previous
descriptor-by-value backport remains. Only the adapted local source digest changes;
the upstream snapshot and upstream source hash do not.

Do not loosen the CN comparison, bypass preprocessing, turn off D4, substitute
a CPU result, or accept the original failed public result. These would conceal
a duplicate numerical implementation rather than repair it.

## Evidence

Failure-first device diagnosis used the existing H3+ independent tblite fixture
and the OH independent xTB control on RTX 5090 / CUDA 12.9.1. The H3+ fixture
failed before the repair and passed after it without changing its reference or
tolerances. The existing 25-case selected source/dependency/CPU-force/CUDA suite
passed after repair. Additional tests preserve the shared generated-source
boundary and exercise asymmetric three-atom CUDA force/finite differences.
Diagnostic printf changes were confined to an isolated build and restored; they
are not part of production. This is a correctness repair, not a speedup claim.

Refs: #954, #560.

Agent: ChatGPT (Even-PR Review R7 viemzo2a)
Model: GPT-6 Astra Pro
