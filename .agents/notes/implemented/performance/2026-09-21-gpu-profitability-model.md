# Decision: Share auditable GPU profitability facts across compiler tuners

Status: implemented
Date: 2026-09-21

## Problem

Integral and TensorIR CUDA tuning already had complementary evidence: symbolic
live-range/rematerialization metrics on the integral side and traffic/register/
occupancy/source estimates plus PTXAS calibration on the tensor side. Each tuner
ranked candidates with local ad hoc tuples, so fusion, CSE retention, and
rematerialization could not be compared through one compiler-owned vocabulary.

## Decision

Use common.gpu_profitability.GpuProfitability as the backend-neutral record for
static and compiled GPU cost evidence. Keep values explicit and optional rather
than inventing estimates when a consumer cannot measure a field.

TensorIR uses the shared static ordering to allocate its finite compilation
budget. Its exact emitted launch sequence is now computed from the plan and is
also reused by graph-capture qualification. Integral tuning records symbolic
operation/live-range/rematerialization facts together with PTXAS resources and
uses the shared compiled-resource ordering only after candidates are inside the
existing one-percent complete-endpoint timing noise band.


No weighted scalar score is introduced. Static compilation priority remains
lexicographic and auditable: endpoint semantic traffic, register pressure,
occupancy, launches, live values/work, source size, then generation order.
Compiled tie-breaking starts with spills, occupancy, and registers before local/
shared memory and build/artifact costs. Outside the endpoint noise band, measured
complete-endpoint time remains primary.

## Rejected alternatives

A learned or hand-weighted scalar cost was rejected because its weights would
hide target assumptions and could turn calibration data into an implicit
performance promotion. A second live-range algebra was rejected because the
integral expression layer already computes exact SSA lifetimes and bounded
rematerialization plans.

## Invariants

- Static estimates never promote a production candidate by themselves.
- PTXAS spills remain hard negative evidence where existing resource gates apply.
- Complete-endpoint correctness and performance gates remain authoritative.
- Unknown resource facts stay None; they are not guessed from source shape.
- Profitability contains no architecture-name or GPU-model special cases.
- Scientific equations, precision policy, and fast-math behavior are unchanged.


## Evidence

Device-free TensorIR tests verify that fusion exposes the expected tradeoff:
fewer launches and less semantic traffic with higher estimated register pressure.
Shared-model tests verify that pressure reduction can outrank a no-traffic-win
fusion and that a smaller artifact cannot compensate for measured spills.
Integral autotune tests retain existing baseline/runtime rejection behavior while
persisting the new profitability evidence.

Reproduction:

- PYTHONPATH=python python3 -m pytest -q tests/python/test_gpu_profitability.py
- PYTHONPATH=python python3 -m pytest -q tests/python/test_tensor_cuda_search.py
- PYTHONPATH=python python3 -m pytest -q tests/python/test_codegen.py -k 'autotune or static_model'
- PYTHONPATH=python python3 tools/check_compiler_structure.py

## Consequences

This is the common profitability substrate for #832, not the final production
promotion result. Follow-up slices can calibrate static register/live-range
estimates against real PSSS/high-order-force and DFT/TensorIR workloads without
adding another cost vocabulary.

## Revisit when

Measured workloads show that the lexicographic stage ordering systematically
misses profitable candidates, or when a target-independent calibrated model can
be justified with reproducible complete-endpoint evidence.

## References

- #832
- #682
- #356
- #508
