# Decision: fuse generated stationary weights into CUDA derivative consumers

Status: implemented
Date: 2026-09-20

## Problem

The CUDA stationary-gradient path already generated one-electron, overlap/Pulay and
Coulomb source weights from `StationaryGradientPlan` TensorIR, but executed those
small programs in host-visible tiles. Each tile uploaded gathered D/W values,
downloaded a weight vector, iterated it in Python, packed the scalar weights into
primitive records, and uploaded the records to the derivative consumer. The water
qualification therefore spent about 81 TensorIR executions on source weights before
the final source reduction.

## Decision

Keep `StationaryGradientPlan` and its reverse-generated TensorIR programs as the
only scientific authority for stationary source weights. The stationary CUDA
compiler structurally lowers the generated one-term weight programs into inline
device expressions and embeds both the plan identity and each weight-program
logical hash in generated source.

The consumer ABI supplies AO indices beside derivative-center maps. The serialized
CUDA source owner uploads the validated density D and weighted density W once per
force call and evaluates the generated weight at each primitive derivative record.
No rank-4 Coulomb weight tensor is materialized. The native runtime header owns only
allocation, transfer, lifetime, launch and reduction mechanics; it does not restate
D, -W, or 1/2 D*D formulas.

The first slice deliberately retains host primitive enumeration and the explicit
native final-state D/W snapshot. Replacing that snapshot with a checked
device-resident final-state lease is a separate lifetime/ownership change.

## Rejected alternatives

Hand-writing the three source formulas in `stationary_gradient_cuda.cuh` was
rejected because it would create a second scientific definition that can drift from
TensorIR AD. Precomputing all weights on device was also rejected: Coulomb would
require an unbounded AO-rank-4 materialization or a second tiling path. Keeping the
old host-visible TensorIR tiles was rejected because it preserves the D2H -> Python
-> H2D round trip that #665 exists to remove.

## Invariants

- `StationaryGradientPlan` identity and generated weight-program logical hashes
  remain part of the generated CUDA source contract.
- The inline lowering fails closed on unsupported TensorIR operations, shapes,
  inputs or non-exact constants rather than silently substituting a formula.
- RKS sums one spin block and UKS sums two spin blocks according to the generated
  program; AO-index bindings are consumer ABI, not method policy.
- Nuclear repulsion remains source-independent and does not read D/W.
- Source weights have zero host round-trip bytes and zero standalone TensorIR
  executions in the production stationary CUDA path.
- D/W device storage and the expanded AO-index sideband are included in bounded
  resource admission.

## Evidence

- `tests/python/test_stationary_cuda_lowering.py`: 14 passed, including LDA/PBE/r2SCAN
  RKS/UKS plan/hash binding and runtime-independent source generation.
- `python -m ruff check` on the changed Python/tests: clean.
- `python tools/check_compiler_structure.py`: 247 modules checked, 0 dependency
  errors.
- Direct NVCC sm_120 compilation succeeded for both unpolarized and polarized PBE
  generated stationary artifacts; the latest-master full CUDA library build also
  completed successfully.
- Real-device RTX 5090 qualification against PySCF 2.14.0 passed all six LDA/PBE
  RKS/UKS cases in Slurm job 10433. Water analytic maximum errors were
  `1.12e-11 Eh/bohr` (LDA RKS), `2.02e-11` (PBE RKS), `1.12e-11` (LDA UKS),
  and `4.07e-12` (PBE UKS); H2 RKS errors were approximately `5e-15`.
- The same job records `tensor_executions=1`,
  `stationary_weight_tensor_executions=0`, and
  `stationary_weight_roundtrip_bytes=0` for RKS and UKS. Thus the water source-weight
  TensorIR boundary falls from the #660 production baseline of 81 executions to one
  remaining final source reduction, an 81x reduction. D/W are uploaded exactly once
  per force call (`784` bytes RKS water, `1568` bytes UKS water).
- The first device run (job 10425) executed every fused CUDA diagnostic and the
  low-level failure/recovery gate successfully, but its independent reference step
  failed because that Python environment lacked PySCF; no numerical claim uses that
  failed-oracle run.
- r2SCAN CUDA geometry remains owned by the independent stacked PR #650. A temporary
  #650 + #665 integration tree passes 39 lowering/plan tests and preserves both the
  selector-2 tau geometry lowering and the generated stationary-weight plan/hash
  binding. End-to-end r2SCAN device qualification remains a dependency on #650 rather
  than being duplicated into this performance PR.

## Consequences

This removes the stationary source-weight TensorIR tile executions and their
host-visible weight vectors while adding one bounded D/W upload per force call and
four AO-index slots per primitive record. Primitive enumeration and derivative
kernel launch batching remain separate optimization work under the parent
performance issue.

## Revisit when

A final-state CUDA lease can prove method/state identity, device ordinal, stream
ordering and lifetime through force publication. At that point D/W can remain
resident and the one explicit D/W upload in this decision can be removed without
changing the generated weight semantics.

## References

Issue #665; parent #660; supersedes the host-weight handoff portion of
`.agents/notes/implemented/architecture/2026-09-20-stationary-cuda-emitted-contractions.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol
