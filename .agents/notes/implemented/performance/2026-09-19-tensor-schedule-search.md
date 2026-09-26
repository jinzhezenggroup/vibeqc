# Decision: bounded resource-aware TensorIR CUDA schedule search

Status: implemented (first slice; #508 remains open)
Date: 2026-09-19

## Problem

The opt-in TensorIR tuner listed four complete schedules and allowed at most eight
candidates. This obscured interactions and repeated equivalent work: panel tile
changes do not affect direct cuBLAS GEMM, and switches may not affect a particular
DAG. Searching more tuples without recognizing those equivalences wastes compiler
and device time.

## Decision

Use a structured, immutable schedule space with a bounded Hamming-radius walk.
Its default 128-candidate prefix covers every current axis, with single-axis
changes before higher-order interactions. The default full space has 1,296
combinations; the tuner does not enumerate or compile it exhaustively. The
independent default compilation budget is 12 candidates plus the baseline.

Reuse the existing planner, exact combined host/device numeric budget, source
emitter, compiler timeout, PTXAS parser, paired endpoint runner, artifact cache
and atomic selection evidence. Compare actual aliases/lifetimes/allocations,
GEMM paths, effective panel dimensions and launch threads before compiling an
equivalent plan. Keep reasons for illegal, duplicate, statically rejected,
compile-budget and deadline-limited candidates.

Static register pressure is a scalar-liveness heuristic, not a compiler report.
Occupancy is an upper bound without allocation granularity. The estimator does
not model cuBLAS internals. Global packing panels are not shared-memory tiles;
local/spill memory is unknown until compilation. Source bytes proxy compile cost
without inventing a time prediction. Logical traffic explicitly excludes
packing/provider/cache traffic. These limitations travel with the evidence.

Only complete endpoint-qualified candidates receive shared #459 implementation
profiles. Performance guards bind measured input layouts and the existing
baseline execution identity (including device/runtime/host provenance), rather
than promoting a kernel-only win or a shape-specific hardcoded policy. Missing
facts, different layouts or different runtime identities cannot reuse the
performance promotion. Correctness guards and ordinary baseline fallback remain
separate. Profiles reference the original artifact key and evidence hash in the
same selection artifact; there is no second database or automatic loader.

## Rejected alternatives

- A larger handwritten tuple list: interactions would remain manual.
- Exhaustive Cartesian compilation: most candidates are redundant or too costly.
- Treating panel size as shared-memory usage: that is not this emitter's storage
  model.
- Claiming register/traffic heuristics are measurements: PTXAS and endpoint
  evidence remain necessary.
- Promoting measured shapes globally: the concrete returned plan remains scoped
  to the caller's workload and evidence domain.

## Validation

The targeted suite covers deterministic search/axis coverage, effective-plan
identity, exact buffer accounting, legality/resource/source-cost pruning,
compile budgets, missing PTXAS records, spill/register cliffs, numerical failures,
negative/noisy timing evidence, common specialization selection and fallback for
unmeasured layouts/runtime identities. Tuner execution tests use explicitly
synthetic times and interpreter-backed test doubles, not GPU performance data.

Reproduce compiler/unit validation with:

```sh
PYTHONPATH=.:python python -m pytest -q tests/python/test_tensor*.py \
  tests/python/test_specialization.py tests/python/test_compiler_structure.py \
  tests/python/test_local_profiles.py \
  -k 'not test_native_source_identity_and_probe_abi_match_checkout'
python tools/check_compiler_structure.py
```

This compiler-only selection passed 316 tests, skipped 37 opt-in tests and
deselected the one native-build-dependent check. The compiler structure check
covered 188 modules with zero dependency errors; Ruff and `git diff --check`
also passed.

The excluded native source-identity/ABI check was attempted and failed because
this isolated checkout has no built `libvibeqc`. It still needs a matching native
build/CI run; it is not classified as passing or deleted from the suite.

A separate CUDA 12.4 / GCC 11 / sm_80 compile-only check built the baseline and
one admitted candidate for each of five families: direct GEMM (65x129 times
129x97), output-transposed packed GEMM at the same dimensions, a 513-element
multiply/add/reduction, its generated JVP, and the existing CC virtual-residual
fragment (2 occupied, 8 virtual). All ten artifacts passed the PTXAS gate, with
no spills. No GPU preparation/execution or performance promotion was performed.
The device was occupied by other allocated work; no allocation was interrupted.

For those 128-candidate searches, respectively 117, 96, 123, 123 and 123 schedules
were pruned as equivalent before compilation. This demonstrates avoided compiler
work, not improved endpoint speed. Generated sources, PTXAS logs and artifact
metadata remain in the ignored local compile cache; no binaries or release assets
are published. Existing `tools/tensor_cuda_examples.py --mode compile` and
`--mode tune` remain the supported CLI for reproducible CC-fragment qualification.

## Remaining #508 work

A cheap representative-timing shortlist before full endpoint qualification,
measured traffic, expanded executable reduction/vector/staging strategies,
cost-model calibration, and allocated-GPU evidence that at least two different
workloads qualify different winners remain open. CPU scheduling stays in #471;
DFT grid semantics stay in #168. No speedup or complete #508 closure is claimed.

## References

#508; #146; completed #136; #459; #203; `docs/tensor_cuda.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
