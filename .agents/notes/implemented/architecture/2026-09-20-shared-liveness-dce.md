# Decision: shared backward liveness before backend lowering

Status: implemented
Date: 2026-09-20

## Problem

VibeQC had output-reachability logic embedded in TensorIR while other compiler IRs either had no reusable liveness model or intentionally treated provider calls as opaque. Issue #673 needs dead-code elimination that can later be reused across TensorIR, IntegralIR, MethodIR, and boundary programs without assuming that every operation is safe to delete.

## Decision

Add a backend-neutral backward liveness analysis under vibeqc_compiler.common.liveness. Every analyzed operation declares its reads, writes, and one of three effect contracts: PURE, EFFECTFUL, or OPAQUE. Only pure operations whose outputs are not demanded can be removed. Effectful and opaque operations become conservative liveness roots.

TensorIR is the first consumer. Its immutable SSA primitives are already pure, so Program.live_nodes now delegates output reachability to the shared analysis and the existing dead_nodes rewrite consumes that result.

## Rejected alternatives

- Relying only on NVCC/LLVM DCE was rejected because it cannot remove upstream IR nodes, host argument preparation, or influence compiler scheduling.
- Adding an effect field directly to common.ProgramIR.PlanCall in the first slice was rejected because ProgramIR serialization/identity is already a contract and its provider calls are intentionally opaque. A later ProgramIR optimization must supply or version an explicit provider effect contract.
- Treating unknown calls as pure was rejected because diagnostics, stores, atomics, failure semantics, or external effects could be lost.

Missing effect metadata defaults to OPAQUE, not PURE. TensorIR explicitly
marks its known immutable primitives PURE at the adapter boundary. EFFECTFUL
and OPAQUE operations may have no SSA results (for example a check, store or
synchronization); their reads still retain transitive dependencies. Pure
output-free operations remain invalid rather than needing a fabricated token.

## Invariants

- Unknown/opaque operations are retained.
- Analysis inputs are ordered SSA-like records; forward/cyclic reads and duplicate value producers fail closed.
- Multi-output operations are retained as a unit when any output is live.
- Scientific/runtime policy remains outside the common liveness module.
- Tensor equation identity and numerical results do not change when dead retained definitions are removed.

## Evidence

Targeted validation on the implementation branch ran test_compiler_liveness.py, test_tensor_ir.py, test_tensor_execution.py, test_tensor_autodiff.py, test_tensor_ad_program.py, and test_tensor_precision.py: 159 passed. The complete tests/python/test_tensor_*.py suite then passed with 593 passed and 116 skipped.

python3 tools/check_compiler_structure.py reports zero dependency errors.

## Consequences

The compiler now has one conservative effect-aware liveness primitive. Later #673 slices can layer constant propagation, branch folding, dead argument elimination, and provider-specific pruning on top without reimplementing reachability in each scientific subsystem.

## Revisit when

Version the effect model only if a new IR needs finer distinctions such as idempotent diagnostics, removable writes, memory aliases, asynchronous effects, or region-local effect scopes.

## References

- #673
- #181
- #396
- #459

---

Agent: ChatGPT
Model: GPT-5.6 Sol
