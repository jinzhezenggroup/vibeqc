# Decision: Keep generated one-electron CUDA device results caller-owned

Status: implemented
Date: 2026-09-21

## Problem

The real VibeQC CuMetal endpoint gate reached the production RHF one-electron
value kernel but the pinned typed PTX importer rejected it before launch with
`aggregate PTX device return has missing, partial, or overlapping fields`.
The generated value path returned `PairGeometry`, `ST`, and the runtime policy
accumulator by value from device helpers. NVIDIA CUDA accepts those internal
returns, but they are not part of VibeQC's scientific contract and they create
an unnecessary compatibility boundary for PTX consumers.

## Decision

Generated one-electron CUDA structs are now written into caller-owned output
objects instead of being returned by value. `make_pair`, the S/T helper family,
the generated value policy, and the shared AO-pair contraction all follow this
rule. Scalar attraction helpers remain scalar-returning because they do not
exercise the aggregate ABI.

The policy inventory advances to version 3 and records both caller-owned
device struct results and scalar/pointer view transport, so generated-artifact
identity cannot silently confuse the old and new internal ABIs.

The same compatibility rule now applies at CUDA kernel entry points and across
the one-electron device call graph. The public host launchers keep typed
OneElectronDeviceView / OneElectronWeightView arguments, but value and
derivative kernels receive scalar/pointer parameters and forward only the fields
actually consumed by each device helper. Device code no longer reconstructs or
passes either heterogeneous view aggregate. Output pointers are likewise passed
individually instead of through an Outputs pointer array. This avoids both PTX
aggregate parameter loads and local heterogeneous-record spill/reload without
adding allocations, transfers, synchronization, or global mutable state.

## Invariants

- S/T/V formulas, normalization, component ordering, nuclear-charge sign, and
  FP64 arithmetic are unchanged.
- The change is internal to generated/native CUDA source; no public Calculator,
  C ABI, method, or scientific capability is added or removed.
- CUDA compatibility providers must not require VibeQC to weaken precision or
  substitute a CPU scientific path.
- Generated policy/source identity changes whenever this internal ABI changes.

## Evidence

The failing CuMetal job for PR #704 reached the production endpoint and reported
the typed lowering error on the generated one-electron pair kernel. Local source
validation after the change passes the independent one-electron value suite and
CuMetal workflow tests, Ruff, compiler dependency checks, and `git diff --check`.
After the entry-ABI repair, CUDA 12.9 `sm_120` PTX compilation of both
`one_electron_values.cu` and `one_electron_derivatives.cu` emits two value and
four derivative kernel entries whose parameters are scalar or pointer PTX
parameters; none of those `.entry` signatures contains an aggregate `.b8`
parameter. Focused one-electron/CuMetal workflow tests pass locally (21 passed,
98 skipped where an installed native test library is unavailable). The
Apple/CuMetal exact-head workflow remains the authoritative execution gate for
translator execution.

## Rejected alternatives

- Mark the real endpoint benchmark allowed-to-fail: this would restore a green
  badge without executing the endpoint and would defeat the purpose of #704.
- Fall back to CPU: provider fallback would make the measured endpoint dishonest.
- Lower scientific precision: the endpoint is intended to exercise production
  FP64 semantics, so precision is not traded for translator compatibility.

## Revisit when

If the supported CuMetal PTX importer accepts aggregate device returns with a
qualified numerical corpus, the caller-owned ABI may still remain: it is a
simple portable device boundary and avoids depending on compiler-specific
aggregate return lowering.

## References

- PR #704
- `python/vibeqc_compiler/integral/one_electron_cuda.py`
- `python/vibeqc_compiler/integral/one_electron_policy_cuda.py`
- `src/runtime/cuda_ao_pairs.cuh`
- `.github/workflows/cumetal-cuda.yml`

Agent: ChatGPT
Model: GPT-5.6 Sol


## Independent validation of the published flat ABI

The separately prepared f8a7331e value-only patch was initially inspected during
review. Before publication, the concurrent bdacdca5 repair supplied a more complete
shared-body solution covering value and derivative entries. That published repair
was preserved; this review adds validation rather than replacing it.

On integrated source b11a695d, all 121 selected workflow, emitted one-electron
value/derivative and launch-contract tests passed without skips. The new launch
regression compiles and executes the actual entry bodies, host argument lists,
binding macros and shared scheduling bodies. Only the scientific pair evaluator
is replaced by a capture probe. It checks all 17 view fields, all four output
pointers, optional null outputs, both schedules, noncanonical pair maps, two
systems, partial blocks/warps and exact-once ownership. It is an ABI/scheduling
test, not CUDA numerical execution.

Fresh NVCC 12.4 / sm_80 PTX compilation used exact-source regenerated helpers
and both real production translation units. The two value entries have 24/21
scalar or pointer parameters; the four derivative entries have 29/29/26/26.
No entry contains an aggregate .b8 parameter. Ruff and formatting checks pass.
These results do not establish Apple's Clang/PTX/CuMetal execution or speed.

The earlier Apple job 106162926229 failed all three endpoints at an indirect
parameter load. New exact-head Apple endpoint CI must pass before overall LGTM
or auto-merge. No scientific tolerance, IEEE64 policy, or backend was weakened.

A later exact-head Apple failure in job 106232040555 reached thread_pairs_flat
but CuMetal typed lowering then reported ptr<device, i8> versus i64 mismatches
inside the entry's device call graph. The follow-up removes device-side
OneElectronDeviceView, OneElectronWeightView, and output-pointer aggregates
rather than weakening the verifier. Focused one-electron source/codegen coverage
after this change passes 26 tests with 148 environment-dependent CUDA tests
skipped on the validation host; Ruff, clang-format, and git diff --check also
pass. Apple/CuMetal execution remains the authoritative acceptance gate.

Agent: ChatGPT
Model: GPT-6 Astra Pro