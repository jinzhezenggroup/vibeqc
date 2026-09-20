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

The policy inventory advances to version 2 and records
`device_struct_results=caller_owned_out_parameters`, so generated-artifact
identity cannot silently confuse the old and new internal ABIs.

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
The Apple/CuMetal exact-head workflow remains the authoritative execution gate
for this compatibility repair.

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
