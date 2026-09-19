# Decision: reuse bounded generated CUDA consumers for final CCSD derivative contraction

Status: implemented
Date: 2026-09-19

## Problem

The complete CPU validation chain in #532 proves the CCSD Lagrangian, Lambda,
orbital Z-vector, overlap/Pulay and AO cotangent mathematics, but its final
`integral_derivatives` oracle materializes coordinate-major AO derivative arrays.
For ERIs this is O(3Natom * NAO^4) storage and is unsuitable as the next GPU
production boundary. Reimplementing CC-specific derivative recurrences would also
duplicate the generic generated integral mathematics already qualified by #141
and #144.

## Decision

Keep the upstream RHF/CC/Lambda/Z and generated AO-weight ownership unchanged.
At the final coordinate derivative boundary, offer two explicit backends:

- `cpu`: retain the dense native derivative oracle as an independent validation
  path and regression reference;
- `cuda`: contract the fixed AO overlap/hcore weights through the existing
  generated CUDA S/T/V consumer and the ERI cotangent through existing #144
  consumers. Dense mode transforms a complete AO N^4 cotangent first. Shell mode
  uses generated C^4 block transforms and immediately contracts each public-AO
  shell quartet, so no complete AO N^4 cotangent is retained. Compute nuclear
  repulsion derivatives directly in O(Natom^2) scalar work.

A private post-HF bridge exposes the already-qualified one-electron CUDA consumer
to an owned `NativeSource`, mirroring the existing weighted-ERI bridge. The
bridge accepts arbitrary S/T/V cotangents and reports measured device/host
numeric storage, transfer bytes, upload count and synchronizations. It does not
register a native method or change public Calculator capabilities.

The complete-gradient endpoint exposes `derivative_backend="cpu"|"cuda"`,
`eri_weight_mode="dense"|"shell"`, and an independent per-call
`derivative_stage_budget_bytes`. CUDA failure has no CPU fallback. Shell mode is
an explicit bounded fallback; it currently trades many quartet launches for
lower AO-weight residency and therefore carries no speed claim.
`gradient_capabilities()` describes this internal force endpoint while
`method_capabilities("rccsd")` deliberately remains energy-only.

## Rejected alternatives

- Materialize the dense derivative tensor on GPU and copy/contract it later:
  preserves the O(3N*N^4) storage problem and adds device traffic.
- Add CCSD-specific one-/two-electron derivative kernels: duplicates generated
  recurrence mathematics and creates a second scientific owner.
- Mark RCCSD public/native forces supported now: the upstream response and AO
  weight chain is still host-side and dense, and the validation endpoint remains
  <=12 AO. Capability promotion requires a separate native prepared ownership
  contract and scalable weight flow.
- Silently fall back to the CPU derivative oracle on CUDA budget/device failure:
  hides the requested execution path and invalidates resource/performance evidence.

## Invariants

The CUDA and CPU derivative paths consume the exact same immutable AO cotangents,
use energy-gradient sign throughout, and add nuclear repulsion once. Hcore
cotangents apply identically to kinetic and attraction derivatives; overlap has
its own cotangent. Full ordered ERI cotangents retain the existing #144 convention.
No coordinate projection may hide a mismatch. CUDA stages are sequential and
bounded individually; caller weights/result, source ownership, CUDA context and
allocator overhead stay explicit exclusions.

SCF, CC, Lambda, Z and orbital-stationarity gates remain backend independent.
A failed derivative consumer cannot publish a `CCSDGradientResult`.

## Evidence

Host-side composition tests replace the GPU consumers with independent dense
oracle contractions and require exact agreement with the existing CPU endpoint,
including physical component decomposition and resource aggregation. A separate
block-transform test compares arbitrary generated shell-quartet cotangents with
slices of the dense C^4 transform, while shell-mode composition forbids entry to
the complete AO N^4 weight consumer. The opt-in
real-device tier compares arbitrary S/T/V and ERI weight contractions with the
dense derivative oracle, then complete H2/H2O/NH3/CH4 gradients with retained
analytic references. It also checks two CUDA stage budgets, dense-vs-shell ERI
weight parity, and rejects a one-byte budget without entering the dense CPU
derivative oracle. Exact executed results
belong in the PR record; configuration or compilation alone is not device evidence.

## Consequences

The dominant coordinate-by-AO derivative tensor is removed from the CUDA path,
but dense MO integrals/cotangents, CPU response solving and Python orchestration
remain. Dense mode also retains a full AO ERI cotangent; shell mode avoids that
specific allocation at the cost of more launches. Therefore this is bounded GPU
**derivative consumption**, not a resident
GPU CCSD gradient or a general-size performance qualification. Complete endpoint
measurements, not kernel timings, determine later promotion.

## Revisit when

A native CCSD prepared state owns resident amplitudes/Lambda/Z and can produce or
stream AO derivative cotangents without full dense MO/AO materialization. At that
point move this same generated consumer contract behind the native method API and
retain CPU-oracle/real-device parity tests as acceptance gates.

## References

#153, #532, #525, #144, #141; `docs/ccsd_gradient.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol
