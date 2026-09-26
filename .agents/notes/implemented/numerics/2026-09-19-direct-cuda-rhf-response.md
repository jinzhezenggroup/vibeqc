# Decision: reuse prepared exact CUDA J/K for RHF response

Status: implemented
Date: 2026-09-19

## Problem

The shared #179 response layer had a conventional CPU backend and a CUDA DF
backend, while #178 already supplied conventional CUDA second-integral
primitives. Substituting the DF response into a conventional Hessian would
change the Hamiltonian, not close the direct-CUDA gap.

## Decision

`CudaDirectJKBackend` is a thin adapter over the existing `FockPlan`, with
explicit exact J/K, derivative order zero, FP64 and zero screening. Its action
accepts signed symmetric response densities and returns raw J/K. It does not
rerun SCF, subtract nearly equal Fock matrices, introduce integral recurrences,
or duplicate #179's RHF operator or multi-RHS solver. The actual immutable
source shell records, representation and conventional reference identity are
bound explicitly. Owned plan cleanup and borrowed source lifetime are separate.

The current complete response invocation is host-orchestrated: AO/MO transforms,
Krylov vectors/orthogonalization/GMRES and returned matrices remain on the host.
FockPlan also prepares one-electron data and computes Fock/energy outputs not
needed by this consumer. These costs are not a hidden all-GPU or speedup claim.
Retained direct-J/K device allocations obey the supplied budget; preparation
peaks, host result copies, solver and complete Hessian budgets are not covered.

## Rejected alternatives

- Reuse CUDA DF response as if it were exact: it changes the operator.
- Treat a response density as a normalized, positive ground-state density:
  it may be indefinite, have negative eigenvalues and a zero metric trace.
- Add another CUDA integral formula or CPHF implementation: working provider
  and solver owners already exist.
- Label the adapter a GPU Hessian or device-resident solve: nuclear RHS,
  response transforms/Krylov storage, and complete directional assembly still
  have their own integration and qualification work.
- Hash the whole geometry/basis on every action: NativeSource already freezes
  scientific fields, so reuse its immutable identities and live-source guard.

## Invariants

The conventional Hamiltonian, zero screening, unscaled J/K, current reference,
and FP64 provider must match. CUDA failures are propagated, not retried through
a CPU/DF oracle. Outputs are not symmetrized to hide an operator defect. UHF/KS
and molecular Hessian/HVP capability remain disabled. New geometry/state needs
new owners; only compatible references may reuse response/recycling state.

## Evidence

- CPU adapter tests exercise exact source binding, invalid controls/densities,
  detached diagnostics, immutable output, closed-source/backend behavior,
  unsupported-provider/creation failures, incomplete results and clean replay.
- Shared CPU response plus Hessian weights/assembly/response regression:
  **93 passed, 11 skipped** (device-only cases and one existing optional tier).
  An explicit CPU-only-library check rejects CUDA with `NotImplementedError`.
- Real RTX 5090 (`sm_120`, driver 580.95.05): direct response device tests plus
  the existing DF response test: **10 passed**. Raw signed J/K checks include
  H2, LiH, water and f-shell HeH fixtures; RHF CPHF checks use independent MO
  matrices, orbital finite differences and all three shared multi-RHS modes.
  The action path forbids CPU shell-tile and CPU direct-response callbacks.
- Device memcheck on native-state/lifetime and impossible-budget/replay cases:
  **2 passed; 0 errors; 0 bytes leaked**. This is not a sanitizer claim for
  every historical GPU path.
- This change modifies no native/CUDA source. Device checks used a pinned copy
  of the existing CUDA library, SHA256
  `eecfc179ca4ce1098699a395b75fc5f06cffc4c8a14f9ab0c51d973f61c5220c`.
  It came from the existing `501a8f4` checkout's build, not a fresh CUDA build
  of this Python-only branch. No complete-endpoint performance was measured.

Reproduce the device suite in a finite Slurm allocation with
`VIBEQC_RESPONSE_CUDA_TEST=1`, a compatible explicitly selected CUDA library,
and `PYTHONPATH=python:.`; run
`python -m pytest -q tests/python/test_response_direct_cuda.py tests/python/test_response_cuda.py`.
The independent fixture provenance is retained under `tests/reference_data/posthf`.

## Consequences and revisit conditions

This closes the direct-CUDA **J/K action adapter** gap, not #179 or #180 as
umbrellas. Next compose a genuinely directional nuclear RHS and #178 weighted
second-integral/HVP consumers, including full density/energy-weighted-density
and overlap response. Move transform/vector execution into a shared resident
backend without creating a second solver. Qualify a complete conventional RHF
HVP against the tiny analytic Hessian and directional analytic-force differences
before exposing larger/full-Hessian or public derivative capabilities.

## References

#178, #179, #180, #449; `docs/response.md`; `docs/hessian.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
