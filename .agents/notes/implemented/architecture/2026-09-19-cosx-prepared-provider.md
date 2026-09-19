# Decision: Execute COSX through a DFT-owned prepared Fock composite

Status: implemented
Date: 2026-09-19

## Problem

The native CUDA COSX arithmetic is qualified and the #202 strategy layer can
represent COSX explicitly, but the ordinary PreparedFockPlan has an intentional
exact/DF provider model. Making SCF import DFT/COSX would reverse the existing
dependency direction, while simply adding COSX to the old exact-vs-DF branch
would risk silently routing it through DF.

A production candidate also needs one bounded device-budget contract covering
both the J provider and COSX K.

## Decision

Add dft::PreparedCosxFockPlan as the owner of fixed-density CUDA COSX
composition. It consumes one mixed ResolvedFockBuild, constructs the dedicated
COSX MolecularGrid directly from FockCosxSpec, preflights the COSX storage,
then allocates the remaining admitted device budget to a J-only
scf::PreparedFockPlan.

The adapter returns ordinary scf::DirectJkMatrices. Restricted execution builds
K from the spin-summed density; unrestricted execution builds independent
alpha/beta K matrices. Standard assemble_fock and contract_fock_energy remain
the only coefficient/energy assembly logic.

The regular PreparedFockPlan explicitly rejects SeminumericalCosx before its
exact/DF source selection. Thus making cuda.cosx executable cannot activate a
fall-through path.

The cuda.cosx registration becomes Executable only in CUDA builds. CPU COSX
remains a correctness oracle. Public C Fock ABI, AUTO selection, SCF iteration,
batching and derivatives remain unsupported.

## Rejected alternatives

- Add DFT/COSX directly to scf::PreparedFockPlan: rejected because it reverses
  the subsystem dependency and expands the old exact/DF owner beyond its
  intended boundary.
- Treat COSX as DF in the provider variant: rejected as a silent mathematical
  approximation change.
- Give J and COSX independent unbounded allocations: rejected because the
  composed provider would have no auditable resource admission.
- Promote SCF/force capability together with fixed-density K: rejected because
  the discrete COSX derivative and warm prepared-state lifecycle are separate
  acceptance work.

## Invariants

- One mixed ResolvedFockBuild remains the mathematical authority.
- COSX grid materialization must exactly reproduce every FockCosxSpec field.
- COSX device bytes are admitted before allocation; the remaining total budget
  bounds the J provider.
- The legacy PreparedFockPlan must reject any present COSX term.
- RHF uses spin-summed COSX K and UHF uses independent spin K.
- All coefficients are applied once by the shared Fock/energy assembly.
- No first derivative, SCF, AUTO, public ABI or batching capability follows
  from fixed-density provider execution.

## Evidence

vibeqc_cosx_fock_provider_tests compares CUDA RI-J/COSX-K RHF and UHF matrices
against independent CPU DF J and Slice-A CPU COSX K oracles, then compares the
shared two-electron energy and assembled Fock. It checks resolved grid identity,
bounded total device accounting, insufficient-budget rejection and rejection
by the legacy PreparedFockPlan.

The lower-level native COSX kernels remain independently checked by
vibeqc_cosx_cuda_tests, including spherical d/f public AO expansion.

## Consequences

Hybrid DFT can now consume a real fixed-density COSX exchange provider without
adding COSX branches to SCF. The next slice can add prepared lifecycle and
host-SCF integration in DFT space, while derivatives remain explicitly absent.

## Revisit when

A common spatial/exchange provider layer moves the native COSX ownership out of
DFT without reversing dependencies, or when complete analytic COSX derivatives
are available for a separately qualified force provider.

## References

- #202
- #246
- PR #572
- PR #575
- docs/fock_build.md
- docs/cosx_reference.md

Agent: ChatGPT
Model: GPT-5.6 Sol
