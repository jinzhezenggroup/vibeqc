# Decision: Bootstrap GFN2-xTB with a pinned embedded runtime

Status: implemented
Date: 2026-09-19

## Problem

The semiempirical compiler roadmap in #499-#505 deliberately excludes SCC iteration,
eigensolver/library policy, occupations, convergence state, and public runtime
admission. Compiler representability therefore could not by itself make GFN2-xTB
executable. GFN2 also owns an intrinsic minimal basis, while VibeQC previously
required every System to contain Gaussian shells.

A production CPU endpoint additionally needs a reproducible LP64 LAPACKE/CBLAS
provider without inheriting arbitrary process-global BLAS state.

## Decision

Bootstrap the runtime from xTBloom commit
`5a67cc59ace94c8296e873503b2ae1298e7c2861` as a VibeQC-owned, statically linked,
transitional source provider. No installed xTBloom library or `xtb` executable is
required at runtime.

Admit molecular CPU `gfn2-xtb` energy and analytic forces first. Charge and
standard shared-orbital restricted open-shell states are supported; CUDA, ragged
batch execution, and explicit two-channel unrestricted GFN2 remain separate gates.

Permit atom-only System descriptors for intrinsic-basis methods, while HF, DFT,
and MP2 explicitly continue to require Gaussian shells. Gaussian basis or ECP
descriptors are rejected for GFN2.

Linux wheels reuse xTBloom's reviewed private
`scipy-openblas32==0.3.34.0.0` provider boundary. A small sibling shim gives
auditwheel a dependency edge; wheel repair vendors and collision-renames the
provider cohort. The package does not publish scipy-openblas32 as a runtime
Python dependency.

Compiler-generated GFN2 owners from #504/#505 should replace duplicated
handwritten scientific equations after independent qualification. SCC iteration,
convergence policy, occupations, eigensolver selection, and public capability
admission remain runtime responsibilities.

## Rejected alternatives

- Shelling out to `xtb` or requiring an installed xTBloom runtime: breaks the
  self-contained runtime and weakens identity/failure control.
- Supplying STO-3G as a placeholder basis: changes the scientific model.
- Blocking all runtime work on #504/#505: couples runtime policy to compiler
  representability unnecessarily.
- Silently using arbitrary process-global BLAS: risks ABI/interposition and
  reproducibility problems.

## Invariants

- No hidden external xTB/xTBloom runtime dependency.
- GFN2 uses its intrinsic basis; Gaussian methods still reject atom-only input.
- Current public GFN2 capability is CPU-only; CUDA requests fail explicitly.
- Multiplicity maps to unpaired electrons without silently selecting a different
  two-channel model.
- Energy/forces require independent oracle and derivative gates.
- Pinned provider source participates in VibeQC build identity.

## Evidence

H3+ matches the independent tblite 0.7.0 golden energy to machine precision
(`-0.8989438125591576 Eh` versus `-0.8989438125591571 Eh`). The OH radical
matches the xTB 6.7.1 golden energy and force at approximately 1e-9 atomic units.
An OH bond-direction finite difference converges to the analytic force with an
error of about 3e-9 Eh/bohr at a 1e-4 bohr step.

The focused Python suite plus existing calculator regression passed, all native
CTest targets passed, and an auditwheel-repaired wheel independently reproduced
the H3+ result with the private OpenBLAS cohort.

## Consequences

The bootstrap vendors a broader xTBloom source snapshot than the final GFN2
owner needs and currently carries extra compile work. Linux wheels also gain the
private OpenBLAS cohort. These costs are accepted for the first complete runtime
and can be reduced without changing the public scientific contract.

## Revisit when

Revisit scientific ownership when #504/#505 generated kernels are qualified.
Revisit runtime admission for ragged batching, explicit unrestricted states,
warm starts, richer diagnostics, and native CUDA under #560.

## References

#499, #504, #505, #560

Agent: ChatGPT
Model: GPT-5.6 Sol
