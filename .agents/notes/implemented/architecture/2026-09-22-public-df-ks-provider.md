# Decision: bind public KS energies to the shared DF provider

Status: implemented
Date: 2026-09-22

## Problem

DF J/K providers and KS iteration machinery existed, but public DFT descriptors
rejected DF and the resident CUDA KS owner borrowed only a direct-J stream.
Removing the admission checks alone would still execute the wrong Hamiltonian.

## Decision

Resolve DF explicitly in the native Fock specification and construct the same
PreparedFockPlan used by existing shared DF consumers. CUDA KS borrows its DF
stream and device-pointer J adapter. CPU global hybrids select DF for both J
and K. Auxiliary descriptor pointees are copied and geometry changes rebind
both sets of centers. No new integral or XC equations are introduced.

## Invariants

Density/Fock matrices remain resident during CUDA iteration. DF metric cutoff,
source ownership, bounded value tiles and provider budgets remain authoritative.
Final-state identities retain the actual fitted Fock specification. Unqualified
conventional derivative snapshots and whole-KS resource inventories fail closed.
The direct-J solver-region prototype cannot receive a fitted source.

## Rejected alternatives

A host-staged KS/DF adapter would introduce matrix traffic each iteration.
Silently accepting mixed CPU-DF/CUDA-KS requests would obscure backend ownership.
Reusing the conventional force consumer would omit metric/auxiliary response.
Reusing the direct KS resource estimate would give a false allocation contract.

## Evidence

`tests/python/test_dft_df_public.py` covers independent PySCF energies, CUDA
residency diagnostics, warm replay, moved auxiliary centers, and failure gates.
See `docs/user/dft_density_fitting.md` for the explicit scientific acceptance limits.

## Revisit when

DF stationary force composition, mixed precision, global CUDA hybrid execution,
or the combined DF+KS allocation inventory gains independent qualification.
