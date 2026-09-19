# Decision: Keep first COSX SCF slice energy-only and DFT-owned

Status: implemented
Date: 2026-09-19

## Problem

The fixed-density RI-J/COSX-K provider establishes the physical Fock operator,
but #246 also requires RHF/UHF energy execution. Reusing the generic host HF
driver directly would require scf::PreparedFockPlan and its derivative/final
state contract, while moving COSX implementation into SCF would reverse the
existing DFT-to-SCF dependency.

The first iterative endpoint must also avoid publishing a DIIS-extrapolated
matrix as a physical converged state.

## Decision

Add DFT-owned run_cosx_rhf and run_cosx_uhf entry points over
PreparedCosxFockPlan. They reuse the common initial-density helpers, DIIS,
reference generalized eigensolver and AO commutator definitions, while every
physical J/K build goes through the same mixed prepared provider.

The convergence gate requires all of:
- energy change below the requested tolerance;
- density change below the requested tolerance;
- physical commutator residual below min(1e-9, density_tolerance).

After the iterative gate succeeds, the endpoint rebuilds the unextrapolated
physical RI-J/COSX-K Fock, diagonalizes it, reconstructs the integer-occupation
density, rebuilds the physical Fock again, and rechecks energy, density drift
and commutator before retaining a converged result.

RHF uses the spin-summed density convention already qualified by the provider.
UHF retains independent alpha/beta densities and unit occupations.

## Deliberate boundary

This slice rejects compute_forces=true and ScfOptions proposal hooks. It does
not add batching, public C ABI/method registration, AUTO COSX selection,
screening/fitting or hybrid-functional dispatch. All arithmetic remains FP64.

A failed trajectory returns its last density and never executes the converged
two-build finalization path.

## Rejected alternatives

- Reuse scf::solver::run_rhf_host_plan by making PreparedCosxFockPlan pretend to
  be PreparedFockPlan: rejected because it would weaken typed provider
  ownership and derivative assumptions.
- Move COSX into SCF merely to reuse the existing driver: rejected because it
  reverses dependency direction.
- Declare convergence from DIIS energy/density alone: rejected because the
  retained state must also satisfy the physical target operator.
- Enable forces with empty or borrowed exact/DF derivatives: rejected because
  energy and derivatives must use identical COSX semantics.
- Enable proposal hooks immediately: rejected until proposal validation/trial
  evaluation is explicitly bound to this target operator and its resource
  model.

## Invariants

- Every physical Fock in the trajectory uses one resolved RI-J/COSX-K model.
- A converged result is rebuilt from the unextrapolated physical Fock.
- Returned E and D pass an independent physical commutator check.
- Warm starts cannot change COSX grid/provider identity.
- Force requests fail before an iterative calculation is published.
- Nonconverged runs never execute finalization or claim convergence.

## Evidence

vibeqc_cosx_scf_tests runs cold and warm RHF and UHF fixtures, strict warm
validation, one-iteration failure behavior and force rejection. The returned
RHF/UHF densities are independently evaluated with CPU density-fitted J and the
Slice-A CPU discrete COSX K oracle; energy and physical commutator must match
the retained endpoint.

Lower-level device and fixed-density provider correctness remains separately
covered by vibeqc_cosx_cuda_tests and vibeqc_cosx_fock_provider_tests.

## Consequences

#246 now has an internal energy-only RHF/UHF execution path suitable for
method-level hybrid integration experiments. The next correctness milestone is
not performance promotion: it is complete discrete COSX first derivatives and
provider-consistent forces, or alternatively explicit method exposure that
keeps forces unsupported.

## Revisit when

Proposal hooks, batching or public method integration have their own
provider/lifetime contracts, or when the complete COSX analytic derivative is
qualified against finite differences of the same discrete model.

## References

- #246
- PR #572
- PR #575
- PR #578
- docs/cosx_reference.md
- docs/fock_build.md
- tests/native/test_cosx_scf.cpp

Agent: ChatGPT
Model: GPT-5.6 Sol
