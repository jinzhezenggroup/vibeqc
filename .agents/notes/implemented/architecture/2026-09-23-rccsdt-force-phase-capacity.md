# Decision: admit the complete retained RCCSD(T) force lifetime

Status: implemented
Date: 2026-09-23

## Problem

The first conventional CPU RCCSD(T) public-force owner qualified its numerical
results but passed the same remaining budget to multiple response stages while
retaining their outputs. Its reported capacity was the maximum of selected
triples/Lambda/GMRES scratch diagnostics. That omitted simultaneous parameter,
raw Hamiltonian, denominator, canonicalization and orbital-response data, and
the generated Hamiltonian/Fock arenas. Hamiltonian output copies also include
an occupied-virtual RHS beyond the four square matrices and ERI tensor.

Corrected Lambda added dense energy-source copies and retained a packed source
through its independent residual check without extending its resource bound.
The outer triples owner omitted the newly retained physical RHF reference.

## Decision

`plan_rccsdt_force_cpu` performs checked, allocation-free admission before any
force tensor or generated response kernel. Its named triples, Lambda,
parameter, raw-provider, response and derivative phases include borrowed
molecule/reference/CC data once and all previous outputs that remain alive.
The plan reuses generated arena size functions and the existing MO provider
capacity contract; it includes source scratch, curvature-check matrix copies,
GMRES, shell transform stages, copied response outputs, and final force data.
The ordinary triples response page capacity of at most 16 is reserved before
execution. The plan is a conservative admitted bound, not a minimum possible
allocation or a measured process-memory value.

`max_bytes` now denotes the complete force allowance. The public owner passes
the total correlation budget and reports the maximum of earlier preparation/
energy phases and the complete force plan. It does not add borrowed state a
second time. Actual retained vector capacities are used where available.
Prepared execution resource observations include the admitted force capacity
and orbital GMRES workspace. The existing measured-GMRES fields keep their
separate meaning; no measured complete-endpoint value is fabricated.

Corrected Lambda projects directly into one packed energy-source vector, keeps
that vector charged through both residual evaluations, and shares its capacity
calculation with the composed planner. Shape/finiteness admission remains
explicit. Force and orbital-response results move to their published owners.
These ownership changes leave the force mathematics and scientific gates intact.

## Evidence

The real CPU Release library is linked into an independent allocation-intercept
probe. Physical H2O and NH3 runs exercise generated triples, corrected Lambda,
Hamiltonian/Fock/orbital kernels and conventional derivative contraction. The
test verifies that nested observed allocations plus retained inputs fit the
plan, runs at the exact admitted cap, rejects one byte below before numerical
allocation, and increases the bound when a retained problem vector reserves
extra capacity. It observes allocations larger than 500 kB, preventing a
non-interposing allocator from silently passing.

The public H2O force also runs at its reported complete endpoint budget and
rejects one byte below. Corrected Lambda has a separate exact-cap and early
rejection test. Independent pinned PySCF analytic gradients for H2O/NH3 retain
the 1e-6 Eh/Bohr gate; the existing three-step reconverged energy finite
difference, canonical energy, batch/replay, unsupported-domain and failure
checks remain in the qualification suite.

## Rejected alternatives and boundaries

Increasing the default budget or changing only the final diagnostic would not
enforce simultaneous lifetime admission. Summing separately reported total
capacities would double-charge borrowed references and miss hidden outputs.
The explicit phase composition avoids both errors. Further reductions should
first shorten owner lifetimes or change a schedule and then update the same
allocation tests; they must not relabel retained data as transient scratch.

The qualified public force remains all-electron, closed-shell, conventional
CPU with at most 12 AOs. CUDA, DF, frozen-core, ECP and open-shell force variants
remain unsupported. No full T3 tensor or dense CC Jacobian is materialized;
the small physical orbital-response matrix retains its existing role.
