# Decision: general resident occupied DF work policy

Status: implemented
Date: 2026-09-18

## Problem

PR #443 extended automatic occupied admission from 768/160 to 384/80, with an
exact RTX 5090 product-name check repeated in SCF and response. These useful
benchmark points had become source-level policy, requiring another edit for
every new size or GPU. Reservation and force selection could drift apart.

## Decision

`df_occupied_exchange_preferred` is a CPU-safe, device-independent work policy.
For n orbital AOs, a auxiliaries and occupied rank r, the existing dense
projection and contraction cost 4*a*n^3 FLOPs. Occupied projection and a full
Gram cost at most 4*a*n^2*r; SYRK can lower the Gram cost further. Require at
least a twofold arithmetic reduction, equivalently 0 < r <= floor(n/2).
The twofold margin is conservative headroom for validation/BLAS overhead, not
a calibrated device latency threshold. Division avoids overflow when checking
costs and native BLAS index bounds. Equal orbital/auxiliary dimensions have no
mathematical role in this algorithm and are no longer an admission condition.

Reservation receives a known RHF occupation from the method owner. Unknown
reference/rank and UHF pass zero; high/zero ranks fail the shared work rule.
The same hint travels through native tile planning, plan creation/cache identity
and versioned common resource queries. Old shape-only query ABIs remain valid
and never guess the method from dimensions. Explicit occupied mode retains its
conservative two-full-factor reservation for all references.

Automatic storage is optional. If its charge would force a host-raw resident
plan into partial/streamed tiles or make the budget fail, the planner retries
with the original dense reservation and passes a zero hint to creation. Dense
generated sources cannot use the resident occupied implementation and receive
no automatic charge. Packed sources may drop the optional SCF reservation
without removing their independently requested all-Q U storage. Common resource
accounting uses the returned admission hint, including a separately queried
singleton cold-recovery plan. Actual factor allocation uses the final ranks.

`qualified_resident_rhf_exchange` combines this work rule with actual resident
capacity. SCF, seed factorization, final physical K and force response share it.
It requires singleton non-streamed storage, full AO rows, reserved factors and
an all-Q occupied projection. Host-raw plans require full auxiliary scratch;
packed sources also require raw storage and sufficient logical rank capacity.
Automatic occupied execution is RHF only. Dense response can independently
borrow existing full J/K scratch under compatible singleton shell/BLAS controls:
this avoids repeated Q projections even when occupied factors are unavailable,
requires no new allocation and does not authorize factor reuse. Packed scratch
still requires qualified occupied factors. Explicit diagnostic controls retain
their checked behavior. No device query or product-name check is necessary for the
ordinary FP64 BLAS algorithm.

## Invariants

- Work/capacity selection never authorizes a density factor. Exact owner,
  model, token, density, solve epoch and device-generation validation remains.
- Seed eigenspectrum and full-matrix reconstruction tolerances are unchanged.
- Full-rank metric and the exclusive final-projection lease are still required
  for whitening inversion. Truncated metrics use the raw response projection.
- Packed response still verifies simultaneous rank-squared scratch capacity;
  failed factor validation uses the bounded raw loader.
- Explicit dense, occupied, panel, schedule and probe controls remain available.
- Derivative schedule, pair layout and primitive packet policies are separate;
  their remaining benchmark-specific gates are tracked in #445.

## Rejected alternatives

Reserving at a hypothetical rank one before reference/rank is known charges
UHF and high-rank jobs for storage they cannot consume; under tight budgets it
can force streaming or reject an otherwise valid job. The review exposed this
compatibility regression, so method-aware hints replace that initial design.

Adding another endpoint preserves the maintenance problem. Replacing endpoints
with an unexplained AO-size threshold would merely conceal it. Coupling this
policy to the generated derivative tuning tables would mix independent
algorithms: those profiles describe angular classes and primitive signatures,
not dense versus occupied BLAS work. A device latency model without matched
endpoint measurements would imply unsupported precision.

## Evidence

CPU tests compare contraction FLOP counts across even/odd dimensions, unequal
auxiliaries and rank boundaries, including 384/80, 512/123 and 768/160. They
cover native indexing overflow, invalid dimensions, batches, explicit controls
and tight-budget UHF/high-rank and optional-factor fallback boundaries. CUDA selector tests exercise
actual versus advertised projection capacity, missing storage, residency,
RHF/UHF and explicit fallbacks without querying a device identity.

The changed-geometry OH/UHF replay can execute strict final-state corrections,
which advance the determinant beyond the retained canonical factors. Its route
test now requires host-trace correction evidence before expecting dense fallback;
the initial canonical replay must still use occupied factors, and independent
energy/force tolerances are unchanged.

Numerical validation and measured results are recorded in the PR. The dedicated
automatic endpoint test uses independent PySCF energies and full analytic
forces at 96/192/384/768 AOs, with cold/warm solves and executed provenance.
Practical unequal-auxiliary tests exercise geometry changes and tighter force
gates. This policy does not close #439's separate clean performance ledger.

## Consequences and revisit conditions

New dimensions and GPUs need no whitelist edit. Small contractions or different
FP64 hardware can have a latency crossover not predicted by FLOPs; complete
endpoint measurements, including iterations and provenance work, remain the
acceptance criterion for a performance claim. Revisit the work margin with
matched device/workload profiles, or add a centralized calibrated policy when
those profiles exist. Preserve explicit overrides for that comparison.

## References

- PR #443 and issues #439 / #206.
- [Occupied response derivation](2026-09-15-occupied-df-response.md).
- [Final projection lifetime](2026-09-16-df-tuning-and-projection.md).
- [Current occupied contracts](../../../../docs/developer/df_occupied_cuda.md).
