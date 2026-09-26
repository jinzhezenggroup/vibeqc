# Decision: qualify final occupied K independently of a raw response lease

Status: implemented
Date: 2026-09-26

## Problem

The single packed fitted-B owner admitted occupied SCF K but its final physical
Fock validation still required `packed_raw`. This conflated value execution
with the stronger raw-owner contract needed to lend a final projection to
force response. An earlier 768-AO/3712-auxiliary moved-warm trace spent 4.04 s
in one final dense K, compared with about 0.85 s per iterative occupied K.

The fitted occupied force consumer also repeated two spectral products to
apply a second inverse metric root, although the immutable value plan already
owned that symmetric root. This did not repeat the expensive AO transform, but
still performed avoidable rank-squared/auxiliary work and copied the factors.

## Decision

Use value-only work/capacity qualification for single-B final K. Exact retained
coefficients still require the matching final token, device generation and
solver status, and entry-for-entry density equality. The existing bounded
corrected streamed path also admits single-B: owner, model, solve epoch and
occupation must match, both generations must advance together by at most 16,
and the reconstructed factor must pass spectral, discarded-norm and complete
maximum/RMS density checks. Algebraic factors carry occupation once, with
exchange weight one. Strict physical Fock/final-state validation still runs.

Keep the raw projection lease separate. Single-B final K never publishes it,
including after successful correction. Every final-K attempt first revokes an
old lease, and outstanding corrected uploads/contractions drain on failure.

For an already qualified full-rank fitted occupied response, compute `U = S X`
using the existing symmetric root X and compiler-emitted GEMM. It replaces two
spectral GEMMs and one device copy with one GEMM into disjoint existing staging.
Raw and truncated consumers keep the spectral response. The `spectral` control
retains the old arithmetic for same-binary ablations; controls are included in
checkpoint/resource provenance. No new response allocation is introduced.

## Rejected alternatives

- Broadening the raw-resident qualification would falsely authorize a raw-A
  response lease for a plan that owns only B.
- Reusing the previous physical Fock would omit required final validation.
- Treating arbitrary corrected generations as current would admit detached
  factors; the bounded identity and density reconstruction proof remain.
- Promoting PR #939's factorized derivative fusion has no demonstrated endpoint
  benefit in its retained practical-auxiliary or 768-AO experiments. It changes
  the derivative consumer, not final K, and needs requalification on the newer
  fitted response path before any promotion.
- Recovering discarded raw metric directions from fitted B is invalid. The
  shortcut is restricted to the existing full-rank fitted consumer.

## Invariants

All complete endpoint samples, including cold, moved and diagnostic replays,
pass independent energy/force gates. DF and direct have different Hamiltonians,
so each is compared with its own independent reference. Timing comparisons use
identical orbitals, coordinates, convergence controls, properties and host
threads; they retain actual iteration and Fock counts. Instrumented trace
replays are excluded from clean timings. Warm DF ablations alternate order
from one frozen density within one owner.

## Evidence

The reproducible runner is `benchmarks/compare_df_direct_endpoint.py`.
Host lowering tests compare root layouts and full response adjoints with
independent dense algebra; native tests inject token, density, generation and
capacity failures and compare accepted J/K with a CPU raw-integral oracle.
Molecular GPU tests cover both orbital-sized and practical JKFIT auxiliaries,
forced correction, dense/spectral controls, cold/warm/moved states and response
lease/capacity counters. The qualification receipt records measured endpoints,
source/library identity, independent errors and executed work.

The [retained qualification](../../../../benchmarks/results/df-final-occupied-endpoint-20260926/README.md)
includes 124 complete large endpoints on the same sm_120 Release library and
8 host threads. At 768 AOs / 3712 auxiliaries, five frozen warm medians are
13.443 s (dense final/spectral), 10.254 s (occupied final/spectral) and 9.843 s
(occupied final/root), all with five SCF iterations. Direct takes 3.056 s with
one iteration. Moved-warm improves from 11.648 s to 8.042 s at three DF
iterations. Separate trace work removes one 705,481,932,800-FLOP root GEMM and
one 760,217,600-byte copy without increasing the 1,980,551,200-byte response
scratch. All measured independent E/F errors remain below 2.65e-10 Eh and
1.53e-10 Eh/Bohr. Native and forced-correction memchecks report zero errors.

Six fixed-budget packed-cache replay assertions fail identically with the
pre-change library. They are retained as an existing regression in the receipt,
not suppressed or counted as passing validation of this change.

## Consequences

The explicit single-B/fitted route becomes cheaper without broadening the
default storage or response selector. Its remaining occupied exchange and
response still contain substantial dense FP64 work; asymptotic DF scaling alone
is not an endpoint speed guarantee against screened direct J/K.

## Revisit when

A compatible final fitted projection lease can remove repeated occupied
projection with explicit layout, lifetime and budget ownership, or a derivative
consumer demonstrates an independently accurate complete-endpoint win. Revisit
root selection if conditioning gates or measured larger-size performance expose
a regression; the spectral path remains available.

## References

- [Current occupied CUDA contracts](../../../../docs/developer/df_occupied_cuda.md)
- [PR #939 assessment](https://github.com/jinzhezenggroup/vibeqc/pull/939#issuecomment-5844952104)
- [Previous fitted-response acceptance](https://github.com/jinzhezenggroup/vibeqc/pull/1371#pullrequestreview-5324823011)
