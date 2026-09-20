# Decision: contract one total RCCSD(T) response through the shared RCCSD derivative consumers

Status: implemented
Date: 2026-09-21

## Problem

#155 A established the complete conventional RCCSD(T) orbital/metric response: CCSD baseline, direct standard-(T), corrected-Lambda, orbital-energy denominator response, canonical-gauge response, and one total occupied-virtual RHF Z-vector. The remaining nuclear-gradient step could either duplicate the existing RCCSD AO/nuclear derivative machinery or consume that already-qualified total response. A duplicated gradient owner would create a second set of AO back-transforms, derivative contractions, CUDA consumers, memory gates, and component bookkeeping for the same mathematical h/g/S cotangents.

## Decision

The complete RCCSD(T) gradient endpoint treats `BoundCCSDTOrbitalResponse` as the only owner of correlated response mathematics. `BoundCCSDTGradient` adds no CC, Lambda, canonicalization, or Z equations. It accepts the immutable total h/g/S weights and decomposed components from that owner, then reuses the existing RCCSD derivative-consumer implementation for:

- MO-to-AO one-electron, overlap, and ERI weight transforms;
- the independent dense CPU derivative oracle;
- generated bounded CUDA one-electron derivative contraction;
- dense or shell-streamed weighted-ERI CUDA contraction; and
- nuclear-repulsion derivatives.

The result energy is rebuilt as RHF + converged RCCSD correlation + the same standard canonical (T) scalar. Publication requires the corrected-Lambda residuals, the single total Z residual, full orbital stationarity, current source/reference identities, and successful final derivative contraction. The endpoint remains internal and does not activate native/Public Calculator RCCSD(T) force capability.

## Rejected alternatives

A second `BoundCCSDGradient`-style constructor that regenerated CCSD(T) parameter pullbacks and solved another Z-vector was rejected because it would permit the force path to diverge from the reviewed #155 A response and could double-count orbital response. A separate handwritten PySCF-like CCSD(T) nuclear-gradient implementation was rejected because it would duplicate both the generated response equations and the existing integral-derivative consumers. PySCF remains an independent acceptance oracle only, never a runtime production dependency.

## Invariants

- Exactly one total physical occupied-virtual Z solve belongs to a successful RCCSD(T) gradient.
- Same-space canonicalization and direct denominator response are included before that Z solve and are never re-added during nuclear contraction.
- Nuclear derivative consumers see only final h/g/S cotangents; they do not know CCSD(T) equations.
- CPU and CUDA derivative backends contract the same total weights. A CUDA failure does not fall back to the dense CPU derivative oracle.
- Returned components are unprojected; translation or rotation errors cannot be hidden by post-processing.
- Public/native RCCSD(T) forces stay disabled until #155 C separately qualifies method registration, bindings, batch behavior, and failure propagation.

## Evidence

Exact-head Release CPU validation uses pinned PySCF 2.14.0 as an independent analytic oracle. H2O and NH3 both execute fresh RHF -> RCCSD -> corrected Lambda -> total RCCSD(T) response -> nuclear derivative contraction and meet the #155 maximum-force gate of `1e-6 Eh/bohr`. The tests also require nonzero direct-triples, delta-Lambda, and denominator-force components and require the component sum to reconstruct the unprojected final gradient.

The new endpoint is exercised in `tests/python/test_ccsd_t_complete_gradient.py`; the independent oracle and complete-energy finite-difference evidence remain in `tests/python/test_ccsd_t_gradient_validation.py`.

## Consequences

The final contraction layer stays method-agnostic and substantially smaller than a duplicated CCSD(T) gradient implementation. The current internal qualification retains the existing small conventional all-electron closed-shell RHF boundary and <=12-AO dense validation scope. GPU response ownership before the derivative consumer is still distinct from merely running the final contractions on CUDA.

## Revisit when

Revisit the ownership split only if a future native CC owner can retain the complete response and derivative state without crossing the current Python validation boundary, or if DF/frozen-core/open-shell/ECP variants require genuinely different final derivative semantics rather than different providers.

## References

- #155
- `.agents/notes/implemented/numerics/2026-09-20-ccsdt-orbital-response.md`
- `tools/vibeqc_cc/triples_complete_gradient.py`
- `tools/vibeqc_cc/complete_gradient.py`
- `tests/python/test_ccsd_t_complete_gradient.py`
- `tests/python/test_ccsd_t_gradient_validation.py`

Agent: ChatGPT
Model: GPT-5.6 Sol
