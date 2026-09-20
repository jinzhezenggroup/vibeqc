# Decision: bind mixed-J KS resources to both nonlinear stages

Status: implemented
Date: 2026-09-20

## Problem

The CUDA AUTO mode gives the mixed-J stage and strict FP64 refinement separate
iteration budgets. Reserving and reporting only one stage's diagnostic history
allows the retained vector to grow beyond the capacity reported at preparation.
The Python planner also omitted the precision policy, giving AUTO the FP64
resource identity and history allowance.

## Decision

Reserve AUTO native diagnostic history for two full iteration budgets plus the
four bounded strict-final-closure iterations. Propagate Calculator precision
into the KS request; give AUTO a distinct resource identity and budget both the
retained and exported history for that maximum. Preserve the default FP64
identity and capacity behavior. Reject CPU AUTO and unknown precision modes.

The concurrent ECP RKS final-closure condition is preserved alongside the mixed-J
transition: UKS and ECP RKS both retain the strict physical-state closure gate.

## Evidence

Ten new capacity/identity/argument-propagation regressions fail before the repair
and pass after it. Combined resource suites: 21 passed, seven opt-in real-device
tests skipped. The CUDA KS host translation unit passes C++20 syntax compilation
with CUDA headers. These results do not substitute for exact-head GPU numerical
qualification of the mixed-J implementation; the PR remains draft for that gate.

## Invariants

No SCF equation, convergence tolerance, derivative admission, or strict-refinement
iteration budget is relaxed. A different resource policy must not reuse the FP64
plan identity. Updating the strict-closure iteration bound requires updating its
matching AUTO host-capacity allowance and regression.

## References

- #603
- python/vibeqc/resources_ks.py
- src/dft/cuda_ks.cpp
- tests/python/test_ks_mixed_resources.py
