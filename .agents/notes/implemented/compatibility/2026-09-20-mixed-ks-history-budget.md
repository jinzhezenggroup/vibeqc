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
AUTO also remains on the host-controlled KS path when the experimental
`VIBEQC_CUDA_KS_CHUNK=2` mode is selected, so device chunking cannot bypass the
FP32 mixed-J stage or its independent FP64 refinement.

## Evidence

The capacity/identity/argument-propagation regressions pass together with the
existing KS resource and chunk-control suites: 27 passed, seven opt-in tests
skipped. Ruff, clang-format and diff checks are clean.

On the merged scientific tree `f03ed249` (latest master `6efda7f4`), a full CUDA
13.0 / RTX 5090 (`sm_120`) monolithic build of `vibeqc_dft_api_tests` completed
and the real-device endpoint suite exited successfully. LDA/PBE RKS/UKS AUTO
all executed FP32 mixed-J work followed by strict FP64 refinement, including with
`VIBEQC_CUDA_KS_CHUNK=2` requested. PBE RKS H2 remained
`-1.152064375339672 Eh`.

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
