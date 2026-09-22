# Decision: ri mp2 lagrangian consumer

Status: implemented
Date: 2026-09-22

## Decision

Let response consumers request chemist-notation MO blocks through a common interface, with a CPU RI provider retaining transformed and whitened three-center values. Pull relaxed weights back through the shared inverse-square-root VJP.

## Invariants and rejected alternatives

The current ElectronInteractionSource contract and native work counters are retained. Move existing Coulomb-metric factor/response mathematics to an integral owner instead of adding post-HF dependencies on SCF. The bounded derivative adapter delegates to current generated s/p/d/f derivatives and rejects unbudgeted high-l fallback. Do not reinstate the recovered duplicate recurrence.

## Evidence and remaining qualification

The CPU library builds and the native gradient suite passes, including independent finite-difference A/M pullbacks and dense derivative contractions. Step sizes 1e-3, 2e-4 and 4e-5 produced directional errors approximately 4.4e-14, 6.8e-13 and 1.8e-12. The convergence check therefore admits a scale-dependent subtraction-roundoff floor while retaining its independent 2e-8 absolute gate. Full RI-MP2 force endpoint admission, rank-crossing and larger memory/work gates remain required. This slice does not claim a complete public RI-MP2 force method.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
