# Decision: force-aware numerical policy is evidence-driven and fail-closed

Status: implemented
Date: 2026-09-21

## Problem

Energy-smallness is not a safe proxy for derivative-smallness. NUM03 needs a numerical policy that can allocate screening and XC quadrature effort against explicit energy and force targets without duplicating the schedule-selection/runtime ownership of #168/#798 or the stationary CUDA execution work under #660.

The policy also needs to distinguish mathematical target identity from execution effort. A candidate grid is an approximation to one strict finite-basis/XC/grid target; changing the candidate grid must not silently redefine the target being audited.

## Decision

- Represent observable error as an energy absolute error plus the full per-atom, per-Cartesian-component absolute force-error tensor. max_abs, RMS, per-atom max, and per-atom L2 aggregations are derived without component cancellation.
- Resolve the numerical mathematical target with NumericalTargetModel: method, geometry, orbital-basis identity, XC-functional identity, strict grid identity, derivative semantics, and optional density-fitting identity. Backend, schedule, screening threshold and adaptive state are execution/numerical-effort choices and are excluded from target identity.
- Record omitted-work evidence in ContributionLedger. Estimator and independently observed strict-reference error are distinct fields. Componentwise absolute summation is available as a conservative allocation envelope, but it does not convert empirical inputs into rigorous bounds.
- Use paired grid/screening differences only as a limited-domain empirical estimator. Calibration requires multiple molecular families; whole-family leakage into holdout evaluation is rejected. False-success, overconservatism and energy/force coverage are reported explicitly.
- Adapt only with a hysteretic controller. Missing evidence, branch mismatch or other uncertainty tightens/retries or falls back to strict; it cannot silently pass. Grid/mask transitions are recorded separately from smooth-branch finite-difference evidence.
- Final verification uses an independently observed error against the policy's declared strict target level. strict_reproducible mode bypasses adaptive relaxation.
- The real benchmark separates fixed-AO-density XC quadrature experiments with a rebuilt moving grid/partition, reconverged full energy-plus-force endpoints, and estimator-policy cost. Strict oracle time is excluded from production-policy cost only when used solely for offline validation; paired endpoints required by the estimator are charged.

## Rejected alternatives

- Energy-only screening: rejected because small energy contributions can carry significant geometry derivatives.
- Treating a paired difference as a certified bound: rejected; the estimator has no universal proof and is labelled empirical.
- Changing the target identity with every candidate grid: rejected because it makes candidate-vs-target error ill-defined.
- Charging the strict validation oracle to every adaptive production call: rejected for performance accounting. The oracle is required for benchmark truth, not for every production decision. If the policy actually tightens to strict, its cost is charged.
- Duplicating #798 schedule/cache/profile plumbing: rejected. NUM03 consumes public execution contracts and owns error allocation, not schedule selection.

## Invariants

1. Energy and force targets remain independent; a small energy estimate cannot override a force-target miss.
2. Actual strict-reference error is never populated from an estimator.
3. Empirical evidence is never described as rigorous/certified.
4. Uncertainty fails closed through tighten/retry/strict fallback.
5. Moving-grid and partition response are part of the analytic derivative target.
6. Smooth-branch finite differences and discrete grid/mask switching are reported separately.
7. Performance success requires complete energy-plus-force endpoint improvement at matched actual force accuracy. Isolated kernel savings are insufficient.
8. The public strict target remains reproducible and can be selected unconditionally.

## Evidence

Local managed-worktree validation before native/qz execution:

- tests/python/test_force_aware_numerics.py: 8 passed.
- Ruff on the new policy, tests and benchmark harness: clean.
- Benchmark CLI/pure-Python case+level construction: clean.
- Combined accuracy regression: 57 passed; 5 native-dependent tests could not start because this Windows managed worktree has no native library/build. Those failures occur at library loading before tested accuracy logic and are rerun in the qz build.
- A deliberate unit counterexample has energy error below target while force error is above target; the controller tightens.
- A synthetic held-family test explicitly exercises both false-success and overconservative reporting.

Real qz/Inspire measurements are retained in PR #828 and its compact persistent benchmark artifact rather than duplicating mutable benchmark numbers in this rationale note.

## Consequences

A paired empirical estimator can cost an additional complete endpoint. It is therefore possible, and acceptable, for the first complete benchmark to show no end-to-end speedup even when a looser individual endpoint is faster. Such a negative result remains part of acceptance evidence rather than being replaced by a kernel-only claim.

The core policy is intentionally independent of one DFT scheduling implementation, so #798/#168 can change how a validated schedule executes without changing NUM03 target or error semantics.

## Revisit when

- a mathematically proved derivative bound replaces the empirical paired envelope;
- a public DFT target-model contract supersedes NumericalTargetModel;
- stationary execution exposes a reusable derivative-aware shell/grid/auxiliary ledger;
- calibration evidence supports a materially broader method/basis/geometry domain; or
- #798/#168 exposes a stable public schedule contract that can be consumed without transferring schedule ownership into NUM03.

## References

- GitHub issue #175 (NUM03)
- #173 accuracy model/evidence contract
- #163 complete KS gradients
- #168 and PR #798 schedule/autotuning execution
- #234 spatial execution ownership
- python/vibeqc/force_aware_numerics.py
- tools/vibeqc_numerics/force_aware_benchmark.py
