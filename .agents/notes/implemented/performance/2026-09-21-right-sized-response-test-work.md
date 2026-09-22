# Decision: right-size response contract tests instead of extending timeouts

Status: implemented
Date: 2026-09-21
Agent: ChatGPT
Model: GPT-6 Astra Pro

## Problem

PR #856 exposed a core job that completed 6403 tests but exhausted its job
budget. Long test counts alone did not explain the cost: large physical response
cases repeated a three-strategy multi-RHS matrix, immutable-state checks rebuilt
validated CCSD(T) responses, and rejection-only Hessian tests requested a dense
numerical oracle they never inspected. Worker-local fixture replication adds
another cost, addressed separately by the scheduling proposal in #859.

## Decision

Keep this change orthogonal to job scheduling, timeouts and production code:

- Retain the original RKS/UKS molecular response fixtures, independent Libxc and
  libcint actions, energy matching, transpose checks, three-step reconverged
  finite differences, and one full-size recycled multi-RHS replay.
- Qualify all three multi-RHS strategies on native H4 / H4+ instead of replaying
  every strategy on the larger physical fixture. H4 has a four-dimensional RKS
  response; H4+ has alpha/beta blocks of dimensions four and three. These are
  genuinely coupled native problems, not scalar H2 mocks. Check an independent
  true residual before using the single-RHS solution as the multi-RHS reference.
- Use a 12 x 4 x 8 atom-centred grid for these fixed-grid response tests. Both
  implementations use the identical grid; this is not a quadrature-convergence
  claim. Independent production-grid convergence tests remain unchanged.
- Separate the small live RHF state from the dense Hessian oracle fixture. Only
  tests that compare numerical Hessians request the latter. A regression guard
  forbids dense-oracle evaluation by the preflight fixture.
- Reuse the already validated immutable CCSD(T) response for frozen-attribute
  checks; retain fresh construction for forged-state rejection and numerical
  stationarity/weight tests.
- Preserve per-test JUnit timings in the existing debug artifacts, including
  every Python shard through its matrix-derived filename.

No existing test node is deleted. No numerical tolerance is loosened. This
change neither adds a CI shard nor changes routine/full qualification selection,
job budgets, concurrency, or public capabilities.

## Evidence

Matched before/after on node3 with four Slurm CPUs, Python 3.11.14, PySCF 2.14.0,
single-thread BLAS/OpenMP, pytest-xdist 3.8.0, `-n 4 --dist=worksteal`, and coverage
enabled. Both runs used the same Release CPU library and production code from
`4bcd60fde0437e9bd03f5ff9fdbb461ca8b910c3`.

The measured endpoint consists of these five complete test files:
`test_response_native_rks.py`, `test_response_native_uks.py`,
`test_ccsd_t_orbital_response.py`, `test_cc_triples_lambda_response.py`, and
`test_hessian_block.py`. Both LDA and PBE physical cases were included, even those
normally deselected by routine CI.

| Measurement | Baseline | Right-sized tests |
| --- | ---: | ---: |
| Passing tests | 60 | 65 |
| Pytest wall time | 227.79 s | 139.63 s |
| Complete process wall time | 228.43 s | 139.90 s |
| Sum of JUnit test/fixture elapsed times | 634.504 s | 452.695 s |

The complete-process reduction is 38.8%. The aggregate test/fixture elapsed
reduction is 28.7%; it is not a CPU-time measurement. XML comparison confirms
all 60 original node IDs remain and adds four small-system strategy cases plus
the preflight-fixture guard. There were no failures, errors or skips.

The RKS reconverged endpoint takes 114.38 -> 47.00 s (LDA) and 119.80 -> 47.21 s
(PBE); the UKS counterpart takes 61.41 -> 24.82 s and 66.72 -> 25.09 s. The new
small-system matrices also execute and are included in the total above.

This is one controlled hotspot-suite comparison, not a claim that all GitHub
Actions jobs become 38.8% faster. Hosted-runner and cache variance still require
full CI evidence. Raw logs/XML remain in ignored `.artifacts/ci-runtime/` on the
validation worktree; the reproducible command is:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1   python -m pytest tests/python/test_response_native_rks.py   tests/python/test_response_native_uks.py tests/python/test_ccsd_t_orbital_response.py   tests/python/test_cc_triples_lambda_response.py tests/python/test_hessian_block.py   -n 4 --dist=worksteal --durations=0 --cov --cov-report=   --junitxml=.artifacts/pytest-hotspots.xml -q
```

Set `PYTHONPATH=python:.` and `VIBEQC_LIBRARY` to the Release CPU build for an
in-tree run. The baseline and optimized worktrees must use the same production
revision when comparing timings.

## Rejected alternatives

Raising the job timeout is emergency headroom, not a work-reduction strategy.
A grid-only plus `loadfile` prototype passed all 60 tests but took 246.52 s,
slower than the 228.43 s baseline: serializing both large functional cases within
one file offset its reduced setup work. This negative result is retained rather
than using only favorable per-test timings. Merely adding another shard likewise
does not remove redundant computations, and is owned by #859 rather than this PR.

## Revisit when

New physical coverage requires a larger grid/molecule, an algorithm-specific
large-system regression is demonstrated, or per-test artifacts show that another
contract matrix dominates elapsed work. Add a focused physical regression then;
do not multiply every state/validation/strategy permutation by a large endpoint.

## References

PR #856 (timeout observation); PR #859 (independent shard-locality work).
