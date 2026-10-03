# Ordered AO-owner search qualification

This is a small, isolated geometry-reduction improvement, **not** closure of the
large PBE0 reference gap. Both variants use force-only local AO selection at
`1e-16`, dense SCF, 256-point tiles, 512 MiB device / 256 MiB host caps and a
16 MiB AO-map reserve inside the host cap. Indexed derivative pages and the
shared tile planner are not included.

Same-allocation baseline/candidate jobs 5581 (24 atoms) and 5582 (96 atoms):

| Atoms | Phase | Baseline seconds | Candidate seconds | SCF iterations |
| --- | --- | ---: | ---: | --- |
| 24 | cold | 274.198740 | 137.620109 | 21 / 21 |
| 24 | warm median | 4.734867 | 4.685815 | 1 / 1 |
| 24 | moved | 27.220673 | 27.106669 | 12 / 12 |
| 24 | moved-warm median | 4.726525 | 4.679087 | 1 / 1 |
| 96 | cold | 750.176532 | 584.307066 | 29 / 27 |
| 96 | warm median | 64.503557 | 63.449513 | 1 / 1 |
| 96 | moved | 253.660078 | 268.924869 | 12 / 13 |
| 96 | moved-warm median | 64.277945 | 63.544118 | 1 / 1 |

Each warm phase has five samples. Warm reductions are only 1.04% / 1.63%; the
96-atom changed-geometry endpoint **regresses**. Shared compiler caches and cold
iteration differences prevent a causal cold speedup claim. Do not mix these
absolute times with other allocations. References are reused numerical arrays,
not fresh GPU4PySCF timings or renewed XC backend provenance.

All 48 native calls pass every same-geometry reference-repeat gate: maximum
energy / force error `1.0595613e-10` / `3.2839231e-11`, against unchanged
`1e-8` / `1e-7` gates. AO work, discovery/cache counts and budgets are paired
equal except elapsed discovery time. All Becke pair work remains unchanged.

The standalone verifier pins both production-source/library identities and
the actual measured checkout base `a718695de`, rather than relabeling these
as executions of a later publishing commit:

```bash
python benchmarks/results/pbe0-ordered-ao-owners-20261003/verify.py
```

`campaign.json.xz` losslessly stores 40 exact UTF-8 members; `storage.json`
records every member hash. It includes complete endpoint arrays/iterations,
the production patch, launch/build recipes, host work-probe logs, native basis
ownership, initial **failed** job5579 and subsequent qualification logs.
Job5588 passes six native routing and eight independent PBE0 force checks,
then six routing tests in each of memcheck/initcheck/synccheck/racecheck with
zero errors/hazards. Job5589 passes four formerly missing UKS/r2SCAN analytic
oracles after separate packaged-artifact deployment. Job5592 verifies the
actual 24/96-atom basis owner labels are monotone; it is not a timing run.

Slurm accounting is disabled and older completed jobs have left the controller;
the retained final controller receipt is for 5588 only. Other job identities
come from original launch and endpoint logs, not reconstructed accounting.
All GPU work was submitted through finite `main` / `gpu:5090:1` allocations.
