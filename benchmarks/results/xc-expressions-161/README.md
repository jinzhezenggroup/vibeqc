# XC expressions and derivatives (#161)

Scientific revision: `ac7c15219acb1c8a5eab68d640cb4291d4089ec8` (clean at every
record). These are explicit XC candidates; none is promoted and no public DFT
method is registered. See `docs/xc_expressions.md` for expression provenance,
energy/feature units, derivative ordering and the finite-domain contract.

All 42 cases passed: seven functionals × two spin modes × three output schedules.
Each record contains typical and named boundary fixtures, a fixed-feature
consumer, changed inputs, and ten interleaved CPU/CUDA workflow timings.
There are **420 raw timing samples**, each consuming all requested outputs.
Typical maximum normalized error is **0.000886** against a passing threshold of
one (`atol=1e-11`, `rtol=1e-10`). Boundary maximum is **0.018858**, using the
separately documented `atol=1e-8`, `rtol=2e-6` and independent exchange oracle.

| Variant | Maximum registers | Maximum stack bytes | Maximum spill load + store bytes | Observed CPU-interpreter / CUDA consumer ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline, one output/kernel | 138 | 0 | 0 | 0.977–4.032 |
| fused, all outputs/kernel | 254 | 56 | 128 | 1.092–13.632 |
| split, eight outputs/kernel | 254 | 40 | 104 | 1.079–9.657 |

The eight-output PBE candidate is spill-free, but the corresponding polarized
PBE-correlation candidate still spills. Therefore neither maximal fusion nor
one fixed grouping is assumed to win. The baseline is spill-free throughout.
Per-function resources and explicit `spill_free` fields are in each record.
These observations do not select a production winner.

The timing scope is **host feature arrays -> bounded upload -> native XC
kernels -> download -> weighted output consumption** for 8,192 points with
256-point tiles and five repetitions. Inputs are repeated physical reference
features, not an SCF trajectory or a molecular quadrature result. The CPU route
is VibeQC's interpretable DAG, not Libxc/GPU4PySCF. No molecular energy, force or
competitive DFT speedup is claimed. Compilation, program construction, native
preparation, transfer and kernel timing are recorded separately. Python object
and CUDA module/driver storage are outside the numeric budget; observed device
allocation deltas are retained. Peak memory is explicitly unavailable, so these
records cannot support performance promotion.

The actual device was RTX 5090 on `node3`, UUID
`GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, NVIDIA driver string `580.95.05`,
runtime integer `12090`, driver integer `13000`, NVCC/PTXAS CUDA 12.9.86.
This native runtime creates no cuBLAS handle or workspace; provider retained
allocation is zero. Every real GPU test, sanitizer and timing run used Slurm's
`main` partition with `--gres=gpu:5090:1` and a finite time limit.

Validation logs record:

- 853 full Python passes and 174 explicit skips; the later capability regression
  is included in the 79 passing focused tests (final CI covers the final tree).
- All 10 native CPU suites passed.
- All 44 opt-in CUDA tests passed, covering every fixture and derivative,
  7/31-point tiles, empty/partial tiles, replay, changed inputs, failures,
  independent concurrent owners, vacuum energy and consumer pruning.
- Compute Sanitizer repeated all 44 tests: **zero errors, zero bytes leaked**.
- Pinned PySCF 2.14.0/Libxc 7.0.0 fixture generation reproduced byte for byte.
- All pre-commit hooks passed.

Raw Libxc discrepancies are preserved in `reference_diagnostics`. Boundary
exchange uses separate analytic energy/gradient/Hessian formulas because Libxc's
reduced-variable evaluation can produce a large nonzero cross-spin derivative
where spin separability requires exact zero. Correlation remains checked against
unmodified Libxc, and typical-domain comparisons use Libxc for every entry.
Directional finite differences, independently derived Hessian symmetry, spin
exchange, restricted chain rules and AO matrix-potential factors are also tested.

Reproduce this matrix from the scientific revision:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/validate_xc.py --tier endpoint \
  --points 8192 --tile-points 256 --repeats 5 --output /tmp/xc161-final-evidence
```

Each case retains `report.json` and `contract.json`; compiler resource diagnostics
are extracted into the retention audit. Source is
deterministically regenerated from the contract and scientific revision rather
than duplicated here. `manifest.json` hashes every raw archived file.

Routine log/XML files named in this historical account are now represented in
[the retention audit](../retention-238/migration.json), with extracted measurements,
diagnostic conclusions, and exact original Git/checksum identities.
