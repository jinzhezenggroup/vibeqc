# #310 ordinary-device final eigensolve qualification

> **Checkout retention (2026-09-21):** `raw-evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue310-final-eigen/raw-evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue310-final-eigen/raw-evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

For the standard raw-evidence bundle, verify or unpack it against the retained
member manifest with:

```bash
python -m tools.unpack_evidence benchmarks/results/issue310-final-eigen \
  --archive .artifacts/issue310-final-eigen/raw-evidence.zip \
  --output .artifacts/issue310-final-eigen-unpacked
```

The fused CUDA DF finalizer replaces only its required CPU-reference Fock
eigensolves (and RHF canonical export) with the existing ordinary FP64 Xsyevd
provider. Physical Fock rebuilds, density projection and the complete analytic
force chain remain. Cold overlap/core setup and the independent host iterative
Fock API are outside this increment; verified final-state retention remains #311.

## Separate provider-only endpoint comparison

Both selections retain the same #309 lazy/cached preparation and frozen warm
seed. Baseline explicitly restores actual CPU-reference final solves; candidate
uses the ordinary device provider. Five interleaved pairs per workload, RHF
spherical def2-SVP and matching auxiliary basis, 1 GiB DF allowance, RTX 5090.

| AO / batch | Clean warm endpoint | Reference (s) | Device (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 96 / 1 | energy | 0.086677 | 0.019830 | 4.371 |
| 96 / 1 | force | 0.615482 | 0.540784 | 1.138 |
| 96 / 4 | energy | 0.324612 | 0.057174 | 5.678 |
| 96 / 4 | force | 1.926040 | 1.560712 | 1.234 |
| 192 / 1 | energy | 1.107644 | 0.069145 | 16.019 |
| 192 / 1 | force | 4.245735 | 3.175263 | 1.337 |

Every iteration/retry branch matches. Actual warm overlap/core reference calls
are zero for both selections; final reference/device calls per RHF item are
1/0 at baseline and 0/1 for candidate, with no observed fallback. Separate
intrusive traces verify these counts and are excluded from clean timing.
All cold, unchanged, energy/complete-force and changed-geometry samples remain
in the archive, including setup/destruction and original-geometry restoration
between changed samples. Replay energy/full-force gates pass at 1e-9 Eh and
1e-8 Eh/Bohr against separately prepared cold endpoints. These same-model gates
are not external GPU4PySCF parity or a claim that all remaining work is done.

## Qualification, resources and retained failures

26 CPU native and 176 protocol checks pass (19 CUDA-only skips there). Five
GPU native and 125 GPU Python checks pass. Analytic degenerate matrices at
2/12/32/96/192/384/513 AOs, batch 1/4, verify eigenvalues and invariant occupied
projectors; small independent CPU oracles and physical-reference export check
scientific conventions. Molecular tests include RHF/UHF, Cartesian/spherical,
cold/warm/changed geometry, full forces, bad neighbors and recovery. Memcheck
of the native eigensystem suite reports zero errors and zero leaked bytes.
All real-device tests, queries and benchmarks use finite Slurm allocations.

The metric cuSOLVER handle/parameters now survive to plan teardown. One lazy
serialized workspace is explicitly charged per bucket, with a full additional
workspace for an independent cold retry. Real CUDA 12.9.1 queries at 1–1536 AOs
show about 0.5 MiB fixed cost even for tiny matrices. The shape allowance is
1 MiB + 16 n^2 doubles, with actual device/host queries checked before allocation.
Tiny DF budgets remain explicit OOM cases; feasible parity/recovery fixtures
include the existing half-budget value-plan partition. No CPU fallback hides
resource rejection. Original handle-lifetime, underestimated-capacity and test
expectation failures are archived separately with their source commits and
resolutions, so failed attempts cannot be mistaken for accepted measurements.

A separate one-repeat fresh-calculation component probe (Slurm 9485) retains
actual J/K/force operations, tensor counters, final solve transfer bytes and
CUDA event intervals for 96/192 AOs. These intrusive cold records are diagnostic,
not warm timings. Graph-capture entries are not executed kernel time; event
intervals may include host-induced device idle time. At 96 AOs the named force
fraction exceeds one because other cold-call overhead changes between samples;
that ratio is not a valid percentage of the clean endpoint.

Source/library identity is checked against the compiled scientific hash. Every
original result, trace text, qualification log and reproduction/query script is
hashed in the standard ZIP manifest, restored and compared byte for byte.

```bash
python -m tools.unpack_evidence benchmarks/results/issue310-final-eigen \
  --archive .artifacts/issue310-final-eigen/raw-evidence.zip \
  --output /tmp/issue310-final-evidence
```

Rebuild the manifest source with CUDA 12.9.1, Release, architecture 120 and AOT
shells disabled as in the baseline recipe. The restored `reproduction.sh`
records all nine #206 invocations. Adjust checkout/interpreter paths and use
fresh outputs, then run with `srun --partition=main --gres=gpu:5090:1 --nodes=1
--ntasks=1 --time=00:30:00 bash <script>`, preserving scheduler visibility.
Remaining cold consumers, full state-consistency/correction work, larger and
constrained timing domains, and the #206/#308 acceptance matrix stay open.
