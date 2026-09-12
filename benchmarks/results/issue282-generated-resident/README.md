# Generated resident DF evidence (#282, resident slice)

This checkpoint permits a generated source to retain its transformed tensor
when the complete value-plan allowance fits. It changes forward storage policy,
not the DF model or its derivative. It does not yet optimize the constrained
streamed traversal or generated force-response schedule.

The binary's SHA-256 and native checkpoint are in `summary.json`. It was built
from `8697318` plus the archived resident patch; native code was then committed
as `787304d`. The archive retains the patch and CMake cache, and raw probes
record exact source/library hashes. All GPU commands used finite Slurm jobs.

Three unprofiled energy/energy-plus-force pairs use a 512 MiB DF sub-budget:

| AO | energy median (s) | energy + force median (s) | iterations | maximum energy difference (Eh) | maximum force difference (Eh/bohr) |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.444077 | 1.329604 | 34 | 6.03e-12 | 2.42e-11 |
| 192 | 4.347016 | 9.100057 | 39 | 8.65e-12 | 7.32e-11 |

Differences compare the pre-optimization compatibility arrays in
`../issue283-component-baseline`. Paired energy and force solves use identical
iteration counts. Separate diagnostic pairs use 256 MiB (96 AO) and 512 MiB
(192 AO). Each fresh solve materializes exactly one full raw/transformed tensor;
subsequent resident J/K calls generate no tiles. Graph construction is not
counted as replay. The pre-residency positive-budget 96-AO diagnostic timed out
before completing its first energy solve in ten minutes; that failed attempt
is retained and is not a measured speedup denominator.

The full five-repeat energy-plus-force matrix completed with an 8 GiB declared
DF sub-budget. Its original command left optional numerical error limits unset;
the independent all-repeat audit in `../issue282-streamed-panels/summary.json`
passes at 1e-9 Eh and 1e-8 Eh/bohr. Every value-plan diagnostic reports resident
execution. All cross-engine warm-iteration branches remain unmatched.

| AO / batch | VibeQC warm median (s) | GPU4PySCF warm median (s) | max energy error (Eh) | max force error (Eh/bohr) |
|---:|---:|---:|---:|---:|
| 96 / 1 | 1.109108 | 0.244805 | 6.66e-12 | 3.53e-11 |
| 96 / 4 | 3.772757 | 0.973100 | 5.97e-12 | 3.81e-11 |
| 192 / 1 | 7.999114 | 0.331100 | 9.33e-12 | 7.84e-11 |
| 192 / 4 | 31.972642 | 1.326801 | 2.81e-11 | 7.18e-11 |

These warm endpoints remain slower than the optimized host compatibility
route. The resident policy does not establish a general force-speed advantage;
generated response is still expensive. No matched GPU4PySCF speed claim is made.

The preceding 1 GiB matrix passed both 96-AO cases but rejected both 192-AO
cases before SCF. The existing batched one-electron preparation preflight
reserves a conservative shell-quartet-sized metadata bound. All failures and
logs are retained. Using 8 GiB demonstrates the resident implementation; it
does not prove that the actual endpoint needs 8 GiB. The value-plan memory
diagnostics exclude force staging and opaque CUDA library retention.

Validation: four GPU native suites, 14 global resource GPU tests, 37 generated
derivative GPU tests and 71 host checks passed. The derivative tier includes
Cartesian/spherical RHF/UHF, geometry changes, auxiliary-only centers,
independent PySCF, two finite-difference steps, rank crossing and failed-item
isolation. The resource tests compare actual forward tile decisions with the
global plan; generated response no longer implies streamed forward storage.

The hash manifest inventories 39 archived files (216,275 compressed bytes).
Every member was restored and compared byte for byte:

```bash
python -m tools.unpack_evidence benchmarks/results/issue282-generated-resident \
  --output build/issue282-generated-resident-restored
```

Use a fresh output directory. Reproduction commands are recorded in matrix
manifests. Run the force probe with `--repeats 3 --memory-budget-bytes 536870912`
and an explicitly selected frozen library. Diagnostic traces additionally use
`--component-trace-dir NEW_DIR`; ordinary performance samples do not enable
tracing. The full matrix used the pinned GPU4PySCF environment and
`--memory-budget-bytes 8589934592 --repeats 5` in a 25-minute allocation.

The resident implementation is one #282 slice. Complete streamed reuse,
energy-only batch evidence, improved preparation accounting and integration
acceptance remain outstanding.
