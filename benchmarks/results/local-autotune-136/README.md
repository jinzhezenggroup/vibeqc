# User-local autotuning acceptance (#136)

The supported CLI tuned water/def2-SVP from the generic CUDA baseline on an
allocated RTX 5090 (`sm_120`), atomically activated a local native library, and
exported it. A later process automatically selected that profile. Import into
an independent cache passed the same compatibility checks and reproduced the
energy and analytic forces.

## Measured outcome

Source revision: `02b6897`. CUDA toolkit/NVCC 12.9.86; runtime 12090; driver 13000.
Slurm jobs 8962 (tuning) and 8963 (reuse/import). The native source fingerprint
and complete hardware/toolchain identities are in `profile.json` and
`evidence.json`. The baseline and candidates are Release builds with fast
compilation disabled; the native compiler cache was already populated.

The bounded quick run selected PPPS and PSPS, covering 26.33% of measured active
primitive work. This run explicitly limited coverage to two classes and four
schedules per consumer; it compiled no absent f-shell classes.

| Proposal | Complete endpoint result | Decision |
| --- | --- | --- |
| PPPS Fock | 107.531 ms → 65.053 ms; 1.653× speedup; bootstrap 95% lower bound 1.627× | Accepted |
| PSPS Fock, after PPPS | 0.971× speedup; lower bound 0.962× | Rejected |
| PPPS/PSPS force | No schedule passed the developer tuner's gates | Rejected |

Each endpoint comparison used six samples per side in balanced fresh-process
ABBA order. Timings include the complete warm energy/analytic-force replay from
a frozen converged density; startup, cold convergence, and warmup are excluded.
The accepted endpoint has maximum energy error `1.28e-13` Eh, force error
`6.50e-14` Eh/Bohr, and translation residual `1.24e-14` Eh/Bohr. Every replay
converged on the same one-iteration SCF branch.

The final PPPS native object passed 14 independent libcint fixtures across
RHF/UHF direct and persistent wrappers (56 executions), with maximum absolute
Fock error `6.70e-16`. All source, object, cubin, schedule, and driver hashes,
compiler resources, raw samples, and rejected candidates remain in
`evidence.json`. The final accepted library exactly reproduced the binary used
for endpoint acceptance.

`diagnostics.json` proves automatic selection from the tuning cache. `reuse.json`
and `reuse-{baseline,local}.json` record fresh-process imported-profile use:
energy error `4.27e-14` Eh and force error `1.01e-13` Eh/Bohr against the official
baseline. The same job confirmed that a cached-ERI no-op and an exhausted tuning
budget preserve the active index byte-for-byte, incompatible architecture
metadata falls back safely, and clearing profiles preserves immutable libraries
while deactivating their use.

The archive's SHA-256 is
`8ae1ec8d34b7b4229b874473a426e9a6f9c9e1eadab990e61ef66f9258461442`.
Its binary is intentionally kept outside Git; `profile.json` records its hash.
The checked-in metadata is evidence, not an installable binary bundle.

## Reproduce

Build the matching source with CUDA enabled, `CMAKE_BUILD_TYPE=Release`,
`VIBEQC_CUDA_FAST_COMPILE=OFF`, and the local CUDA toolkit. Install the Python
package with its `autotune` optional dependency. Set `VIBEQC_LIBRARY` to that
baseline library, `CUDA_PATH` to the toolkit, and an isolated
`VIBEQC_PROFILE_CACHE`. Preserve the scheduler's device visibility.

Use this Angstrom XYZ input:

```text
3
Water
O 0 0 0
H 0 -0.757 0.587
H 0 0.757 0.587
```

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:30:00 env OMP_NUM_THREADS=1 \
  python -m vibeqc autotune --quick water.xyz --basis def2-svp \
  --portable-baseline --max-classes 2 --compile-jobs 4 --repeats 6 \
  --budget-seconds 1600 --export local-profile.zip

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:10:00 env OMP_NUM_THREADS=1 \
  python benchmarks/results/local-autotune-136/verify_reuse.py \
  local-profile.zip build/local-reuse-check
```

The second command requires a newly created output directory and preserves its
logs and reports. A no-winner result remains valid behavior on different
hardware or under different timing conditions.

## Regression checks

The local Python suite passes 647 tests (74 skipped); all nine native CPU suites
pass. The shared code generator now covers normalized Fock expression graphs
and value-only Coulomb lookup strides independently of geometry scratch sizes.
The exact native-object gate exposed these integration issues before any
incorrect profile was activated. The two affected golden shard hashes include
a documented correctness update in the integral-IR artifact fixture.
