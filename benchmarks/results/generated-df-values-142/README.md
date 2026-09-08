# Generated DF value promotion (#142)

The generated raw Coulomb metric `M[P,Q]` and three-center tensor
`A[mu,nu,P]` passed the numerical, native integration, resource, and public
endpoint gates below. Generated values are selected by default. The bounded
source selects a primitive-reduction warp per output; the bulk compatibility
builder retains one thread per output. Explicit reference, auxiliary, and
component controls remain available for comparisons.

## Environment and artifact identity

Measurements used an NVIDIA GeForce RTX 5090 through Slurm's `main` partition,
CUDA 12.9.86, and the Release native library compiled for `sm_120`, with AOT
four-center kernels enabled and CUDA fast compilation disabled. The native
build used one split-compilation thread. `native-resources.json` records the
exact CMake settings, compiler version, tested library and CUDA core object
hashes, and entry-point resource usage.

The final default-policy rebuild changes the host policy and library identity.
Its CUDA core object hash is identical to the full-matrix build.
`automatic.json` records the promoted library hash and a separate Slurm job
with both DF overrides absent. It selected `generated_rys` / `primitive`,
passed the native DF suite, and reproduced complete spherical M/A and RI-J/K
references across raw pair/auxiliary tiles of 7/3. Individual validation reports
retain `production_promoted: false` because running a validator does not itself
change production policy; `promotion.json` records the selection decision.

## Numerical and integration gates

| Gate | Coverage | Result |
| --- | --- | --- |
| Isolated values | 162 libcint fixtures, 67,679 primitive records, 64/128/256 threads; all 16 metric and 64 three-center signatures | Passed; maximum Cartesian error 1.836e-12, independently projected spherical error 9.369e-13 |
| Native source | 32 runs: complete Cartesian/spherical s/p/d/f, long contractions, duplicate auxiliary functions, reference plus all three mappings, two tile layouts, batch 2 | Passed; maximum M/A errors 5.299e-12 / 1.684e-12 |
| Native RI-J/K | Independent NumPy factorization/contractions at identical threshold, RHF and UHF densities | Passed; maximum J/K errors 6.402e-12 / 2.219e-12; duplicate metric rank 4 of 5 |
| Public endpoints | 20 RHF water/UHF OH comparisons, batches 1/2, budgets 0/8/16 MiB, spherical STO-3G orbitals and def2-SVP auxiliary basis | Passed; maximum generated/reference energy error 2.544e-12 hartree, force error 1.751e-10 |
| Geometry and budgets | Every warm sample, cross-budget energies/forces, every item's changed geometry, reported memory peaks | Passed |
| Automatic selection | Unset DF overrides, native suite and spherical source probe | Passed, Slurm job 8973 |

The source fixtures deliberately vary per-item geometry and primitive offsets.
Raw tiles include split AO rows and final partial tiles; the probe also checks
empty blocks and invalid ranges. Independent J/K replay uses bounded partial
row panels rather than one-row panels: its pair tile is
`min(nbf*nbf, max(raw_pair_tile, nbf*(nbf/2+1)))`. Both actual tile dimensions
are recorded, so raw reconstruction and J/K timings must not be conflated.

The endpoint matrix exposed an existing open-shell rejection in the shared
CUDA one-electron setup. Both single and batch builders now use general spin
packing because these integrals are spin independent. A native H2+ regression
and the UHF OH endpoints cover the fix.

## Schedule selection and performance

`downselect.json` compares all three mappings on a complete RHF endpoint.
Its preliminary `changed_warm` phase restored the original geometry and is
explicitly excluded; cold, initial warm, and changed-geometry measurements
support the downselection. The final endpoint driver passes changed coordinates
on every replay and checks fixed-geometry energy/force consistency.
Component mapping regressed substantially and was rejected from automatic
selection. The final endpoint matrix compares primitive and auxiliary mappings;
the bulk path is measured once because those overrides only affect the bounded
source. CUDA context initialization is preconditioned before timing. Cold
execution, changed-geometry rebuild, and three warm executions before and after
the geometry change are recorded separately, and every sample is numerically
checked.

Primitive-source speedups over the retained Hermite evaluator across positive
budgets and batch sizes are:

| Endpoint | Cold execution | Geometry rebuild | Warm execution |
| --- | --- | --- | --- |
| RHF | 1.624–1.633x | 1.617–1.620x | 1.515–1.538x |
| UHF | 2.048–2.057x | 2.063–2.069x | 1.640–1.703x |

These are complete energy/force endpoint measurements on the stated small
systems, not universal hardware or basis guarantees. The cached bulk route is
approximately unchanged; its warm timings do not measure integral generation.
For complete raw source tensors with full-size tiles, primitive mapping took
2.450 ms versus 44.949 ms for Cartesian values, and 10.348 ms versus 355.212 ms
for spherical values. See `source.json` for partial-tile timings separately.

## Resource and placement boundaries

The standalone generated fixture uses 130 registers and a 160-byte stack.
Native value entry points retain the reference branch and report up to 255
registers and 26,528 bytes of stack; derivative entries report up to 52,672
bytes. These native figures are not the isolated generated helper's footprint.
`LOCAL=0` is not evidence that there are no spills or stack accesses. Full
standalone ptxas/cuobjdump output and native resource records are archived.

Positive-budget sources generate and publicly transform requested tiles on
CUDA. Transform metadata is copied to the device; metric values are staged
through the host and returned to CUDA for the existing cuSOLVER factorization.
The bulk compatibility route stages raw M/A through the host, applies the
existing public transform there, then uploads the J/K input. Neither route is
claimed to have entirely device-resident setup.

The maximum conservative reported host-plus-device peak in the endpoint
promotion gate was 1,763,116 bytes, below both positive budgets. These are the
existing source/plan diagnostics, including host staging and reported force
scratch; they exclude CUDA context/driver allocations (including implicit
kernel stack storage) and the validation driver's
complete reconstructed reference tensors. This experiment establishes the
budget gate for these fixtures, not a new process-wide memory accounting
contract. Metric factorization, rank/condition diagnostics, derivative
recurrences, and direct-SCF gates are unchanged.

## Reproduction

Use a Python environment with NumPy, pytest, and PySCF, and an AOT-enabled CUDA
Release build. Set `VIBEQC_LIBRARY` to its `libvibeqc.so`, set `PYTHONPATH` to
the repository and its `python` directory, and set `OMP_NUM_THREADS=1`.
Expose the CUDA runtime libraries through the normal loader path. Each real
GPU invocation must preserve Slurm's assigned `CUDA_VISIBLE_DEVICES`.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  python tools/validate_df_values.py --local --directory build/df-value-matrix \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc --architecture sm_120

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 \
  python tools/validate_df_source.py --probe build/cuda-release/vibeqc_df_value_probe \
  --directory build/df-source-release

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:30:00 \
  python tools/validate_df_endpoints.py --output build/df-endpoints-release.json \
  --routes primitive auxiliary --repeats 3

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env -u VIBEQC_DF_VALUES -u VIBEQC_DF_VALUE_MAPPING \
  build/cuda-release/vibeqc_density_fitting_tests

python tools/summarize_df_promotion.py \
  --directory benchmarks/results/generated-df-values-142
```

The full Python suite passed 659 tests (74 skipped), all nine native CPU suites
passed, and the native CUDA DF suite passed under Slurm. Full GPU matrix jobs
were 8965 (isolated), 8971 (source), and 8974 (corrected endpoints).
