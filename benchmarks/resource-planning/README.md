# Resource inventory evidence (#203)

The scripts check declared allocation bounds against actual owned capacities
and verify energies/forces. They do not benchmark performance or treat process
RSS / free GPU memory as provider ownership.

Build a CPU Release library with `VIBEQC_ENABLE_CUDA=OFF`, then run:

```bash
PYTHONPATH=python:. python benchmarks/resource-planning/cpu_inventory.py \
  --build build --output /tmp/cpu-resources.json
```

The standalone C++ helper interposes `new`/`delete` only in the audit executable.
Input parsing and normalization precede measurement. The measured interval
includes fleet preparation, two serialized solves and destruction, including
integral/Jet/recurrence temporaries absent from ordinary iteration samples.
Every case must release all measured storage and agree with the CPU endpoint
within `1e-9` Hartree.

The checked-in [CPU receipt](cpu-receipt.json) records these byte counts; all
tracked allocations were released after destruction:

| Case | Predicted host peak | Observed C++ heap peak |
| --- | ---: | ---: |
| H2 RHF | 62,152 | 5,304 |
| Ragged RHF | 788,504 | 463,992 |
| Ragged UHF | 788,408 | 463,880 |
| Ragged RI-RHF | 793,112 | 464,488 |
| H2 RI-UHF, def2-SVP | 2,459,232 | 1,391,224 |

Build CUDA Release for the assigned GPU architecture with
`VIBEQC_CUDA_FAST_COMPILE=OFF`. Run all real GPU work through the scheduler:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 env PYTHONPATH=python:. VIBEQC_PROFILE=off \
  LD_LIBRARY_PATH=/path/to/cuda/targets/x86_64-linux/lib \
  VIBEQC_NVCC=/path/to/nvcc OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/resource-planning/cuda_inventory.py --build build-cuda \
  --cache /tmp/tensor-resource-cache --output /tmp/cuda-resources.json
```

The CUDA audit checks ragged RHF/UHF direct, resident DF, and regenerated DF
against ordinary corresponding cold/warm histories and CPU oracles. It also
executes HF and recomputed TensorIR concurrently resident under one budget,
checks native ledger charge release, and emits deliberate infeasible plans.
Opaque/runtime exclusions remain in each receipt; DF solver/library allowances
are conservative policies rather than measured peaks. Fast compilation builds
are rejected as resource receipts.

The checked-in [CUDA receipt](cuda-receipt.json) uses CUDA 12.9, Release `-O3`,
`sm_120`, and `VIBEQC_ENABLE_AOT_SHELLS=OFF`. The supported small-HF inventory
does not use AOT shell kernels. All 45 production CUDA Python checks and 12
applicable native suites passed. The additional AOT profile-selection test
expects a tuned AOT registry on this GPU and fails with this deliberately
disabled registry; it is outside this build's validation scope.

| Case | Planned numeric device capacity | Observed cold / warm numeric peak |
| --- | ---: | ---: |
| Ragged RHF direct | 69,872 | 37,208 / 37,208 |
| Ragged RHF resident DF | 335,651,116 | 612,864 / 83,666 |
| Ragged RHF regenerated DF | 335,659,316 | 615,006 / 66,036 |
| Ragged UHF direct | 92,944 | 49,632 / 49,632 |
| Ragged UHF resident DF | 335,651,116 | 613,104 / 85,146 |
| Ragged UHF regenerated DF | 335,659,316 | 615,246 / 67,516 |

DF's numeric bounds include conservative solver/recovery capacity; the
separate opaque-library allowances are excluded from this table. They are
deliberately loose for these small fixtures. The largest energy/force
differences from the CPU oracle were below `5e-13` Hartree / `1.1e-10`
Hartree/bohr. All ledger charges were released after cache destruction. The
joint HF/TensorIR case used at most 135,280 tracked device bytes under a 137,696
byte global plan and selected recomputation.

Focused tests include actual native allocation rejection and rollback:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 env PYTHONPATH=python:. VIBEQC_PROFILE=off \
  LD_LIBRARY_PATH=/path/to/cuda/targets/x86_64-linux/lib \
  VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_TENSOR_CUDA_TEST=1 \
  VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so" VIBEQC_NVCC=/path/to/nvcc \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_hf_resources_cuda.py \
  tests/python/test_tensor_cuda_execution.py
```

See [the API and accounting contract](../../docs/resource_planning.md) for
supported dimensions, lifetimes, retry boundaries and excluded overhead.

Use the same CUDA installation for NVCC, the HF library and the runtime library
path. On a host with several toolkits, generated TensorIR libraries can otherwise
load a system cuBLAS that is incompatible with the HF library's cuSOLVER.
