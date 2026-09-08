# CG09: prepared FP64 TensorIR CUDA evidence

These records validate issue #146 at clean implementation revision
`766bc67417e3a559423231687139ab97497574d0`. The preceding implementation commit
is `77d54bd5584b584048b7388ceaa555247c725fa6`. The second commit includes the
retained-provider allocation correction described below. Results from the
earlier budget model are not used for promotion.

## Validation

| Gate | Result |
| --- | --- |
| Complete Python regression suite | 721 passed, 99 skipped |
| Native CPU suites | 9 passed |
| Real CUDA tests | 25 passed |
| Compute Sanitizer on those CUDA tests | 0 errors; 0 bytes leaked |
| Fixed tensor endpoint cases | 18 passed (6 equation/shape buckets × 3 scales) |
| Maximum absolute error, including constrained plans | `2.5579538487363607e-13` |
| Numerical gates | `atol=1e-11`, `rtol=1e-10` |
| Raw interleaved baseline/candidate samples | 1,152 |
| Maximum generated-kernel registers / stack / spills | 42 / 0 bytes / 0 bytes |
| Candidate native compilation times | 2.99–3.21 seconds |

GPU execution used finite Slurm allocations on `main` with
`--gres=gpu:5090:1`. The final endpoint job was **8996**; the independent CUPTI
allocation audit was **8995**. The device was an NVIDIA GeForce RTX 5090,
`sm_120`, UUID `8e9c9e1ae183258c0b3a03a5ddebb2f8`. Exact NVCC/PTXAS,
driver/runtime/cuBLAS and host identities are retained in the JSON records.
Other architectures have not received numerical validation in this evidence.

The fragments use fixed supplied tensors, with no molecular Hamiltonian or CC
iterations. Their occupied/virtual populations are `(3,7)`, `(5,17)`, `(8,31)`;
each runs at scales `0.001`, `1`, and `100`. `tools/vibeqc_tensor/cuda_fixtures.py`
and the exported equation files define the seed, populations, exact factors
and reproducible values. The independent CPU TensorIR interpreter supplies
each reference; separate GPU tests cover every intermediate of the smaller
explicit-loop reference examples.

## Conservative selection

| Fragment / nocc,nvir | Selected plan | Paired median speedup across the three scales |
| --- | --- | --- |
| Denominator update / 3,7 | Validated view/elementwise fusion candidate | 1.108–1.139× |
| Denominator update / 5,17 | Validated view/elementwise fusion candidate | 1.089–1.113× |
| Denominator update / 8,31 | Baseline | No candidate passed every gate |
| Virtual residual / all three shapes | Baseline | No candidate passed every gate |

Selection requires CPU/unfused-GPU parity, spill/resource limits, the shared
CG01 noise/>2% gate, and a bootstrap speedup lower bound above one on **every**
provided scale. Each `*-tuning.json` retains accepted and rejected candidates,
all raw samples, compiler identities and resource records. These are warmed
complete-program timings including validation, host layout staging, transfers,
packing, cuBLAS, generated kernels, result allocation and error checks.
Startup and synchronized section profiles are reported separately and do not
decide selection. Unselected optimizations do not become the default.

## Allocation accounting and the provider audit

The largest selected-plan capacity bound is **110,984,456 bytes**. The largest
forced constrained-plan bound is **106,230,536 bytes**, including zero explicit
cuBLAS workspace, partial panels and forced root recomputation. Every successful
native tensor allocation matches `plan.allocation_bytes`; its measured
provider delta fits the separately charged allowance. Adding both plus the
conservative host capacity never exceeds the advertised plan peak.

CUPTI showed these retained device allocation requests during `cublasCreate`:

```text
1024 bytes
131072 bytes
67108864 bytes
```

Installing a 4 MiB user workspace did not release them. Neither the audited
ordinary GEMM nor strided-batched GEMM requested another allocation. This is
why a zero user workspace does **not** imply a zero-cost cuBLAS handle.
`provider-allocation-audit.cu` and its raw log preserve the independent audit.

Each plan that uses cuBLAS now charges a minimum/default **96 MiB provider
allowance** in addition to its explicit workspace. Preparation checks the
provider's allocation delta before creating tensor storage and cleans up a
rejected handle. In this endpoint run the provider check measured **69,206,016
bytes** per handle, including its observable allocator overhead. Kernel-only
plans charge no cuBLAS allowance. Tests cover CPU rejection of an omitted
allowance and deliberately bypass that CPU guard to check native rollback.

Budgets also include resident tensors/constants/outputs, index tables, arena
reuse, packing panels, T/R/DIIS reservations, input staging, bounded host
validation scratch and one output set. The recorded peaks are conservative
capacity bounds within that scope. General CUDA context/module/stack storage,
provider host metadata, caller-owned arrays and allocation page rounding remain
outside it; device-wide deltas are reported separately. They are not presented
as portable total-VRAM upper bounds.

Tuning retains the baseline and one candidate concurrently, each with its own
plan budget. The largest sum of their capacity bounds here is **221,476,880
bytes**. CPU reference data is outside production-plan accounting. Prepared
production batches explicitly check the sum of all retained per-system peaks.

## Reproduction

Use a source checkout, the recorded CUDA toolchain, and `PYTHONPATH=.:python`.
The endpoint command was:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env OMP_NUM_THREADS=1 PYTHONPATH=.:python python tools/tensor_cuda_examples.py \
  --mode tune --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --cache /tmp/tensor146-provider-cache --output /tmp/tensor146-provider-evidence
```

CUDA tests use the same finite Slurm form with
`VIBEQC_TENSOR_CUDA_TEST=1`, `VIBEQC_NVCC=/path/to/nvcc` and
`python -m pytest tests/python/test_tensor_cuda_execution.py -q`.
Sanitizer adds `compute-sanitizer --tool memcheck --leak-check full
--error-exitcode 99` before that Python command. The raw test logs are included.
The CPU library was built with CUDA disabled in `build/cpu`; the existing
benchmark test also expects `build/libvibeqc.so`, which pointed to that library.

The optional allocation audit links the toolkit's CUPTI SDK and cuBLAS:

```bash
nvcc -std=c++17 -arch=sm_120 -I"$CUDA_PATH/extras/CUPTI/include" \
  provider-allocation-audit.cu -L"$CUDA_PATH/extras/CUPTI/lib64" \
  -Xlinker -rpath -Xlinker "$CUDA_PATH/extras/CUPTI/lib64" \
  -lcublas -lcupti -o /tmp/provider-allocation-audit
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:05:00 \
  /tmp/provider-allocation-audit
```
