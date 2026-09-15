# Reproduce issue 382

From the repository root, build the final source with CUDA 12.9.1:

```bash
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_FLAGS=-lineinfo -DVIBEQC_BUILD_TESTS=ON
cmake --build --preset cuda-release-sm120 --target vibeqc \
  vibeqc_df_occupied_response_tests vibeqc_df_shell_pairs_tests --parallel 6
```

Use environments providing VibeQC's test dependencies or GPU4PySCF 1.8.1,
PySCF 2.14.0 and CuPy 14.2.0. Set `VIBEQC_LIBRARY` to the built library and
include its directory and CUDA `lib64` in `LD_LIBRARY_PATH`. Exact measured
runner bytes are retained as `.py.txt`; copy to a temporary `.py` path before
executing. The final endpoint runner is also `benchmarks.df_policy_endpoint`.
All real device work must preserve Slurm-assigned `CUDA_VISIBLE_DEVICES`.

```bash
mkdir -p .artifacts/reproduce382
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc '
    export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export VIBEQC_RESOURCE_CUDA_TEST=1
    python -m pytest -q tests/python/test_df_shell_derivatives_cuda.py \
      tests/python/test_df_occupied_response_cuda.py \
      tests/python/test_df_resident_response_cuda.py
    python -m benchmarks.df_policy_endpoint --aos 768 \
      --control VIBEQC_DF_DERIVATIVE_PAIRS --policies full symmetric auto \
      --repeats 5 \
      --reference benchmarks/results/issue377-379-df/gpu4pyscf/water-32mer-4s4-def2-svp-spherical.json \
      --output .artifacts/reproduce382/768-clean.json
  '
```

Repeat for 192/384 using the matching `water-octamer-s4` and
`water-hexadecamer-2s4` reference names. Run each native executable from the
build's executable directory inside the same finite Slurm allocation.
For component/work evidence, make a **separate** invocation with `--trace`,
`--repeats 1` and `VIBEQC_DF_SHELL_COUNTERS=1`. For the block experiment set
`VIBEQC_DF_DERIVATIVE_PAIRS=packed`, select
`--control VIBEQC_DF_PACKED_AO_BLOCK_ROWS --policies 64 128 256 384`.
Never enable tracing/counters in the clean timing run.

```bash
cp benchmarks/results/issue382-packed-df/reproduction/gpu4pyscf.py.txt \
  .artifacts/reproduce382/gpu4pyscf_reference.py
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 bash -lc '
    export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python .artifacts/reproduce382/gpu4pyscf_reference.py 768 \
      .artifacts/reproduce382/gpu4pyscf-768-clean.json
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output .artifacts/reproduce382/768-profile \
      python .artifacts/reproduce382/gpu4pyscf_reference.py 768 \
      .artifacts/reproduce382/gpu4pyscf-768-profile.json --profile
  '
nsys export --type=sqlite --output .artifacts/reproduce382/768-profile.sqlite \
  .artifacts/reproduce382/768-profile.nsys-rep
python benchmarks/results/issue382-packed-df/reproduction/reduce_profile.py \
  .artifacts/reproduce382/768-profile.sqlite \
  .artifacts/reproduce382/768-profile-summary.json
```

The GPU4 environment also needs its `cutensor/lib` on `LD_LIBRARY_PATH`.
For process peaks run `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv,noheader,nounits -lms 100` inside the allocation while the
diagnostic process is active; stop the sampler afterward. Record process IDs,
include cold setup in the scope, and query residency before owner teardown.
Sanitizer uses `compute-sanitizer --tool memcheck --error-exitcode 99` around
the native shell executable and the packed 96-AO pytest case, under `srun`.

Historical measurements used Slurm 9660 (old baseline), 9661/9662 (Slice A
tests/timings), 9663/9664 (Slice B tests/timings), 9665 (sanitizer), 9666 (block
selection), 9667 (final tests/timings/work), and 9668 (fresh GPU4 clean/profile).
The `*-source.patch` files reconstruct native measured snapshots against
`f1703085532229747e54c2d4df3cd16371a00c93`. Use `git apply` in a separate
checkout of that base; do not stack the four snapshots. `endpoint-v1.py.txt`
is the early runner; `endpoint-final.py.txt` adds in-owner memory snapshots.
The baseline source/library identity is also linked from the #381 evidence.
