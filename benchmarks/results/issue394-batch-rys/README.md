# Complete low-angular Rys derivative candidates (#394)

> **Checkout retention (2026-09-21):** `raw-evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue394-batch-rys/raw-evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue394-batch-rys/raw-evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

For commands below that previously unpacked the checkout directly, pass the
restored archive explicitly:

```bash
python -m tools.unpack_evidence benchmarks/results/issue394-batch-rys \
  --archive .artifacts/issue394-batch-rys/raw-evidence.zip \
  --output .artifacts/issue394-batch-rys-unpacked
```

One compiler-owned Gaussian-moment lowering now emits `000/001/002/100/101/110/200`.
All six added classes share one independently generated two-root evaluator. The
existing production manifest remains unchanged; #404 owns combined endpoint
qualification and #206 owns fresh GPU4PySCF comparisons.

Measured implementation: `a6de84f48ad15b13512218393494f3ef9dace402`, based on
post-#411 `ed764eb`. Source fingerprints in this bundle identify that measured
revision. The subsequent retirement of the SSS-only runtime override changes
native dispatch and checkpoint compatibility, not the measured candidate
mathematics; its validation is separate. FP64, normalization, response layouts,
screening, convergence, and force semantics are preserved.

## Qualification

- All **42/42** class × lowering × schedule identities compile with production
  optimization (CUDA 12.9.1, sm_120, no fast compile). No spills or stack storage
  are reported. Two-root candidates use 126–128 registers per thread.
- **404 Python tests** and **30 native CPU suites** pass. The Python suite
  includes actual GPU one/two-root evaluation, 75-digit F0–F3 reconstruction,
  interval/asymptotic boundaries and extreme/subnormal arguments, libcint
  Cartesian/spherical contracted derivatives, signed contractions, and an
  independent 90/400-digit center-differentiation oracle of the closed SSS integral.
- Every candidate passes native CPU derivative holdouts across all three
  response layouts, both public representations, partial auxiliary panels,
  varied exponents/contraction lengths, and shared physical atom ownership.
- Compute Sanitizer memcheck (with leak checking) and initcheck report zero
  errors for the complete 42-candidate holdout executable.

`qualification.json` binds these results to commands, source/library/executable
hashes, generated headers and local log hashes. Routine logs and binaries remain
in `.artifacts/`. `raw-evidence.manifest.json` verifies the compressed raw timing,
work and numerical rows without converting source work into elapsed-time claims.

The 42 candidate objects total 9,883,376 bytes; the eighteen added two-root
objects account for 4,090,080 bytes of that diagnostic family. These are independent
translation-unit object sizes, including repeated common code/data, **not** a
prediction of production-library or process-memory growth. Full-library growth
and complete endpoints belong to the combined #404 build. The per-candidate
resource record retains shared memory, registers, stack/spills, object sizes and
theoretical resource occupancy; it makes no achieved-occupancy claim.

The subsequent selector retirement has separate [runtime validation](runtime-validation.json):
one native shell-pair suite, five CUDA checkpoint cases and four complete
energy/force cases pass with the retired variable deliberately set. The native
fixtures remain polynomial outside the manifest domain. The record binds these
checks to their source, library and finite Slurm allocation.

## Batch handoff

The retained 384/768 real signature distributions are measured separately.
Selection requires at least a 3% improvement over the actual post-#411 manifest
control in both profiles. In particular `000:rys:compact` is already the control;
its historical gain over polynomial is not new work. The source-work model
reports root evaluations and active-component moment states, with zero
coefficient-cache/convolution work for the Rys lowering.

The batch report proposes Rys/compact for `001/002/100/200` and retains the
existing choices for `000/101/110`. These are isolated class decisions; no
complete energy-plus-force improvement or GPU4PySCF superiority is claimed here.
#404 must qualify one combined mapping against a fresh pinned post-#411 baseline,
with clean repeated complete endpoints and unchanged operator/state gates.

The initial v1 probe overlapped host correctness compilation and used the
historical all-polynomial ranking control. Its raw samples are retained as
unqualified diagnostics. v2 uses the correct production control. The final
campaign reuses the exact v2 candidate objects but relinks the independent CPU
oracle built from the measured checkout, removing uncertainty about a borrowed
oracle binary's source identity. No compilation overlaps either clean campaign.

## Reproduce

Use the measured source commit and a Python environment containing NumPy,
PySCF, mpmath and pytest. Generation itself needs none of those numerical oracles.
From the checkout:

```bash
cmake -S . -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build/cpu -j 8
python tools/generate_df_kernels.py --derivatives \
  --output build/df-batch/generated/generated_df_derivatives.cuh \
  --shell-output build/df-batch/generated/generated_df_shell_derivatives.cuh
python tools/benchmark_df_derivatives.py \
  --profile benchmarks/results/issue395-df-work/work/384.json --pair-mode symmetric \
  --profile benchmarks/results/issue395-df-work/work/768.json --pair-mode packed \
  --directory build/df-batch/run --nvcc /path/to/cuda/bin/nvcc \
  --generated build/df-batch/generated --oracle-library build/cpu/libvibeqc.so \
  --compile-jobs 4
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  build/df-batch/run/benchmark --qualify-only
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode=1 \
  build/df-batch/run/benchmark --qualify-only
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  compute-sanitizer --tool initcheck --error-exitcode=1 \
  build/df-batch/run/benchmark --qualify-only
python -m tools.unpack_evidence benchmarks/results/issue394-batch-rys \
  --archive .artifacts/issue394-batch-rys/raw-evidence.zip
```

The batch helper itself allocates one finite Slurm job after compilation. Preserve
Slurm's device visibility. Run the root GPU tests with `VIBEQC_RESOURCE_CUDA_TEST=1`
inside a finite allocation and provide `CUDACXX`. Regenerate the rounded table
with `python tools/generate_df_rys2_table.py`; its output is byte-reproducible.
