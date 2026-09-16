# Exact packed DF values: experiments and endpoint reproduction

This is the work in progress for #409. Physical CUDA DF preparation accepts
`VIBEQC_DF_VALUE_STORAGE=auto|dense|packed`; unset/`auto` remains dense. The
explicit packed path stores separate immutable raw A and whitened B in exact
unit-weight lower-triangular AO-pair order. It preserves FP64, native metric
factorization, all raw metric directions and the existing single Gram.

The [partial evidence](../../results/issue409-packed-values/README.md) keeps
both endpoint gains and regressions, changed SCF work, resource observations,
source identities and remaining qualification. Packing is not automatically
selected, and the current matrix does not establish a general endpoint win.

## Complete warm endpoints

Build before starting clean GPU measurements, using the retained build flags
when reproducing an archived result. `run_endpoints.py` needs a Python environment
with the normal VibeQC benchmark dependencies, a CUDA-enabled native library and
the library search path for that CUDA installation. From the checkout root:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:00:00 \
  python benchmarks/experiments/issue409-packed-values/run_endpoints.py \
  --library build/cuda/libvibeqc.so --output .artifacts/packed-warm-new
```

The wrapper performs at least seven alternating dense/packed pairs with normal
convergence, then a separate intrusive component pass. It checks independent
energy/full-force references through `benchmarks.df_policy_endpoint`, preserving
the existing `1e-9 Eh / 1e-8 Eh/Bohr` gates. Trace collection and process sampling
do not overlap clean calls. The JSON preserves timing samples, representation
priming, convergence and frozen input identities. Slurm device visibility is
passed through unchanged.

Without `--seed-dir`, one cold dense solve generates a new checkpoint for each
AO size, shared by both policies and observables. A fresh seed can change SCF
work and does not reproduce an archived branch by definition. A caller may
supply a directory containing `96.checkpoint`, `192.checkpoint`, etc. to freeze
an existing input. Transient checkpoints are not committed. Restrict the matrix
with `--aos 192`, use `--forces-only`, or set an identical total DF budget through
`--df-budget BYTES`. The constrained 768-AO campaign used 12884901888 bytes;
its seven pairs need a longer finite allocation (90 minutes).

The retained `reproduction/measured-*.txt` files preserve the exact historical
warm, cold, changed-geometry, unequal-basis and capacity harness bytes. They
include workstation paths and should be adapted deliberately for a new run.
`run_endpoints.py` is the maintained portable warm entrypoint. A new run must
record its own identities; copying old medians or headers is not reproduction.

## Qualification and evidence collection

Run the native density-fitting/occupied-response suites and
`tests/python/test_df_packed_values_cuda.py` against the chosen library inside
finite Slurm. The molecular tests require `VIBEQC_RESOURCE_CUDA_TEST=1` and
`PYTHONPATH=python:.`. The separate failed-neighbor tier requires
`VIBEQC_DF_DERIVATIVE_CUDA_TEST=1`; a missing flag skips that tier. Compute
Sanitizer also runs inside Slurm. CPU-safe resource and independent-reference
admission tests need no GPU allocation.

The collector selects compact numerical evidence from the original campaign:

```bash
python benchmarks/experiments/issue409-packed-values/collect.py \
  --artifacts .artifacts/issue409 --output .artifacts/packed-review-new --partial
```

`--partial` explicitly permits a draft with missing cells listed in its manifest.
Without it, collection rejects an incomplete matrix. Completed cells must retain
all seven pairs, normal convergence and unchanged numerical gates. Collection
does not grant qualification or select a representation. Raw logs, traces,
checkpoints and binaries remain transient under the
[evidence retention policy](../../../docs/evidence_retention.md); compact raw
samples, original hashes and measured-source reconstruction stay reviewable.

## Fixed-input projection and producer experiments

`protocol.json` records the fixed-input candidates and the later endpoint
gates. `projection.cu` compares the current dense occupied projection with
three bounded unpack widths and two shared-memory triangular iterators. All
arms produce the same `U[mu,i,Q]` and use one unchanged SYRK plus mirror.
The timed input coefficients use the native SCF's column-major convention.

The nonlinear triangular AO map cannot be described by constant tensor
strides in a single dense cuTENSOR contraction. The direct candidates are
ordinary FP64 CUDA SIMT kernels, with no CUTLASS/cuTENSOR performance claim.

`capture.cpp` is a bounded intrusive preload shim. It retains one immutable B
and up to four eager C/U/K snapshots, skipping CUDA graph construction. Its
captures and endpoint timings are diagnostic only. It must link cuBLAS and
the CUDA runtime explicitly (`--no-as-needed -lcublas -lcudart`) so the real
functions remain visible when Python loads the native library locally.

`producer.cpp` generates each lower AO row directly through the existing
public-basis source, then whitens all packed pairs in one GEMM and contracts J
with unit-weight packed values. It allocates no dense raw/transformed tensor.
The small independent test supplies X and D; a production implementation must
retain native metric setup, source identity, raw ownership, and resource checks.
In particular, the injected NumPy metric map is an oracle input, not a proposed
production provider. Packed J sums both off-diagonal density entries, including
for nonsymmetric diagnostic D. Raw A remains independent of the truncated B.

Build all binaries before clean GPU timings. Example commands from the checkout
root, with `trial-dir`, CUDA installation and baseline library paths supplied
by the caller:

```bash
nvcc -std=c++17 -O3 -arch=sm_120 projection.cu -lcublas -o trial
c++ -std=c++20 -O3 -Iinclude -Isrc -I<cuda>/include producer.cpp \
  -L<baseline-library-directory> -L<cuda>/lib64 -lvibeqc -lcublas -lcudart -o producer
PYTHONPATH=python python run_projection.py prepare \
  --captures <capture-directory> --executable <trial> --output <fresh-trial-directory>
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=python python run_projection.py run --output <fresh-trial-directory>
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:05:00 \
  env PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python validate_producer.py --executable <producer> --library <libvibeqc.so> \
  --output <fresh-producer-directory>
```

The source filenames above are relative to this experiment directory; use their
full paths when running from the checkout root. Set the library search path
for the chosen CUDA installation and preserve Slurm's device visibility.

The projection preflight charges the full common trial arena, including the
simultaneously retained control and packed inputs, both U/K results, bounded
unpack scratch and a library allowance. Standalone candidate byte estimates
are reported separately. They exclude a complete molecular plan's raw A,
response buffers, metadata and solver state. Logical loads are distinguished
from hardware DRAM traffic; no process-memory improvement is claimed here.

Before integrating a candidate, retain numerical and sanitizer evidence and
the original input/library/source hashes. Direct packed generation at larger
sizes, native cache/representation identity, packed force response, full
cold/changed/warm work counts, and capacity crossover remain separate gates.
