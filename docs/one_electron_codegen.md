# Generated one-electron CUDA values

The one-electron compiler emits normalized contracted overlap S, kinetic T,
and nuclear attraction V for Cartesian s/p/d/f shells. Production consumers
write S and Hcore=T+V directly. The handwritten CUDA implementation remains
the default and the independent CPU implementation remains an oracle.

## Selecting a candidate

Set these variables **before creating a prepared batch**:

```bash
export VIBEQC_ONE_ELECTRON_VALUES=generated
export VIBEQC_ONE_ELECTRON_VALUE_MAPPING=thread
```

`VIBEQC_ONE_ELECTRON_VALUE_MAPPING=shell_warp` selects the alternative
shell-pair schedule. `VIBEQC_ONE_ELECTRON_VALUES=reference` selects the
handwritten implementation in the same binary. Unrecognized explicit value
selections also retain that implementation. These controls participate in
prepared resource and checkpoint runtime-policy identities. Changing either
selection invalidates the cached native direct plan before the next execution.
Explicit resource plans still require a matching policy identity. Create a new
prepared batch for cold A/B comparisons.

Both candidates run on CUDA, including public real-spherical d/f bases,
RHF/UHF and the standalone one-electron construction used by density fitting.
They do not delegate integrals to a CPU backend. First derivatives continue
through the existing analytic/handwritten implementation.

## Scientific and execution boundaries

`tools/vibeqc_codegen/one_electron_values.py` builds traceable, pruned Hermite
DAGs from `IntegralIR`. It includes Gaussian decay and radial prefactors.
Kinetic raising can reach internal ket powers of five without widening the
public f-shell dimensions. Attraction carries an independent nuclear center
and an explicit signed nuclear-charge factor. Its positive-term Boys series
and stable downward/upward recurrences cover orders zero through six.

`one_electron_cuda.py` shares S/T subexpressions and cuts reusable pair-geometry
roots out of the nucleus-dependent graph before emitting code. Primitive-pair
decay, exponent sums and relative shifts are therefore computed outside the
on-device nuclear loop. The emitted header and its 48-operator-signature
inventory are build artifacts:

```bash
python tools/generate_one_electron_kernels.py \
  --output /tmp/generated_one_electron_values.cuh \
  --inventory /tmp/one_electron_inventory.json
```

`src/scf/cuda/one_electron_values.cu` owns bounded contraction loops and output
consumers in a separate translation unit. It borrows the existing normalized
basis and sparse Cartesian expansions, so component order and real-spherical
normalization stay with the basis layer. The internal launcher can optionally
write separate contracted T/V matrices for diagnostics; ordinary HF allocates
only S/Hcore outputs.

| Schedule | Ownership | Storage |
| --- | --- | --- |
| `thread` | One thread owns a lower-triangular AO pair | Scalar primitive accumulators |
| `shell_warp` | One warp owns a shell pair, lanes cover AO components | Scalar primitive accumulators |

Each owner writes both symmetric matrix entries, and writes the diagonal once.
Partial blocks and components beyond one warp are explicitly bounded. Batch
indices reference global shell/atom/primitive arrays and per-system matrices.
There are no atomic matrix accumulations or per-nucleus host launches.

The new layer retains no geometry cache. Prepared HF reuses its fixed topology,
and recomputes integral data whenever positions change. Existing topology
comparison includes nuclear charges, primitive exponents and coefficients,
shell ownership and AO expansions. The standalone DF entry points upload
current geometry for every invocation.

## Validation and promotion

`query_integral_capability(request, backend="cuda_one_electron_values")` checks
the new primitive executor's semantic boundary. This is separate from the
legacy quartet-task CUDA ABI and from numerical or production acceptance.

CPU tests compare every Cartesian component with independent PySCF/libcint
S/T/V, including same-center, nearly coincident, intermediate and distant
nuclear centers, charge signs, exchange symmetry, translation, normalization
and long signed contractions. They also compile and execute the emitted
arithmetic on a host compiler to test lowering and geometry substitution.

The raw GPU runner reuses the shared CUDA compiler, finite Slurm executor,
resource parser, block-error schema and numerical-driver protocol:

```bash
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/validate_one_electron_values.py \
  --directory /tmp/one-electron-gate --nvcc /path/to/cuda/bin/nvcc

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc 'PYTHONPATH=python:. \
  VIBEQC_LIBRARY=$PWD/build-cuda/libvibeqc.so \
  VIBEQC_ONE_ELECTRON_CUDA_TEST=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/python/test_one_electron_values_cuda.py -q'
```

The raw gate covers 83 complete S/T/V fixtures, 11,497 primitive records, and
independent Cartesian/spherical projections at `atol=1e-11, rtol=3e-12`.
The endpoint gate compares both schedules to the same-binary handwritten
CUDA path across RHF/UHF, Cartesian/spherical, direct/DF and one/three-item
batches. It checks cold, unchanged, moved and restored geometries. Once this
GPU tier is enabled, runtime errors fail rather than becoming hardware skips.

`benchmarks/one_electron_values_gate.py` uses the shared interleaved A/B runner
and retains every sample, iteration count and final convergence diagnostics.
Cold starts and changed geometries measure actual integral construction;
unchanged geometries are a cache non-regression control. Its timings describe
whole HF endpoints and cannot establish integral-kernel speedups. Default
production selection requires numerical, resource and endpoint timing evidence
for the particular operator/shell family; source emission alone cannot promote
the complete family.

## RTX 5090 evidence

The [archived gate](../benchmarks/results/one-electron-values-rtx5090/validation-summary.json)
records the release library's source identity and binary hash. Validation passed
13 native tests, 17 CUDA endpoint cases, 41 checkpoint tests and 14 CUDA resource
tests; the GPU suites had no skips. The raw blocks' largest absolute error was
`7.14e-14` for the 83-fixture matrix. The broader CPU codegen/IR checks passed
403 tests with 48 unavailable optional tiers skipped.

| Native schedule | Registers/thread | Stack | Spills | Shared memory |
| --- | ---: | ---: | ---: | ---: |
| `thread` | 180 | 176 B | 0 B | 0 B |
| `shell_warp` | 200 | 176 B | 0 B | 0 B |

The native generated TU compiled in 112.38 seconds and produced a 4,100,328-byte
object from a 3,100,551-byte generated header. The standalone raw fixture used
116 registers/thread and compiled in 57.60 seconds. PTXAS logs and raw timing
samples are retained alongside the summaries.

Five interleaved A/B pairs per workload on the three-item `sp8` batch gave:

| Schedule | Cold, reference/candidate | Unchanged geometry | Changed geometry |
| --- | --- | --- | --- |
| `thread` | 9.748 / 9.938 ms | 2.436 / 2.433 ms | 12.783 / 12.925 ms |
| `shell_warp` | 9.793 / 9.730 ms | 2.453 / 2.430 ms | 12.741 / 12.709 ms |

These differences do not pass the shared significance gate. Both schedules
remain opt-in; no integral-kernel or whole-HF speedup is claimed.

The latest gate also switches reference/thread/shell-warp selections on an
existing plan and compares with fresh plans. Both native reuse predicates now
invalidate the plan on policy changes, keeping actual execution consistent
with recorded Python runtime controls. The resource tier was initially disabled
by a missing enable flag in review job 9075; its enabled run passed in job 9076.

Run the checkpoint and large f-containing endpoint suites in separate Python
processes. After the full endpoint matrix, the live parent's CUDA context can
leave insufficient memory for a checkpoint subprocess. This behavior reproduces
with the older pre-one-electron library and is recorded in the archive. The
isolated checkpoint and resource suites pass on the current library.
