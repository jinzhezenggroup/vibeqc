# Incremental Coulomb factors and exact-target RHF cleanup

Both bundles measure clean source `dfc9e55f13bc55b4961d7ab5255064a66e38df7b`.
They accept the experimental numerical implementation, without promoting a
production selector or claiming a general speedup. CPU and CUDA each retain
three complete samples for H2, water and LiH, using pinned independent post-HF
fixtures. The publication manifests validate every selected file's checksum.

| Maximum checked error | CPU | CUDA | Absolute gate |
| --- | ---: | ---: | ---: |
| Exact final energy versus independent reference, Hartree | 2.56e-13 | 2.71e-13 | 2e-9 |
| Exact final AO density versus independent reference | 9.71e-9 | 9.71e-9 | 4e-7 |
| Final force versus direct exact solve, Hartree/Bohr | 3.18e-10 | 3.18e-10 | 3e-7 |
| Same-approximation J | 3.56e-15 | 2.23e-15 | 3e-11 |
| Same-approximation K | 1.78e-15 | 1.78e-15 | 3e-11 |

The dense reference reconstruction also checks conditional pair residual entry
and Frobenius bounds, with a 3e-10 allowance for independent integral and
floating-point differences. These are tensor diagnostics under the PSD
assumption, not certified relaxed-observable bounds. Production factor/consumer
code never receives the dense reference tensors.

Median wall time in seconds, from plan setup through converged final forces:

| Backend/case | Direct exact | Fixed rank + exact cleanup | Staged rank + exact cleanup | Separate DF |
| --- | ---: | ---: | ---: | ---: |
| CPU H2 | 0.01031 | 0.01450 | 0.01506 | 0.01233 |
| CPU water | 3.44231 | 3.60521 | 3.62912 | 3.54239 |
| CPU LiH | 1.92905 | 2.05746 | 2.04732 | 2.00184 |
| CUDA H2 | 0.05636 | 0.06370 | 0.06398 | 0.13815 |
| CUDA water | 24.21631 | 22.50190 | 23.17767 | 1.47593 |
| CUDA LiH | 15.68202 | 14.79051 | 15.31901 | 1.32120 |

Small CPU systems pay additional initialization overhead. The CUDA results
describe this particular mixed execution and exact FockPlan endpoint, including
CPU raw columns, pivot/PSD decisions, host eigensolves and explicit transfers.
They are not a comparison with a fully GPU-resident integral factorization.
Compilation/cache lookup, destruction and independent reference checks are
outside the solve clock. CUDA section events are enabled; `device_ms=0` means
an overall device interval was not measured. Section times alone are not the
whole-solve metric. DF changes the Hamiltonian: its energy/force differences
are recorded separately and are not used to pass exact-target accuracy gates.

Staged ranks grow 2→3 for H2, 3→26 for water and 3→21 for LiH. Fixed rank uses
the same final 1e-5 pair-diagonal threshold; staged execution first uses a 0.1
threshold and at most three pivots. Stage records expose fresh commutator
residuals, the electronic-energy jump at identical density, and actual pivot
and column hashes. Old factors are retained; rank changes carry no DIIS/Krylov
history. Final convergence always switches to the original unscreened target
provider. Truncated-factor derivatives remain unsupported.

Every factor/initializer fits the declared 9 MiB host and 101 MiB device
budgets. The CUDA water factor arena is 4,204,544 bytes, with 8,388,608 bytes
of observed retained cuBLAS storage inside a conservative 96 MiB allowance.
Per-sample resource plans account for both factor mirrors and numeric work.
The separately owned exact FockPlan's native resources are recorded separately.
CUDA context/driver/kernel stack, Python metadata, host BLAS overhead and the
tiny independent reference tensors are outside these numeric-buffer budgets.

Validation accompanying the source: 88 CPU tests and 6 scheduled CUDA tests
passed; Compute Sanitizer's native owner suite passed with zero errors and
zero leaked allocations; `uvx pre-commit run --all-files` passed.

## Reproduction

Use Python with NumPy and the repository's `python` and `tools` import roots.
Set `VIBEQC_LIBRARY` to the CPU or CUDA native build for the corresponding
command. Each envelope records the exact library binary hash and each target
record includes its native source identity. These runs used the existing
validated #202 production libraries; low-rank CUDA code is compiled separately
from the measured source through the shared native cache. NVCC 12.9.1 and
`sm_120` are used by this hardware-specific driver.

```bash
export PYTHONPATH=python:tools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/validate_low_rank.py --backend cpu --samples 3 \
  --output .artifacts/low-rank-cpu-reproduction

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  python tools/validate_low_rank.py --backend cuda --samples 3 \
  --output .artifacts/low-rank-cuda-reproduction
```

Use an empty output directory and preserve Slurm's assigned device visibility.
Optional `--publish <new-directory>` validates and publishes a selected bundle
only from clean measured source. No run logs, profiler dumps or binaries are
retained here.
