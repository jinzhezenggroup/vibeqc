# WB97M-V CUDA energy and forces

The Python `Calculator` and prepared-batch interface compose the complete
all-electron FP64 WB97M-V energy and stationary analytic forces on CUDA:
semilocal meta-GGA, 15% short-range exchange, 100% long-range exchange
(`omega=0.3`), and self-consistent VV10. Nuclear repulsion, overlap/Pulay,
AO motion, quadrature-point motion and Becke partition response are included.

Numerical acceptance covers restricted H2, unrestricted H3, and spherical
def2-SVP water, including independent GPU4PySCF forces and reconverged energy
finite differences. Performance at the HF README's water-cluster sizes is
not qualified. The retained [qualification evidence](../../benchmarks/results/wb97mv-cuda-20260926/README.md)
records completed measurements and incomplete attempts separately.

```python
from vibeqc import Calculator

calculator = Calculator(
    method="wb97m-v", basis="def2-svp",
    basis_representation="spherical", device="cuda",
)
water = [("O", (0.0, 0.0, 0.0)),
         ("H", (0.0, 1.43, 1.11)), ("H", (0.0, -1.43, 1.11))]
result = calculator.singlepoint(water, properties=("energy", "forces"))
print(result.energy, result.forces)  # Hartree and Hartree/Bohr; coordinates in Bohr
```

Use `method="wb97m-v-uks"` and an appropriate multiplicity for unrestricted
spin. Energy-only requests avoid derivative work. Forces are the negative
energy gradient. No CPU integral derivative, reference SCF or finite difference
is part of the production force path.

The complete Python force consumer admits built-in STO-3G and def2-SVP, or
explicit all-electron s/p/d bases, in Cartesian or spherical representation.
ECPs, density fitting and mixed precision are outside this force contract.
The backend-neutral native C method registry continues to advertise energy
only; Python owns the compiled stationary composition. The private snapshot
bridge is not a public C force API.

NVCC is required for the generated geometry and final-reduction modules;
set `CUDA_PATH` or `CUDACXX` when it is not on `PATH`. Cold timing includes
compilation when the artifact cache is empty. Prepared batches retain derivative
owners on unchanged geometry and rebuild them on geometric changes. A live
SCF-generation token is checked before and after force assembly; failures
discard Python-owned derivative scratch and preserve per-item error reporting.

The consumer has explicit limits of 128 atoms, 1024 AOs, four million quadrature
points, and s/p/d angular momentum. Additional derivative numeric storage is
bounded by 1 GiB device and 2 GiB host capacity; admission can fail below the
shape limits when its conservative inventory exceeds these allowances.
These bounds exclude existing SCF state, compiler processes, CUDA modules and
driver-managed recurrence stacks. Host work includes snapshot validation and
exports, tiling, and total-density/VV10-active-domain packing. The VV10 cutoff
is `rho >= 1e-8` on both pair legs, using the same active-branch convention as
SCF. Finite memory bounds do not imply scalable endpoint work: the retained
generic exchange/derivative provider and quadratic VV10 pairs remain material
performance costs.

## Reproduce acceptance and timing

Run real-GPU commands through the local Slurm partition, preserving its device
visibility. Point the Python package and `VIBEQC_LIBRARY` at the same checkout
and its Release CUDA build. The opt-in tests compare independent GPU4PySCF
energies and grid-responsive analytic forces, reconverged energy finite
differences, warm replay, geometry rebuild and failed-neighbor isolation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env VIBEQC_TEST_WB97MV_CUDA=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_wb97mv_complete_cuda.py -q
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  build/vibeqc_cuda_fock_provider_tests --range-response-only
```

The benchmark uses the HF README's water geometries, 3/6/12/24/48/96 atoms,
spherical def2-SVP, and three interleaved synchronized complete warm SCF plus
analytic-force repeats. Each engine restarts from its own fixed post-cold
density. Both use the explicitly stated moving atomic quadrature, including
partition response. Default quadrature is 48 radial × 16 polar × 32 azimuthal
points per atom. Every paired result must satisfy `|dE| <= 1e-8 Eh` and
`max|dF| <= 1e-7 Eh/Bohr`; cold and priming samples are also checked.

```bash
export README_BENCHMARK_PYTHON=/path/to/benchmark-env/bin/python
export VIBEQC_LIBRARY=$PWD/build/libvibeqc.so
export README_BENCHMARK_OUTPUT=$PWD/.artifacts/readme-wb97mv
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:35:00 \
  bash benchmarks/run_readme_benchmarks.sh wb97mv
python tools/render_readme_wb97mv.py \
  --raw-directory .artifacts/readme-wb97mv \
  --destination .artifacts/readme-wb97mv-figures
```

The `wb97mv-reference` runner group measures GPU4PySCF independently when a
native failure or timeout prevents a paired result. `README_BENCHMARK_POINT_TIMEOUT`
sets the finite per-point limit (default 900 seconds). Failed, incomplete and
timed-out points remain in the retained evidence and never become accepted
timings or speedup claims. Use the retained result records, rather than mere
API admission, to assess qualification at a particular size.
