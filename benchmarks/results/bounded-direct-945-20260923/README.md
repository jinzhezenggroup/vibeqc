# Bounded Direct-HF qualification (#945)

All 18 allocated-GPU regressions passed without skips. Automatic batch-2 water
 tetramer/def2-TZVP exceeds the one-GiB fixed task budget; RHF and UHF energies,
analytic forces, and per-class work counts pass their explicit gates.

`automatic-uhf.json` intentionally preserves the original failed comparison to
an unstable PySCF stationary point. `evidence.json` independently recomputes the
gates against `uhf-aligned-oracle.json`: a stable PySCF state related to the CUDA
localized state by an exact molecular symmetry. Both original forces and the
orthogonal operation/permutation are retained. No VibeQC density enters PySCF.

Reproduce from the repository root with a source-matched Release/sm_120 library:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env -u VIBEQC_BOUNDED_DIRECT_STREAMING -u VIBEQC_DIRECT_TILE_VALIDATION \
  -u VIBEQC_AOT_SHELL_CLASSES PYTHONPATH=python:. \
  VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/results/bounded-direct-945-20260923/reproduce.py \
  --output .artifacts/automatic-rhf.json
```

Add `--uhf --oracle benchmarks/results/bounded-direct-945-20260923/uhf-aligned-oracle.json`
for the cation doublet. `reproduce_uhf_reference.py` independently regenerates the
stable CPU oracle; exact molecular symmetry may select an equivalent localized
state with different atom labels. The accepted force tolerance is 2e-7 hartree/bohr.

The focused suite is `tests/python/test_bounded_direct_high_l_cuda.py`, enabled
with `VIBEQC_RESOURCE_CUDA_TEST=1` inside Slurm. `regressions.json` retains all
18 test names and high-l work/oracle errors. Resource and kernel summaries come
from a separate intrusive Nsight pass and are not clean speedup measurements.
Full traces and routine logs remain ignored local artifacts.
