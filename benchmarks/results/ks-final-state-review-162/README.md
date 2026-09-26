# Verified KS final-state review validation

Clean source `5129a8db666efcf3dedacc8a24f2dbf205b96bd7`; RTX 5090, CUDA 12.9.86, Release,
sm_120, AOT shells disabled. Slurm job 9628 used `main` and
`--gres=gpu:5090:1` with an explicit finite time limit and preserved device visibility.

- Native KS/DFT/final-state/eigenframe selection: **10/10 passed**.
- Python KS diagnostics, resources and options: **47 passed, 1 opt-in skip**.
- Compute Sanitizer memcheck of the complete native KS test: **0 errors**.

The GPU test validates LDA/PBE RKS and UKS snapshots, empty beta occupations,
requested/unrequested W, density/orbital/Fock identities, exact transfer
accounting, stale tokens, warm replay, failed solves, resource ownership and
recovery. Additional C API regressions reject old single/batch tokens after
null output, invalid ABI and wrong result-count requests. The orbital copy
now converts CUDA column-major storage to the detached row-major contract.

`summary.json` binds source revision/dirty state, binary and build identities,
device/toolchain, commands and results. The verified archive contains full
logs, all measured source hashes and the allocation-only reproduction runner.
Restore with `python -m tools.unpack_evidence
benchmarks/results/ks-final-state-review-162 --output /tmp/ks-review`.
Invoke the restored runner inside a finite Slurm GPU allocation, passing the
source directory, a new output directory, and `ks`; adjust its Python path to
the validation environment. Build with CMake Release, CUDA on, AOT shells off,
NVCC 12.9 and `VIBEQC_CUDA_ARCHITECTURES=120`.

This establishes the internal handoff for #163. It does not implement nuclear
gradients or extend the public C/Python result ABI.
