# RHF resident reconstruction through CUDA relaxation — 2026-09-21

Refs #180, PR #753.

This note records the bounded conventional RHF slice that consumes converged
CUDA-resident response rotations before host publication, reconstructs D1/W1 in
the shared response owner, and imports those matrices device-to-device into the
generated first-integral relaxation contraction. Scalar, sequential, blocked,
recycled multi-RHS and bounded full-Hessian consumers share the same path.

## Validation

Source implementation commit before this evidence-only note:

- `d4a6a4bcaee8623b73449c76db08786c89c4e59c`

Native build:

- CUDA toolkit: 12.9
- target: `sm_120`
- `vibeqc` shared library built and linked successfully from the PR checkout
- changed native files pass clang-format
- Ruff, evidence retention, method manifest, compiler structure, CUDA ownership
  and SCF structure gates pass

Host regression against the rebuilt native library:

- `tests/python/test_response_krylov.py`
- `tests/python/test_hessian_hvp.py`
- `tests/python/test_hessian_block.py`
- `tests/python/test_hessian_directional.py`
- result: **72 passed**
- focused Krylov run: **32 passed**

Real-device Slurm qualification:

- Slurm job: `10580`
- GPU: NVIDIA GeForce RTX 5090
- driver: 580.95.05
- explicit `VIBEQC_RESPONSE_CUDA_TEST=1`
- explicit `VIBEQC_TEST_CUDA_ARCH=sm_120`
- suites:
  - `tests/python/test_hessian_directional_cuda.py`
  - `tests/python/test_hessian_relaxation_cuda.py`
  - `tests/python/test_hessian_block_resident_cuda.py`
- result: **15 passed, 3 skipped, 0 failed** in 74.50 s
- the three skipped cases are the parameterized external-PySCF Hessian-oracle
  cases from `test_hessian_block_resident_cuda.py`; PySCF is not installed in
  this Slurm Python environment. The resident CUDA paths themselves were not
  skipped.

## Boundary

This slice does not claim an end-to-end all-device molecular Hessian. It removes
D1/W1 host round-trips from the CUDA relaxation consumer, but compatibility
response publication remains, #178 compact second-integral HVP tiles still
publish to host, and final molecular HVP/full-Hessian assembly remains host-side.
Those boundaries are owned by the stacked follow-up rather than hidden by this
PR.

Agent: ChatGPT
Model: GPT-5.6 Sol
