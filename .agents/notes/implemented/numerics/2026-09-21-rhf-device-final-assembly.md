# RHF CUDA device-final HVP/Hessian assembly — 2026-09-21

Refs #180. Follow-up to #753.

This slice keeps the final conventional RHF HVP composition on CUDA after the
already-qualified resident response reconstruction. Generated relaxation exports
its Cartesian result through a borrowed device pointer, #178 generated
second-integral consumers export compact coordinate tiles without host
publication, and a bounded CUDA accumulator combines those terms with a native
nucleus-nucleus HVP. The scalar/multi-RHS endpoint downloads only the final
molecular HVP.

For a full Hessian, each completed HVP column is copied device-to-device into a
column-major CUDA matrix owner. Block HVP host publication is suppressed and the
raw full matrix is downloaded once after all columns succeed. No diagonal
substitution or post-hoc symmetrization is introduced.

## Exact implementation head

- `5f431d61e4dc3243258625dc14ebc182379690ac`
- based on master `41be6f8d8db7b74336810cf3e27c5b33595bbc52`
- commit records `Agent: ChatGPT` and `Model: GPT-5.6 Sol`

## Validation
Exact-head static gates:

- `git diff origin/master...HEAD --check`: passed after EOF cleanup
- Ruff on changed Python implementation/tests: passed
- clang-format dry-run on changed native CUDA/runtime headers: passed

Exact-head host regression:

- `tests/python/test_hessian_hvp.py`
- `tests/python/test_hessian_block.py`
- result: **16 passed**

Earlier same implementation before the master-only rebase additionally ran the
response/Hessian host group (`response_krylov`, `hessian_hvp`, `hessian_block`,
`hessian_directional`) with **72 passed**.

Exact-head real-device Slurm qualification:

- Slurm job: `10597`
- GPU: NVIDIA GeForce RTX 5090
- driver: `580.95.05`
- CUDA compiler: `12.9` (`sm_120`)
- suites:
  - `test_hessian_directional_cuda.py`
  - `test_hessian_relaxation_cuda.py`
  - `test_hessian_second_cuda.py`
  - `test_hessian_block_resident_cuda.py`
  - `test_hessian_device_assembly_cuda.py`
- result: **24 passed, 3 skipped, 0 failed**
- the three skips are existing external-PySCF oracle cases in the environment
  where PySCF is not installed; the resident/device-final CUDA paths were run.

The new device-final suite specifically verifies scalar HVP, sequential/blocked/
recycled multi-RHS HVP, and bounded full-Hessian assembly. It asserts zero host
publication of scientific component HVPs in the CUDA assembly path, zero #178
compact-result tile downloads, no CUDA-relaxation host result publication, and
for the full Hessian one final matrix D2H with every HVP column copied D2D.

## Remaining boundary

This does **not** claim a completely all-device Hessian workflow. Directional
H1/S1 publication, nuclear-RHS/metric preparation, projected Krylov
least-squares/scalar control and compatibility response publication still cross
or reside on the host. The slice removes the downstream final-assembly host
boundary; public Calculator exposure, production-size qualification and DFT
Hessians remain separate gates.

Agent: ChatGPT
Model: GPT-5.6 Sol
