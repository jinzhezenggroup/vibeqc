# Stationary CUDA RKS gradient diagnostic

Date: 2026-09-19

## Problem

Issue #163 C1 needs a bounded native CUDA lowering of the same compiler-owned `StationaryGradientPlan` already used by the CPU diagnostic. The CUDA route must include one-electron, Coulomb, XC AO motion, physical grid motion, Becke partition-weight response, overlap/Pulay and nuclear sources without adding a second PBE-specific scientific implementation or calling the CPU/interpreter consumers from device execution.

## Decision

Add an internal, opt-in complete RKS CUDA diagnostic rather than enabling public `Calculator` forces. The runtime binds only a current native CUDA KS snapshot, requires CUDA wire v3 with the owner's exact grid prescription/raw atomic measures and measured snapshot-export work, and revalidates the state before result publication.

Scientific arithmetic remains shared/generated: integral first derivatives come from the existing derivative generator with CUDA emission, TensorIR supplies plan weights/final reduction, AO/XC geometric pullbacks come from common compiler graphs, and CPU/CUDA share a two-pass Becke adjoint traversal. The handwritten CUDA owner provides bounded record traversal, normalization, scatter/reduction, stream/lifetime control, device validation and transactional failure handling.

The grid task ABI gains a lease-scoped borrowed view of D-contracted density jets. The consumer must complete on the producer stream before releasing the task; the XC work arena invalidates this view before reuse.

Invalid device ordinals are rejected with `cudaGetDeviceCount` before constructing/storing a CUDA `Context`. This avoids a failed `cudaSetDevice` poisoning later owners during partial destruction.

## Rejected alternatives

- A separate handwritten PBE CUDA gradient formula: duplicates the CPU/compiler science and blocks reuse by later MethodIR primitives.
- Calling B1/CPU interpreter per grid tile: violates C1 and hides host derivative work behind a CUDA label.
- Full coordinate-by-grid-by-AO Jacobians: unnecessary and unbounded; generated VJP-style contractions reduce into `3*Natom` sources.
- Public force enablement in this slice: C2 still owns production endpoint/batch/provider qualification; UKS/ECP/hybrid/meta-GGA and higher angular momentum remain unsupported here.

## Invariants

- Source order is exactly `one_electron`, `coulomb`, `xc_ao`, `xc_grid`, `xc_weight`, `overlap_pulay`, `nuclear` from the shared plan.
- No CPU derivative/interpreter consumer is entered during complete CUDA execution.
- Borrowed grid pointers never outlive the task lease; geometry completes on the producer stream.
- Failure publishes no partial result; the source owner is poisoned until explicit reset.
- The diagnostic is direct, all-electron, real FP64 RKS LDA/PBE on the declared stable grid branch and s/p AO domain only.
- Public DFT force capability remains disabled.

## Evidence

On an RTX 5090 / CUDA 12.9 (`sm_120`), H2 and asymmetric-water LDA/PBE complete gradients were compared source-by-source against independent PySCF/libcint/libxc analytic gradients with grid response. The largest total-gradient error observed was `2.02e-11 Eh/bohr` (water/PBE); H2 LDA/PBE errors were `5.38e-15 Eh/bohr`. Multistep reconverged finite differences reached sub-`1e-9 Eh/bohr` raw errors in the recorded coordinate checks.

Host/structure qualification included the shared Becke adjoint tests, compiler ownership/structure checks, focused CPU stationary regressions and the new device-free CUDA lowering guard. The real-device failure/recovery test passed after the invalid-device admission fix. Native CUDA KS/snapshot v3 qualification passed. `compute-sanitizer --tool memcheck` over H2 LDA/PBE plus the late source-failure/recovery path reported `ERROR SUMMARY: 0 errors`.

Local detailed evidence is intentionally ignored under `build-cuda/stationary-evidence/`; reviewed tests and this decision record carry the durable acceptance contract.

## Consequences and revisit conditions

C1 now has a complete bounded RKS diagnostic path, but this is not C2/public force qualification. Revisit before production enablement to remove avoidable host record packing/TensorIR staging, qualify ragged public batches and complete resource/failure timing, and independently validate UKS and any ECP/DF/hybrid/meta-GGA/higher-angular-momentum provider chain.

## References

- Issue #163 C1
- `docs/stationary_cuda_diagnostic.md`
- `python/vibeqc/_stationary_cuda.py`
- `python/vibeqc_compiler/method/stationary_cuda.py`
- `src/dft/stationary_gradient_cuda.cuh`

Agent: ChatGPT
Model: GPT-5.6 Sol
