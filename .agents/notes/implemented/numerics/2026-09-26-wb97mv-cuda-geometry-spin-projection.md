# Decision: Project MethodIR spin layout at the CUDA geometry boundary

Status: implemented
Date: 2026-09-26

## Problem

PR #1389 passed an RKS unpolarized FunctionalSpec into a geometry point lowerer
that requires polarized features. This rejected RKS source generation. The
native GridTaskView always supplies alpha/beta features; its density setup
already splits an RKS density into equal spin blocks.

## Decision

The stationary wrapper projects only the FunctionalSpec spin field to
`polarized` before geometry lowering. The original MethodIR, component weights,
range omega, provenance and other semantic fields remain intact. Direct
`functional=4` lowering still requires an explicit polarized FunctionalSpec.

Reconstructing a named WB97M-V spec would erase custom exact weights and omega;
changing MethodIR itself would break its density and source-weight convention.
Neither is an acceptable substitute for adapting the existing two-spin ABI.

The geometry point expression retains the pinned production Maple graph and
Libxc work-density, sigma and tau policy. The sigma threshold is converted to
float before hexadecimal emission because Python exponentiation's static type
can include an integer; this preserves the emitted binary64 constant.

## Acceptance boundaries

- Deterministic RKS/UKS generation, explicit semantic admission, and a custom
  weighted/range-parameter fixture protect the spin projection.
- `test_stationary_wb97mv_cuda_geometry.py` executes the actual generated CUDA
  wrapper, grid features and AO/tau pullback on LiH/STO-3G (s and p AOs), for
  both RKS and UKS positive fixed densities. AO-center and owner-grid-point
  derivatives are separately compared with PySCF 2.14.0 / Libxc 7.0.0 energy
  differences using a `2e-5` Bohr step and `atol=3e-8, rtol=3e-7`.
- Explicit grid weights remain fixed in this independent gate; the partition
  source is zero and the summed translation derivative must be below `2e-12`.
- GPU gates run on Slurm `main`, `--gres=gpu:5090:1`, with a finite ten-minute
  limit and CUDA 12.9. `VIBEQC_TEST_RANGE_CUDA=1` enables both stack slices.

This qualifies only the semilocal geometry slice. It does not qualify a complete
WB97M-V force endpoint, reconverged SCF forces, SR/LR/VV10 composition, or a new
partition-weight derivative implementation. Public capability remains unchanged.

## References

- PRs #1388 and #1389
- `python/vibeqc_compiler/method/stationary_cuda.py`
- `python/vibeqc_compiler/xc/geometry_cuda.py`
