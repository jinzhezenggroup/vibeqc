# Decision: keep CUDA providers external to Linux Python wheels

Status: implemented
Date: 2026-09-18

## Problem

VibeQC's native CUDA build normally links `CUDA::cudart`, `CUDA::cublas`,
and `CUDA::cusolver`. Carrying that contract unchanged into a Python wheel
made the wheel depend on the build image's complete NVIDIA provider SDK and
left hard CUDA `DT_NEEDED` entries in `libvibeqc.so`. The CUDA manylinux
image used by cibuildwheel intentionally contains NVCC/compiler runtime headers
without the complete cuBLAS/cuSOLVER development package, exposing that the
source-build linkage boundary was not a suitable distribution boundary.

The wheel also needs to coexist with environments that obtain CUDA user-space
libraries from NVIDIA's separately distributed Python packages rather than a
system toolkit, without ever treating a toolkit `libcuda` stub as the NVIDIA
kernel driver.

## Decision

Keep two explicit CUDA linkage modes.

Native SDK/source builds retain the existing CMake provider targets and link
`CUDA::cudart`, `CUDA::cublas`, and `CUDA::cusolver` normally. Python
wheel builds set `CMAKE_CUDA_RUNTIME_LIBRARY=None` and use hidden Implib.so
ELF trampolines for the curated CUDA host symbols VibeQC actually references.
Those trampolines lazily resolve the exact CUDA-12 SONAMEs
`libcudart.so.12`, `libcublas.so.12`, and `libcusolver.so.11`; the wheel
does not carry ordinary provider `DT_NEEDED` edges.

Wheel compilation uses NVCC's CUDA runtime/compiler headers plus VibeQC-owned
minimal cuBLAS/cuSOLVER ABI declarations. These declarations contain only the
opaque types, enum values, and function signatures used by VibeQC; native builds
continue to compile against NVIDIA's full headers.

At Python runtime, Linux installs declare the reviewed NVIDIA CUDA 12
user-space packages as base dependencies. This is required because the
CUDA-bearing `libvibeqc.so` executes NVCC fatbinary/function registration
constructors while the DSO is loaded, before a caller selects a CPU or CUDA
calculation path. The historical `vibeqc[cuda12]` extra remains as an empty
compatibility alias rather than the ownership boundary. Before loading
`libvibeqc.so`, the package discovers `site-packages/nvidia/*/lib` and
preloads the reviewed provider cohort by absolute path so later SONAME lookup
reuses those objects. `libcuda.so.1` is never sourced from a Python package or
toolkit stub; the kernel driver remains system-owned.

The vendored Implib.so templates are direct binary-generation inputs and are
therefore hashed as part of `VIBEQC_SOURCE_IDENTITY`, in addition to the
provenance manifest and VibeQC wrapper generator.

Runtime JIT/autotuning is a separate developer-toolchain boundary. It may still
require NVCC/PTXAS and a full development toolkit; the Python runtime-provider
dependencies do not claim to provide that toolchain.

## Rejected alternatives

- Vendor CUDA provider shared libraries into the VibeQC wheel. This would make
  the artifact much larger, duplicate NVIDIA's independently distributed
  packages, and blur driver/runtime ownership.
- Keep hard CUDA `DT_NEEDED` edges and stage provider wheels into the build
  image. That made a successful wheel build depend on an ephemeral provider
  installation and did not solve import/runtime portability.
- Treat `auditwheel --exclude` alone as sufficient. Excluding a dependency from
  repair does not remove the ELF dependency or make provider discovery work.
- Load only `libvibeqc.so` in wheel CI. Implib resolution is lazy, so a
  load-only test can pass without exercising any CUDA provider symbol.
- Make wheel packaging change the normal native build. Source/benchmark builds
  should continue to use the toolkit's canonical CMake targets and headers.

## Invariants

- A repaired Linux CUDA wheel must not bundle cudart, cuBLAS, cuSOLVER,
  cuSPARSE, nvJitLink, or `libcuda`.
- The wheel's `libvibeqc.so` must have no CUDA provider `DT_NEEDED` entries.
- Linux wheel metadata must declare the reviewed CUDA 12 user-space provider
  cohort as runtime dependencies; CUDA registration occurs during DSO load.
- The installed-wheel test must install the base wheel without a CUDA extra and
  execute a provider-backed CUDA runtime call; a no-driver/no-device status is
  acceptable, but merely opening the library is not.
- `libcuda.so.1` always comes from the host driver.
- Native SDK builds retain ordinary CUDA toolkit linkage.
- Every vendored template byte that can change the trampoline binary participates
  directly in VibeQC source identity.
- New CUDA host API calls must be added deliberately to both the minimal
  declaration surface and the curated trampoline symbol set.

## Evidence

On CUDA 12.9 / RTX 5090, a wheel-mode build using the minimal host declarations
linked successfully with `-Wl,-z,defs`. `readelf -d libvibeqc.so` reported
only the standard C/C++ runtime dependencies and no CUDA provider
`DT_NEEDED`; `nm -D --undefined-only` reported no CUDA/cuBLAS/cuSOLVER
undefined symbols.

With `LD_LIBRARY_PATH` unset, Python preloaded the `nvidia-*` provider
cohort and the provider-free library reported CUDA runtime 12.9 and driver 13.0
on an RTX 5090. H2/STO-3G CUDA RHF then converged in two iterations at
-1.1167143251757694 Ha. These checks exercise the installed-provider ABI path;
they are compatibility evidence, not a performance claim.

The repository's native CUDA 12.9 production compile, CuMetal CUDA path,
pre-commit checks, and CPU gcc/clang builds remain independent regression gates.
The dedicated `Python wheels` workflow additionally verifies the repaired
payload, provider exclusion, ELF dependency boundary, and installed-wheel
provider smoke.

## Consequences

The wheel gains a small maintained ABI declaration/symbol inventory and a
vendored MIT-licensed trampoline generator template set. Adding a CUDA host call
now requires keeping that inventory synchronized. The provider DSOs remain
independently distributed and are never bundled into VibeQC, but Linux wheel
installation now brings in the reviewed CUDA-12 user-space cohort because
loading the CUDA-bearing native DSO requires cudart registration support. A
future split between CPU and CUDA DSOs could make those providers optional
again.

## Revisit when

Revisit this boundary if NVIDIA publishes a stable packaging/linkage mechanism
that removes the need for local trampolines, if VibeQC splits CPU and CUDA code
into independently loaded DSOs so providers can become truly optional, if
VibeQC targets a new CUDA major with different provider SONAMEs or ABI, if
non-Linux CUDA wheels are added, or if runtime JIT is intentionally made
self-contained from Python-distributed toolchain components.

## References

- PR #436
- `pyproject.toml`
- `.github/workflows/wheels.yml`
- `cmake/VibeQCCudaImplib.cmake`
- `src/runtime/nvidia_host_api/`
- `python/vibeqc/_cuda_runtime.py`
- `cmake/3rdparty/implib_manifest.json`
