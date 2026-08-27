# CUDA generated-shell profiles

VibeQC separates **correct CUDA execution** from **measured generated-kernel
performance**. A shell schedule tuned for one compute capability is not silently
reused on another GPU.

## Profile selection

CMake resolves one generated-shell profile for the first entry in
`CMAKE_CUDA_ARCHITECTURES`.

- An exact non-empty manifest profile, such as `sm_120`, enables its measured
  generated Fock and force kernels.
- A target without an exact profile selects `portable_cuda`. That profile
  generates an empty registry, disables generated shell classes, and retains
  the validated generic CUDA direct-J/K and analytic-force paths.
- An explicitly requested incompatible profile is rejected rather than compiled
  for another SM.

For example, a single-target Ampere build can be configured with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

Until an `sm_80` profile is measured and added to the production manifest, the
configuration reports `portable_cuda`. This is a correctness fallback, not an
A100 performance claim.

## Configuration controls

Use automatic exact-match resolution by default:

```bash
-DVIBEQC_AOT_PROFILE=auto
```

Force the generic portable path, including on an architecture with a tuned
profile:

```bash
-DVIBEQC_AOT_PROFILE=portable_cuda
```

Disable generated shell AOT explicitly:

```bash
-DVIBEQC_ENABLE_AOT_SHELLS=OFF
```

Require an exact tuned profile and fail configuration otherwise:

```bash
-DVIBEQC_AOT_REQUIRE_TUNED_PROFILE=ON
```

An explicit tuned profile must match the normalized target:

```bash
-DCMAKE_CUDA_ARCHITECTURES=120 \
-DVIBEQC_AOT_PROFILE=sm_120
```

`VIBEQC_AOT_CODEGEN_ARCHITECTURE` remains a deprecated compatibility alias for
`VIBEQC_AOT_PROFILE`.

## Current boundary

This first stage resolves **one** profile. For a multi-architecture CMake list,
only the first entry controls generated-shell selection. The same binary may
still contain CUDA code images for multiple targets, but it does not yet contain
independent tuned registries and schedules for each target. Issue #9 tracks
profile-suffixed kernel bundles and runtime dispatch.

The portable fallback also does not make schedule enumeration target-aware.
The next codegen stages are:

1. probe the allocated GPU and construct a capability record containing compute
   capability, warp size, register and shared-memory limits, occupancy limits,
   and SM count;
2. pass that target record into schedule enumeration and resource gates instead
   of using RTX 5090-oriented defaults;
3. tune algorithm and schedule jointly, including subset/Wick versus
   cooperative Rys lowerings, primitive/root/component ownership, data
   placement, and persistent-worker counts;
4. validate winners on real molecular workloads after synthetic pruning;
5. emit independent profile-suffixed registries into multi-architecture
   binaries and choose one once during CUDA context initialization.

Performance claims remain restricted to the exact device, software stack, and
benchmark artifacts used to accept a tuned profile.
