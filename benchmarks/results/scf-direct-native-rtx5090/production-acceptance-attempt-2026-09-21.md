# #240 production-acceptance audit attempt — 2026-09-21

Status: **retained evidence, not final acceptance**

This note preserves production-build and incremental-rebuild evidence gathered while advancing #240. It must not be used to close #240 because `master` advanced during the run and the submitted runtime batch was not captured into this retained record.

## Audited source and environment

- Audited source SHA: `41be6f8d8db7b74336810cf3e27c5b33595bbc52`
- Host: `node3`, Linux 6.8.0-136-generic x86_64
- CMake 4.2.1; Ninja 1.13.0.git.kitware.jobserver-pipe-1
- GCC 11.4.0; CUDA 12.9 (`Build cuda_12.9.r12.9/compiler.36037853_0`)
- Release, `sm_120`, `VIBEQC_CUDA_FAST_COMPILE=OFF`, `VIBEQC_COMPILER_CACHE=off`
- `VIBEQC_CUDA_COMPILE_JOBS=2`, total Ninja parallelism 4
- code-generation Python: `/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python`, NumPy 2.4.6

The first empty-build attempt let CMake select `~/.local/bin/python3.11`; generated-grid code generation failed immediately because that interpreter did not have NumPy. The successful run below explicitly pinned the known-good interpreter.

## Structural gate

`tools/check_scf_structure.py --json` reported 222 shared SCF modules, 815 dependency edges, and 0 dependency errors. `src/scf/cuda_rhf.cpp` was 4,803 physical lines / 275,087 bytes; `src/scf/cuda/df_plan_setup.cpp` was 830 lines / 44,843 bytes. The residual driver therefore remains a documented legacy orchestration exception rather than silently satisfying the 600-line review target.

## Empty production build

Configuration:

```text
cmake -S <source> -B <empty-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVIBEQC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DVIBEQC_CUDA_FAST_COMPILE=OFF \
  -DVIBEQC_COMPILER_CACHE=off \
  -DVIBEQC_CUDA_COMPILE_JOBS=2 \
  -DPython3_EXECUTABLE=/home/jzzeng/codes/vibeqc-torch-boundaries/.venv/bin/python

cmake --build <empty-build> --parallel 4 -v
```

| Measurement | Result |
| --- | ---: |
| Configure wall time | 3.16 s |
| Configure max RSS | 188,160 KiB |
| Full build wall time | 2,183.29 s |
| Full build max RSS | 2,742,912 KiB |
| `libvibeqc.so.0.1.0` size | 369,505,968 bytes |
| `vibeqc_weighted_eri_probe` size | 32,992 bytes |
| Direct-native device-link commands captured | 1 |
| Build result | PASS |

The host was shared with other active CUDA compilation campaigns, so the wall time is a reproducibility receipt for this configuration, not a performance comparison.

## Representative incremental rebuilds

Each implementation was touched only to advance its mtime and the `vibeqc` target was rebuilt with the same Release tree. Git content remained unchanged.

| Edited owner | Wall time | Recompiled owner | CUDA compile actions | Device-link actions |
| --- | ---: | --- | ---: | ---: |
| `src/scf/cuda/rhf_graph.cpp` | 2.87 s | `rhf_graph.cpp.o` | 0 | 0 |
| `src/scf/cuda/rhf_bucket.cpp` | 3.58 s | `rhf_bucket.cpp.o` | 0 | 0 |
| `src/scf/cuda_rhf.cpp` | 8.43 s | `cuda_rhf.cpp.o` | 0 | 0 |

Each edit relinked `libvibeqc`, but none recompiled a CUDA translation unit or reran the direct-native device link. A post-rebuild structure check again reported 222 shared SCF modules and 0 dependency errors, and `git status --porcelain` was empty.

## Why this is not final #240 acceptance

While the production build was running, `master` advanced from the audited `41be6f8` to `43d6354f087f847d3b0984d5f9827ef540517c9d`, four commits ahead. The intervening files include native/build-relevant changes such as `src/integrals/hvp_assembly_runtime.cu`, `src/integrals/first_gradient_runtime.cuh`, and `src/scf/weighted_eri_runtime.hpp`. Therefore the `41be6f8` receipt is intentionally **not** promoted to the exact-final-SHA acceptance table.

A Slurm runtime batch was submitted as job 10615 requesting one RTX 5090 and configured to run the complete native CTest set, the retained HF/DF/MP2 CUDA Python endpoint suite, and `tools/validate_weighted_eri.py` against the same production library. Its result was not captured into this retained record, so no CTest, Python endpoint, or weighted-reference pass claim is made here.

## Remaining close-out gate

Before #240 is closed, rerun the audit on one stabilized final SHA and retain: the empty Release CUDA 12.9/sm_120 build; library size and direct-native device-link receipt; Graph/bucket/driver incremental touched-object sets; structural audit; native CTest; HF/DF/MP2 Python CUDA endpoints; and weighted-integral/reference validation.

Only after those exact-final-SHA results are captured should `docs/scf_module_boundaries.md` be promoted from pending production acceptance and #240 closed.

---
Agent: ChatGPT
Model: GPT-5.6 Sol
