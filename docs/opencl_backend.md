# Experimental OpenCL compiler/runtime contracts

The optional OpenCL adapter executes generated scalar expressions and FP64
reductions on a real GPU. It does not provide an OpenCL HF/DFT calculation.
CUDA remains the production accelerator backend and the default selection.
No OpenCL SDK or Python extension is required by ordinary builds or imports;
the adapter loads the installed OpenCL ICD with `ctypes` only when constructed.

## Target audit and evidence

The available non-CUDA API on this machine is NVIDIA OpenCL on the RTX 5090.
This is a second compiler/runtime API on the same NVIDIA hardware, so it does
not establish portability to another vendor. Domestic-GPU validation is
**not-run**; no domestic GPU or toolchain was available for this issue.

The [execution archive](../benchmarks/results/opencl-rtx5090/report.json) records
the exact queried device and compiler identity, generated sources, compile/link
logs, seven event timings per kernel, resources and executable-cache roundtrips.
It also records Python/NumPy/PySCF versions, source-file hashes, the base commit
and dirty-worktree status. The integral source identity stays separate from its
scientific equation identity and workgroup schedule identity.

| Property | Audited boundary on this target |
| --- | --- |
| Device | NVIDIA GeForce RTX 5090, UUID `8e9c9e1ae183258c0b3a03a5ddebb2f8` |
| Driver / runtime | `580.95.05` / `OpenCL 3.0 CUDA` |
| Compiler / language | ICD online compiler supplied by this driver; queried `OpenCL C 1.2 `, compile and link available |
| FP32 / FP64 | FP32 is required by OpenCL 1.2; FP64 extension and nonzero double capability queried; this lowering uses FP64 |
| Subgroups | NVIDIA attribute query reports width 32; scalar kernels require no subgroup intrinsics |
| Workgroup limits | Device maximum 1024 threads; compiled smoke kernels report maximum 256 and preferred multiple 32 |
| Local memory | Device reports 49,152 bytes; dynamic arguments and compiled total are bounded before launch |
| Atomics | Integer extensions are enumerated in the report; FP64 atomic add is unsupported by this adapter |
| Math | OpenCL C builtins supply scalar math; the smoke exercises arithmetic and square root; full integral/Boys math coverage is pending |
| Streams / events | Native in-order queues, explicit cross-queue dependencies and producer flushes, event waits and profiling |
| Graphs / device enqueue | Unsupported; ordinary host submission is the correctness path |
| GEMM / eigensolve / Cholesky | No native provider configured; explicit unsupported result before submission |
| Profiling resources | Native event timestamps and queried workgroup/local/private memory; registers and spills are unknown |

The compiler has no separately queried version string; its identity is the
online compiler packaged with the exact driver above. Resource queries reported
one local byte and zero private bytes for the smoke kernels. These are retained
as vendor-reported values, not interpreted as register or spill counts.
There is no portable OpenCL 1.2 register/spill query and no external profiler
evidence in this gate.

Vendor documentation was checked on 2026-09-09. NVIDIA's
[OpenCL page](https://developer.nvidia.com/opencl) states OpenCL 3.0 conformance
for R465 and later drivers on Linux/Windows x86. The adapter's ABI and queue
semantics were checked against Khronos OpenCL-Headers and OpenCL-Docs;
[document provenance](../benchmarks/results/opencl-rtx5090/vendor-docs.json)
retains exact blob identifiers, URLs, file hashes and retrieval date. OpenCL 3.0
does not imply all optional 2.x features, so queried capabilities take precedence
over a version-number assumption.

## Scientific and execution contracts

`ScalarCEmitter` holds the existing C-family scalar arithmetic emitter.
`CudaEmitter` preserves its CUDA import/API, while `opencl_lowering.py` emits
structured OpenCL address spaces, arguments, work-item indexing and FP64 policy.
There is no CUDA source-text replacement. `ScalarKernel` names an ordered
primitive input ABI and existing scientific graph roots; emission preserves
the scientific hash while source and schedule hashes vary with the target.

`backend.TargetInfo` represents unknown subgroup and resident-workgroup limits
with `None`. CUDA continues to supply its known limits. `RuntimeCapabilities`
and `ExecutionShape` validate workgroup size, local workspace and explicit
optional requirements before execution. Synthetic widths 8/16/32/64 and unknown
widths are tested without importing an SDK. A missing subgroup width permits
scalar workgroups and rejects subgroup-dependent schedules.

`OpenCLRuntime` implements separate `clCompileProgram`/`clLinkProgram` calls,
GPU queries, typed context-owned buffers/programs/queues/events, bounded blocking
transfers, padded launches, event profiling and multi-level FP64 reductions.
Reduction scratch and intermediate results stay on device without floating-point
atomics. An event retains submitted objects until completion; forged, stale and
cross-context handles are rejected. Failed compilation preserves diagnostics and
releases the program. Cleanup attempts every native release after vendor errors.
An indeterminate reduction wait failure drains and invalidates the runtime;
a terminal execution failure releases its intermediate resources and raises.

The library boundary uses `LibraryRequest` for explicit FP64 dimensions, row/
column layout, workspace limit and normalized residual tolerance. A future
provider must report workspace and native error status, and expose numerical
residuals for eigensolves/factorizations before its result can be accepted.
The current `UnsupportedLibraryProvider` implements no operation and raises for
GEMM, symmetric eigensolve and Cholesky. It never delegates to CPU arithmetic.

`LocalArtifactCache` is explicitly selected and local. Each bounded atomic
record includes backend, compiler, language/options, driver/runtime, device,
scientific/source/schedule identities, ABI and binary checksum. Loading checks
file ownership, shared-write permissions, regular-file type, size and identity;
symlink records are refused. The cache reuses the existing atomic publication
helper and does not change CUDA official/user profile precedence or automatically
download/activate executable artifacts from a remote profile.

## Reproduction and numerical boundary

CPU contract tests and an ordinary CPU build need no OpenCL installation:

```bash
PYTHONPATH=python:. python -m pytest \
  tests/python/test_runtime_backend.py tests/python/test_opencl_lowering.py -q
cmake -S . -B build-cpu -DVIBEQC_ENABLE_CUDA=OFF \
  -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 4
ctest --test-dir build-cpu --output-on-failure
```

All actual GPU execution, including the OpenCL ICD, uses a finite Slurm job:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 bash -lc 'set -e
  export PYTHONPATH=python:.
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VIBEQC_OPENCL_TEST=1
  python -m pytest tests/python/test_opencl_runtime_gpu.py -q
  python tools/validate_opencl_backend.py --directory /tmp/opencl-gate'
```

The Slurm visibility environment is preserved. OpenCL ICDs may ignore CUDA
visibility, so this machine's manual runner requires a single-GPU OpenCL
inventory. Multi-GPU validation first needs an explicit scheduler-assigned
device-to-OpenCL UUID mapping; a device ordinal alone is insufficient.

The smoke first checks `2*x+y`, then lowers the existing normalized DF
`(p_x|p_x)` Coulomb-metric expression with its radial prefactor. It compares
three contracted references from independent PySCF/libcint: asymmetric centers,
coincident centers, and long signed contractions, totaling 67 primitive rows.
Workgroups 16/32/64 exercise partial final groups. The largest absolute error
is `3.50e-13`, within `atol=1e-11, rtol=3e-12`. Every compiled binary is exported,
installed in the private cache, reloaded and checked for identical outputs.

Boys moments are host-generated fixture inputs and primitive outputs are summed
on the host for this reference comparison. This is an expression/compiler/runtime
proof, not a complete on-device integral provider. Separate GPU runtime tests
verify native reductions at sizes 1, 17 and 8197, error cleanup, ownership,
cross-queue events and unsupported operations. Timings are smoke-kernel samples;
no HF endpoint performance or cross-vendor speedup is claimed.

The archived validation passed 353 CPU Python tests (48 optional tiers skipped),
13 native CPU tests and 9 OpenCL GPU tests with zero GPU skips. One Python
source-identity test initially ran before the CPU library was available; its
retry passed with the explicit completed-library path. Both logs are retained.
The relocated CUDA scalar emitter has identical arithmetic implementation after
renaming, and the existing CUDA generator/profile regression tests passed.

## Next native HF integration

1. Select available target hardware and pin its ICD/toolchain/library versions.
   Repeat the capability and Slurm UUID audit, retaining missing hardware as
   not-run and absent provider operations as unsupported.
2. Supply a native linear-algebra adapter through the existing context/provider
   boundary, beginning with FP64 GEMM, then symmetric eigensolve and Cholesky.
   Validate layout, bounded workspaces, native failures and independent residuals.
3. Extend the shared integral graph consumer with on-device Boys evaluation,
   normalized contraction, spherical expansion and matrix write ownership.
   Pass independent raw-block gates before wiring an HF calculation.
4. Add explicit native prepared-plan selection with context-owned topology and
   geometry invalidation. Start with bounded one-system RHF and ordinary queue
   submission; keep graphs optional and expose unavailable methods clearly.
5. Run cold, unchanged and moved-geometry RHF/UHF/batch endpoints with density,
   energy and force gates, resource limits and full timing evidence. Promote
   only validated operator/method/target combinations; this issue promotes none.
