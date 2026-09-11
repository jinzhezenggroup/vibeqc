# RCCSD fixed-amplitude GPU evidence

This is #149 A: the same #148 energy, physical R1/R2 and intermediate TensorIR
evaluated on a real H100. It also records tested CC state primitives. It does
not establish a resident iterative solver, public RCCSD API, supported batches
or complete GPU molecular convergence. B/C remain unverified.

## Archived raw records

The expanded numerical traces, frozen source files, and historical execution
records are preserved byte for byte in [raw-evidence.zip](raw-evidence.zip).
The [manifest](raw-evidence.manifest.json) binds the archive, every member's
size/SHA-256, and migration source commit. Paths below such as `numerical/`,
`frozen-source/`, `window/`, and `minimum-window/` refer to the restored tree;
`source-snapshot.json`, the review, and acceptance summary remain readable here.

Verify or restore into a new directory with the standard-library verifier:

```bash
python -m tools.unpack_evidence benchmarks/results/rccsd-149-a
python -m tools.unpack_evidence benchmarks/results/rccsd-149-a \
  --output build/cc149-history
```

Restoration refuses an existing destination and validates every byte before
writing. Frozen files remain historical records; current source fixes do not
rewrite their recorded validation identity.

## Source and scope

The tested checkout was based on #148 C commit
`5f31c4289db59853e4f64a942a51b4cccc68a3cd`, with upstream
`73be8676e5dd0bf3882e52807530655eb4732ef5` integrated before testing.
The recorded revision alone is insufficient to identify this then-uncommitted
candidate: `source-snapshot.json` records 283 source hashes and nine A-owned
files, retained under `frozen-source/`. The coordinator verified these against
the working files or staged Git blobs (normalizing the Git/Windows checkout
distinction by inspecting actual blobs, not changing recorded hashes).

Later `cuda_resident*` modules are not part of this snapshot or these results.
No host-input ABI test is being represented as persistent-device execution.
After publication, pre-commit reformatted one assertion in the current
`tests/python/test_cc_gpu_state.py` without changing its Python AST and added
terminal newlines to the two window result JSON files without changing their
parsed values. The original frozen test bytes were restored before this archive migration.
They now remain inside the archive, so no formatter exclusions are needed.
Use the frozen copy or original `5cbe3a2` Git blob for that historical test hash;
do not mistake the current formatting-only test bytes for the original bytes.
`transfers_per_execution` records that fixed-amplitude inputs and outputs are
transferred on each call. Kernel section measurements are diagnostic samples,
not a performance ranking or exclusive resource benchmark.

## Numerical and runtime results

`numerical/manifest.json` indexes four fixed nonzero-amplitude cases (random
2o2v, H2 1o1v, water 5o2v, LiH 2o4v) at normal and minimum feasible budgets.
All eight records pass their 5664 individual checks. Maximum absolute error
is `3.552713678800501e-15`. Every executed node is checked against the CPU
interpreter; energy, physical residuals and named intermediates are additionally
checked against pinned independent #148 reference outputs. Per-element gates
remain `atol=1e-11, rtol=1e-10`. One byte below each tested minimum is rejected;
the four failure diagnostics are retained in the manifest.

The GPU was an NVIDIA H100 80GB HBM3, sm_90, driver 595.58.03, CUDA runtime
12.9 / cuBLAS 12.9.1. Exact device data and artifact/source identities are
retained in the numerical records and window device snapshots.

- First monitored window: 2026-09-08 13:01:48–13:02:27 UTC. Parity, eight GPU
  tests and memcheck exited zero. `window/result.json` records command, PID and
  start/end for each phase; `window/memcheck.log` reports zero errors.
- Follow-up minimum-budget window: all four shapes at both budgets passed
  memcheck, zero errors; the updated GPU state tests again passed 8/8. See
  `minimum-window/result.json` and `minimum-memcheck.log`.
- The combined upstream CPU run passed 999 Python tests, with 188 skips, and
  native CTest 11/11. Skips are not device acceptance; the separate real-GPU
  records above establish only the stated A scope.

These were short correctness runs with competing-PID monitoring. No competing
PID interruption was reported. The records do not claim exclusive timing or
memory-performance conditions. No other task was stopped for this validation.

## Portable reproduction

From the intended checkout, activate its own compatible environment, allocate
a CUDA 12.9+ device and set the actual compiler/architecture. Use dedicated new
build output paths and coordinate with any other GPU user before execution.
Do not overwrite this historical evidence:

```bash
export PYTHONPATH=.:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tools.validate_cc_cuda --architecture sm_90 \
  --nvcc /usr/local/cuda/bin/nvcc --cache build/tensor-cuda-cache \
  --output build/cc149-reproduction
VIBEQC_CC_CUDA_TEST=1 VIBEQC_TENSOR_ARCH=sm_90 \
  VIBEQC_NVCC=/usr/local/cuda/bin/nvcc python -m pytest \
  tests/python/test_cc_cuda.py tests/python/test_cc_cuda_state.py -q
compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
  python -m tools.validate_cc_cuda --architecture sm_90 \
  --nvcc /usr/local/cuda/bin/nvcc --cache build/tensor-cuda-cache \
  --output build/cc149-reproduction-memcheck
```

`--compile-only` intentionally records numerical acceptance as false and is
not a substitute for these device executions. Unsupported or infeasible plans
must fail explicitly rather than fall back silently.

Two uninvolved read-only reviewers found no P0/P1/P2 issues; their checks and
limitations are recorded in `review.md`. The evidence above remains a fixed-amplitude acceptance record,
not completion of the full #149 issue.
