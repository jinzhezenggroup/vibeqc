# GPU RCCSD implementation and acceptance

This slice builds on the merged **PR #215** and its audited #148 A/B/C
implementation. The historical GPU evidence used CPU commit
`5f31c4289db59853e4f64a942a51b4cccc68a3cd`. That CPU implementation supplies
the physical equations; #149 does not introduce a second CC equation source.
Historical evidence preserves its original source identities; the active code
integrates current master. Shared native reference/provider and resident
TensorIR interfaces are coordinated with #193.

## A: fixed-amplitude equations

`tools.vibeqc_cc.cuda.PreparedRCCSDResidual` compiles the complete #148
energy/R1/R2 DAG with the #146 planner and executor. `trace=True` retains every
live node as an output, with unchanged node definitions. The resulting longer
lifetimes and host output storage are charged by the same planner. It is an
audit mode, not a solver benchmark.

The runner uses the exact nonzero amplitudes and Fock/MO integrals from the
checked #148 reference file. Every GPU node is compared to the CPU interpreter;
named physical residuals, energy, and intermediates are additionally compared
to the pinned independent reference. Per-element tolerances remain
`atol=1e-11, rtol=1e-10`, with the #138 absolute energy/residual limits
`1e-8/1e-9`. Normal and minimum feasible numeric budgets are exercised, and
one byte below the minimum must fail before CUDA allocation.

```bash
export PYTHONPATH=.:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tools.validate_cc_cuda --architecture sm_90 \
  --nvcc /usr/local/cuda/bin/nvcc --cache build/tensor-cuda-cache \
  --output build/cc149-reproduction
VIBEQC_CC_CUDA_TEST=1 VIBEQC_TENSOR_ARCH=sm_90 \
  VIBEQC_NVCC=/usr/local/cuda/bin/nvcc python -m pytest \
  tests/python/test_cc_cuda.py tests/python/test_cc_cuda_state.py -q
```

Use the architecture of the allocated GPU. `--compile-only` does not touch a
GPU and records numerical acceptance as false. Real execution requires an
assigned validation window. Each record contains the generated-library hash,
source identities, plan/lifetimes/budget, device/toolchain identity, raw repeat
and section-profile measurements, numerical errors, and transfer accounting.
Section profiling synchronizes operations and is not a performance ranking.
Module/context/allocator overhead follows #146's explicitly excluded scope;
the device-memory delta is reported separately from numeric buffer capacity.

The fixed-amplitude API uploads all feeds and downloads all outputs on each
call. It must **not** be used as a host-driven CC iteration or advertised as
GPU-resident solving. It registers no public RCCSD method.

## Remaining B/C acceptance

- The solver must keep amplitudes, physical residuals, denominators, integrals,
  workspaces and DIIS histories on device. Only small diagnostics may control
  iterations on the host. The #193 resident interface is a prerequisite.
- `gpu_state.iteration_program` defines the damped Jacobi update in TensorIR
  separately from the physical equations. Shifts affect denominator inputs;
  the final expanded equation must be freshly evaluated without them.
- `AmplitudeSnapshot` owns immutable amplitudes tied to an exact reference
  identity. Geometry, generation, and orbital changes invalidate reuse even
  at identical shape. No cross-geometry transport is implemented.
- `src/cc/cuda_state.cuh` provides allocation-free DIIS Gram construction via
  cuBLAS, a bounded GPU coefficient solve, extrapolation and physical maximum
  residual reductions. It is preparatory code until integrated and validated
  through the complete solver. Production storage must be a #146 reservation.
- Public RCCSD must use its own method identifier, preserving all existing
  values and the reserved RCCSD_T entry. Energy-only capabilities must reject
  forces, unsupported references, precision and frozen-core configurations.
- Single-system endpoints precede supported homogeneous prepared batches;
  every item requires isolated T/DIIS/status. Ragged batching is unsupported
  unless separately implemented and validated.
- Snapshot lifetime, repeated execution, independent contexts, partial batch
  failures and full VibeQC HF→integrals→GPU RCCSD require end-to-end evidence.
  Neither kernel parity nor successful compilation completes #149.
