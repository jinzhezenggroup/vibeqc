# GPU RCCSD solver, energy-only API and isolated batches (#149 B/C)

This document records the #149 B/C implementation state and its explicit
boundaries. The A slice (fixed-amplitude kernel parity) is `rccsd_gpu.md`; the
scientific B/C CPU baseline is `rccsd_bc.md`. The physical equations, inputs,
denominators and final acceptance are exactly #148's — the code described here
only changes which backend evaluates the residual, and how energy-only
single points and batches are exposed.

## B: bounded ordinary-stream GPU solver

`tools.vibeqc_cc.gpu_solver.PreparedGPUSolver` owns the #148 preparation and
two compiled #146 plans:

- the **primary** plan is `gpu_state.iteration_program` — the `shared`-form
  physical DAG plus `next_t1/next_t2 = t + (1-damping)*R/D` with resident
  denominator inputs `d1/d2`;
- the **replay** plan is `build_ccsd_program(..., form="expanded",
  diagnostics=False)`, the independent expanded DAG used for final acceptance
  and re-evaluated freshly with unshifted denominators.

Both plans are composed and bounded by `gpu_state.solver_plans` under the
`SolverOptions.max_bytes` budget (primary + replay + caller-supplied provider
peak). An infeasible combined peak raises before any CUDA allocation. The
`denominators`, `AmplitudeSnapshot` warm-start identity, and the DIIS state
reservation from `gpu_state.py` continue to apply unchanged.

The control law in `gpu_solver.solve_gpu` is a faithful transcription of
the #148 CPU loop `tools.vibeqc_cc.solve`: the MP2-like initial guess, damped
shifted-denominator Jacobi update, host CC DIIS (bounded history, drop-oldest
on singular systems, restart counting), and the two-stage acceptance
(energy change **and** physical `max|R1|,|R2|` on the shared DAG, then a fresh
expanded DAG reproduction). The GPU evaluates only the energy/residual tensors;
the host forms the Jacobi trial and drives DIIS exactly as the CPU path does.
Results are the same owned `CCSDResult` (status/reason/energy/amplitudes/
history/provenance/replay inputs and `.write()`), so `tools.replay_ccsd`
reproduces a GPU result on the CPU solver within cross-backend rounding, without
new AO work. The primary-plan history omits the CPU `energy_components` and
adds backend timings; the physics and acceptance are identical.

### Residency boundary (honest, not hidden)

The #146 executor `PreparedCuda.execute` uploads every feed and downloads every
output on each call; it does not expose device-resident pointers. A complete
resident loop where only small residual scalars cross the host is #193's
resident provider interface. This module therefore runs a **host-controlled
ordinary-stream iteration**: amplitudes and residuals are staged through host
buffers each iteration, and that per-iteration transfer volume plus the
`graph_status = "ordinary-stream: ..."` fact are recorded in the result
provenance. It is an honest bounded GPU solver and convergence endpoint, not a
claim of resident acceleration. `src/cc/cuda_state.cuh` DIIS/`max-norm`
kernels remain preparatory until the #193 interface provides resident device
pointers.

Failures are explicit: provider failure raises (never a silent CPU fallback);
numerical overflow is a `nonfinite` result; the iteration limit is a
`not_converged` result carrying the last finite state. Compilation requires an
explicit `CudaCompilerAdapter` and cache — CUDA is never implicit.

## C: energy-only API and isolated batch semantics

`tools.vibeqc_cc.api` exposes:

- `method_capabilities("rccsd")` → energy-only, `supports_batch=False`;
- `energy(snapshot, provider, backend="cpu"|"cuda", ...)` → single point,
  force requests raise `NotImplementedError`;
- `batch_energy(problems, ...)` → independent per-item states; one item's
  failure is captured as an `error` item and never corrupts a neighbor;
  heterogeneous system shapes run independently without padding.

The native C-ABI registry (`VIBEQC_METHOD_*`) is deliberately unchanged: there
is no `VIBEQC_METHOD_RCCSD` identifier, and no native C++ CC solver or resident
executor exists yet — those are #193's resident-interface prerequisite. The
reserved `VIBEQC_METHOD_RCCSD_T` stays unavailable. Registering ABI methods or
a native ragged `supports_batch` without the resident interface would advertise
capabilities that do not exist, so this slice exposes the method at the same
Python facade tier as the #148 `solve` entry point and documents that boundary.

## Validation status

Local (numpy-only) verification is complete and passing for all non-device
paths: `tests/python/test_cc_api.py` (9) and the plan/preparation layer of
`tests/python/test_cc_gpu_solver.py` (4 skipped device tests) plus the existing
#148 GPU-state tests.

Real-device GPU validation runs through `tools/validate_cc_gpu_solver.py`
(five committed endpoint molecules, converged energy against pinned PySCF
2.14.0, independent expanded residual gate, and CPU replay parity):

```bash
python -m tools.validate_cc_gpu_solver --output build/cc149-bc-results \
  --cache build/tensor-cuda-cache --nvcc /usr/local/cuda/bin/nvcc \
  --architecture sm_90
VIBEQC_CC_CUDA_TEST=1 VIBEQC_TENSOR_ARCH=sm_90 \
  VIBEQC_NVCC=/usr/local/cuda/bin/nvcc python -m pytest \
  tests/python/test_cc_gpu_solver.py -q
```

This run requires an allocated GPU validation window on qz/inspire. It has not
been executed from this workstation (which has no CUDA toolchain); until it
passes there, the real-device convergence gate is honestly **not-run**, not
fabricated. The `compile_only` flag records planning/compilation evidence
without claiming numerical acceptance.