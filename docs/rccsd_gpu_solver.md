# Experimental CUDA RCCSD validation solver

This document records the historical ordinary-stream validation helper that
preceded #149 B/C. Resident T/R iteration was later delivered by #555, and the
native `VIBEQC_METHOD_RCCSD` owner plus homogeneous prepared batches are now
implemented by #149 C. The A slice (fixed-amplitude kernel parity) and current
production boundary are summarized in `rccsd_gpu.md`; the scientific CPU
baseline is `rccsd_bc.md`. The physical equations, inputs,
denominators and final acceptance are exactly #148's — the code described here
only changes which backend evaluates the residual, and how energy-only
single points and batches are exposed.

## Experimental ordinary-stream GPU solver

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
output on each call; it does not expose device-resident pointers. Issue #149 step 2 requires a complete
resident loop where only small residual scalars cross the host. That execution
contract is not implemented by this experimental helper. This module therefore runs a **host-controlled
ordinary-stream iteration**: amplitudes and residuals are staged through host
buffers each iteration. Provenance reports bytes per evaluation and separate
primary/replay evaluation counts: an ordinary iteration evaluates both the
current and trial amplitudes, uploading all integral inputs on both calls. The
`graph_status = "ordinary-stream: ..."` fact is recorded in the result
provenance. It is an honest bounded GPU solver and convergence endpoint, not a
claim of resident acceleration. `src/cc/cuda_state.cuh` DIIS/`max-norm`
kernels remain preparatory until a resident #149 iteration can use device
pointers.

Failures are explicit: provider failure raises (never a silent CPU fallback);
numerical overflow is a `nonfinite` result; the iteration limit is a
`not_converged` result carrying the last finite state. Compilation requires an
explicit `CudaCompilerAdapter` and cache — CUDA is never implicit.

## Internal energy-only facade and isolated batch helper

`tools.vibeqc_cc.api` exposes:

- `method_capabilities("rccsd")` at this historical helper tier → energy-only,
  `supports_batch=False`; the native slice C registry now reports batch support;
- `energy(snapshot, provider, backend="cpu"|"cuda", ...)` → single point,
  force requests raise `NotImplementedError`;
- `batch_energy(problems, ...)` → independent per-item states; one item's
  failure is captured as an `error` item and never corrupts a neighbor;
  heterogeneous system shapes run independently without padding.

At the time this helper was introduced, the native C-ABI registry deliberately
had no `VIBEQC_METHOD_RCCSD`; that historical boundary is why this module never
claimed native execution. Slice C subsequently adds an independent additive
RCCSD identifier and a native resident solver/owner. The separate native
`VIBEQC_METHOD_RCCSD_T` CPU/CUDA energy owner is documented in `rccsd_t.md`.
This ordinary-stream module remains an independent validation backend,
not the production registry implementation.

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

RTX 5090 qualification now passes on clean source
`b7dcace8263b5a12b5aa9639c47679317fb71cae`: all five molecular endpoints,
fresh native RHF-to-CUDA-CC endpoints, CPU replay and the full CC test
selection (201 passed, 8 explicit skips). The source-bound results and
reproduction are retained in
[`rccsd-149-bc-review`](../benchmarks/results/rccsd-149-bc-review/README.md).
The `compile_only` flag compiles both plans without executing CUDA or
claiming numerical acceptance. All local GPU runs use finite Slurm allocations.
