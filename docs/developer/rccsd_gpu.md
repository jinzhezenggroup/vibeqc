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

## Boundary inherited from the pre-resident slices

The fixed-amplitude and host-staged solvers established equation identity,
reference validation, memory planning and replay, but they deliberately did
not satisfy #149 B: T/R and integral inputs crossed the host boundary on each
evaluation. The resident implementation below is the replacement for that
historical limitation. The remaining open requirements after B are native
method registration, homogeneous prepared batches and full public/API evidence;
those are tracked under C.

`gpu_state.iteration_program` remains the single damped-Jacobi TensorIR owner,
`AmplitudeSnapshot` still forbids shape-only reuse across reference changes, and
cross-geometry amplitude transport remains unsupported.

## B: resident single-system solver

`tools.vibeqc_cc.resident_solver.PreparedResidentCCSD` now binds the existing
#149 primary iteration plan to the #420 resident TensorIR ABI. The physical
energy/R1/R2 equations and damped Jacobi proposal remain generated from #148;
no second CC residual implementation is introduced.

The primary owner uploads Fock/integral blocks, denominators and initial T1/T2
once. Across iterations it retains current/trial amplitudes, physical residuals,
DIIS vectors/errors, Gram/system scratch and reduction buffers inside the exact
#146 reservation. The resident post-run action computes R1/R2 max norms on the
same stream. Host control reads only the correlation energy and two residual
maxima; trial residual tensors feed GPU DIIS directly and are never staged to
the host. `src/cc/cuda_state.cuh` supplies the shared Gram solve support plus
history compaction and slice extrapolation for separately pinned T1/T2 spans.

The control sequence preserves the CPU solver's semantics:

1. run the exact physical equations at the current amplitudes;
2. check energy-change and physical residual gates from scalar diagnostics;
3. if unconverged, save the current finite state and device-copy generated
   `next_t1/next_t2` into the pinned input spans;
4. run the physical equations at that exact trial state;
5. append the trial amplitude/residual pair to resident DIIS, dropping the
   oldest pair and retrying after an ill-conditioned solve as the CPU `_DIIS`
   policy does;
6. form the extrapolated current amplitudes on device;
7. on a convergence candidate, download final T once and execute the separately
   retained expanded physical TensorIR replay on GPU before publishing success.

An arithmetic or solver failure never invokes the host-staged CC solver. The
last finite device amplitudes are retained separately and downloaded only for a
failure/result record. `diis_size=0` is supported as resident Jacobi rather than
a backend switch.

The internal energy facade accepts `backend="cuda-resident"`. It remains
energy-only. Slice C now additionally exposes the production native owner through
`VIBEQC_METHOD_RCCSD` / `Calculator(method="rccsd")`; force requests remain
unsupported. `PreparedResidentCCSD` itself
stays open after convergence. Its immutable `owner_identity` binds the
reference/integrals/equations/artifact/device; a separate
`solved_state_identity` additionally hashes the replay-qualified final T1/T2.
A new solve clears that solved identity before any mutation and republishes one
only after a fresh expanded replay. A later Lambda/gradient consumer can
therefore fail closed on stale or merely prepared owners rather than mistaking
owner identity for a solved amplitude state. The integral provider may be released after preparation:
all numerical inputs needed by the resident/replay owners have already been
copied or retained under their own lifetime.

### Transfer boundary

On the audited H2O endpoint (15 current iterations, 29 total primary runs), the
resident path measured an initial large H2D upload of 11,920 B. Thereafter the
reported large per-iteration H2D and D2H volumes are both exactly zero. Each
resident run returns only its 4-byte ABI status; current-state control reads
three FP64 scalars plus the reduction error status. DIIS pivot/arithmetic checks
add only small integer transfers. The final accepted amplitudes total 880 B and
are downloaded once for expanded replay/result publication. This is a transfer
claim for the TensorIR owner, not a whole-process PCIe or CUDA-context claim.

RTX 5090 / sm_120 validation covers H2, H2O and CH4 convergence against pinned
energies/amplitudes, retained solved-owner amplitudes, provider release after
setup, explicit nonconvergence and resident zero-DIIS execution. H2O agrees with
its pinned amplitudes to about 6.5e-12 max absolute error and its pinned total
energy to about 2.9e-13 Eh.

## C: native/public promotion

Slice C promotes restricted closed-shell CCSD as a genuine native method without
reusing the reserved `VIBEQC_METHOD_RCCSD_T` identifier. `VIBEQC_METHOD_RCCSD=12`
is an additive method id; all earlier values remain unchanged. The registry reports
energy only and prepared-batch support. A force buffer/request therefore fails
explicitly instead of returning an HF derivative under a CC label.

`src/methods/rccsd_method.cpp` owns the public HF -> conventional MO blocks ->
RCCSD lifecycle. It accepts only all-electron closed-shell RHF, FP64, unscreened
conventional integrals, no frozen core and no density fitting. The physical CC
energy/residual and independent expanded replay are generated at build time from
the same #148 TensorIR. The generated topology is extent-independent: occupied
and virtual dimensions are runtime values rather than a finite molecular shape
whitelist.

The CUDA native owner in `src/cc/cuda_solver.cu` uploads large Fock/integral,
denominator and initial-amplitude inputs once, then retains T1/T2, physical R1/R2,
DIIS histories and workspaces in one bounded device arena. Host iteration control
reads only scalar energy/residual/DIIS status. A convergence candidate must pass
the separately generated expanded physical replay on GPU before success. CUDA
provider or solver failure never falls back to CPU CC.

Prepared RCCSD batches intentionally admit one homogeneous `(nocc,nvir)` shape.
Each input owns an independent prepared calculation, amplitudes, DIIS and status;
an invalid or failed item cannot corrupt its neighbours. Changed geometry is
re-prepared and starts from the deterministic MP2-like amplitudes. Shape equality
alone never authorizes cross-geometry amplitude reuse. Ragged shapes are reported
as unsupported and should be split by the caller.

The correlation diagnostic appends CC iteration count, DIIS restarts, physical and
replay residual maxima, correlation energy and CUDA movement counters behind the
existing struct-size ABI boundary. A normal `NOT_CONVERGED` execution retains its
last finite energy and CC diagnostics; arithmetic/backend failures invalidate the
result. Older diagnostic callers continue to receive only the prefix they sized.

The production path is correctness/resource qualified, not a performance
leadership claim. Direct producer-to-consumer HF/AO2MO device leases and validated
cross-geometry orbital/amplitude transport remain future optimizations rather than
requirements for the declared energy-only capability.

The final #149 C endpoint matrix and exact reproduction commands are recorded in
[`benchmarks/results/rccsd-149-c`](../../benchmarks/results/rccsd-149-c/README.md).
The committed CPU and RTX 5090 records cover `(nocc,nvir)=(1,1)` and `(5,2)` at
64/128/256 MiB with the same source identity; the 64 MiB CUDA rows are explicit
pre-allocation budget rejections while the two larger budgets converge.

### Ordinary-stream comparison on the same H2O endpoint

With cached compilation artifacts, both the existing ordinary-stream GPU solver
and the resident solver converged in 15 current iterations to the same pinned
energy (2.84e-13 Eh absolute difference) and amplitudes (about 6.5e-12 maximum
absolute difference from the retained reference). The scientific control law is
therefore not changed to obtain residency.

The ordinary plan executed 29 primary evaluations. Its conservative transfer
accounting gives 11,920 B uploaded per primary evaluation, 1,768 B of complete
plan outputs per primary evaluation, plus one 11,040 B expanded-replay input:
356,720 B primary+replay H2D and 51,272 B full-plan D2H upper bound in total.
The resident owner instead reports 11,920 B of initial large H2D, zero
per-iteration large H2D/D2H, 524 B of scalar control D2H, and one 880 B final
amplitude download. The resident ABI itself additionally returns 4 B per run.

Cached wall times in this one observation were 1.09 s ordinary and 0.73 s
resident, but this PR does **not** promote a speedup: compilation is excluded,
fixtures are tiny, CUDA context/module state is shared with the process, and no
representative molecular benchmark matrix has been qualified yet.

### Reuse and isolation

A resident owner is reusable after convergence: calling `solve()` again starts
from the retained converged amplitudes and does not repeat the one-time large
input upload. `amplitude_snapshot()` detaches the current T1/T2 together with
the exact reference identity for a later explicit warm start. A warm snapshot
is accepted only by the identical `ReferenceSnapshot`; geometry, generation or
orbital changes are rejected before JIT compilation/device upload. Supplying an
identity-bearing warm snapshot together with raw T arrays is also rejected.

Two independently prepared owners may coexist on one device. The RTX 5090 test
keeps H2 and H2O owners alive simultaneously, verifies distinct resident-state
identities, and converges both without cross-state contamination. These tests
establish single-system owner reuse/context isolation; they are not the
homogeneous native prepared batch required by #149 C.
