# Issue #162 implementation and acceptance record

This is an intermediate record. **Issue #162 is not complete.** The original
RKS/UKS CPU/CUDA prepared-execution scope and both issue addenda remain the
acceptance contract. Gradients are #163; DF is not advertised by this work.

PR #306 publishes the validated resident XC/SCF and native ragged batch
commits `05951fb` and `afad4e4`. Its next stage adds the public #203 KS resource
request/CLI, allocation-owned CUDA shape queries and one persistent ledger
covering both prepare and execute. CPU budgets include all retained grids,
bases, providers and warm states plus serialized setup/SCF workspace. Device
budgets sum all concurrent item arenas; geometry rebuilds retire old owners
before allocation. Failures retain their preparation or execution evidence.

Resource-stage validation on 2026-09-14: 25/25 native CPU tests; 83 selected
Python resource/calculator/batch/HF tests passed, with 3 optional skips;
10 CUDA ownership tests passed. The existing C++ heap audit, extended through
the native KS method adapter, passed ten HF/KS cases including >16-AO water,
cold/replay/changed/restored geometry and complete release after destruction.
On the Slurm RTX 5090, four native tests and 24 KS/HF CUDA resource cases
passed. Six CUDA KS resource cases also passed Compute Sanitizer memcheck
in 124.79 s with zero errors and zero bytes leaked. These capacity checks
do not replace the full #138 numerical/workload evidence gate.

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  benchmarks/resource-planning/cpu_inventory.py --build build/cpu \
  --include-ks --output /tmp/ks-cpu-resources.json
```

PR #305 is merged as `15d6936390723edf9e9eb0c91fecde4390490573`.
Implementation `05951fb` is integrated onto that actual squash commit. It retains
all CPU corrections: stable extreme-spin/gradient point algebra, evaluated
UKS returned states, OH occupation stabilization, separate public density and
physical residual diagnostics, and permanent independent endpoint evidence.
The merge-linked issue closure was corrected because the full #162 acceptance
scope is still incomplete.

The prior integration on `c08927e` passed 24 native CPU tests, 97 RTX 5090 point
references and 32 public CPU/CUDA matched-grid endpoint/API cases. Those are
historical measurements; the integration onto the final merged API is being
revalidated. CPU occupation stabilization still needs matching CUDA SCF
implementation and coverage before full cross-backend parity is established.

Validation of `05951fb` on 2026-09-14 passed 25 native CPU tests, 166 selected
Python CPU tests (3 optional skips), 10 CUDA-ownership inventory tests, four
native CUDA tests, and all 32 public CPU/CUDA matched-grid SCF cases. CUDA
execution used an RTX 5090 allocated through Slurm, CUDA 12.9 and sm_120.

The next working-tree change adds native prepared ragged KS scheduling,
geometry rebuilds, resident/frozen/cleared/imported last-good seeds and an
additive per-item SCF diagnostic query. CPU validation passed 25 native tests
and 47 selected Python tests (3 optional skips). Two subsequent derivative-
capability regressions also pass: PBE energy requires first AO jets, LDA only
AO values, and neither energy-only method requires nuclear derivatives.
The matching CUDA rebuild passed. Its four native tests and all 45 combined
public batch/independent-SCF cases pass on the Slurm-allocated RTX 5090.
Compute Sanitizer memcheck of all five CUDA batch cases passed in 239.50 s,
with 0 errors and 0 bytes leaked. This covers ordinary replay, geometry
rebuilds, item failure/recovery, frozen and cleared seeds, atomic source-metric
seed import and iteration-limit handling for both functionals/spin conventions.

```
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_PROFILE=off OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 99 --leak-check full \
  .venv/bin/python -m pytest tests/python/test_dft_batch.py -q -k cuda
```

## Implemented and checked so far

- Existing #285 CPU RKS and #214 identical-grid XC integrator/oracles retained.
- Shared CPU/CUDA LDA/PBE point evaluator with stable positive-density PBE
  algebra, no LDA tail substitution, and explicit C2 spin endpoint identity.
  See [the numerical domain](../../../docs/xc_scf_domain.md).
- Independent 97-point E/V fixture: Libxc 7 interior and 450-digit mpmath
  original-formula tail/spin derivatives. CPU and RTX 5090 point tests pass.
- Native ordinary-stream device-buffer XC now reuses the generated AO kernel,
  minimal D/gradient ingredients and shared point coefficients. It consumes a
  method-owned exact-size arena and uploads host quadrature once; iteration
  enqueue has no bulk staging, allocation or synchronization. See
  [the device contract](../../../docs/xc_native_cuda.md).
- RTX 5090 fixed-density CPU/CUDA RKS/UKS E/V, spin finite differences,
  Cartesian/spherical f shells, empty-spin/vacuum tails, device-produced
  density, stale generation/grid rejection, failure isolation and arena
  canaries pass. Compute Sanitizer memcheck reports 0 errors and 0 leaks.
- The #202 exact direct provider now exposes its ordinary stream and a
  validated device-buffer enqueue seam. It reuses common specification checks
  and the existing J/K kernel. Resident RKS/UKS J/K through f agrees with the
  independent CPU integrals; absent terms, alias/shape rejection and numerical
  failure recovery pass alongside the existing DF/response provider checks.
  Both new device test executables pass Compute Sanitizer memcheck with full
  leak checking: 0 errors and 0 bytes leaked for each executable.
- Native ordinary-stream GPU SCF now composes the #202 J provider, XC arena,
  existing device matrix/DIIS/eigensolver/density kernels, and current physical
  state. All iteration matrices remain resident; only scalar records are read.
  Exact state/XC arena allocations are charged to the existing #203 ledger.
- Registered CPU/CUDA LDA/PBE RKS/UKS single-system execution reports the
  actual backend and rejects forces. Energy-only GPU results do not download
  a final density. Compatible last-good warm density is retained on the device;
  failed or exhausted runs cannot replace it. CPU prepared replay also reuses
  its compatible last-good density with fresh DIIS.
- RTX 5090 KS tests pass for LDA/PBE H2 RKS and H/H2+/H3 UKS, including
  independent CPU reconstruction of returned E/D/residual, same-geometry
  resident replay, changed-geometry normalization, stale grid rejection,
  numerical failure recovery, iteration limits and exact arena accounting.
  The native SCF executable passed Compute Sanitizer memcheck with full leak
  checking: 0 errors and 0 bytes leaked.
- CPU LDA/PBE UKS through the method registry and the #202 Coulomb-only
  strategy, with independent spin occupations, Fock/residual evaluation,
  trace-normalized warm density and per-run DIIS state.
- KS convergence gates energy, density change and physical residual; the
  residual gate is at most 1e-9. RKS checks its final physical rebuild too.
- Opt-in normalized DIIS metric avoids the absolute-pivot failure at tight
  residuals. HF retains its original default DIIS behavior.
- Native wrong-factor, XC double-counting, spin swap, empty-spin, stale-grid,
  stale-AO, changed-geometry warm and failed-run tests pass.
- Stable independent matched-grid CPU/CUDA SCF tests pass for LDA/PBE H2,
  He, water (RKS), H, Li and H2+ (UKS), with two independent PySCF initial
  guesses. The public CUDA backend must actually execute for CUDA cases.
  The extended test module passed 28 CPU/CUDA endpoint/API cases; four additional
  water/def2-SVP cases also passed above the small eigensolver dimension.

Validation on 2026-09-13, using GCC 11.4, Python 3.13.9 and PySCF 2.14.0:

```
cmake --build build/cpu -j 8
ctest --test-dir build/cpu -j 6 --output-on-failure
# 23/23 passed

PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so OMP_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests/python/test_calculator.py \
  tests/python/test_xc_integration.py tests/python/test_xc_expressions.py \
  tests/python/test_dft_scf.py -k 'not cuda'
# 130 passed, 3 skipped, 25 deselected

cmake --build build/cuda --target vibeqc_xc_point_cuda_tests -j 2
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:02:00 build/cuda/vibeqc_xc_point_cuda_tests
# 72 independent LDA/PBE SCF-domain E/V points passed

cmake --build build/cuda --target vibeqc_dft_cuda_tests \
  vibeqc_cuda_fock_provider_tests -j 2
# Full native CUDA library and both test executables built successfully.
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'build/cuda/vibeqc_dft_cuda_tests && \
  build/cuda/vibeqc_cuda_fock_provider_tests'
# Native device-buffer LDA/PBE RKS/UKS E/V and state gates passed.
# CUDA independent J/K: DF layouts/selection and direct through-f values,
# s/p derivatives PASS.

PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/python/test_xc_integration.py \
  -k identical_grid_independent
# 24 passed, 15 deselected. Existing #214 reference/hash checks retained.
```

Additional native SCF / public adapter validation on 2026-09-13:

```
cmake --build build/cuda --target vibeqc_ks_cuda_tests vibeqc_dft_api_tests -j 2
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:05:00 ctest --test-dir build/cuda --output-on-failure \
  -R '(vibeqc_ks_cuda_tests|vibeqc_dft_api_tests)'
# 2/2 passed, including actual CUDA public RKS/UKS execution.

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_PROFILE=off OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/python/test_dft_scf.py -q
# 28 passed; both backends and both independent PySCF guesses.

ctest --test-dir build/cpu --output-on-failure -R '(dft|uks|xc_point)'
# 5/5 passed after the public adapter change.
```

The CUDA build uses `/group/software/cuda-12.9.1/bin/nvcc` (12.9.86), sm_120,
Release, AOT shells disabled for these independent-provider development
checks. The point test transfers explicit point inputs/outputs; it is not
evidence for GPU XC integration, resident SCF or complete endpoint timing.
The separate `vibeqc_dft_cuda_tests` executable now supplies native GPU
fixed-density integration evidence. It has passed under finite Slurm jobs,
including `/group/software/cuda-12.9.1/bin/compute-sanitizer --tool memcheck
--error-exitcode 99 --leak-check full`; this does not establish full CUDA SCF.
Formatting, evidence-retention, compiler/SCF dependency and CUDA ownership
checks pass. The new point algebra is honestly classified as maintained
scientific code in the CUDA ownership ledger.

Earlier unconstrained exploratory OH calculations are not accepted independent endpoints: native
LDA/PBE converges, while PySCF's stricter convergence flag remains false for
some degenerate-grid traces, including Newton traces despite tiny energy
differences. Preserve this distinction when recording branch/guess evidence.
The later #305 CPU OH records use the explicit C2v reference subgroup and
pass the full unrestricted AO residual gate; see the source-bound correction
record. They do not establish the earlier unconstrained or CUDA traces.

## Immutable model-options increment (2026-09-14)

`KsOptions` reuses the existing `FunctionalSpec` and `GridSpec`, resolves only
audited LDA/PBE spin compositions, records the effective native tail policy,
and derives the minimal AO order from functional ingredients. Grid counts,
per-element radial scales and XC tile capacity are copied by native prepare
and included in both model identity and the shared resource request. A changed
model cannot execute an existing Python prepared owner. The additive C method
option respects legacy descriptor size; unsupported families/domains fail
before constructing scientific owners. See [the public contract](../../../docs/ks_options.md).

Validation of the model implementation:

- CPU native: 25/25 tests passed, including caller-storage snapshot, malformed
  options, non-DFT rejection and a legacy descriptor with an ignored invalid
  tail pointer. Python: 71 passed, 3 optional skips, 37 deselected across options,
  KS/HF resources, calculator and native batches.
- Slurm RTX 5090: 4/4 native tests and 32 selected CUDA options/resources/batch/
  independent SCF tests passed. Custom-grid LDA/PBE RKS/UKS agree with both
  independent PySCF guesses within 1e-8 Eh; tiles 31/128 preserve the discrete
  endpoint and device ledgers match the selected shapes exactly.
- All four CUDA custom-grid cases passed Compute Sanitizer memcheck with full
  leak checking: 0 errors and 0 bytes leaked. Log:
  `/tmp/vibeqc-162-options-memcheck.log`.
- The complete C++ heap audit still bounds and releases all ten HF/KS cases:
  `/tmp/vibeqc-162-ks-options-cpu-heap-20260914.json`. New copied radius tables
  are included in the host metadata allowance.
- Pre-commit and ten ownership tests passed. The final public-symbol export
  annotation passed its CPU API rebuild/test and matching CUDA rebuild:
  4/4 CUDA native tests and all four CUDA custom-grid option cases passed.

## CUDA occupation and larger open-shell state increment (2026-09-14)

CUDA UKS now applies the CPU's stationary occupation policy: after energy and
the per-spin physical residual pass, a cycling density enables a 0.1-Eh
virtual-projector shift for subsequent orbital proposals. The unshifted
physical Fock, energy and commutator remain tied to the current density, and
a subsequent density-change gate is still required. Each replay resets the
control state. Existing matrix scratch is reused; the policy adds no device
buffer, matrix transfer or synchronization. A cumulative proposal counter
lets the native regression verify that this policy actually executed.

LDA/PBE OH now has independent matched-grid CPU/CUDA endpoints with two PySCF
guesses, using the explicit C2v subgroup and the full unrestricted AO residual
gate. Both STO-3G and >16-AO def2-SVP are covered. Ragged OH/H def2-SVP batches
exercise resident replay, changed geometry, frozen seeds, failure isolation
and restoration.

That failure regression exposed coincident O/H centers: their AO metric can
remain nonsingular while the nuclear energy becomes infinite. CUDA KS now
rejects nonfinite prepared one-electron/nuclear data and checks the complete
physical energy before convergence or caching. The failed item reports a
numerical failure without publishing a finite residual alongside an infinite
energy; its last-good seed and neighboring items remain usable.

Validation: 30 CPU Python SCF/batch cases passed, plus both stricter numerical
failure-status regressions; 2/2 CUDA native tests and 38 CUDA independent SCF,
batch, model-option and resource cases passed. Ten ownership tests and
pre-commit passed. GPU runs used finite Slurm jobs on the RTX 5090. The full
native CUDA KS executable and both larger-open-shell batch cases passed
Compute Sanitizer memcheck with full leak checking: 0 errors and 0 bytes
leaked for each process. The two Python batch cases took 279.21 seconds under
memcheck; the complete scheduled job stayed within its 15-minute allocation.
Log: `/tmp/vibeqc-162-stabilization-memcheck.log`.

Physical ownership comparison against merged `15d6936`: scientific CUDA
+154/-0 lines, runtime CUDA +194/-4 lines; no reclassification or scientific
retirement. Current ownership is reproducibly generated from the source tree and
`docs/cuda_ownership.json`; CI retains the generated report as an artifact.

## Public physical KS diagnostic increment

The C/Python result path now retains the native physical KS snapshot:
occupations, AO-metric electron counts, actual grid/tile/AO order, final energy
components, per-spin maximum convergence measures, Fock-build count and every
iteration of the reported solve. UKS history records whether each orbital
proposal used occupation stabilization. CPU RKS final validation remains
separate from its original historical iterations. The first energy difference
is absent in Python JSON because there is no preceding energy.

The additive C summary/history query validates every output before writing,
preserves legacy result-array strides, and cannot expose an earlier item's
record after a failed or invalid replay. Python snapshots contain immutable
dataclasses/tuples and survive later execution unchanged. The adapter moves
the exported native history to the C handle rather than copying it at each
layer. See [the public contract](../../../docs/ks_diagnostics.md).

CPU validation: 25/25 native tests; seven initial diagnostic tests plus
the explicit cold-retry case; 93 selected KS/HF resource, calculator, model,
batch and independent-SCF regressions (three optional skips); and eight model
option cases with actual diagnostic grid/tile checks. All ten complete native
C++ heap cases fit their plans and release tracked allocations:
`/tmp/vibeqc-162-diagnostics-cpu-heap-20260914.json`.

CUDA validation passed 4/4 native tests and all 46 selected KS/HF diagnostic,
model, resource, batch and independent-SCF cases. Full native KS and all seven
CUDA diagnostic cases passed Compute Sanitizer memcheck: zero errors and zero
bytes leaked in each process. Every GPU run used a finite Slurm RTX 5090 job.

The independent H3+ RKS diagnostic cases exposed a CUDA DIIS failure: after
history errors became nearly dependent, the kernel returned the current Fock
instead of retrying with recent independent errors. LDA/PBE energy was stable
while residuals remained about 1e-8/1e-7 after 150 steps. Normalized KS DIIS now
retires the oldest ring entry and retries, matching CPU policy without moving
matrices or adding storage. The unnormalized HF slot order/fallback is retained.
These same independent H3+ component/physical-residual cases now pass. Warm
fallback also releases its discarded exported history before a second solve,
preserving the existing two-history resource bound.

This increment does not yet supply complete timing/transport records across
preparation, discarded warm attempts, geometry rebuild and finalization.
Those remain part of the original acceptance work below.

## Remaining work against the full issue

1. Preserve the implemented method-level #203 resource contract as richer
   model options/diagnostics are added; bind new capacities and identities.
2. Retain the larger-solver closed/open-shell and replay/failure coverage in
   permanent source-bound endpoint evidence: water and OH def2-SVP now pass
   independent CPU/CUDA checks, and OH/H ragged batches pass state checks.
3. Include the now-tested native CPU/CUDA ragged batch paths in the full
   workload/evidence runner. Active/converged/failed isolation, stable order,
   force rejection, warm/frozen/imported replay and changed-geometry rebuilds
   pass; failed items preserve valid warm states.
4. Preserve the implemented immutable functional/grid/model options and
   numerical identity as diagnostics and evidence are completed. Geometry,
   basis, spin/charge, grid and functional/regularization changes must retain
   the validated invalidation and current-density generation gates.
5. Complete measured component costs and transport diagnostics, including
   common provider preparation, discarded attempts and final outputs. Public
   physical residual/density change, occupations, grid/model and numerical
   history/component snapshots are implemented; the enclosing result reports
   the actual backend.
6. Publish permanent source-bound resource evidence alongside the final #138
   workload records. Public KS budgets now include prepare/execute observation,
   all J/grid/XC/state/DIIS owners and host setup bounds. The standalone CPU
   heap audit includes recurrence/eigensolver temporaries; CUDA ledger tests
   verify exact persistent capacities, prepare failure, rebuild and release.
7. Validate CPU/CUDA fixed-density E/V and full RKS/UKS endpoints, failure
   isolation, numerical state, replay, changed geometry and distinct guesses.
   Retain all HF gates. Run every real-GPU test/sanitizer/benchmark through
   finite-time Slurm jobs; do not override assigned CUDA_VISIBLE_DEVICES.
8. Extend #138 evidence runner for cold setup, changed/fixed geometry,
   energy-only, batch 1 and ragged batches. Record grid size, actual backend,
   iteration history, component costs, H2D/D2H bytes and synchronization,
   including host quadrature and final outputs in complete timings.
   The shared entry point is `benchmarks/validation_gate.py`, with protocol
   helpers in `tools/vibeqc_validation/{schema,performance,fixtures}.py`.
9. Keep staged results in PR #306 with exact required AI attribution, complete
   review/CI and the requirement-by-requirement audit before claiming completion.

Workspace: `/home/jzzeng/codes/vibeqc-issue-162`, branch
`codex/issue-162-completion`, starting from upstream `e32c6d4`. The user's
original worktree `/home/jzzeng/codes/qc-mixed-precision` is untouched.
