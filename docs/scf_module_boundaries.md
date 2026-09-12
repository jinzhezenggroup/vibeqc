# HF module decomposition (#240)

The independent CPU reference arithmetic, initial-state preparation, iteration
control and stationary force assembly now have explicit interfaces in
`src/scf/reference/`, `src/scf/initial_guess/`, `src/scf/solver/` and
`src/scf/gradient/`.
They retain the original loop order, occupation factors, Jacobi thresholds,
overlap rejection threshold, and UHF frontier-rotation policy. Issue #240
remains open for host graph/bucket control and the final production
build/runtime gates below. The retained scientific kernels now have bounded
CUDA owners; the remaining direct host driver compiles as ordinary C++.

## Responsibility inventory

The source baseline is `9f5b625`. The table describes ownership groups, not
mathematical equivalence or a new replacement implementation. The scientific
CUDA classifications and exact source anchors remain in `cuda_ownership.json`
and `cuda_ownership_current.json`, under #231.

| Source / group | Functions and state | Destination / current boundary |
| --- | --- | --- |
| `rhf.cpp`: reference linear algebra | `Matrix`, `EigenResult`, `index`, `identity`, `multiply`, `transpose`, `symmetric_eigen`, `symmetric_orthogonalizer`, `generalized_eigen`, `dot`, `solve_linear` | Extracted to `reference/linalg.*`; no HF, CUDA, or external numerical-library dependency. |
| `rhf.cpp`: mean-field reference contractions | `density_from_orbitals`, `energy_weighted_density`, restricted/unrestricted electronic energies, joined/split spin matrices, `commutator_residual`, density/residual RMS | Extracted to `reference/mean_field.*`; full dense AO conventions and spin factors remain explicit. |
| `rhf.cpp`: core/warm initial state | `spin_occupations`, `mix_open_shell_frontier_orbitals`, `normalize_spin_density`, `prepare_initial_density`, `prepare_initial_uhf_density` | Extracted to `initial_guess/density.*`; callers still own physical state/topology validation. |
| `rhf.cpp`: iteration control and safeguards | `Diis`, generations, proposal validation, `safeguarded_update`, finalization, `run_rhf_host_plan`, `run_uhf_host_plan` | Extracted to `solver/diis.*`, `solver/proposal_control.*` and `solver/mean_field_driver.*`. The #186 proposal contract, independent physical residual checks and convergence/finalization order are unchanged. Legacy CUDA/DF recovery orchestration still uses the shared DIIS owner. |
| `rhf.cpp`: provider semantics and accounting | `CpuScfIntegralDataView`, integral capacity sampling, `build_fock`, `build_uhf_focks`, prepared strategy entry points, CUDA DF plan preparation | The common CPU driver consumes #202's `PreparedFockPlan` with no direct/DF storage branches. `PreparedFockPlan::cpu_observation_capacity()` now owns the prior capacity accounting, counting shared AO data once. Legacy CUDA/DF provider orchestration remains in the compatibility driver. |
| `rhf.cpp`: stationary forces | `analytic_forces`, `analytic_uhf_forces`, DF/CUDA gradient adapters and final force assembly | Provider-based stationary assembly is extracted to `gradient/hf_gradient.*`, taking the provider's explicit positive derivative rather than owning a plan. Legacy CUDA/DF gradient adapters and finalization remain; their provider, overlap/Pulay, one-electron and nuclear terms must remain distinct. |
| `rhf.cpp`: compatibility entry points | Public RHF/UHF CPU/CUDA wrappers and CPU-build CUDA stubs | Keep existing method/ABI signatures, failure behavior and per-item ordering. |
| `cuda_rhf.cu`: scientific reference/fallback/specializations | `Dual`, `Dual3`, `MixedPrecisionFloat`, angular/Hermite/Coulomb workspaces, primitive/contracted one-/two-electron values and gradients, order-specific Fock/force kernels | Retained one-electron and direct numerical families now have bounded shared headers and CUDA launch owners. The #231 classifications and arithmetic are preserved; no scientific formulas are retired by the move. |
| `cuda_rhf.cu`: queues, work descriptors and compaction | `ActiveShellQuartetTile`, `DirectTileValidationRecord`, `PsssResidentTask`, pair bounds, descriptor validation, bounded page ranges and queue kernels | Host partitioning and page ranges use `cuda/queue_plan.*`. Device validation, density bounds, compaction, generated/resident tasks, bounded pages, scans and diagnostics now have separate `cuda/direct_*` owners. Fused native consumers use `direct_bounded_dddd.cu`, `direct_bounded_exact_force.cu` and `direct_bounded_fallback.cu`; host launch sequencing remains in the C++ driver. |
| `cuda_rhf.cu`: generic SCF kernels and libraries | Density/Fock/update/convergence kernels, inactive-eigensolver profiles and launch selection | Generic eigensolver execution lives in `cuda/eigensolver.cpp` / `eigensolver_kernels.cu`. Shared matrix, density, DIIS, convergence and state kernels now have separate `cuda/scf_*_kernels.*` owners; public/direct basis transforms use `basis_transform_kernels.*`. Scientific integral/Fock kernels now have separate CUDA owners; host bucket control remains in `cuda_rhf.cpp`. |
| `cuda_rhf.cu`: host planning and replay | `DeviceBatch`, `HostBatch`, `ArenaLayout`, `CudaResources`, `CudaRhfBucketPlan`, integral source implementation, graph construction and bucket dispatch | Packed/POD contracts and host topology/arena planning already have separate owners. DF source/export uses `cuda/df_source*` / `df_integral_export*`. Stream/graph/arena lifetime uses `cuda/resources.*`; matrix-library execution borrows `matrix_library.*`. Direct J/K host ownership uses `direct_jk.cpp` / `direct_jk_plan.hpp`; one-electron host exports use `one_electron_export*.cpp`. Graph construction and bucket dispatch now compile as `cuda_rhf.cpp`; their finer responsibility split remains pending. |
| Former `cuda_density_fitting.cu`: metric and storage planning | `SetupBuffers`, `CudaDensityFittingJkPlan`, checked sizes, cuSOLVER setup, plan creation/release and diagnostics | Extracted to `cuda/df_plan*`, `df_setup_internal.hpp`, `df_runtime.*` and `df_metric_kernels.*`. The public handle remains opaque; source transfer and retained metric factors keep their single owner. |
| Former `cuda_density_fitting.cu`: J/K execution | `build_coulomb`, `build_exchange`, tile gather/transpose/reduction kernels, RHF/UHF host/device/item entry points | Extracted to `cuda/df_coulomb.cpp`, `df_exchange.cpp`, `df_jk*`. Resident, host-backed and source-backed execution retain the same provider semantics and memory sub-budget. |
| Former `cuda_density_fitting.cu`: force integration | `execute_cuda_density_fitting_generated_force_response` | The adapter now lives in `cuda/df_force_response.cpp`. The #205 source-backed response borrows device factors through `df_response_weights.*` / `df_gradient_bridge.*`; the host-value compatibility adapter retains its CPU metric path. |
| Former `cuda_density_fitting.cu`: iterative replay | `DeviceSolver`, `DeviceIterationGraph`, `PersistentScfState`, eigensolve wrappers, RHF/UHF device SCF loops and graph-tail kernels | Extracted to `cuda/df_scf_state.hpp`, `df_scf_library.*`, `df_rhf_scf.cpp`, `df_uhf_scf.cpp` and `df_scf_kernels.*`. Host replay/graph control compiles in C++; only equations and the graph-tail kernel require CUDA compilation. |

## Dependency and size gates

`tools/check_scf_structure.py` runs in pre-commit and exposes `--json` for the
current shared-module include graph and sizes. Reference numerics may depend
only on reference numerics and standard headers. Initial-guess preparation may
add core/integral data interfaces. Neither layer may include the method driver,
public ABI implementation, CUDA providers, or later DFT/CC methods. The solver
may consume the shared provider/types/proposal interfaces and gradient assembly;
gradient assembly cannot acquire solver state. Tests cover
relative and angle-bracket include spellings and the reverse dependency edge.

New shared CPU modules should remain below 600 physical lines per source file;
a larger module requires a documented responsibility and build-cost argument.
This is a review target, not a claim that legacy source units already pass a
structural-size gate. At the baseline, `rhf.cpp` contains 3,413 lines / 165,000
bytes, `cuda_rhf.cu` 19,754 lines / 1,048,372 bytes, and
`cuda_density_fitting.cu` 3,098 lines / 160,928 bytes. The original DF unit is now
removed; its largest replacement is the 549-line setup transaction. The large
direct host unit (`cuda_rhf.cpp`, 4,865 lines after MP2 integration) still exceeds the target and
remains unfinished work under #240. Its graph/bucket coupling explains the staged extraction,
not an exemption from the issue's final acceptance criteria.

## Direct-native integration with MP2

The merge with upstream `b69ec53` preserves every physical-reference export,
capacity, provider-accounting and cleanup change from the original CUDA driver.
Its added and deleted host tokens match the upstream patch, and the layered
audit still verifies the preceding numerical and consumer extractions. The
shared `tensor/cuda_error.hpp` keeps `DeviceAllocationError` and `cuda_check`
unchanged while allowing the reference-export bridge to compile as ordinary
C++ without importing device kernels from `tensor/cuda_runtime.cuh`.

At integration source `c5301f3`, development and optimized Release builds each
pass 22 native suites and 214 Python cases without skips. These cover HF/DF
energies and forces, warm/failure semantics, mixed precision, final-Fock reuse,
physical-reference export, public CPU/CUDA MP2, identical-orbital components and
permutations, allocation status/rollback, CUDA tile replay, and an independent
14-AO reference with a partial final virtual block. Each library also passes
weighted-integral validation with 3,349 records, 141 tiles and four runs.
Slurm allocations 9302 and 9303 preserve scheduler-assigned device visibility.

The current source selection passes 545 checks with 50 optional skips; its six
deselected fixed-native fixtures pass against both rebuilt libraries. Five
actual CMake graph checks verify architecture intent and the ten-owner device
link with standalone angular force, including global RDC mode. All hooks pass.
`benchmarks/results/scf-direct-native-rtx5090/integration.json` retains this
integration evidence separately from the historical pre-MP2 cold-build,
incremental-build, binary-size and matched-runtime measurements below. The
integration reused existing build trees and supplies no new cold-build claim.

## Validation and remaining acceptance

The first extraction is checked with the existing RHF/UHF, direct/DF, spherical,
warm-batch, proposal, precision, and public native CPU tests. CUDA builds and
allocated-device endpoint tests are also required because `rhf.cpp` supplies
host control and finalization for native CUDA execution. Function movement must
preserve arithmetic order; new scientific behavior belongs in a separate fix.

A local compiler-cost sample used GCC 11.4.0, `-O3 -DNDEBUG`,
`VIBEQC_HAS_CUDA=0`, and `CCACHE_DISABLE=1`. Each translation unit was compiled
once with a warm filesystem cache, using its existing Ninja command and a
separate object/dependency-file destination. These are compiler-work samples,
not whole-build or molecular runtime measurements:

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| Baseline `rhf.cpp` | 2.781 | 141,960 |
| Extracted driver `rhf.cpp` | 2.223 | 116,520 |
| `reference/linalg.cpp` | 0.805 | 19,136 |
| `reference/mean_field.cpp` | 0.504 | 11,432 |
| `initial_guess/density.cpp` | 0.667 | 14,416 |

Separate implementation edits now compile their own small object. The combined
serial compiler work increases from 2.781 to 4.199 seconds in this sample, and
the summed object size increases from 141,960 to 161,504 bytes. This extraction
therefore establishes narrower rebuild ownership, not a cold-build speedup.
The representative CUDA rebuild after editing these CPU implementations
compiled only the affected C++ objects and build-identity object, then linked;
it did not compile a CUDA kernel object. A before/after full CUDA build study
remains part of the later decomposition acceptance.

Final #240 acceptance still needs the remaining inventory groups moved behind
stable interfaces, a complete before/after largest-file inventory, cold and
representative incremental build measurements with touched-object sets, and
unchanged direct/DF energy/force and batch/failure gates. A successful CPU
extraction alone is insufficient to close the issue.

## CPU solver and gradient extraction

The common host loops now compile once against the prepared-provider interface
instead of being private templates embedded in `rhf.cpp`. Their reference
arithmetic remains independent of generated backends. A comparison against
`64b1551` verifies unchanged arithmetic in 16 moved functions; the explicitly
recorded interface changes move derivative acquisition to the caller and
numerical-capacity observation to the provider owner. The DIIS augmented solve,
singular fallback, proposal generation, damping schedule, three-failure limit,
per-item history, physical convergence comparator and final unextrapolated Fock
rebuilds are preserved.

The source header `types.hpp` now directly includes the public enum definitions
it uses; its previous reliance on include order surfaced when proposal control
became an independent translation unit. This changes no struct layout or ABI.
The largest new implementation is `solver/mean_field_driver.cpp`, below 300
lines. The compatibility driver is approximately 2,530 lines versus 3,413 at the
initial inventory. This is a CPU ownership improvement; it does not satisfy the
remaining large-CUDA-file acceptance criteria by itself.

A second GCC 11.4 Release sample disabled ccache for each affected translation
unit. These are individual compiler-work samples with a warm filesystem cache,
not a whole-build measurement:

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| Before: `rhf.cpp` | 2.233 | 116,520 |
| Before: `fock_prepared.cpp` | 1.165 | 48,432 |
| After: `rhf.cpp` | 1.110 | 38,680 |
| After: `fock_prepared.cpp` | 1.270 | 49,488 |
| `solver/diis.cpp` | 0.580 | 10,632 |
| `solver/proposal_control.cpp` | 0.880 | 20,136 |
| `solver/mean_field_driver.cpp` | 1.506 | 59,128 |
| `gradient/hf_gradient.cpp` | 0.329 | 3,200 |

Summed compiler work for these owners rises from 3.398 to 5.676 seconds;
summed object bytes rise from 164,952 to 181,264. A captured implementation-only
mean-field-driver edit rebuilds that C++ object and `c_api_tuning.cpp` (source
identity), then relinks the library and dependent probes/tests. It recompiles
no CUDA kernel object. Full cold-build and device-link measurements remain
part of the later CUDA decomposition acceptance.

## CUDA topology, planning and eigensolver extraction

The next runtime move removes approximately 2,100 lines from `cuda_rhf.cu`.
Host topology packing, checked arena sizing, bounded queue partitioning and
queue diagnostics compile in ordinary C++ owners. Generic native/library
eigensolver dispatch borrows the existing cuSOLVER handles and workspaces;
only six narrow launch wrappers and the unchanged native Jacobi/instrumentation
kernels require CUDA compilation. The largest new implementation is 458 lines.
Shared headers contain POD layouts or narrow contracts; there is no umbrella
header containing the remaining scientific recurrence implementations.

A source audit against `bd657ec` verifies 26 function bodies, including the
remaining bucket executor, after two explicit interface changes: borrowed
library resources and host-callable launch wrappers. Each wrapper preserves
launch geometry, stream, shared bytes and argument ordering. Provider input
sanitization, inactive-state profiling, Jacobi rotations, stable eigenpair
sorting and last-error checks retain their existing order. Dependency gates
reject method-driver imports in these runtime owners and direct queue-policy
imports in the generic eigensolver owner.

Validation completed on the extracted implementation: 16 native CPU tests,
3 native CUDA runtime/provider/composition tests, and 106 Python GPU checks
with no skips. GPU execution used Slurm on the RTX 5090. The Python checks
cover RHF/UHF direct/DF Fock composition, checkpoint replay, public batch
behavior and the larger def2-TZVP water eigensolver endpoint.

Compiler samples below use GCC 11.4 Release and NVCC 12.9 sm_120 with the
explicit development fast-compile mode enabled, disabling ccache for each
invocation. These are single compiler-work samples with a warm filesystem
cache on a shared machine; production cold-build, device-link and performance
acceptance remain outstanding.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 43.269 | 9,891,176 |
| extracted_cuda_rhf | 41.937 | 9,815,088 |
| arena | 0.674 | 15,832 |
| topology | 1.120 | 26,440 |
| queue_plan | 0.683 | 10,568 |
| queue_profile | 0.681 | 10,864 |
| eigensolver | 0.781 | 5,520 |
| eigensolver_kernels | 2.826 | 267,552 |

Summed compiler work rises from 43.269 to 48.702
seconds; summed object bytes rise from 9,891,176 to
10,151,864. Actual implementation-only edits to
`topology.cpp` and `eigensolver.cpp` each rebuild their own C++ object plus
`c_api_tuning.cpp` source identity, then relink dependents. Neither edit
recompiles a CUDA kernel. The exact source was restored and rebuilt after
each probe.

`cuda_rhf.cu` remains 17,658 lines. Scientific direct Fock/force kernels,
device queue execution, graph/bucket control, source-backed DF integration,
and `cuda_density_fitting.cu` decomposition are still required for full #240
acceptance. The lower CUDA ownership line count from moving host code into
C++ is a structural move, not retirement of scientific arithmetic.

## DF source and tensor-export ownership

The source boundary now separates `df_source_setup.cpp` (validation, transforms,
metadata and current metric setup), `df_source.cpp` (bounded replay and public
source diagnostics), and explicit Cartesian tensor export in
`df_integral_export.cpp` / `df_integral_export_batch.cpp`. Their three kernel
launch wrappers and generated-policy basis/public-layout contractions live in
`df_source_kernels.cu`. Shared allocation registration is in
`metadata_upload.hpp`, because both direct J/K and DF already used that helper.
This avoids making a direct provider depend on private DF source state.

The mathematical DF policy, metric/raw/public layouts, primitive contraction,
coordinate mapping, partial tiles, export budget calculation, allocation
failure cleanup, and diagnostic strings remain unchanged. An audit against
`4b2083a` verifies 29 function bodies and all value/response specialization
arguments in three launch wrappers. The device record layout and public opaque
handle remain unchanged; only the owning translation units move. No generated
or handwritten scientific formula is copied into a second maintained file.

`cuda_rhf.cu` decreases from 17,658 to 16,018 lines for this move. The largest
new implementation is 437 lines. The CUDA scientific ownership ledger counts
existing bounded layout contractions in their new owner and distinguishes
host launch wrappers; reduced counted CUDA host lines do not constitute
scientific-code retirement.

The following development compiler-work sample uses ccache-disabled NVCC 12.9
sm_120 fast-compile mode and GCC 11.4 Release. The baseline is the exact
`4b2083a` source copied to a separate evidence path and compiled against
unchanged parent headers, so absolute compiler source paths differ. These
single-invocation samples are not production cold-build or runtime evidence.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 41.761 | 9,896,208 |
| extracted_cuda_rhf | 37.381 | 8,652,736 |
| df_source_setup | 1.672 | 71,448 |
| df_source | 0.943 | 29,584 |
| df_integral_export | 1.423 | 52,328 |
| df_integral_export_batch | 1.526 | 59,216 |
| df_source_kernels | 4.376 | 2,225,240 |

Aggregate compiler work rises from 41.761 to 47.320 seconds;
object bytes rise from 9,896,208 to 11,090,552.
Scientific direct kernels, GPU queue execution, graph/bucket control, and
`cuda_density_fitting.cu` planning/SCF ownership remain under the full #240
acceptance criteria.

This extraction passed 36 CPU structure/ownership/source checks, three native
GPU DF/provider/composition tests, and 34 Python GPU endpoint/resource checks
without skips. The allocated RTX 5090 runs cover values, complete forces,
Cartesian/spherical RHF/UHF, geometry replay, rank crossings, failed neighbors,
and positive device budgets. Actual implementation-only edits to source setup
and batched tensor export each rebuilt only their C++ owner and source identity,
then relinked; neither edit recompiled a CUDA kernel. The exact source was
restored and rebuilt after each probe.

After integration with the device force-response implementation and current
upstream, the combined source extraction passed 75 CPU structure/ownership
checks, three native GPU tests, and 38 opt-in Python GPU endpoint/resource
tests with no skips. The 29-function and three-wrapper audit still passes.

## DF plan, J/K and persistent solver ownership

The former 3,076-line `cuda_density_fitting.cu` is removed. Its existing
implementations now compile under these separate responsibilities:

| Owner | Responsibility |
| --- | --- |
| `df_plan.cpp`, `df_plan_setup.cpp`, `df_plan_lifetime.cpp` | Public opaque handle, setup transaction and teardown. |
| `df_runtime.*`, `df_setup_internal.hpp`, `df_plan_internal.hpp` | Checked sizes/status mapping, temporary setup allocations and sole plan storage layout. |
| `df_coulomb.cpp`, `df_exchange.cpp`, `df_jk.cpp` | Bounded mathematical J/K and public host/item/device adapters. |
| `df_force_response.cpp` | Borrow retained device factors for source response, or the existing explicit host-value compatibility path. |
| `df_scf_state.hpp`, `df_scf_library.*` | Persistent allocations, graph handles and cuBLAS/cuSOLVER workspace integration. |
| `df_rhf_scf.cpp`, `df_uhf_scf.cpp` | Host replay, graph capture/fallback, convergence and result publication. |
| `df_metric_kernels.*`, `df_jk_kernels.*`, `df_scf_kernels.*` | Unchanged device equations/reductions and 18 exact launch wrappers. |

The largest new implementation is the 549-line setup transaction. It remains
one transaction to preserve validation, source transfer, allocation failure
cleanup, metric factorization, diagnostics and publication order. No kernel
owner includes host plan state. The new dependency checks reject that edge as
well as imports of the RHF method driver into shared DF runtime owners.

A source audit against `bd9f3de` checks 54 function bodies, five storage layouts
and all 18 launch wrappers. Arithmetic order, streams, launch dimensions,
shared bytes, reduction order, public signatures and the persistent-state
lifetime are unchanged. Default arguments live only in the shared declaration.
The host cuBLAS J/K composition remains native scientific code; moving it from
CUDA to C++ does not retire its mathematics from #231's ownership model.

The compiler-work sample below uses ccache-disabled NVCC 12.9 sm_120 development
fast-compile mode and GCC 11.4 Release, with a warm filesystem cache on a shared
machine. The baseline is the exact parent source in its separate DF-source
worktree; absolute source paths differ. These are individual compiler samples,
not production cold-build, device-link or molecular runtime measurements.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_density_fitting | 6.381 | 456,488 |
| df_coulomb | 0.833 | 10,760 |
| df_exchange | 0.863 | 14,008 |
| df_force_response | 0.885 | 14,928 |
| df_jk | 1.017 | 30,848 |
| df_jk_kernels | 2.411 | 86,984 |
| df_metric_kernels | 2.337 | 35,024 |
| df_plan | 0.744 | 6,224 |
| df_plan_lifetime | 1.096 | 19,608 |
| df_plan_setup | 1.426 | 54,080 |
| df_rhf_scf | 1.314 | 39,888 |
| df_runtime | 1.182 | 32,248 |
| df_scf_kernels | 2.475 | 164,376 |
| df_scf_library | 0.978 | 12,200 |
| df_uhf_scf | 1.361 | 43,368 |

Aggregate compiler work rises from 6.381 to 18.922
seconds; object bytes rise from 456,488 to 564,544.
This extraction narrows rebuild ownership; it does not establish a cold-build
speedup. The direct scientific/queue/graph groups in `cuda_rhf.cu` and the full
production build/runtime acceptance remain unfinished under #240. Earlier
sections record the state at each intermediate extraction checkpoint.

Validation of this DF runtime extraction passed 16 native CPU tests,
83 structure/ownership/source checks, three native GPU tests and 138 Python GPU
tests with no skips. GPU checks ran through Slurm on the RTX 5090 and cover the
complete derivative, resource, Fock-composition, checkpoint-replay and native
CUDA-runtime suites. Hooks and the function/layout/launch audit pass.

Actual implementation-only edits to `df_plan_setup.cpp`, `df_scf_library.cpp`
and `df_exchange.cpp` each rebuild only that C++ owner plus the source-identity
object, then relink dependents. They compile no CUDA kernel object. Each probe
restores and rebuilds the exact validated source before the next edit.

## Shared SCF device kernels

State initialization and solver-result routing, generic matrix operations,
density/warm-state preparation, DIIS, physical convergence, and public/direct
basis transforms now have six distinct kernel owners in `src/scf/cuda/`.
The 34 host-callable wrappers preserve launch geometry, shared bytes, streams,
arguments and the two retained-density template specializations. Shared launch
and convergence constants live in `scf_constants.hpp`; common kernel owners
cannot include direct queue policy or host plan state.

An audit against `cb12e0a` checks all 36 moved function bodies and the entire
remaining direct CUDA implementation after removing only the definitions and
replacing launch syntax. It also checks the 34 wrapper bodies. The warm-density
block reduction, metric trace normalization, open-shell frontier rotation,
DIIS reduction/solve, mixed-to-target refinement reset, final-Fock reuse and
warm-state restoration retain their exact arithmetic and per-item routing.
No scientific formula is duplicated into another maintained owner.

`cuda_rhf.cu` drops from 16,017 to 15,060 lines; the largest new implementation
is 370 lines. The ownership ledger retains scientific classification for
occupied/energy-weighted density and energy equations, while matrix algebra,
state, DIIS, convergence and launch routing remain runtime code. This still
leaves the direct scientific, device queue and host graph/bucket groups, plus
full production build/runtime acceptance, under the open #240 issue.

The following compiler-work sample uses ccache-disabled NVCC 12.9 sm_120
in development fast-compile mode, with a warm filesystem cache on a shared
machine. The exact parent and extracted source use separate worktrees, so
absolute paths differ. It is not production cold-build, device-link or molecular
runtime evidence.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 37.385 | 8,654,208 |
| extracted_cuda_rhf | 36.564 | 8,440,480 |
| scf_state_kernels | 1.008 | 44,184 |
| scf_matrix_kernels | 1.073 | 128,592 |
| scf_density_kernels | 1.104 | 172,088 |
| scf_diis_kernels | 1.004 | 92,808 |
| scf_convergence_kernels | 1.179 | 226,176 |
| basis_transform_kernels | 1.024 | 93,000 |

Aggregate compiler work changes from 37.385 to 42.957
seconds; aggregate object bytes change from 8,654,208 to
9,197,328. These figures do not establish a cold-build speedup.

The formatted extraction passes four native GPU tests and 123 Python GPU tests
with no skips, covering direct/DF provider composition, checkpoint replay,
RHF/UHF batches, precision provenance, final-Fock reuse, higher-angular-momentum
calculator endpoints and warm-state failure isolation. GPU execution used Slurm
on the RTX 5090. The combined code-generation/structure run passed 376 checks
with 48 opt-in compiler probes skipped; its one stale launch-syntax assertion
was corrected and both affected regression checks pass in the focused rerun.
Hooks and the complete remaining-source audit also pass.

Actual DIIS and convergence implementation edits each rebuild their own CUDA
object plus the source-identity object, then relink. Neither probe compiles
`cuda_rhf.cu` or an unrelated kernel owner. The exact source was restored and
rebuilt after both probes.

## CUDA resource lifetime and matrix-library execution

`resources.*` owns the prepared bucket's stream, graphs, library workspaces and
arena. Its destructor now compiles in ordinary C++, preserving owning-device
selection, graph destruction, library-handle release, stream-ordered frees,
stream drain and host-workspace release in their original order. The resource
field layout is unchanged. Eigensolver and matrix consumers borrow explicit
views; they do not acquire allocation or graph ownership.

`matrix_library.*` consumes only a stream and cuBLAS handle. It preserves the
resolved native/cuBLAS route, column-major strides, active masks and spin-aware
broadcasting. `runtime_support.*` keeps the existing status mappings and
nonempty host-upload behavior. Dependency gates reject imports of the bucket
resource owner into matrix-library execution and reject method-driver imports
into both owners.

The audit against `36b7209` checks eight moved bodies, the resource field and
special-member layout, the two-handle borrowed matrix view, and the entire
remaining direct CUDA body after only definition removal and explicit view
construction. This is an ownership extraction: no equation, fallback decision,
resource budget, public ABI or iteration order changes. Host graph construction
and bucket dispatch, device queues and direct scientific kernel decomposition
remain unfinished under #240.

The direct CUDA implementation decreases from 15,060 to 14,902
lines; the largest new C++ implementation has 78 lines.
The following compiler-work samples disable ccache and use NVCC 12.9 sm_120
development fast-compile mode and GCC 11.4 Release. The exact parent source and
candidate use separate worktrees with different absolute paths, on a shared
machine with a warm filesystem cache. These are not production cold-build,
device-link or molecular runtime measurements.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 36.442 | 8,440,448 |
| extracted_cuda_rhf | 36.396 | 8,428,544 |
| resources | 1.030 | 12,616 |
| matrix_library | 0.382 | 4,584 |
| runtime_support | 0.354 | 1,968 |

Aggregate compiler work changes from 36.442 to 38.163
seconds; aggregate object bytes change from 8,440,448 to
8,447,712. This establishes ownership boundaries, not a cold-build speedup.

Validation passed 88 structure/ownership checks, four native GPU tests,
123 Python GPU tests and 14 allocation-budget GPU tests, with no GPU skips.
GPU execution used Slurm on the RTX 5090. The body/layout audit and hooks pass.
The allocation checks cover prepared execution, failure cleanup and repeated
plan lifetime behavior with the extracted resource destructor.

Actual implementation-only edits to `resources.cpp` and `matrix_library.cpp`
each rebuild only the edited C++ object plus source-identity metadata, then
relink dependents. The observed rebuilds take 1.585 and 1.589 seconds,
respectively, and compile no CUDA kernel object. Both probes restore and rebuild
the exact validated source. Full production acceptance remains under #240.


## Direct device queues and screening

Nine CUDA owners now separate tile validation, density-bound reduction, tile
compaction, generated exact-class tasks, resident-bra tasks, bounded pages,
bounded generated tasks, scans/retries and diagnostic counters. Five shared
headers contain only indexing, task encoding, physical screening, page-tail
bounds and profiling helpers; the largest is 199 lines. Queue implementations
consume borrowed packed metadata and cannot import host bucket resources or
integral recurrence implementations.

The exact-parent audit against `fda0f69` preserves 53 function/type definitions,
26 launch wrappers, the unclassified-slot sentinel, and the entire remaining
CUDA implementation after explicit launch-interface substitutions. It checks
macro-spliced launch sites as well as ordinary calls. RHF/UHF and Fock/force
specializations retain their original bodies and launch geometry. The bounded
generated wrapper exposes only the materializing specializations already used
by the driver; per-class overflow and exact-page recovery remain unchanged.

The physical density/Schwarz gates, conservative page tails and mixed-precision
contribution cutoff remain classified as scientific code in the #231 ledger.
Queue bookkeeping and diagnostic counts remain runtime code. This move does
not retire a screening formula or duplicate an integral evaluator.

`cuda_rhf.cu` decreases from 14,902 to 13,172 lines. The largest new CUDA
implementation is 241 lines. Host bucket/graph construction, host generated
launch sequencing and the direct scientific/force kernels remain under #240;
fused native bounded consumers still enumerate and drain their own work.

The following ccache-disabled compiler samples use NVCC 12.9 sm_120 development
fast-compile mode, a warm filesystem cache and separate parent/candidate
worktrees with different absolute paths on a shared machine. They measure
individual compiler invocations, not production cold builds, device linking,
resource acceptance or molecular runtime.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 36.337 | 8,428,512 |
| extracted_cuda_rhf | 33.013 | 8,041,040 |
| direct_tile_validation | 2.683 | 151,032 |
| direct_density_bounds | 2.713 | 174,536 |
| direct_tile_compaction | 2.831 | 325,744 |
| direct_generated_tasks | 2.670 | 143,560 |
| direct_resident_tasks | 2.669 | 121,144 |
| direct_bounded_pages | 2.878 | 366,888 |
| direct_bounded_tasks | 2.872 | 390,904 |
| direct_queue_scan | 5.236 | 322,976 |
| direct_queue_diagnostics | 2.667 | 150,408 |

Aggregate compiler work changes from 36.337 to 60.233
seconds; aggregate object bytes change from 8,428,512 to
10,188,232. The reduced scope of an implementation edit
comes with additional aggregate compiler work in this development sample.

Validation passed four native GPU tests and 137 Python GPU tests with no skips,
including the 14 allocation-budget cases. Six exact-parent comparisons cover
fixed, resident and paged RHF/UHF execution with ragged batches of three,
STO-3G, descriptor validation, cold/warm state and explicit changed geometry.
All 96 energy/force comparisons pass their existing tolerances; the largest
absolute difference across the compared arrays is `3.55e-14`.
Both GPU gates ran through Slurm on the RTX 5090. These correctness comparisons
do not establish production runtime acceptance for the remaining #240 work.

The combined source/structure suite passed 350 checks with 48 opt-in compiler
probes skipped. Its one stale page-screening source-location assertion was
updated for the extracted owner and passes in a focused rerun. The additional
ownership/publication checks, refreshed source snapshot, hooks and complete
body/launch audit pass.

Implementation-only comment edits to `direct_queue_scan.cu` and
`direct_bounded_pages.cu` each trigger only their own CUDA object and the
source-identity C++ object, followed by relinks. The ordinary cache-enabled
Ninja rebuilds take 1.751 and 1.606 seconds. Neither rebuild invokes the
compiler for `cuda_rhf.cu` or an unrelated CUDA owner. Both probes restore and
rebuild the exact source; the dependency scope is distinct from the uncached
compiler-work samples above.


## Direct provider host APIs and one-electron export

The direct public-AO provider now owns its lifecycle, bounded metadata/scratch
allocation, input validation, host uploads/downloads and item/batch APIs in
`cuda/direct_jk.cpp`, with the opaque plan layout in `direct_jk_plan.hpp`.
Its destructor preserves device selection, stream drain, allocation release
and stream destruction in their original order. Download fences still protect
local output buffers during exceptions; failed uploads retain the outer guard's
stream fence before caller-owned pageable memory can expire.

The three direct consumer kernels use `direct_jk_kernels.hpp` launch contracts.
They remain a 151-line fragment of the retained contracted-ERI owner, with no
recurrence duplication. Host input/specification checks and provider semantics
remain native after moving out of the CUDA-only inventory; the SCF migration
ledger records that fact explicitly. No scientific retirement is claimed from
the change in file extension or the reduced CUDA-source count.

Single-system and homogeneous-batch one-electron exports compile separately in
`one_electron_export.cpp` and `one_electron_export_batch.cpp`. They preserve
Cartesian staging, open-shell acceptance, coordinate offsets, optional response
outputs and release/error routing. The generated value route and retained Dual
response/nuclear kernels are unchanged. `one_electron_view.*` constructs only
a borrowed normalized-metadata view; it does not own positions or allocations.
Host implementations cannot import recurrence fragments or SCF resource owners,
and consumer interfaces cannot acquire the direct provider plan.

An audit against `bd1c614` checks the entire direct J/K host implementation,
plan fields and destructor, three unchanged kernel bodies, five launch wrappers,
five one-electron bodies and the complete remaining CUDA implementation.
The direct CUDA source decreases from 13,172 to 12,837 lines, and the former
500-line direct J/K and 239-line one-electron include fragments no longer carry
host orchestration into that translation unit. The largest new C++ owner is
393 lines. Direct recurrence/force ownership, graph/bucket control and full
production acceptance remain open under #240.

These ccache-disabled compiler samples use NVCC 12.9 sm_120 development
fast-compile mode and GCC 11.4 Release, a warm filesystem cache, and separate
parent/candidate worktrees with different absolute paths on a shared machine.
They do not measure production cold builds, device linking or molecular runtime.

| Translation unit | Seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 33.066 | 8,041,040 |
| extracted_cuda_rhf | 31.383 | 7,910,992 |
| direct_jk | 1.524 | 80,640 |
| one_electron_view | 0.078 | 1,424 |
| one_electron_export | 1.582 | 51,248 |
| one_electron_export_batch | 1.713 | 57,128 |

Aggregate compiler work changes from 33.066 to 36.280
seconds; aggregate object bytes change from 8,041,040 to
8,101,432. These figures establish the compilation
tradeoff of the extraction; full #240 build acceptance remains separate.

Validation passed 396 source/structure/ownership/publication checks, with
48 optional compiler probes skipped. The formatted CUDA build, hooks and
complete body/layout/launch audit pass. Four native GPU tests and 138 Python
GPU tests passed with no skips through Slurm on the RTX 5090, covering direct
and DF providers, fixed-density response, complete HF derivatives, resource
budgets, failed-item isolation and checkpoint/geometry replay.

Implementation-only comment edits to `direct_jk.cpp` and
`one_electron_export_batch.cpp` each rebuild only that C++ owner and the
source-identity object before relinking. The ordinary cache-enabled Ninja
rebuilds take 1.610 and 1.597 seconds. Neither probe invokes
CUDA compilation. Each probe restores and rebuilds the exact validated source.


## Retained one-electron and numerical CUDA owners (#240 slice)

The native Dual response kernel and scalar/cooperative one-electron force
kernels now compile in `one_electron_reference.cu` and
`one_electron_force_reference.cu`. Their former included fragments are removed.
`nuclear_kernels.cu` owns the existing nuclear energy/response/force equations;
`direct_pair_cache.cu` owns primitive-pair Gaussian geometry preparation.
Host callers retain the same launch geometry, stream and dynamic shared memory
through narrow forwarding interfaces.

Five private headers separate retained overlap/kinetic primitives, attraction,
attraction gradients, normalized AO contraction, and density-weighted force
contraction. Shared scalar/AD, Gaussian geometry, Cartesian indexing, Boys,
Hermite and Coulomb recurrence headers contain only their respective numerical
building blocks. `integral_limits.hpp` preserves the existing constants without
queue policy. `one_electron_force_workspace.hpp` preserves the host/device layout
used to allocate three Hermite axis tables in dynamic shared memory. Ordinary
`inline` provides shared header linkage; existing forced/no-inline decisions and
all numerical expressions are retained.

The source audit compares 28 definitions, four complete numerical chunks, both
old kernel fragments, five new and two existing forwarding wrappers, constants,
workspace layout and the entire remaining CUDA implementation against parent
`58314e8`. It permits formatting, header linkage and explicit launch substitutions.
Dependency guards reject queue policy, host plans and resource ownership in the
new numerical/kernel owners, and reject operator contractions in shared numerical
headers. Existing source-location checks follow the moved definitions.

`cuda_rhf.cu` decreases from 12,837 to 11,384 lines. New CUDA implementations are
47–130 lines and shared/private numerical headers are at most 288 lines. The
ownership ledger retains each formula's scientific or measured-exception status;
this decomposition does not promote generated derivatives or retire native
formulas. The plain limits header is outside the CUDA-only source census.

Development compiler samples (NVCC 12.9, `sm_120`, fast-compile mode, ccache
explicitly disabled) use individual compiler invocations and warm filesystem
cache. The parent and candidate use separate worktrees with different absolute
source paths on the shared machine:

| Unit | Compiler seconds | Object bytes |
| --- | ---: | ---: |
| baseline_cuda_rhf | 31.465 | 7,910,992 |
| extracted_cuda_rhf | 29.849 | 7,227,600 |
| one_electron_reference | 2.611 | 1,309,840 |
| one_electron_force_reference | 3.037 | 1,761,008 |
| nuclear_kernels | 1.078 | 169,144 |
| direct_pair_cache | 0.995 | 83,136 |

Aggregate sampled compilation changes from 31.465 to
37.570 seconds. Separate owners add compiler setup and instantiated device
code, while narrowing implementation rebuilds. These are development samples,
not full cold-build, production resource, device-link or runtime acceptance.

The full issue still requires direct scientific/force ownership, host bucket and
graph decomposition, and production build/device-link/runtime acceptance.


Actual implementation-only comment edits to `one_electron_force_reference.cu`
and `direct_pair_cache.cu` each rebuild exactly their CUDA object and the source
identity C++ object, followed by relinks. The large direct CUDA owner and other
kernel objects stay unchanged. These ordinary cache-enabled Ninja probes take
1.618, 1.610 seconds, respectively; exact source bytes are restored and rebuilt
after each probe. They are separate from the uncached compiler samples above.

Validation uses the RTX 5090 through Slurm. Four native GPU suites pass. The
initial Python allocation reaches its finite 20-minute limit after reporting
177 passing cases. Its collected order confirms the complete 167-case prefix
(provider/resource/replay/runtime and derivative modules); the entire final
21-case value-schedule module then passes in a fresh allocation. This covers all
188 distinct cases, with ten value cases repeated during recovery and no skips.
Twelve exact-parent scalar/cooperative, fixed/resident/paged RHF/UHF scenarios
pass 192 energy/force comparisons, with maximum absolute difference
`2.9976021664879227e-14`. The formatted build, hooks and 420 source/structure/
ownership/publication checks pass; 48 optional compiler probes are skipped.

## Retained direct kernels and C++ host control

The remaining direct arithmetic now lives in 24 bounded `direct_native_*.cuh`
headers: Cartesian/Hermite/Coulomb recurrences, sparse pair families, psss/psps/
ppss/dsss gradients, order-specific gradients and contractions. Their largest
header is 438 lines. Eleven consumer helper headers separate contraction,
density and symmetry handling from nine CUDA launch owners: cached tensors,
Schwarz bounds, packed Fock, angular Fock, reference force, bounded dddd,
bounded exact force, bounded fallback and angular force. The largest new CUDA
owner is 386 lines. Existing J/K and weighted-ERI fragments now compile as
`direct_jk_kernels.cu` and `weighted_eri_kernels.cu`.

`cuda_rhf.cu` becomes the ordinary C++ file `cuda_rhf.cpp`, decreasing from
11,384 to 4,792 lines. It contains no device/kernel declarations or CUDA launch
syntax. Its graph/bucket responsibilities still exceed the structural target
and remain separate follow-up work. Numerical headers reject host plans and
queue policy; consumer owners reject host resource lifetime; the C++ driver
has an explicit launch-interface include allowlist. The ownership census
excludes this ordinary C++ driver. That classification change retires no
scientific formulas; all retained numerical owners remain in the #231 ledger.

An exact-parent, three-stage source audit checks 77 numerical definitions,
46 consumer definitions, 17 consumer launch adapters, both complete former
fragments, two weighted launch adapters and three existing J/K wrappers. It
also checks data layouts, launch bounds, forwarding and the complete remaining
host body. Recursive angular dispatch and mixed-precision instantiations are
preserved. The bounded dddd path retains six instantiations; the bounded
fallback retains only its four existing force instantiations. Allocation,
launch/error order, screening, spin factors and final-Fock reuse are unchanged.

NVIDIA builds share ten direct owners through the resolved
`vibeqc_direct_native` archive. `direct_angular_force.cu` compiles separately
with relocatable device code disabled: NVCC 12.9 needs whole-program
compilation to propagate the resident kernels' launch-bound register ceilings
into their retained callees. A dedicated object target preserves this rule
even when whole-library separable compilation is requested. Kernel launch
bounds and no-inline annotations are unchanged. The default-enabled
`VIBEQC_CUDA_DIRECT_DEVICE_LINK` option can disable the ten-owner archive for
comparison; CuMetal and other compilers keep standalone modules. Real,
virtual-only and combined architecture requests retain their requested flags.
Split-compile experiments now apply to the direct owners instead of the
removed monolithic CUDA source.

The source/ownership/publication suite passed 433 checks with 48 optional
compiler skips. Five additional tests inspect actual CMake/Ninja build graphs:
real, virtual and combined architecture requests, explicit standalone mode,
and the angular-force exclusion under whole-library separable compilation.
The three-stage source audit and repository hooks pass.

The standalone extraction passed four native GPU suites and 139 Python cases
(Slurm 9280). Another 37 response cases and the weighted primitive validator
passed in Slurm 9281; the latter covers 3,349 records, 141 output tiles and four
budget/route runs, with libcint values and finite differences. An experimental
device-linked build passed the same native/Python and weighted gates in Slurm
9286. Its preceding run used incomplete opt-ins (121 passes and 18 skips) and
is not used as the complete GPU gate. The final ten-owner CMake development build passes four native suites, all
139 Python cases and the weighted validator in Slurm 9292. Slurm 9293 also
passes 96 endpoint comparisons with PTX-only images for every moved direct
owner; unchanged owners in that probe retain development images.

Exact-parent prepared endpoints compare six fixed/resident/paged RHF/UHF
scenarios, each containing a ragged three-system batch, cold execution, changed
geometry and repeated warm execution. Standalone and experimental linked
builds each pass 144 energy/force comparisons against the parent; maximum
absolute differences are `3.3084646133829665e-14` and
`4.618527782440651e-14`. Two per-item mixed-precision cases pass on each of
those three libraries, asserting actual FP32 work and strict FP64 refinement.
These are development-build numerical gates, not production speed claims.

### Optimized compiler samples

NVCC 12.9 targets `compute_120` plus `sm_120` with `-O3`, fast-compile mode
disabled and the compiler cache disabled. The parent is the exact `a6311fe`
implementation (also present at `671baef`). At most two compiler invocations
run concurrently on this shared host with a warm filesystem cache; aggregate
figures sum invocation wall times. The host driver uses GCC 11.4 Release.

| Scope | Aggregate compiler seconds | Object bytes |
| --- | ---: | ---: |
| Parent monolithic CUDA owner | 1,087.030 | 40,964,624 |
| Eleven standalone direct owners | 1,170.006 | 63,982,360 |
| Ten relocatable direct owners | 501.965 | 28,110,992 |
| Standalone angular force (required with either mode) | 205.726 | 8,973,568 |
| Extracted ordinary C++ host driver | 5.343 | 245,456 |

The standalone angular-force row is already included in the eleven-owner row;
it is added to the ten-owner row for the shared-code arrangement. That bounded
device link takes 12.475 seconds and produces a 9,952,280-byte link object.
Standalone compilation adds aggregate work and binary storage. Bounded linking
reduces duplication while retaining per-owner source compilation. These
samples do not establish complete cold-build or runtime acceptance. Fresh
whole-library Release builds, incremental object probes and endpoint runtime
checks are recorded separately when completed. Full #240 also requires the
remaining host graph/bucket decomposition.

The optimized changed objects were also linked against the identical untouched
development components to isolate this extraction's compiler effects. These
hybrid libraries pass six prepared-endpoint scenarios and 144 comparisons per
variant against the parent (maximum absolute differences `3.241851231905457e-14`
for standalone owners and `3.3306690738754696e-14` for the ten-owner link).
Four additional cases per library pass: per-item RHF/UHF mixed precision,
independent d/f PySCF comparisons and final-Fock reuse. Slurm 9289 records this
check. The hybrid library sizes are 111,937,968 bytes for the parent,
135,211,096 bytes for standalone extraction and 118,255,072 bytes for the
bounded link. Complete Release acceptance remains separate from these
partially optimized library measurements.

### Complete Release validation

Fresh parent (`671baef`) and extracted (`59c3e9f`) builds use GCC 11.4,
NVCC 12.9.1, `-O3`, `compute_120` plus `sm_120`, fast-compile mode disabled,
compiler cache disabled, two CUDA jobs and four total Ninja jobs. Both build
trees start empty. The builds run concurrently on a shared host with a warm
filesystem cache. The parent finishes in 2,143.872 seconds; extraction finishes
in 1,509.839 seconds. These single samples describe this build configuration.
The extraction was uncommitted when configure began and was committed unchanged
during the build; the evidence records source commit `59c3e9f` separately from
the configure-time HEAD. The subsequent `<cstdlib>` portability fix (`e7d8414`)
rebuilds the host/identity objects before runtime checks.

Complete Release library sizes are 116,304,304 bytes for the parent and
122,613,128 bytes for extraction, an increase of 5.4%. Slurm 9296 passes four
native suites, 139 Python GPU cases and the weighted validator (3,349 records,
141 tiles, four budget/route runs). Slurm 9298 compares six fixed/resident/paged
RHF/UHF scenarios: each has a ragged three-system batch, cold execution,
changed geometry and five warm repeats per geometry. All 288 energy/force
comparisons pass; maximum absolute error is `5.3512749786932545e-14`. Four
additional cases pass per library: actual RHF/UHF mixed precision and strict
FP64 refinement, independent d/f PySCF references, and final-Fock reuse.
Both libraries also pass 37 response cases with the required explicit opt-in
in Slurm 9299; those cases were skipped in the preceding 9298 run.

| Endpoint | Parent warm median (ms) | Extracted warm median (ms) |
| --- | ---: | ---: |
| RHF fixed | 19.241 | 19.161 |
| RHF resident | 19.167 | 18.897 |
| RHF paged | 19.771 | 19.419 |
| UHF fixed | 18.073 | 18.594 |
| UHF resident | 18.191 | 18.150 |
| UHF paged | 18.175 | 18.036 |

Changed-geometry warm medians also differ by less than one millisecond. These
short endpoint samples show no material runtime overhead in the exercised
scenarios; they do not establish a general performance improvement. Input
geometries, all comparison records, individual timing samples, library hashes,
build commands and scope limits are retained in
[`validation.json`](../benchmarks/results/scf-direct-native-rtx5090/validation.json).

Actual implementation comment edits were rebuilt in the same Release tree with
compiler cache disabled. Each edit was then restored byte-for-byte and rebuilt:

| Edited implementation | Rebuild seconds | Compiled objects and device link |
| --- | ---: | --- |
| `cuda_rhf.cpp` | 6.975 | C++ driver and source identity; no CUDA compile or device link |
| `direct_schwarz_kernels.cu` | 90.759 | Its CUDA object and source identity; direct archive device link |
| `direct_angular_force.cu` | 206.855 | Its CUDA object and source identity; no device link |

All timings include the shared library and dependent executable relinks.
[`incremental.json`](../benchmarks/results/scf-direct-native-rtx5090/incremental.json)
retains the complete touched-object and relink lists. No unrelated CUDA source
is recompiled. The angular-force exclusion therefore preserves its standalone
compiler contract during both cold and incremental builds. Full #240 remains
open for the host graph/bucket decomposition and its final combined inventory.
