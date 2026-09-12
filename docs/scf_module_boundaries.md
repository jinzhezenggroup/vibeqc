# HF module decomposition (#240)

The independent CPU reference arithmetic and initial-state preparation now
have explicit interfaces in `src/scf/reference/` and `src/scf/initial_guess/`.
They retain the original loop order, occupation factors, Jacobi thresholds,
overlap rejection threshold, and UHF frontier-rotation policy. This extraction
does not complete #240: the CUDA driver, provider planning, shared iteration
control, and gradient assembly still require the remaining moves below.

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
| `rhf.cpp`: iteration control and safeguards | `Diis`, generations, proposal validation, `safeguarded_update`, finalization, `run_rhf_host_plan`, `run_uhf_host_plan` | Remaining solver extraction. Preserve the existing #186 proposal contract and independent physical residual checks. |
| `rhf.cpp`: provider semantics and accounting | `CpuScfIntegralDataView`, integral capacity sampling, `build_fock`, `build_uhf_focks`, prepared strategy entry points, CUDA DF plan preparation | Continue to consume #202's `PreparedFockPlan` / provider interfaces; resource observations belong to the owner of each allocation. |
| `rhf.cpp`: stationary forces | `analytic_forces`, `analytic_uhf_forces`, DF/CUDA gradient adapters and final force assembly | Remaining reusable gradient boundary. Keep explicit provider, overlap/Pulay, one-electron and nuclear terms separate. |
| `rhf.cpp`: compatibility entry points | Public RHF/UHF CPU/CUDA wrappers and CPU-build CUDA stubs | Keep existing method/ABI signatures, failure behavior and per-item ordering. |
| `cuda_rhf.cu`: scientific reference/fallback/specializations | `Dual`, `Dual3`, `MixedPrecisionFloat`, angular/Hermite/Coulomb workspaces, primitive/contracted one-/two-electron values and gradients, order-specific Fock/force kernels | Move by the existing #231 scientific ownership regions; do not copy formulas or turn the independent oracle into generated production arithmetic. |
| `cuda_rhf.cu`: queues, work descriptors and compaction | `ActiveShellQuartetTile`, `DirectTileValidationRecord`, `PsssResidentTask`, pair bounds, descriptor validation, bounded page ranges and queue kernels | Remaining bounded-task runtime interface, with validated offsets and shared buffer lifetimes. |
| `cuda_rhf.cu`: generic SCF kernels and libraries | Density/Fock/update/convergence kernels, inactive-eigensolver profiles and launch selection | Remaining matrix-kernel and eigensolver modules; scheduling choices must not select a different physical operator. |
| `cuda_rhf.cu`: host planning and replay | `DeviceBatch`, `HostBatch`, `ArenaLayout`, `CudaResources`, `CudaRhfBucketPlan`, integral source implementation, graph construction and bucket dispatch | Separate common views from ownership, then move ordinary host policy/planning to C++ wrappers. Preserve #111's existing compilation-unit selection. |
| `cuda_density_fitting.cu`: metric and storage planning | `SetupBuffers`, `CudaDensityFittingJkPlan`, checked sizes, cuSOLVER setup, plan creation/release and diagnostics | Remaining DF plan/metric module; preserve resident versus source-backed/host-backed distinctions. |
| `cuda_density_fitting.cu`: J/K execution | `build_coulomb`, `build_exchange`, tile gather/transpose/reduction kernels, RHF/UHF host/device/item entry points | Remaining DF execution module using the same provider semantics and explicit memory sub-budget. |
| `cuda_density_fitting.cu`: force integration | `execute_cuda_density_fitting_generated_force_response` | Continue #143/#205 integration through `cuda/df_gradient_bridge.*`; its current host response staging must remain visible until #205 replaces it. |
| `cuda_density_fitting.cu`: iterative replay | `DeviceSolver`, `DeviceIterationGraph`, `PersistentScfState`, eigensolve wrappers, RHF/UHF device SCF loops and graph-tail kernels | Remaining common solver/graph boundary; lifetime ownership and per-system failure isolation precede structural moves. |

## Dependency and size gates

`tools/check_scf_structure.py` runs in pre-commit and exposes `--json` for the
current shared-module include graph and sizes. Reference numerics may depend
only on reference numerics and standard headers. Initial-guess preparation may
add core/integral data interfaces. Neither layer may include the method driver,
public ABI implementation, CUDA providers, or later DFT/CC methods. Tests cover
relative and angle-bracket include spellings and the reverse dependency edge.

New shared CPU modules should remain below 600 physical lines per source file;
a larger module requires a documented responsibility and build-cost argument.
This is a review target, not a claim that legacy source units already pass a
structural-size gate. At the baseline, `rhf.cpp` contains 3,413 lines / 165,000
bytes, `cuda_rhf.cu` 19,754 lines / 1,048,372 bytes, and
`cuda_density_fitting.cu` 3,098 lines / 160,928 bytes. The latter two still exceed
the target and remain explicit unfinished work under #240. Their existing
kernel/template and runtime coupling explains the staged extraction, not an
exemption from the issue's final acceptance criteria.

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
