# Fock provider dispatch audit

Audit baseline: `3da58410bb02a903ea6341a7caac0afc9314355b` (after #239).
The implementation progress below supersedes the baseline audit. Mixed
provider SCF is executable; a complete DFT SCF method is not registered.

## Implementation progress after the audit

The CPU provider integration now executes all exact/DF J/K combinations,
independently absent terms and finite coefficients. `CpuFockProviderView` binds
the existing immutable integral owners; `CpuFockPlanView` validates both sources
before execution and shares one provider call when both terms use the same
source. It introduces neither another integral cache nor another solver.

CPU RHF/UHF direct and fitted entry points now use the same iteration and
finalization code. The generic `run_cpu_fock_strategy` endpoint also accepts
explicit mixed semantics. Fock, energy and first derivatives use one binding;
the DF metric cutoff and complete spectrally truncated response are retained.
The source owner's DF buffers remain included in CPU resource observations.

Single-item method and fleet dispatch now call `run_fock_strategy`. Standard
CUDA HF still uses its established fused solver. CUDA DF raw services now
accept a typed output selection for resident, streamed, tiled, batch, item and
device-pointer execution. Unselected device outputs may be null and are never
accessed; host outputs are empty. Plan scratch capacity remains accounted and
available for subsequent selections.

`CudaFockProviderView` now binds existing direct and DF plan items to the same
`BasicFockPlanView` composition used by the CPU. The direct provider reuses the
existing contracted-ERI value/Dual evaluator and retains metadata, AO matrices,
and compact gradient storage without molecular ERI tensors. Bounds and computed
outputs reject nonfinite results transactionally. Absent or zero-weight
responses do not evaluate unused quadratic terms.

General CUDA requests select `CudaIndependent`: CUDA one-electron generation,
direct/DF J/K and matching two-electron response feed the shared host SCF
iteration/finalization code. Standard HF keeps its fused CUDA schedules. The
DF derivative reuses the generated external-weight service and its full
spectrally truncated metric response, accepting signed coefficients. It
requires symmetric densities; all raw providers accept nonsymmetric inputs.

Validation includes 15 CPU native suites, through-f CUDA direct comparisons
and signed d/f responses, independent DF device layouts and source/result failure
cases. The composition suite checks every exact/DF/absent pair, both spins,
signed derivatives, resident/source-backed DF storage, distinct batch items,
and SCF/replay/changed-geometry endpoints. Exact run logs and build provenance
are retained in the untracked artifact directory.

`PreparedFockPlan` now owns the shared views and existing sources. Independent
CUDA single/fleet replays retain it per item, with exact immutable-input and
backend-variant compatibility checks and transactional replacement. CPU SCF
preserves its transient ERI lifetime. Reuse tests cover unchanged requests,
changed geometry/coefficient/auxiliary/cutoff/device identity and failed
replacement, alongside the original spin/pair numerical endpoints.

The additive C `fock.h` and Python `FockPlan` expose independent choices,
fixed-density values/responses, complete native SCF, explicit warm replay and
requested/resolved/actual-source diagnostics. Mathematical identity remains
separate from execution/source identity. `FixedDensityMeanField` connects
native unit J and absent K to the executable semilocal XC integrator; LDA and
PBE retain independent fixed-density fixture and energy-variation checks.
See [the public contract](fock_build.md) for ownership and output semantics.

The [production evidence](../benchmarks/results/fock-strategies/README.md)
compares 40 complete endpoints per backend against the audit baseline. Energies
and raw matrices are unchanged; maximum force differences are below 1e-14.
Complete endpoint median ratios are 1.0077 on CPU and 1.0028 on CUDA. The
retained raw/provider and alignment experiments explain and resolve a CPU DF
code-layout regression without changing its arithmetic.

The sections below preserve the baseline audit; they describe the coupling
before these implementation changes.

## Existing mathematical contract

`src/scf/fock_build.hpp` already defines `FockBuildSpec`, independent Coulomb
and exchange term specifications, spin conventions, operator/range parameters,
approximation identity and derivative order. `ResolvedFockBuild` keeps the
mathematical request separate from backend, schedule, screening and DF metric
threshold. Absent terms are canonicalized; malformed parameters and unsupported
range operators fail before execution.

Restricted density includes double occupation. Unrestricted J consumes the
total spin density and K consumes each matching-spin density. Raw J/K matrices
are unscaled; assembly applies each requested coefficient once. Fixed-density
two-electron energy/derivative assembly contains the additional one-half energy
factor. One-electron, Pulay and nuclear terms belong to complete method assembly.

The current independent consumer is `build_exact_direct_jk`: a CPU reference
over an explicitly supplied dense chemists'-order ERI tensor. The dense tensor
is an existing CPU reference representation, not a proposed production CUDA
memory model. Its tests cover absent terms, arbitrary coefficients, spin
conventions, fixed-density derivatives and preflight rejection.

## Dispatch and ownership map

| Boundary | Existing responsibility | Remaining coupling |
| --- | --- | --- |
| `methods/hf_method.cpp:resolve_hf_options` | Translate legacy method/DF options into one resolved request | Both terms inherit one approximation and one backend |
| `HfPreparedSingle::execute` | Select existing CPU/CUDA RHF/UHF solvers | Method code still branches on `legacy_density_fitting` |
| `HfPreparedBatch` / `scf::FleetPlan` | Own compatible buckets, geometry and warm state | Separate direct/DF booleans select the whole bucket |
| `scf/rhf.cpp` | CPU iterations and final energy/force assembly | Direct/DF entry points remain separate; exact entry preflight requires standard complete HF |
| `scf/cuda_rhf.cu` | Persistent direct-HF queues, generated/fallback kernels and solver state | Fused CUDA consumers require the standard coupled HF coefficients |
| `scf/cuda_density_fitting.cu` | Prepared metric/three-center data, bounded J/K, response and device SCF | Public execution helpers always request both J and K |
| `scf/fock_build.cpp` | Validate capabilities and mathematical/execution identity | Nonstandard CUDA terms and independently fitted terms are rejected |

The public method ABI does not yet expose independent J/K provider choices or
the resolved strategy diagnostics. Its existing combined DF enum is an explicit
approximation choice: AUTO does not authorize changing an exact Hamiltonian to
a fitted one.

## Reusable DF primitives

The CPU implementation already separates private `build_coulomb` and
`build_exchange` contractions. Likewise its response implementation separates
the Coulomb quadratic derivative from matching-spin exchange quadratic
derivatives. Reuse the existing metric pseudoinverse and its spectrally
truncated Frechet response; an independent J/K selection must not change the
retained metric subspace or omit metric response.

`cuda_density_fitting.hpp` exposes host-returning batched and item-level raw
J/K execution, plus device-pointer variants on the plan's stream. The latter
perform no mandatory D2H. Their implementation calls separate `build_coulomb`
and `build_exchange` services, but currently requires every output pointer and
executes all terms. Independent-term execution should reuse these services and
skip unrequested work explicitly. The plan's retained and scratch allocations
must remain visible to the existing resource accounting.

The direct CUDA implementation exposes a fused HF Fock path and a diagnostic
Fock-only iteration mode, not a general raw independent J/K provider interface.
The diagnostic mode is not an interchangeable production provider and cannot
be substituted for an independently validated raw consumer.

## Acceptance work identified by the baseline audit

1. Provide typed independent provider execution and coefficient-aware
   assembly, retaining the established fused path for standard HF requests.
2. Integrate exact and fitted combinations, their matching derivative
   contributions, complete prepared identities and failure isolation.
3. Expose requested/resolved semantics and backend/schedule diagnostics while
   preserving existing public defaults and approximation authorization.
4. Connect the available fixed-density XC consumer to the common J/K boundary.
   Complete RKS/UKS and hybrid methods remain owned by #162/#165, which are
   currently open; this refactor must not imply those methods are complete.
5. Validate identical-approximation raw matrices before exact-versus-fitted
   comparisons, then complete CPU/CUDA energy/force, replay, changed-geometry,
   ragged-batch and dispatch-overhead checks under matched final accuracy.

Unsupported range-separated operators remain explicit until #166 supplies
validated values and derivatives. Future providers must extend this common
boundary rather than introduce another method-specific selector or cache.
