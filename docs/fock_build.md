# Shared Fock construction contract

The internal C++ boundaries in `src/scf/fock_build.hpp` and
`src/scf/fock_provider.hpp` separate a mathematical J/K request from its
resolved execution strategy and prepared integral sources. CPU and CUDA
providers can be selected independently. The additive public `vibeqc/fock.h`
and Python `FockPlan` interfaces expose these choices while the legacy method
descriptors retain their density-fitting defaults.

## Public prepared API

`FockPlan` owns normalized geometry, orbital/auxiliary data and native sources.
Create a new plan when those inputs or mathematical coefficients change:

```python
from vibeqc import FockBuildSpec, FockPlan
from vibeqc_compiler.dft import NativeAO

atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]  # Bohr
spec = FockBuildSpec.hf(coulomb="density_fitted", exchange="exact")
with NativeAO(atoms, basis="sto-3g") as basis:
    with FockPlan(basis, spec, device="cpu") as plan:
        result = plan.solve()
        replay = plan.solve(initial_density=result.density, compute_forces=False)
        fixed = plan.evaluate(result.density, derivative=True)
        diagnostics = plan.diagnostics
```

`solve` uses the shared SCF iteration/finalization driver for the declared J/K
energy. It includes no XC. Its forces are complete negative geometric energy
derivatives in Hartree/Bohr, including one-electron, nuclear and Pulay terms.
The optional warm density must satisfy the existing overlap-metric occupation,
Hermiticity and electron/spin trace checks; it is never silently normalized.
Every solve starts fresh DIIS history. Sources persist across replay, and a
failed solve cannot replace another result or retain a failed iterate.
`compute_forces=False` skips response execution and works on either value-only
or derivative-capable plans. The latter retain their already prepared response
data. Use `FockBuildSpec.hf(derivative_order=0)` to avoid preparing that data.

`evaluate` returns unscaled J/K, Fock matrices including Hcore, one- and
two-electron energies and nuclear repulsion. Its optional `gradient` is only
the fixed-density **two-electron** geometric derivative, not a complete force.
Restricted inputs have shape `[AO,AO]`; unrestricted inputs have shape
`[2,AO,AO]`. Returned NumPy arrays own immutable storage. Input densities are
snapshotted once for native execution and result identity.

The plan's mathematical `identity` includes the normalized basis, effective
auxiliary basis, canonical terms, derivative capability, screening, DF metric
cutoff and FP64 precision. Unused auxiliary and absent-term metadata are
canonicalized away. `execution_identity` additionally records the requested
specification, native source hash, backend/device, source mappings and buffer
policy. Result identity includes both identities and the evaluated density;
SCF identity also includes seed, convergence controls and output selection.
These identities do not use or extend the legacy HF checkpoint format.
Both result classes retain detached `diagnostics` after plan closure; the
property returns a copy so caller edits cannot change recorded provenance.

`diagnostics` distinguishes the resolver's preferred standard-HF schedule from
the actual independent plan's source schedule. On CUDA, the public plan uses
CUDA integral/J/K/response consumers with host DIIS and eigensolves. Standard
`Calculator` HF continues to use the established fused CUDA solvers. CUDA
device-byte diagnostics cover explicit source buffers and exclude modules,
driver/library-private storage and recurrence stacks.

The C API offers the same ownership and output rules through
`vibeqc_fock_plan_create`, `evaluate`, `solve`, `diagnostic`, `last_error` and
`destroy`. Context and system handles may be destroyed after creation.
Descriptors are versioned and caller-owned output buffers must be disjoint.
Publication is transactional, including SCF nonconvergence. C callers must
serialize calls and destruction; Python serializes them with a per-plan lock.

## Fixed-density semilocal consumer

`FixedDensityMeanField` combines a native unit-Coulomb, absent-exchange request
with the existing `FixedDensityXC` integrator:

```python
from vibeqc import FixedDensityMeanField, FockBuildSpec, FockTerm
from vibeqc_compiler.xc import FixedDensityXC, functional

spec = FockBuildSpec(
    derivative_order=0, coulomb=FockTerm(), exchange=FockTerm(present=False)
)
with NativeAO(atoms) as basis, FockPlan(basis, spec) as plan:
    consumer = FixedDensityMeanField(plan, FixedDensityXC(functional("PBE")))
    combined = consumer.integrate(grid, density)
```

Here `grid` is a compatible molecular grid and `density` uses the plan's spin
layout. The result contains full fixed-density energy (including nuclear
repulsion) and Fock matrix. Both consumers use one owned density snapshot;
the J/K contraction is implemented only in the native provider. LDA and PBE
are checked against the independent pinned XC integration fixtures and
density-direction energy variations. This is not DFT SCF and supplies no
geometric XC forces; those remain with #162/#165.

## Densities, operators, and coefficients

`FockBuildSpec` version 1 records spin, requested derivative order, and
independent Coulomb and exchange terms. Each term records presence, a signed
Fock coefficient, the full-/short-/long-range operator and range parameter,
and the exact or density-fitted approximation.

All raw matrices are FP64, row-major, in the caller's public AO representation.
The exact CPU consumer accepts nonsymmetric finite density matrices without
silently symmetrizing them. ERIs use chemists' `(ij|kl)` ordering:

- `J_ij = sum_kl D_total,kl (ij|kl)`.
- `K_spin,ij = sum_kl D_spin,kl (ik|jl)`.
- Restricted density includes double occupation: `F = H + J[D] - 0.5 K[D]`.
- Unrestricted densities have unit occupation:
  `F_alpha = H + J[D_alpha+D_beta] - K[D_alpha]`, and similarly for beta.

`build_exact_direct_jk`, `CpuFockPlanView::build`, and `CudaFockPlanView::build`
return unscaled J/K matrices.
`assemble_fock` applies the requested coefficients exactly once. An absent
term has no raw output allocation. Presence and a zero coefficient are
different requests: a present zero-coefficient term still requests its raw
matrix.

`contract_fock_energy` and `CpuFockPlanView::energy_derivative` use the same
resolved terms and coefficients for energy and fixed-density derivatives:

```text
RHF: dE_2e = 0.5 D : (c_J dJ + c_K dK)
UHF: dE_2e = 0.5 c_J D_total : dJ
             + 0.5 c_K (D_alpha : dK_alpha + D_beta : dK_beta)
```

One-electron, overlap/Pulay, and nuclear-repulsion derivatives remain outside
this two-electron consumer. The force is the negative total derivative.

## Resolution and supported execution

`resolve_fock_build` preflights the complete request before any contraction.
Unsupported versions, spin layouts, operators, derivative orders, or provider
combinations fail explicitly. Merely having an enum value does not make that
operator executable.

| Consumer | Executable scope |
| --- | --- |
| Exact CPU raw J/K | Full range, RHF/UHF density conventions, independently present J/K, arbitrary finite coefficients, values and first derivatives |
| CPU DF raw J/K | Full range, both spin conventions, independently present J/K and finite coefficients |
| Shared CPU SCF solver | All exact/DF pairings, independent terms and coefficients, matching first derivatives |
| CUDA fused direct HF | Complete standard RHF/UHF pair and first derivatives; existing fused kernels, device storage, streams, and graphs |
| CUDA DF raw services | Independently selected J/K, resident/streamed/tiled/batch/item/device-pointer execution |
| CUDA direct raw services | Independent terms, public Cartesian/spherical AOs through f, nonsymmetric densities, batch/item execution and matching first derivatives |
| CUDA DF SCF adapter | Existing standard HF DF pair, using existing DF implementations |
| Independent CUDA SCF | Any exact/DF/absent pair and signed coefficients; host DIIS/eigensolves with CUDA integrals, J/K and two-electron derivatives |
| SR/LR operators | Rejected; #166 supplies these additional mathematical operators |

The CPU reference consumes already materialized four-center integral and
derivative tensors. This refactor does not make that algorithm bounded or
on-demand. CUDA keeps its existing persistent, quartet, and bounded execution
paths; independent mathematical terms do not require separate GPU launches.

`FockProviderCapabilities` reports the supported independent full-range value
and first-derivative strategies. The generated DF response requires symmetric
densities and preflights this before either provider executes; raw matrices
still accept nonsymmetric densities. The capability query does not replace the existing system/basis preflight or probe whether a CUDA device is
available. Existing basis limits and backend initialization still apply.

## Prepared state and compatibility

A prepared HF calculation resolves its request before execution and retains
it in `ScfOptions::resolved_fock_build`. CPU iteration, final-density rebuilds,
and analytic two-electron forces consume the same immutable resolved value.
CUDA direct bucket compatibility also includes this value.

`CpuFockProviderView` borrows a prepared exact or fitted integral owner.
`CpuFockPlanView` binds one source to J and one to K, verifies both providers
before executing either, and invokes a shared source once for a complete pair.
The views never own or mutate a geometry cache. Their immutable owners must
outlive them; changing geometry or the AO/auxiliary representation requires new
bindings. CPU and CUDA plan views instantiate the same `BasicFockPlanView`,
so source selection, absent-term handling, preflight and shared-source dispatch
are implemented once. `CudaFockProviderView` borrows one batch item in an
existing direct or fitted CUDA plan, together with the immutable DF geometry
and response metadata. Item calls preserve neighboring source and scratch
state. DF value and response cutoffs must agree with the resolved request.
The quadratic response retains the full Frechet derivative of the truncated
metric inverse, including retained/discarded-space mixing.

`PreparedFockPlan` owns these views and their existing integral/source data.
It supplies one-electron data to the shared host SCF consumer as well as raw
J/K and fixed-density two-electron response to other consumers. Independent
CUDA single/fleet execution retains one owner per item. Exact compatibility
compares normalized atom/coordinate, shell/primitive, representation, charge,
spin and auxiliary snapshots; resolved semantics, device, buffer allowance
and relevant generated-kernel variants are also checked. No hash collision or
recycled object address can validate a source. A complete replacement is built
before swapping the cache. Convergence thresholds and warm density do not
invalidate integrals. CPU SCF keeps its existing transient integral lifetime
so retaining every reference ERI tensor does not change fleet memory policy.

The mathematical request (`spec`) is distinct from backend/schedule fields.
Screening, precision, and the DF metric cutoff are explicit resolved fields;
an unused exact-provider metric cutoff is canonicalized away. Zero screening
retains the existing unscreened reference semantics used by post-HF exports. Geometry,
orbital basis, auxiliary basis, and device-resource ownership remain in the
enclosing existing prepared plans and their compatibility checks.

The old DF selector still first chooses the DF approximation, then selects
its backend. `AUTO` does not authorize changing exact into DF or DF into exact.
`run_fock_strategy` centralizes single-item dispatch. Legacy CPU entry points
delegate to the common CPU solver, while existing standard CUDA HF schedules
remain fused. The `CudaIndependent` schedule explicitly identifies its host
SCF control; it does not imply a fused device iteration. Its exact provider
reuses the existing contracted-ERI evaluator without retaining a molecular ERI
or derivative tensor. Its fitted provider reuses the generated external-weight
response, including signed coefficients and the complete metric response.
Source-backed DF plans regenerate bounded tiles; resident/host-streamed DF
plans use the same typed binding. CPU ragged fleets accept explicit resolved independent requests
and retain per-item failure isolation and warm-state ownership.

## Validation and scope

`vibeqc_fock_build_tests` uses independent pinned two-AO J/K values, distinct
alpha/beta densities, nonsymmetric densities, and non-unit coefficients. It
checks absent terms, derivatives, preflight failures, and resolved identities.
`vibeqc_fock_provider_tests` adds all exact/DF pairings, independently absent
terms, a separate dense DF oracle, central differences through a truncated
metric, molecular SCF/force comparisons, warm replay, changed geometry and
ragged failure isolation. CUDA DF tests cover independent host and resident
device outputs against the same-approximation CPU services. The existing
RHF/UHF, batch, density-fitting, Cartesian, and spherical suites exercise the
standard method endpoints.

`vibeqc_cuda_fock_provider_tests` covers through-f direct matrices, s/p
derivatives, public representations, nonfinite source/result failure, and
independent DF device layouts. `vibeqc_cuda_fock_composition_tests` compares
all exact/DF/absent pairs, both spins, signed responses and separate batch items
against CPU integrals. It checks resident and regenerated DF storage with a
truncated metric, and complete SCF/replay/changed-geometry force endpoints.
The optional `vibeqc_cuda_fock_provider_tests --through-f-response` numerical
tier adds signed d/f UHF responses against the complete CPU derivative tensors
for both public representations and both independently prepared batch items.

`tests/python/test_fock.py` exercises public independent choices, transactional
failure, identity, source lifetime, SCF/replay/force consistency and semilocal XC
composition. `vibeqc_fock_api_tests` exercises the public C lifecycle after
context/system destruction and is also run with CUDA under the scheduler.
The [retained production comparison](../benchmarks/results/fock-strategies/README.md)
contains matched, synchronized CPU/CUDA endpoints, raw samples, quantitative
errors and hardware/build provenance. Complete endpoint medians changed by
+0.77% on CPU and +0.28% on CUDA; all energies and raw J/K matrices are
unchanged, with complete force differences below 1e-14 Hartree/bohr.
This is an architecture non-regression study, not a speedup promotion.
No complete DFT SCF method is advertised.

`tools/benchmark_fock_strategies.py` runs one worker per revision/backend
against production Release builds with `VIBEQC_CUDA_FAST_COMPILE=OFF`. It
records fixed-density direct CPU and DF CPU/CUDA matrices, complete RHF/UHF
SCF/forces, warm replay, changed geometry and four-item batch timings, with
all raw samples and matched-approximation numerical gates. The standalone
`benchmarks/fock_dispatch_probe.cpp` compiles against either revision and
compares the old contraction entry with the new provider boundary. Baseline
CUDA has no independent direct raw API, so its standard direct route is
compared through complete fused HF endpoints instead.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:35:00 python tools/benchmark_fock_strategies.py \
  --baseline /path/to/3da5841-worktree --head "$PWD" \
  --build-relative .artifacts/overhead-cuda-build \
  --output .artifacts/fock-strategy-overhead
```

The runner never publishes its artifact directory or changes a production
selector. Accepted evidence is selected through the repository evidence policy
after accuracy and overhead review.
Optional `--case`, `--spin`, `--approximation` and `--endpoint` filters repeat
a selected endpoint with its normal setup/warmup. The retained CPU bundle
includes an alternating-order focused reproducer and the code-layout diagnosis
that led to the local DF function alignment hint. Python plan identities use
native-normalized controls and device indices, including accepted NumPy scalars.
