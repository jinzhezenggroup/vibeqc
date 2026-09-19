# Physical KS diagnostics

Completed LDA/PBE RKS/UKS results expose an immutable `ks_diagnostic` on both
`Result` and `BatchItemResult`. It contains the actual grid/tile/AO order,
audited SCF domain, alpha/beta occupations, electron counts, physical energy
components, convergence measures and iteration history. The enclosing result's
`executed_backend` identifies the backend that performed the calculation.

```python
from vibeqc import Calculator

result = Calculator(method="pbe-uks").singlepoint(
    [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    charge=1, multiplicity=2,
)
diagnostic = result.ks_diagnostic
print(diagnostic.occupations)           # (1, 0)
print(diagnostic.components.total)      # the returned physical energy
print(diagnostic.physical_residual_max)
print(diagnostic.to_payload())          # all iteration records included
```

Energies are in Hartree. `KsEnergyComponents` contains nuclear, one-electron,
Hartree and XC terms with XC counted once; `total` sums those four terms.
`electrons` measures `Tr(D_s S)` in the AO metric. RKS reports equal spin
counts from half the total density. These are distinct from quadrature
electron counts and are independent of the quadrature's accuracy.

`density_change_max` and `physical_residual_max` are the maximum spin RMS
values used by the physical convergence gates. RKS has one total-density
matrix. UKS gates each spin separately, so an empty beta channel cannot dilute
an unconverged alpha channel. The legacy result's `density_rms` and
`physical_residual_rms` continue to combine the alpha/beta matrix entries in
one RMS.

Each `KsIteration` describes a physical E/F/D state and its proposed density
change. `energy_change` is `None` on the first iteration, because no preceding
energy exists. Later entries retain the measured absolute energy difference.
`occupation_stabilized` records whether that proposal used the stationary
UKS virtual-projector shift; its recorded energy and residual remain physical
and unshifted. CPU RKS performs additional validation/projection after its
iteration loop. The final diagnostic retains those final physical components
and residual separately from the original historical rows.

`fock_builds` includes final validation rebuilds for this solve.
`initial_density_used` describes this solve's actual initial state. When the
prepared batch retries a rejected warm attempt from a cold seed, the snapshot
describes the final attempt; it is not a count or history of all discarded
work. Complete endpoint timing and transport measurement must include every
attempt and preparation/finalization phase separately.

Valid nonconverged batch items retain their actual history. Invalid or
numerically failed items have no diagnostic. A replay invalidates cached C
records before executing; Python snapshots from earlier results remain
unchanged. Unsupported methods and older native libraries return `None`.
The public object stores frozen dataclasses and tuples, not borrowed pointers
into a mutable native history.

The additive C queries are `vibeqc_calculation_get_ks_diagnostic` and
`vibeqc_batch_get_ks_diagnostic`. Legacy result structures and batch-array
strides do not change. Query a size/ABI-initialized `vibeqc_ks_diagnostic` with
NULL history and zero capacity to learn `history_count`. To copy the history,
allocate at least that many `vibeqc_ks_iteration` descriptors, initialize every
descriptor's size and ABI, and query again. Every output is validated before
any output is written. Query and execution calls on a prepared owner must be
serialized by its caller.

The method adapter moves one exported history through to its C handle. The
shared resource inventory covers CPU vector growth or the CUDA history plus
that exported snapshot; public queries do not allocate another native history.
Python result storage is owned by the caller.

## Internal final-state handoff

The native CPU RKS and resident CUDA RKS/UKS owners provide an internal,
versioned final-state handoff for issue #163. CPU UKS and CPU ECP handoffs
remain unsupported. This interface is deliberately absent from the public
C/Python result ABI and does not make public force requests supported.

After a converged solve, `methods/dft_method.hpp` can return an eligibility
token for either a prepared single calculation or one prepared batch item. The
token binds the prepared provider, geometry/basis owner, GridSpec, functional,
spin occupations, device, solve epoch and exact orbital/Fock/density
generation. Every new `begin`, including a failed or nonconverged attempt,
revokes the preceding token before backend work. Rebuilt geometry receives a new
owner even when all matrix dimensions are unchanged. Prepared-call exceptions
and batch items rejected before device submission explicitly revoke their old
eligibility while preserving independent warm-start ownership.

An exact-token read canonicalizes the retained physical, non-DIIS Fock on the
selected backend and detaches `D`, `F`, `C`, orbital energies,
occupations, energy components, grid and provider identity. The read validates
the AO-metric eigenframe, density reconstruction, commutator, electron trace,
idempotency, canonicality, component energy and physical residual before
publication. `W` is constructed on the host only when explicitly requested
after every gate passes. A stale token is rejected before matrix transfer.

`CudaKsTransfers::final_state_d2h_bytes` and `final_state_reads` separate this
explicit handoff from ordinary energy execution. The existing public
`matrix_d2h_bytes` total still includes snapshot matrices, so legacy transport
accounting remains conservative without an ABI change.

## Native CPU stationary-gradient diagnostic

`vibeqc._stationary_cpu.complete_rks_gradient_diagnostic` is an internal,
complete first nuclear-gradient **diagnostic**, not a production native force
endpoint. It consumes a live `StationaryKsState.from_native` lease. The supported
domain is direct, all-electron, integer-occupation real-FP64 LDA/PBE RKS with
s/p AOs, the native version-one unpruned grid and distinct nuclei. Unsupported
angular, spin, backend and derivative capabilities do not inherit support from
this entrypoint. Public `Calculator` DFT properties remain energy-only.

The seven reported sources are one-electron, Coulomb, XC AO-center, XC point
motion, XC partition-weight motion, overlap/Pulay and nuclear repulsion.
`StationaryGradientPlan` supplies the integral weights through TensorIR AD and
checks exactly-once final source coverage. The existing integral graphs generate
native CPU S/T/V, ERI and nuclear-pair derivatives. There is no method-specific
PBE force formula, SCF iteration tape, CPKS solve, HF rerun or PySCF runtime call.

The execution boundary is explicit: SCF, AO jets, exact native SCF-domain XC
point coefficients and generated integral derivatives execute natively.
`execution="reference"` (the default) retains interpreted TensorIR weights,
AO pullbacks and Becke JVPs. `execution="native"` compiles the same TensorIR
weights/reduction, reuses native generated AO-jet pullbacks, and contracts the
Becke adjoint in native CPU code. Python orchestration and NumPy feature/BLAS/map
operations remain; neither selector enables public forces or establishes a
whole-endpoint resource reservation. See [compiled consumer contracts](stationary_native_consumers.md).

```python
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

atoms = [("H", (0.1, 0.2, -0.6)), ("H", (0.2, -0.1, 0.8))]
calc = Calculator(
    method="pbe-rks", basis="sto-3g", device="cpu",
    ks_options=KsOptions(grid=GridSpec(
        radial_points=24, angular_polar=8, angular_azimuth=16,
    )),
    energy_tolerance=1e-12, density_tolerance=1e-10,
)
with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
    energy = batch.execute(strict=True).items[0].energy
    state = StationaryKsState.from_native(batch, basis)
    diagnostic = complete_rks_gradient_diagnostic(
        state, basis, cache=".cache/stationary-cpu",
    )
    gradient = diagnostic.gradient       # dE/dR, Hartree/bohr
    forces = -gradient                   # negate exactly once
    print(energy, diagnostic.components, diagnostic.work)
```

A caller may pass a `CppCompilerAdapter`; otherwise `CXX`, or `c++` when unset,
selects a compatible C++17 compiler. Sources are published atomically before the
shared native-artifact cache hashes and compiles them. Compiler identity,
flags, source and the complete project-header closure participate in reuse.
No generated source or binary belongs in Git.

CPU preparation retains the final evaluated F[D] and a density copy without
adding a Fock evaluation. Explicit snapshot export canonicalizes that actual
Fock and constructs W only after the existing final-state validator passes.
CPU wire version two uses the `UINT64_MAX` device sentinel and additionally
carries the native grid prescription and raw atomic quadrature measures. The
CUDA version-one payload and device semantics are unchanged. Raw measures are
materialized only for explicit CPU export, not retained by ordinary energy
grid execution; the existing physical-state observer includes retained D/F.

Partition response multiplies the generated partition derivative by the native
raw atomic measure. Dividing the final weight by a tiny or zero partition is
not permitted. Frozen source data and before/after lease checks prevent stale,
replayed, changed-geometry, detached or relabeled states from publishing a
complete gradient. A late derivative failure publishes no partial result and
does not corrupt the valid SCF state.

Working records and AO/grid evaluations are tiled. Both routes still visit all
ordered AO quartets. The reference route evaluates partition JVPs for all
`3*Natom` directions; native grid contraction uses two pair passes per point
and atom-sized scratch, with a separate center-validation count. The SCF
reference retains a full molecular grid and dense reference data. Reported
component byte/work bounds are **not** a global ResourceBudget or whole-process
peak-memory guarantee. Compilation, Python/NumPy and any reference-interpreter
overhead must remain visible in timing.

### Qualification

With a current CPU library and the test dependencies installed:

```sh
PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VIBEQC_LIBRARY="$PWD/build/libvibeqc.so" python -m pytest -q \
  tests/python/test_dft_complete_cpu.py \
  tests/python/test_dft_stationary_native.py \
  tests/python/test_dft_stationary_gradient.py
```

The independent PySCF/Libxc/Libcint gate uses the same primitive input and raw
atomic quadrature but its own SCF, integral/XC derivatives and Becke response.
It compares all seven sources and totals for asymmetric water with LDA and PBE.
Every Cartesian component is also checked with three fully rebuilt and
reconverged central-difference steps. The raw maximum gradient gate is
`1e-6 Eh/bohr`; the independent analytic and Richardson gates are `1e-7`.
Translation/permutation, different tile sizes, replay, malformed native inputs,
late provider failure, unsupported domains and a fresh process forbidding
external oracle imports have dedicated tests. The CUDA snapshot tier retains
its existing explicit opt-in; CPU qualification is not CUDA execution evidence.

See the [implementation decision](../.agents/notes/implemented/architecture/2026-09-19-native-cpu-stationary-gradient.md)
for the state, quadrature and compiler-ownership rationale.
