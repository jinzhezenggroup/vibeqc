# Scalar Gaussian ECP baseline

The supported model is the scalar semilocal BSE/NWChem convention. For an
atom with removed core count `Nc`, the ionic charge is `Q = Z - Nc`. The
Hamiltonian contains the Coulomb attraction `-Q/r`, the local residual and
the nonlocal projector differences. Nuclear repulsion is `sum Q_A Q_B/R_AB`;
the electron count is `sum Q_A - molecular_charge`. Atomic number remains
the nuclear identity. Neither the Coulomb tail nor removed electrons are
included a second time in the residual integrals.

The highest supplied angular channel is local. Each lower channel denotes
`U_l - U_local`, with radial terms `c r^(n-2) exp(-alpha r^2)`. The nonlocal
operator sums normalized real spherical projectors over `m = -l..l`. Orbital
normalization and Cartesian/spherical ordering come from the ordinary basis
layer. Supported orbitals are s/p/d/f in Cartesian and real spherical layouts;
projectors are s/p/d/f, local labels at
most g, radial powers 0..4, at most 256 AOs and 128 atoms. Unknown parameter
fields, multiple coefficient rows, spin-orbit formats and higher angular
momenta are rejected. Parameter import performs no online lookup.

## Interfaces and derivatives

`vibeqc_system_create_ecp` is a separate C ABI extension; existing descriptors
retain their layout. It owns core counts and atom-mapped terms.
`vibeqc_system_ecp_integrals` exports local and nonlocal matrices separately,
followed by atom-major xyz derivative matrices for each component.

```python
from vibeqc import Calculator, load_basis
from vibeqc.ecp import ecp_integrals

basis = load_basis("owned-basis-and-ecp.json")
raw = ecp_integrals(atoms, basis)
gradient = raw.contract(arbitrary_real_ao_weights)
refined = ecp_integrals(atoms, basis, radial_points=224, polar_points=44)
print(raw.quadrature_difference(refined))
result = Calculator(basis=basis, device="cuda").singlepoint(atoms)
```

`contract` is the full AO dot product, including nonsymmetric fixed weights,
with no implicit occupancy factor. Its result is an energy derivative; API
forces have the opposite sign. The public raw contraction uses NumPy. The
HF CUDA adapter contracts total RHF/UHF density on the device, combining
ECP derivatives with the existing kinetic, effective-charge attraction,
overlap-Pulay, two-electron and ionic-repulsion derivatives.

IntegralIR records two Gaussian shell slots and a distinct physical ECP
center. Its schema version 4 preserves radial/projector terms and the shared
RawBlock/WeightedDerivative contracts. Generated Gaussian component DAGs
differentiate basis centers analytically before emitting device arithmetic.
`integral/ecp_projector.py` now owns the residual radial integrand, angular
projection reductions and local/nonlocal pair contractions. Radial expressions
include the volume measure (`r^2` cancels the operator's `r^-2`); bilinear
derivatives are differentiated from the value expression. The same generated
C++ header runs in native host tests and the production CUDA adapter.
ECP-center derivatives use generated `dC = -(dA+dB)` after the angular
reduction and before physical atom accumulation, including coincident A/B/C
cases. Value-only calls do not read derivative slots.

CPU uses one radial layer. CUDA batches up to four layers for at most 16 public
AOs and 44 polar points; larger domains retain one layer. Each AO pair
accumulates layers in ascending radial order. This matches the preceding
production path's one-layer launches on the same stream; the old internal
kernel's unused multi-layer thread mapping is not the ordering contract.
The independent `src/integrals/ecp.cpp` CPU implementation remains the public
CPU fallback and a numerical oracle; it intentionally does not call these
generated contractions. The compiler also owns ECP-centered node displacement,
normalized AO primitive/component accumulation, local/nonlocal hcore addition
and the full AO fixed-weight derivative contraction with the energy-to-force
sign. The CUDA adapter supplies total RHF/UHF density without an extra occupancy
factor. Component/primitive/AO reduction order and FP64 storage are unchanged.
`integral/ecp_grid.py` owns the production host Gauss-Legendre recurrence,
mapped radial Jacobian, sphere coordinates/weights, real s/p/d/f harmonics and
Cartesian component coefficient normalization. Scalar arithmetic uses the
common DAG; finite root/grid loops are compiler-emitted host schedules.
Trigonometric operations call the host standard library. The independent CPU
oracle retains its own nodes, harmonics and normalization; its integral path
does not call the generated grid wrapper. The existing grid limits, Newton
stopping policy, node order and append semantics are preserved.

The CUDA adapter retains allocation/launch/scatter, AO expansion metadata and
the method's two-grid convergence policy. It remains conservatively classified
as scientific in the ownership ledger. This does not claim complete adapter
retirement or expanded method support.

Orbital f uses the existing Gaussian DAG, generated component normalization and
molecular real-spherical expansion. The compiler-owned orbital limit also
defines the native ECP constructor boundary. The Python preflight applies the
same limit to both ECP atoms and all-electron atoms in mixed systems. Component
dispatch checks the angular powers before encoding them, so unsupported g
components cannot alias supported lower components. This orbital extension
does not itself expand projector channels, radial powers, element families or methods.

The separate projector extension includes seven orthonormal real f harmonics
(slots 9..15) and compiler-owned f-channel reductions. The native constructor
uses the emitted projector bound; BSE/NWChem local labels may extend through g
so that f is a nonlocal difference. This local label does not enable g orbitals
or g projectors. Each staged radial layer contains 16 projected jets per AO.
CPU stages one layer; CUDA stages up to four within the bounded schedule
described above. The resource inventory includes this projection storage.
Synthetic signed f-channel parameters qualify this operator capability without
claiming a physical heavy-element parameter family.

See the [AO/weight ownership decision](../.agents/notes/implemented/architecture/2026-09-16-ecp-ao-weight-consumers.md)
for the reduction-order and oracle rationale.
The [host-grid ownership decision](../.agents/notes/implemented/architecture/2026-09-17-ecp-host-grid.md)
records the quadrature boundary and independent moment/addition-theorem gates.
The [orbital-f decision](../.agents/notes/implemented/numerics/2026-09-17-ecp-orbital-f.md)
records the separate orbital/projector bounds and f qualification gates.
The [f-projector decision](../.agents/notes/implemented/numerics/2026-09-18-ecp-f-projectors.md)
records the harmonic/storage extension and its independent operator gates.

## Mixed-family physical centers

The bounded NaK qualification combines LANL2DZ Na and Stuttgart RLC K
orbital/ECP records in one 16-AO spherical calculation. Both physical atoms
carry distinct ECP terms, removing 10 and 18 core electrons respectively.
Independent per-center operator blocks, all-center derivatives, atom/AO
permutations, direct RHF/UHF complete forces and budgeted geometry replay
test that parameter and core-count mappings follow their physical atoms.
Na has a nonzero local residual; K has a zero local residual and nonzero
nonlocal projectors. Effective ionic charges supply the Coulomb tails once.

See [the mixed-center qualification](../benchmarks/results/ecp-multicenter-171/README.md)
for exact parameters, geometries, acceptance gates and reproduction. This
bounded evidence does not establish general mixed-family coverage or new
methods, angular limits or resource inventories.

## Numerical and execution boundaries

The baseline uses Gauss-Legendre radial nodes mapped by `r=t/(1-t)`, polar
Gauss-Legendre nodes and uniform azimuth. Full direct RHF/UHF checks the
160/32/64 grid against 224/44/88, separately for local/nonlocal matrices
(absolute 2e-9) and derivatives (2e-8), returning the refined result. Failure
rejects the calculation. This is empirical convergence evidence, not a
rigorous error bound for arbitrary exponents/geometries. Raw exports expose
their explicit grids so discretization and CPU/GPU floating-point errors
can be inspected separately.

CPU stages one radial layer of AO values and projections. CUDA stages at most
four in the bounded small-domain schedule, independently of radial grid length.
The final batch uses its actual remaining layers. OOM in optional four-layer
staging retries the original single-layer schedule; other failures propagate.
The shared compiler policy also controls the resource bound, using Cartesian
storage capacity and actual public AO count for schedule selection.
CUDA consumes device hcore/density/force buffers on the borrowed stream;
raw exports additionally transfer output to host. Complete HF uses bounded
two-grid derivative buffers; the resource planner includes this transient
workspace, core-adjusted occupations and ECP parameter identity. GPU memory
uses the shared native allocation ledger. Prepared topology identity includes
core counts, channel parameters and atom mappings; geometry changes rebuild
one-electron terms. The [radial-schedule decision](../.agents/notes/implemented/performance/2026-09-18-ecp-radial-batching.md)
records ordering, bounded fallback and endpoint qualification.

CUDA HF resource inventory v1 remains limited to at most 16 public AOs (and
its existing DIIS/layout constraints). Orbital-f support does not enlarge that
inventory: larger supported calculations can execute without an explicit
budget, but resource estimation reports `unsupported` and `require_feasible()`
raises. The f budget/replay gate uses a 16-AO spherical fixture; larger f
endpoint measurements qualify numerical execution only.

Direct RHF/UHF are the supported complete methods. ECP density fitting is
explicitly rejected pending its own complete force/budget gates. Complete
canonical MP2 with ECP is also rejected until its reference/provider gates are
validated. The all-electron accuracy-model schema cannot represent an ECP
Hamiltonian, so ECP `resolved_model()` requests fail explicitly. Complete
LDA/PBE RKS/UKS energy-only calculations use the same ECP Hamiltonian, as
described below. Complete DFT gradients remain dependent on #163 and force
requests are rejected. No broad heavy-element validation follows from support
for the parameter format.

## Semilocal DFT energies

`lda-rks`, `pbe-rks`, `lda-uks` and `pbe-uks` accept supported scalar ECP
basis records on CPU/CUDA for the energy observable. The shared one-electron
provider supplies kinetic energy, effective-charge Coulomb attraction and
the local/nonlocal ECP residual exactly once. Electron populations and nuclear
repulsion use effective ionic charges. XC evaluates the valence density in the
ordinary Gaussian AO basis; it does not reconstruct a core density or apply a
nonlinear core correction. Atomic number still determines grid element identity.

AO capability checks permit values and first spatial jets needed by LDA/GGA
energies. They still validate ECP metadata and angular limits. Higher AO jets,
complete nuclear forces, DF/ECP and other DFT methods do not inherit support.

```python
result = Calculator(method="pbe-rks", basis=basis, device="cuda").singlepoint(
    atoms, properties=("energy",)
)
assert result.forces is None
```

The bounded qualification uses installed PySCF 2.14.0 LANL2DZ Na and STO-3G H:
neutral NaH RKS and the +1 doublet UKS cation, Cartesian/spherical s/p orbitals,
and the declared default GridSpec. Independent Libcint/Libxc solves on identical
grid points and weights check total energy, nuclear/one-electron/Hartree/XC
components, spin electron counts and physical residuals. Different reference
initial guesses must reach the same energy. This is a matched-discretization
gate, not a claim of convergence to the continuum XC integral or validation
of arbitrary ECP/functional combinations.

Exact-budget mixed ECP/all-electron batches check cold/warm execution,
changed-geometry rebuilding, independent displaced energies, failed-item
isolation and restoration. Existing KS resource plans include ECP setup
workspace and preserve ECP parameter/core identity.

Run `pytest tests/python/test_ecp_dft.py`; enable GPU cases only with
`VIBEQC_ECP_CUDA_TEST=1`. `tools/qualify_ecp_dft.py --device cpu --output result.json`
(or `--device cuda`) records source/library identities and eight independently
checked energy endpoints. See the
[decision note](../.agents/notes/implemented/numerics/2026-09-18-ecp-dft-energy.md)
and [retained evidence](../benchmarks/results/ecp-dft-171/README.md).

## Stuttgart RLC parameter qualification

The bounded Stuttgart RLC suite uses the installed PySCF 2.14.0
`stuttgart-dz` Na/K orbital and scalar ECP records with STO-3G hydrogen.
The unmodified s/p orbital records give nine molecular AOs. Na removes ten
core electrons and K removes eighteen; both leave ionic charge +1. Neutral
singlet NaH/KH therefore contain two explicit electrons, while their +1
doublet cations contain one. Qualification covers direct RHF/UHF on CPU/CUDA
at the off-axis geometries in `tests/python/test_ecp_stuttgart.py`.

This family has an exactly zero local residual, separate from the nonzero
effective-charge Coulomb attraction, and signed s/p/d projector differences.
The reference loader drops zero local coefficients. The test-only converter
restores the published zero local term (`n=2`, exponent 1, coefficient 0) to
retain the local f channel in the owned schema. Dropping the entire local
channel would incorrectly reinterpret the d projector as local.

The suite pins the combined orbital/ECP parameter hashes, checks separate
Libcint local/nonlocal matrices, compares both quadrature levels and all-center
derivatives at two displacement steps, and checks complete energies/forces
against PySCF. Exact-budget geometry replay checks complete-energy differences
and the CUDA allocation ledger. The supported scope is these parameter records,
states and geometries; other Stuttgart elements, RSC/MDF families, spin-orbit,
DFT/ECP and larger angular domains require separate evidence.

With the pinned reference-test extra and the corresponding native library:

```sh
python -m pytest tests/python/test_ecp_stuttgart.py -q
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_stuttgart.py -q -k cuda
python tools/qualify_ecp_stuttgart.py --device cpu --output stuttgart-cpu.json
python tools/qualify_ecp_stuttgart.py --device cuda --output stuttgart-cuda.json
```

The report binds parameter, fixture, driver and native-library identities to
matrix/energy/force errors, resource bounds and device observations. Timings
describe single synchronous calls and do not establish a speedup.
See the [qualification decision](../.agents/notes/implemented/numerics/2026-09-18-ecp-stuttgart-rlc.md)
and [retained evidence](../benchmarks/results/ecp-stuttgart-171/README.md).

## Bounded heavy-element parameter qualification

The real-parameter qualification covers the installed PySCF 2.14.0 LANL2DZ
Rb, Cs and Au orbital/ECP records, paired with STO-3G hydrogen. These are separate
from synthetic high-angular-momentum operator tests. The reference parameters
are consumed only by tests and the qualification driver; normal execution
continues to use caller-owned basis/ECP records without a PySCF dependency.

| ECP element | Atomic number | Removed electrons | Ionic charge | Molecular AO count |
|---|---:|---:|---:|---:|
| Rb | 37 | 28 | 9 | 13 |
| Cs | 55 | 46 | 9 | 13 |
| Au | 79 | 60 | 19 | 23 |

The Rb/Cs fixtures use the unmodified s/p orbital records, local f label and
s/p/d nonlocal channels. They cover neutral singlet RbH/CsH (10 explicit electrons)
and their singly charged doublet cations (9 explicit electrons), with direct
RHF/UHF on CPU/CUDA. The nuclei are placed off-axis; raw matrix and derivative
gates use two bond geometries per element. Qualification is limited to these
parameter records, states and geometries. It does not establish arbitrary
Rb/Cs/Au chemistry, other LANL2DZ elements, other ECP families, spin-orbit physics,
or a relativistic method beyond the supplied scalar potential.

`tests/python/test_ecp_heavy.py` pins the combined orbital/ECP parameter hashes,
checks removed-core/effective-charge bookkeeping, and compares local and
nonlocal matrices separately with Libcint. Both grid levels, all-center
two-step finite differences, arbitrary nonsymmetric AO weights, complete HF
energies/forces, and geometry replay with complete-energy differences must
pass, using a planned budget where supported. The 13-AO Rb/Cs fixtures fit the
current CUDA resource inventory; their tests check the allocation ledger
against the declared budget.

AuH uses the unmodified s/p/d orbital records with a local g label and real
s/p/d/f nonlocal projectors. The neutral singlet has 20 explicit electrons
(RHF), and the -1 doublet has 21 (UHF). Its f-only Libcint block and all-center
finite differences are checked independently against the difference between
the full potential and the same potential with f coefficients set to zero.
This protects the physical f channel from silent omission or relabeling.
The local g label does not imply a g orbital or nonlocal g capability.

The 23-AO AuH fixture exceeds the CUDA budget inventory's 16-public-AO limit.
Numerical CUDA singlepoints and changed-geometry replay run without an explicit
budget; resource estimation and explicit-budget preparation must reject that
domain. CPU replay retains its planned budget. This qualification does not
extend CUDA resource support or claim a device-memory bound for AuH.
Au uses tighter SCF stopping criteria (`energy_tolerance=1e-12`,
`density_tolerance=1e-10`, at most 200 iterations). The reference must converge
to the same energy from minao, one-electron and atomic initial guesses. The
AuH+ doublet is not qualified: its native SCF solution did not match the
independent references during validation. ECP operator agreement alone does
not establish complete-method agreement for that state.

Run the suite and record a compact endpoint report with:

```sh
python -m pytest tests/python/test_ecp_heavy.py -q
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --output heavy-cpu.json
python tools/qualify_ecp_heavy.py --device cuda --output heavy-cuda.json
python tools/qualify_ecp_heavy.py --device cuda --elements Au --output gold-cuda.json
```

Use the pinned reference-test extra and the corresponding `VIBEQC_LIBRARY` as
described below. The report includes parameter identities, matrix/energy/force
errors, exact library and test hashes, planned bounds and the CUDA ledger.
Unsupported Au CUDA plans are recorded explicitly with no claimed peak bound.
Single-call timings are context for reproduction, not performance claims.
See the [qualification decision](../.agents/notes/implemented/numerics/2026-09-18-ecp-heavy-parameters.md)
for the boundaries and the retained evidence.
The [real f-channel decision](../.agents/notes/implemented/numerics/2026-09-18-ecp-real-f-gold.md)
records the separate AuH parameter, projector and resource-boundary qualification.

## Reproduction and parameter sources

Install the pinned `reference-test` extra (PySCF 2.14.0), build the CPU or CUDA
library and set `PYTHONPATH=python` and `VIBEQC_LIBRARY` to that exact artifact.
For example, build in Release mode with Ninja:

```sh
cmake -S . -B build-ecp-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build-ecp-cpu --parallel
cmake -S . -B build-ecp-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-ecp-cuda --parallel
```

Architecture 120 is the measured RTX 5090 target; select the actual allocated
GPU architecture for other devices. The measurements validate the generic
CUDA build and do not promote an AOT shell profile.

Run `pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py tests/python/test_ecp_f.py tests/python/test_ecp_f_projector.py`; set
`VIBEQC_ECP_CUDA_TEST=1` only on an allocated GPU. The tests use installed
PySCF LANL2DZ Na ECP/orbitals with STO-3G H, asymmetric mixed centers, d shells,
contracted f shells on ECP and all-electron atoms,
Cartesian/spherical layouts, charged open-shell HF, independent ECP center
motion and two finite-difference steps. Synthetic fixtures isolate local and
nonlocal components. Production code never imports PySCF.

The f suite compares all-center derivatives at two displacement steps, including
an independently moving ECP center without its own basis, and nonsymmetric
fixed-weight contractions. Complete RHF/UHF energies and forces are checked
against PySCF for both representations; CUDA additionally checks complete
energy finite differences. Prepared geometry replay is checked under the
planned host/device budget. The native capability test checks f acceptance,
g orbital/projector rejection and acceptance of f projectors through the C API.

`vibeqc_ecp_projector_tests` independently checks the emitted host arithmetic
using a double angular-node sum and the Legendre addition theorem, without
forming the production AO projections. It covers all radial powers 0..4,
local/s/p/d/f channels, signed mixed-exponent terms, center filtering, the
radial origin, and poisoned derivative slots in value-only mode. The Python
ECP suite also compares every radial power and d projectors with Libcint and
finite differences of an independently displaced ECP center on CPU/CUDA.
The f-projector suite includes powers 0..4, signed multi-exponent terms,
all-center finite differences at two steps, nonsymmetric weights, value-only
exports, f-orbital grid refinement, complete Cartesian/spherical RHF/UHF
forces, and budgeted changed-geometry replay with complete-energy differences.

Real numerical parameter tables are read from the test installation and are
not redistributed by this change. PySCF is Apache-2.0; consult its installed
basis-file notices and the original LANL2DZ references before redistributing
those data. User-imported basis records retain their source/version/license
provenance. Synthetic test parameters and this implementation use the
repository's GPL-3.0-or-later license.

`tools/benchmark_ecp.py --device cpu|cuda --output result.json` records matched
NaH matrix/refinement/backend errors, RHF/UHF energies/forces, hardware,
library hash, resource bounds and timing samples. Component endpoint timings
zero the other component's coefficients while retaining the full engine;
they must not be described as isolated CUDA kernel timings or a specialized
schedule promotion.
