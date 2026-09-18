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
projectors are s/p/d, local labels at
most f, radial powers 0..4, at most 256 AOs and 128 atoms. Unknown parameter
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

The one-radial-shell schedule is shared across the supported orbital domain.
The independent `src/integrals/ecp.cpp` CPU implementation remains the public
CPU fallback and a numerical oracle; it intentionally does not call these
generated contractions. The compiler also owns ECP-centered node displacement,
normalized AO primitive/component accumulation, local/nonlocal hcore addition
and the full AO fixed-weight derivative contraction with the energy-to-force
sign. The CUDA adapter supplies total RHF/UHF density without an extra occupancy
factor. Component/primitive/AO reduction order and FP64 storage are unchanged.
`integral/ecp_grid.py` owns the production host Gauss-Legendre recurrence,
mapped radial Jacobian, sphere coordinates/weights, real s/p/d harmonics and
Cartesian component coefficient normalization. Scalar arithmetic uses the
common DAG; finite root/grid loops are compiler-emitted host schedules.
Trigonometric operations call the host standard library. The independent CPU
oracle retains its own nodes, harmonics and normalization; its integral path
does not call the generated grid wrapper. The existing grid limits, Newton
stopping policy, node order and append semantics are preserved.

The CUDA adapter retains allocation/launch/scatter, AO expansion metadata and
the method's two-grid convergence policy. It remains conservatively classified
as scientific in the ownership ledger. This does not claim complete adapter
retirement or expanded method/projector support.

Orbital f uses the existing Gaussian DAG, generated component normalization and
molecular real-spherical expansion. The compiler-owned orbital limit also
defines the native ECP constructor boundary. The Python preflight applies the
same limit to both ECP atoms and all-electron atoms in mixed systems. Component
dispatch checks the angular powers before encoding them, so unsupported g
components cannot alias supported lower components. This orbital extension
does not expand projector channels, radial powers, element families or methods.

See the [AO/weight ownership decision](../.agents/notes/implemented/architecture/2026-09-16-ecp-ao-weight-consumers.md)
for the reduction-order and oracle rationale.
The [host-grid ownership decision](../.agents/notes/implemented/architecture/2026-09-17-ecp-host-grid.md)
records the quadrature boundary and independent moment/addition-theorem gates.
The [orbital-f decision](../.agents/notes/implemented/numerics/2026-09-17-ecp-orbital-f.md)
records the separate orbital/projector bounds and f qualification gates.

## Numerical and execution boundaries

The baseline uses Gauss-Legendre radial nodes mapped by `r=t/(1-t)`, polar
Gauss-Legendre nodes and uniform azimuth. Full direct RHF/UHF checks the
160/32/64 grid against 224/44/88, separately for local/nonlocal matrices
(absolute 2e-9) and derivatives (2e-8), returning the refined result. Failure
rejects the calculation. This is empirical convergence evidence, not a
rigorous error bound for arbitrary exponents/geometries. Raw exports expose
their explicit grids so discretization and CPU/GPU floating-point errors
can be inspected separately.

CPU and CUDA stage only one radial shell of AO values and projections.
CUDA consumes device hcore/density/force buffers on the borrowed stream;
raw exports additionally transfer output to host. Complete HF uses bounded
two-grid derivative buffers; the resource planner includes this transient
workspace, core-adjusted occupations and ECP parameter identity. GPU memory
uses the shared native allocation ledger. Prepared topology identity includes
core counts, channel parameters and atom mappings; geometry changes rebuild
one-electron terms. The first implementation favors a verifiable baseline
and makes no speedup claim.

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
DFT SCF/gradients remain dependent on #162/#163, so DFT/ECP completion is not
claimed. No broad heavy-element validation follows from support for the
parameter format.

## Bounded heavy-element parameter qualification

The real-parameter qualification covers the installed PySCF 2.14.0 LANL2DZ
Rb and Cs orbital/ECP records, paired with STO-3G hydrogen. These are separate
from synthetic high-angular-momentum operator tests. The reference parameters
are consumed only by tests and the qualification driver; normal execution
continues to use caller-owned basis/ECP records without a PySCF dependency.

| ECP element | Atomic number | Removed electrons | Ionic charge | Molecular AO count |
|---|---:|---:|---:|---:|
| Rb | 37 | 28 | 9 | 13 |
| Cs | 55 | 46 | 9 | 13 |

The fixtures use the unmodified s/p orbital records, local f label and s/p/d
nonlocal channels. They cover neutral singlet RbH/CsH (10 explicit electrons)
and their singly charged doublet cations (9 explicit electrons), with direct
RHF/UHF on CPU/CUDA. The nuclei are placed off-axis; raw matrix and derivative
gates use two bond geometries per element. Qualification is limited to these
parameter records, states and geometries. It does not establish arbitrary
Rb/Cs chemistry, other LANL2DZ elements, other ECP families, spin-orbit physics,
or a relativistic method beyond the supplied scalar potential.

`tests/python/test_ecp_heavy.py` pins the combined orbital/ECP parameter hashes,
checks removed-core/effective-charge bookkeeping, and compares local and
nonlocal matrices separately with Libcint. Both grid levels, all-center
two-step finite differences, arbitrary nonsymmetric AO weights, complete HF
energies/forces, and budgeted geometry replay with complete-energy differences
must pass. The 13-AO fixtures fit the current CUDA resource inventory; their
tests check the allocation ledger against the declared budget.

Run the suite and record a compact endpoint report with:

```sh
python -m pytest tests/python/test_ecp_heavy.py -q
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp_heavy.py -q -k cuda
python tools/qualify_ecp_heavy.py --device cpu --output heavy-cpu.json
python tools/qualify_ecp_heavy.py --device cuda --output heavy-cuda.json
```

Use the pinned reference-test extra and the corresponding `VIBEQC_LIBRARY` as
described below. The report includes parameter identities, matrix/energy/force
errors, exact library and test hashes, planned bounds and the CUDA ledger.
Single-call timings are context for reproduction, not performance claims.
See the [qualification decision](../.agents/notes/implemented/numerics/2026-09-18-ecp-heavy-parameters.md)
for the boundaries and the retained evidence.

## Reproduction and parameter sources

Install the pinned `reference-test` extra (PySCF 2.14.0), build the CPU or CUDA
library and set `PYTHONPATH=python` and `VIBEQC_LIBRARY` to that exact artifact.
For example, build in Release mode with Ninja:

```sh
cmake -S . -B build-ecp-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build-ecp-cpu --parallel
cmake -S . -B build-ecp-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-ecp-cuda --parallel
```

Architecture 89 is the measured RTX 4090 target; select the actual allocated
GPU architecture for other devices. The measurements validate the generic
CUDA build and do not promote an AOT shell profile.

Run `pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py tests/python/test_ecp_f.py`; set
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
g rejection and continued rejection of f projectors through the C API.

`vibeqc_ecp_projector_tests` independently checks the emitted host arithmetic
using a double angular-node sum and the Legendre addition theorem, without
forming the production AO projections. It covers all radial powers 0..4,
local/s/p/d channels, signed mixed-exponent terms, center filtering, the
radial origin, and poisoned derivative slots in value-only mode. The Python
ECP suite also compares every radial power and d projectors with Libcint and
finite differences of an independently displaced ECP center on CPU/CUDA.

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
