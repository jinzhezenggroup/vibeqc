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
layer. Supported orbitals are s/p/d; projectors are s/p/d, local labels at
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
The projector/quadrature loops remain a native baseline, not a complete
generated projector lowering. ECP-center derivatives use `dC = -(dA+dB)`
before physical atom accumulation, including coincident A/B/C cases.

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

Direct RHF/UHF are the supported complete methods. ECP density fitting is
explicitly rejected pending its own complete force/budget gates. Complete
DFT SCF/gradients remain dependent on #162/#163, so DFT/ECP completion is not
claimed. No broad heavy-element validation follows from support for the
parameter format.

## Reproduction and parameter sources

Install the pinned `reference-test` extra (PySCF 2.14.0), build the CPU or CUDA
library and set `PYTHONPATH=python` and `VIBEQC_LIBRARY` to that exact artifact.
The measured builds used Release mode and Ninja:

```sh
cmake -S . -B build-ecp-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build-ecp-cpu --parallel
cmake -S . -B build-ecp-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-ecp-cuda --parallel
```

Architecture 89 is the measured RTX 4090 target; select the actual allocated
GPU architecture for other devices. The measurements validate the generic
CUDA build and do not promote an AOT shell profile.

Run `pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py`; set
`VIBEQC_ECP_CUDA_TEST=1` only on an allocated GPU. The tests use installed
PySCF LANL2DZ Na ECP/orbitals with STO-3G H, asymmetric mixed centers, d shells,
Cartesian/spherical layouts, charged open-shell HF, independent ECP center
motion and two finite-difference steps. Synthetic fixtures isolate local and
nonlocal components. Production code never imports PySCF.

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
