# Higher angular momentum: initial g capability (BASIS02)

The initial higher-l path is CPU reference HF with orbital and auxiliary
shells through g (`l=4`), in Cartesian or libcint-ordered real-spherical AOs.
Complete RHF energy/analytic-force endpoints are independently checked for
a loaded g-containing HeH+ basis and for g auxiliary shells in DF-RHF.
These small diagnostic molecules establish implementation correctness; they
do not validate neutral transition-metal chemistry or large-basis performance.

## Execution matrix

| Operation | Orbital l | Auxiliary l | Derivative | Representation | Backend |
| --- | --- | --- | --- | --- | --- |
| S/T/V and conventional ERI/HF | 0–4 | — | Values, first nuclear | Cartesian, real spherical | CPU reference |
| DF metric / three-center integrals | 0–4 | 0–4 | Values, first nuclear | Cartesian, real spherical | CPU reference |
| Raw shell-tile post-HF integral source | 0–4 | 0–4 | Values only | Cartesian, real spherical | CPU reference |
| Explicit bounded S/T/V/DF component emitter | 0–4 | 0–4 for DF | Value and first nuclear | Unnormalized Cartesian primitive | CPU or CUDA scalar code |
| Existing native CUDA HF/DF and production tables | 0–3 | 0–3 | Values, first nuclear | Cartesian, real spherical | CUDA |
| AO/grid jets | 0–3 | — | Spatial orders 0–3 | Cartesian, real spherical | Existing CPU/CUDA routes |
| Second derivatives, weighted four-center GPU executor | 0–3 | Existing separate limits | Existing separate limits | Existing separate limits | Existing backends |

No DFT, h-or-higher, ECP or g native CUDA HF claim is made. Native CUDA
system creation and host packing reject g before fixed-table indexing. The
55-class s–f catalog, 64-bit masks, table inventory, production manifests,
profile identifiers and default compilation units retain their meanings.
Auxiliary shells are never silently dropped. Use `basis_capability` for the
actual backend/operator/derivative/role; mathematical data remains loadable
above the execution boundary.

## Mathematical and resource audit

* Cartesian monomials use descending x, then descending y powers: g has 15
  components. The bounded public component constructor rejects l>4 before
  narrowing to signed loop counters. Shell counts widen before arithmetic.
* g real harmonics are generated from the differentiated Legendre polynomial,
  expanding `(x+i*y)^m`, `z` and `(x*x+y*y+z*z)`. The Condon–Shortley phase is
  removed; imaginary parts supply negative m and real parts nonnegative m.
  The resulting polynomials are normalized in the exact Cartesian Gaussian
  moment metric. Output order is m=-4…4. The six-term m=0 polynomial is fully
  retained; CPU expansions use vectors. Existing s–f coefficients are untouched.
* Native shell-radial and component normalization already generalize to g.
  Signed contractions, all nine spherical functions, orthonormality, and
  mixed-center raw blocks are compared to PySCF/libcint without using VibeQC's
  transform to manufacture the reference.
* CPU Hermite/Coulomb workspaces depend on actual angular orders. g/g kinetic
  values require internal powers through six; g/g/g/g Coulomb values require
  Boys order 16, with order 17 for first derivatives. A positive Boys series
  handles low/intermediate T; upward recurrence starts only above the required
  order plus a margin. Derivatives use `dF_n/dT = -F_(n+1)`.
* CPU conventional ERIs evaluate one representative of each eightfold
  permutation orbit, copying values and physical-atom derivatives to the
  remaining positions. This reduces repeated high-l recurrence work. Full CPU
  HF remains a reference implementation with O(N^4) tensor storage; the resource
  planner derives its scratch bound from the supplied orbital/auxiliary l.
* The production DF Rys root and axis tables remain through f. Opt-in g DF
  component DAGs require `subset_wick`. No larger root family is guessed.

## Opt-in generated components

```python
from vibeqc_compiler.integral.one_electron_derivatives import build_one_electron_derivative_ir
from vibeqc_compiler.integral.bounded_component import emit_bounded_component
from vibeqc_compiler.integral.capabilities import query_integral_capability

ir = build_one_electron_derivative_ir("nuclear_attraction", (4, 4))
report = query_integral_capability(
    ir, backend="cuda_bounded_component", component_indices=(0,)
)
source = emit_bounded_component(ir, ("xxxx", "xxxx"), backend="cuda")
```

One compilation unit contains one explicitly selected component, capped at
30,000 reachable scalar DAG nodes. The CPU exports `evaluate(inputs, outputs)`;
CUDA emits a device function of the same name for a caller-owned launch. Input
exponents precede xyz coordinates of all mathematical operator centers. S/T/V
have two exponents (V includes an independent nuclear center); DF has one
exponent per Gaussian. Inputs must be finite with positive exponents. Outputs
are the value, then center-major xyz gradients, in atomic units. These are raw
unnormalized primitive results; callers supply normalization, contractions,
spherical transformations and arbitrary fixed cotangents. The IR's nuclear
charge is retained. This interface does not dispatch an HF calculation.

CUDA recomputes ancestors in separate per-output helpers while sharing Boys
values, limiting gradient liveness. Neither its scalar schedule nor any g class
is promoted into a production profile. Resource measurements are diagnostic
compile/numeric evidence, not molecular GPU performance claims.

## Reproduce validation

Build the CPU library normally and install the pinned `reference-test` extra
(PySCF 2.14.0). Set `PYTHONPATH=python:.` and `VIBEQC_LIBRARY` to the resulting
library. Use one BLAS/OpenMP thread for reproducible small-oracle timings.

```bash
# Ordinary tier: capabilities, full selected g S/T/V blocks and derivatives,
# signed-contracted raw ERI/DF blocks, and host-compiled scalar code.
python -m pytest -q tests/python/test_high_angular.py

# Complete loaded-basis RHF molecular energy/force evidence.
VIBEQC_HIGH_L_MOLECULAR_TEST=1 python -m pytest -q -s \
  tests/python/test_high_angular.py -k loaded_g

# Explicit allocated RTX 4090 tier (nvcc sm_89); logs compile time, source/
# binary size, registers, stack and spills. This is not compile-only testing.
VIBEQC_HIGH_L_CUDA_TEST=1 python -m pytest -q -s \
  tests/python/test_high_angular.py -k bounded
```

The g matrix is intentionally selected rather than exhaustively compiling all
g quartets in normal CI. The broader unchanged s–f tests and native suite remain
regression gates. Molecular fixture generation uses the actual BSE importer;
the independent oracle uses identical coordinates, charge, exponents,
contractions, representation and approximation.
