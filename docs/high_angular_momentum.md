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
| Explicit bounded S/T/V/DF/full-range ERI component emitter | 0–4 | 0–4 for DF | Value and first nuclear | Unnormalized Cartesian primitive | CPU or CUDA scalar code |
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

Full-range four-center g components use a generic `ShellSignature` only in the
explicit bounded emitter. They are not assigned a legacy shell-class identity,
55-class mask bit, production manifest row, or default compilation unit. The
raw emitter accepts Cartesian primitive signatures; real-spherical transforms
and pulled-back weights stay caller-owned and a spherical raw signature fails
closed before code emission.

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
* On measured implementation commit `1c551ddb46bdf377fcae573eb5e9ca746af0d081`,
  the generic g-s-s-s four-center `xxxx` component was executed on an allocated
  RTX 4090 (compute capability 8.9, driver 550.163.01) with CUDA 12.9.86. The
  `sm_89` build took 2.074 s, produced 50,160 B of source and a 1,210,768 B
  shared object, and PTXAS reported 128 registers, a 48 B stack frame, zero
  spills, 0 B shared memory and 0 B local memory. Six selected high-l CUDA numerical gates passed, including the
  existing S/T/V/DF cases. This is resource and numerical qualification for the
  opt-in scalar route, not a complete CUDA molecular endpoint or production
  promotion.
  The final synchronized PR candidate `af5313627c178ab4a1cd791093dff8dc6e5f6c92`
  could not be re-executed on a GPU because current qz 4090/H200 notebook
  requests were unschedulable under the available node-memory/priority
  constraints. The measured `1c551ddb` implementation and final candidate are
  byte-identical in the bounded emitter, four-center component builder and
  high-angular test source; this is retained as source-equivalence evidence,
  not reported as a final-head rerun.
* The matching CPU qualification rebuilt the native library from the same
  source and ran the high-angular suite with molecular gates enabled: 38 passed
  and 6 CUDA-only cases skipped. Loaded orbital-g HeH+ RHF remained within
  6.15e-11 Hartree/Bohr of PySCF forces; g-auxiliary DF-RHF remained within
  1.92e-13 Hartree/Bohr. The detailed snapshot is in
  `benchmarks/results/high-angular-170-four-center/` and the catalog-retention
  rationale is recorded in the linked Agent Note.

## Opt-in generated components

```python
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
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
exponent per Gaussian. Full-range four-center ERIs use four exponents followed
by the four shell-center xyz coordinates. They are lowered from the same
ShellClassComponentKernel DAG used by CUDA shell codegen; range-separated ERIs
remain on their dedicated operator path. Inputs must be finite with positive
exponents. Outputs are the value, then center-major xyz gradients, in atomic units. These are raw
unnormalized primitive results; callers supply normalization, contractions,
spherical transformations and arbitrary fixed cotangents. The IR's nuclear
charge is retained. This interface does not dispatch an HF calculation.

CUDA recomputes ancestors in separate per-output helpers while sharing Boys
values, limiting gradient liveness. Neither its scalar schedule nor any g class
is promoted into a production profile. Resource measurements are diagnostic
compile/numeric evidence, not molecular GPU performance claims.

### Full-shell CPU first-derivative consumer

`compile_first_derivative_shell` partitions a complete Cartesian shell into
deterministic component tiles of at most 64 entries and compiles each tile with
the same compiler-owned S/T/V or four-center ERI DAG used by the bounded
lowering. `FirstDerivativeShellEvaluator` executes those tiles sequentially and
publishes the complete shell only after every tile succeeds. The numeric budget
therefore covers the complete output plus one live bounded tile rather than an
unbounded full-shell symbolic graph.

The four-center route keeps component normalization, primitive contraction and
physical atom scatter outside the generated recurrence. Its regression compares
the complete d-p-s-s shell, including a partial final tile, contracted primitives,
tight/diffuse exponents and nearly coincident centers against the independent
dynamic-`Jet` CPU oracle in `src/integrals/s_integrals.cpp`. This consumer does
not replace that oracle or promote generated CPU integrals into the default SCF
path; endpoint retirement still requires matched performance evidence.

## Reproduce validation

Build the CPU library normally and install the pinned `reference-test` extra
(PySCF 2.14.0). Set `PYTHONPATH=python:.` and `VIBEQC_LIBRARY` to the resulting
library. Use one BLAS/OpenMP thread for reproducible small-oracle timings.

```bash
# Ordinary tier: capabilities, full selected g S/T/V blocks and derivatives,
# signed-contracted raw ERI/DF blocks, and host-compiled scalar code.
python -m pytest -q tests/python/test_high_angular.py

# Shared-DAG CPU first derivatives, including bounded complete-shell ERI gates.
python -m pytest -q tests/python/test_first_derivatives_native.py

# Cold compile/source-size plus warm generated/oracle micro-timings.
python benchmarks/issue351_full_shell_cpu.py --samples 50

# Complete loaded-basis RHF molecular energy/force evidence.
VIBEQC_HIGH_L_MOLECULAR_TEST=1 python -m pytest -q -s \
  tests/python/test_high_angular.py -k loaded_g

# Explicit allocated RTX 4090 tier (nvcc sm_89); logs compile time, source/
# binary size, registers, stack and spills. This is not compile-only testing.
VIBEQC_HIGH_L_CUDA_ARCH=sm_89 VIBEQC_HIGH_L_CUDA_TEST=1 \
  python -m pytest -q -s tests/python/test_high_angular.py -k bounded
```

The g matrix is intentionally selected rather than exhaustively compiling all
g quartets in normal CI. The broader unchanged s–f tests and native suite remain
regression gates. Molecular fixture generation uses the actual BSE importer;
the independent oracle uses identical coordinates, charge, exponents,
contractions, representation and approximation.

Architecture rationale: [generic high-l four-center components without catalog
promotion](../.agents/notes/implemented/architecture/2026-09-22-generic-high-l-four-center.md).
