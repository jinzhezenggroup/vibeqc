# Compact XC contractions

`vibeqc_compiler.xc` now has one contraction owner between the audited scalar
functional and AO potential, response and explicit geometric consumers. This
implements A–C of #236 for real LDA/GGA on a fixed quadrature and screening
branch. The native CPU candidate is explicit; method capability registration,
stationary molecular forces, GPU XC and Hessians remain with #162/#163/#180.
The value-only Becke partition routine currently supplies no physical weight
Jacobian, so geometry returns independent source partials for that later adapter.

## Scientific contract and ownership

`IngredientContract`, `DerivativeRequest` and `DiscreteEnergyContract` identify
`E = sum_g w_g e_xc(z_g)`, where the scalar returns energy per volume. Density
and spatial AO jets are real. A total D splits as Da=Db=D/2; polarized output
keeps two functional-spin channels. Unpolarized output has one total-density
channel and requires equal spin matrices/directions. Potentials and response
use the full symmetric matrix trace, with both off-diagonal entries present.
`PairSpace` svec coordinates put sqrt(2) on each off-diagonal element.

| Term | Scientific owner | Independent check |
| --- | --- | --- |
| Scalar e, feature gradient/Hessian | Existing audited expressions + Graph AD | Pinned Libxc/formula blocks |
| rho, gradient, sigma, tau | `dft.features` | Native AO and independent orbital/PySCF fixtures |
| Compact potential/response coefficients | `xc.coefficients`, existing Graph AD/CSE | Direct bilinear factor and directional tests |
| AO matrix assembly | `xc.potential.assemble_coefficients` | Pinned independent PySCF E/V matrices |
| Both AO-leg geometric adjoints | `xc.coefficients.jet_pullback_program` | Displaced centers/points/weights, separately and together |
| CPU point lowering/cache | Existing ScalarCEmitter, CppCompilerAdapter, compile_runtime | Native/interpreter gates and standalone ASan/UBSan ABI harness |
| Solver actions | Existing `FixedDensityXCDerivativeKernel` adapter | Full-spin actions, restricted reduction and svec transpose |

The previous hand-written `potential_coefficients` and response coefficient
formulas are retired. Compatibility functions delegate to these common owners;
there is no second production contraction formula. The generic tau-half path
remains for synthetic tests: tau = 1/2 sum_k grad(phi) D grad(phi). It does not
add a meta-GGA. Cross-spin sigma is grad(rho_a) dot grad(rho_b), without an
extra factor two. Nonlinear XC follows complete-density reduction; splitting
D into contributions and summing their separate nonlinear energies is invalid.

For LDA, coefficient generation requests only rho and value AO jets. GGA adds
first jets and sigma, with no tau reduction. Response requests only the active
feature Hessian and includes both its action and the changing sigma Jacobian.
Geometry adds one AO derivative order. Unsupported observables, complex or
asymmetric D, unvalidated physical boundary domains and stale identities fail
explicitly. A structurally empty fixed AO mask contributes constant zero;
nonempty zero density retains the audited vacuum derivative error.

The old potential baseline already used compact panels. The new generator
canonicalizes the same patterns rather than claiming an invented symbolic
compression speedup. LDA assembly uses one matrix product per spin; GGA forms
one combined spatial panel and uses two. Reported coefficient SSA before/after
CSE is distinct from an AO-pair expansion, which is never constructed.

## Explicit native execution

```python
from pathlib import Path
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions

program = NativeContractionProgram(
    functional("PBE"), "response",
    compiler=CppCompilerAdapter(Path("c++")), cache=Path(".artifacts/xc-cache"),
)
with PreparedXCContractions(
    program, basis, grid, tile_points=64,
    resource_budget=ResourceBudget(host_bytes=64 << 20),
) as prepared:
    delta_v = prepared.execute(density, delta_density=direction)["response"]
```

AO evaluation and CPU BLAS are native; generated scalar/coefficient/jet-pullback
functions use checked contiguous point ABIs. NumPy orchestrates bounded panel
reductions and products. There is no production PySCF/Libxc callback. The common
compile cache hashes actual source/compiler/options/binary; endpoint metadata
also binds host source owners, basis, quadrature, functional and mask identity.

`spatial=PreparedSpatialGrid(...)` borrows an existing CPU fixed-mask adapter.
Its certificate must cover the requested jet order. Execution holds the CPU map
lock during internal work, with no yield to caller code; reconfiguration cannot
mix new points and old maps. The prepared XC owner does not close borrowed
basis/spatial objects. Each call recomputes from the supplied D.

The existing response solver adapter accepts `prepared=prepared` for a matching
native response owner. `apply_spin()` keeps functional-spin channels;
`apply()` and `apply_transpose()` retain the restricted total-D convention.
This does not register a new complete KS/CPKS method capability.

## Geometry and memory scope

`GeometryPartials.centers`, `.points` and `.weights` are independent gradients
at fixed D, exponents/coefficients, parameters, regularization and mask membership.
For an AO depending on r-R_A, point motion has positive spatial-jet sign and
its own center motion has the negative sign. The weight partial is unweighted
`e_xc`. `directional(centers=..., points=..., weights=...)` contracts all three
sources exactly once. A physical partition/grid adapter must supply its actual
motion. These partials exclude electronic response, Pulay and other energies;
force is minus the complete gradient, not this partial contribution alone.

The shared `ResourceBudget` preflight covers numeric capacities. For tile B,
N AOs, P points and A atoms, the XC workspace allowance is
`8*(20*N*N + 128*B*N + 256*B)` bytes, plus basis/grid storage,
`8*(16*P + 12*A)` for quadrature/output copies, and molecular setup scratch.
Borrowed spatial requests are composed into the same plan and may conservatively
double-charge inputs. The bound includes immutable copies and full spin D/V;
no global AO table, AO^4 tensor or solver-history tape is retained.

Python object headers/allocator rounding, BLAS workspace/thread stacks, native
scalar stack/code pages and caller-retained old results are explicitly outside
that numeric scope. It is not a whole-process memory cap. Benchmarks separately
record traced allocations and process RSS high-water observations, with their
measurement scopes; none is silently substituted for the numeric plan.

## Validation and reproduction

`test_xc_contractions*.py` extends pinned independent E/V checks with both spin
layouts, f Cartesian/spherical AOs, response at three finite-difference steps,
spin exchange, svec transpose, each geometry source and simultaneous motion,
minimal ingredients, stale masks and negative domains. Existing synthetic
bilinear tests retain tau-half/cross-spin/off-diagonal factor checks.

Run `tools/check_xc_native_sanitizer.py --output .artifacts/xc-sanitizer` to
compile 16 standalone CPU ASan/UBSan ABI harnesses. Its explicit instrumented
flags and source hashes are separate from the production compile cache.

`tools/benchmark_xc_contractions.py` records complete construction/execution,
scalar compilation, source/SSA size and liveness, contraction counts, numeric
plans and scoped observed memory. Independent reference work is outside timed
production calls. It requires a clean measured checkout and matching native
library. `tools/publish_xc_contractions.py` validates and publishes only durable
summaries/samples under the common evidence policy. See the retained evidence
README for exact commands and historical measured revisions. No significant
speedup or automatic production promotion follows from these measurements.
