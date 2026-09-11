# Audited semilocal XC expressions (DFT02)

`vibeqc_compiler.xc` represents LDA exchange, PW92 correlation (ordinary and modified
parameters), PBE exchange and PBE correlation. `LDA_XC_PW` and `PBE` are exact
component sums. This is an explicit scientific tooling API, not a public DFT
method or a promoted kernel schedule. There is no Hartree, nuclear or exact
exchange energy in these outputs.

## Identity, source and licensing

`functional("PBE", spin="polarized")` returns a frozen `FunctionalSpec`. Its
identity includes exact rational composition, spin mode, version, ingredients,
derivative orders, feature layout, exchange/range-separation metadata, license,
source manifest and translated-expression hash. User-defined compositions use
`Fraction`, e.g. 3/4 PBE exchange plus PBE correlation with 1/4 exact-exchange
metadata; the last term is evaluated by a future nonlocal consumer. Unknown
names, operations and orders fail explicitly. Metadata never inserts another
energy term or infers composition from a name.

The mathematical source is Libxc **7.0.0**, under **MPL-2.0**. Exact upstream
Maple expressions, parameter-setting C sources, utility definitions, copyright
notices, license and SHA-256 URL manifest are in `external/libxc-7.0.0/`.
`expressions.py` is an MPL-2.0 translation, with that notice retained. The
formatter excludes these upstream files to preserve their audited bytes.
PBE correlation uses *modified* PW92 constants; ordinary `LDA_C_PW` retains the
original rounded constants. PBE uses beta=0.06672455060314922,
mu=beta*pi²/3, kappa=0.804 and gamma=(1-log(2))/pi². No runtime Libxc call or
Python autograd appears in device execution.

## Scalar and derivative conventions

All quantities use atomic units. The scalar is **energy per volume**
`e_xc = (rho_a + rho_b) epsilon_xc`, in hartree/bohr³. Each derivative acts on
`e_xc`, not `epsilon_xc`.

| Spin mode | Feature-major FP64 input, shape `[feature, point]` |
| --- | --- |
| polarized | rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b |
| unpolarized | rho, sigma, tau (total density, total gradient invariant, total kinetic density) |

`sigma_ab = grad(rho_a) dot grad(rho_b)` has **no factor two**. Total sigma is
`sigma_aa + 2 sigma_ab + sigma_bb`. `tau_s = 1/2 sum D_s,mu,nu grad(phi_mu) dot
grad(phi_nu)`. Tau is unused by this inventory; its derivatives are exact zeros
of a represented expression, not placeholders for an unsupported functional.
Unpolarized lowering substitutes `rho_s=rho/2`, `sigma_ss'=sigma/4`, `tau_s=tau/2`
*before differentiation*. `pack_grid_features` maps DFT01 data and requires equal
spin fields before accepting an unpolarized conversion.

`build_program(spec, order=2)` emits energy, all first partials, then the packed
upper triangular Hessian in lexicographic feature-index order. Explicit
`outputs=((), (0,), (2,3))` prunes differentiation and lowering to just those
roots. Directed mixed partials `(i,j)` and `(j,i)` can both be derived for an
independent symmetry check. `unpack` exposes only complete requested tensors.
Orders beyond two are unsupported.

For a downstream potential matrix, the spin-a gradient coefficient is
`G_a = 2 e_sigma_aa grad(rho_a) + e_sigma_ab grad(rho_b)`; exchange a and b for
spin b. Differentiating density features with respect to each density-matrix
entry gives the quadrature integrand

```text
V_s,mu,nu = e_rho_s phi_mu phi_nu
           + G_s dot [grad(phi_mu) phi_nu + phi_mu grad(phi_nu)]
           + (e_tau_s/2) grad(phi_mu) dot grad(phi_nu).
```

The caller multiplies by grid weights once. There is no further cross-spin
factor and no factor for symmetrizing the already symmetric AO bilinear.
`potential_coefficients` supplies these factors; tests differentiate an actual
AO-density contraction, adding a linear tau term to exercise the half factor.
For unpolarized inputs, `G = 2 e_sigma grad(rho)`.

## Finite domain: `libxc-7.0.0/interior-v1`

There is **no clipping or smoothing**. The supported positive-density domain is
`1e-12 <= rho_a+rho_b <= 1e12`, with each spin fraction at least `1e-10` and
`|grad(rho_s)|/rho_s^(4/3) <= 1e6` for GGA. Density, same-spin sigma and tau must
be nonnegative, finite and real. Sigma must be a physical Gram matrix:
`|sigma_ab| <= sqrt(sigma_aa sigma_bb)` (16 machine epsilons of dot-product
roundoff are accepted without modifying the supplied values).

Exact all-zero vacuum features have zero *energy*. Vacuum derivatives, exact
full polarization and inputs outside these bounds raise `UnsupportedXC`.
This deliberately finite contract is insufficient to evaluate arbitrary
molecular-grid tails without a future, separately versioned limit policy.
The API must not silently discard such tiles. Domain tests cover both sides of
support boundaries, zero/small sigma, extreme exponents and near polarization.
There is no regularization branch whose derivative is omitted.

Squared reduced gradients avoid singular `sqrt(sigma)` differentiation at zero.
PW uses `log1p`; PBE uses `expm1` and `log1p`. Stable primitive nodes survive both
shared algebra rebuild passes. Fractions retain exact source coefficients;
transcendental constants and fractional exponents lower to FP64. Both orders
of optimization/differentiation are checked against the ordinary interpreter.

## Execution, budgets and capability stages

The CPU `program.evaluate(features)` is an interpretable array-DAG diagnostic
path. Generated CUDA uses the existing `CudaEmitter`, structural CSE,
materialization planning and small-integer-power lowering. Variants are:

- `baseline`: one output per kernel;
- `fused`: all requested outputs per kernel;
- `split`: bounded groups (default eight) that reduce derivative live ranges.

All use FP64 and `--fmad=false`, with explicit 64/128/256-thread candidates.
`compile_cuda` reuses `CudaCompilerAdapter`, `compile_runtime`, release resource
parsing and binary/source/header/toolchain hashing. Its contract records the
functional/expression, ordered feature/output sets, layout, groups, threads and
all emitter/runtime source hashes. Compilation does not execute a GPU.
`CudaXC` binds that contract to a checked binary and owns a private persistent
arena, stream and events. It uploads one bounded input tile, runs native
kernels, downloads all requested outputs and reports cumulative device timings.
No execution allocates device storage or falls back to CPU arithmetic.

`plan_tiles` records tile size, host scratch capacity, device IO/error capacity
and budget. The capacity check happens before CUDA preparation. Caller input,
Python object overhead and CUDA module/driver storage are outside the numeric
capacity budget; preparation and observed device deltas are reported separately.
The shared header links cuBLAS, but this runtime creates no cuBLAS handle,
provider allocation or workspace. Empty and partial tiles, repeated execution,
changed input, failures, closure and concurrent independent plans are tested.

`query_capability` separates represented/emitted/compiled/validated/promoted.
Validation requires the shared evidence schema and exact generated/binary
identity. No candidate is automatically promoted. The PBE Hessian's fully fused
candidate spills on the measured RTX 5090; separate and grouped outputs provide
spill-free alternatives. Eight-output PBE correlation still spills, so the
full matrix does not assume a universal grouping. Resource success alone is not
a performance result.

## Reproducing independent evidence

```bash
# Independent generation; requires pinned PySCF 2.14.0 and Libxc 7.0.0.
python tools/generate_xc_references.py tests/data/xc
PYTHONPATH=python:. python -m pytest tests/python/test_xc_expressions.py -q
python tools/validate_xc.py --tier cpu --variants baseline --output build/xc-cpu
python tools/validate_xc.py --tier cuda-compile --output build/xc-compile
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env PYTHONPATH=python:. VIBEQC_XC_CUDA_TEST=1 \
  python -m pytest tests/python/test_xc_cuda.py -q
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/validate_xc.py --tier endpoint \
  --functionals LDA_XC_PW PBE --output build/xc-endpoints
```

Fixtures start from Cartesian spin-gradient vectors, ensuring physical sigma.
Every typical-domain derivative uses the existing `atol=1e-11, rtol=1e-10` gate.
Named boundary fixtures use `atol=1e-8, rtol=2e-6`: very large Hessians and Libxc's
near-polarization subtraction need a relative allowance. Raw Libxc outputs are
always preserved. In extreme spin/gradient combinations Libxc's PBE exchange
`d²e/(d rho_b d sigma_aa)` can be about 1e9 although spin separability requires
exact zero. Boundary exchange is therefore additionally checked with a separate
closed-form energy/gradient/Hessian oracle; combined PBE uses that exchange plus
unmodified Libxc correlation. The diagnostic reports expose both comparisons.
Typical-domain data use Libxc directly for every element.

Additional gates check directional energy and gradient finite differences at
three steps, independently derived Hessian symmetry, spin exchange, restricted
chain rules, source hashes and matrix factors. The CLI archives deterministic
source, contracts, compiler logs, raw per-candidate JSON, schema-validated
numerical stages, memory capacities and interleaved timings. Its endpoint tier
measures the **fixed-feature upload -> XC -> download -> consumption** workflow,
including changed inputs; it makes no molecular SCF or force speedup claim and
cannot promote a public DFT method. Local cache/tuning CLI ownership stays with
CG09/#136; this module supplies explicit candidate inputs and measurements.
