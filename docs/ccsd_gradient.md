# Complete small-system CPU RCCSD gradients

`tools.vibeqc_cc.complete_gradient_validation` connects native RHF, converged
RCCSD, generated Lambda, raw-Hamiltonian and orbital response, and native
analytic integral derivatives. It returns a **complete conventional CCSD energy
gradient for the supported small-system validation scope**, not just fixed-
orbital correlation weights. The public `Calculator`/native method force flags
are unchanged. Perturbative triples are not differentiated by this endpoint.

## Supported scope

The endpoint requires a native CPU source, a real closed-shell all-electron
conventional unscreened RHF reference with occupied and virtual orbitals, no
frozen orbitals, no auxiliary basis/DF, and **at most 12 AOs/MOs**. Cartesian
and real-spherical representations are tested separately, including an explicit
d shell. A Cartesian d shell and a spherical d shell span different spaces;
the tests compare each with its matching independent reference, not with an
assumed identical energy.

The current HF snapshot export performs disclosed host canonicalization. MO
integrals and cotangents, AO cotangents, and the native coordinate-major
integral derivative arrays are dense. These existing small-system interfaces
are deliberately used to validate the complete scientific chain. This is not
a bounded production GPU implementation or a full-molecule memory-performance
promotion. No PySCF, Torch or CuPy calculation is invoked by the endpoint.

## Run it

From a checkout with a matching CPU native library:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DVIBEQC_ENABLE_CUDA=OFF -DVIBEQC_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j2
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build-cpu/libvibeqc.so \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/run_ccsd_gradient.py --case h2o --output water-gradient.json
```

The driver writes a new JSON file and refuses to overwrite an existing one.
Omit `--output` to print JSON. `--input` accepts the complete explicit molecular
schema used by `tools.cc_gradient_fixtures.inputs`; coordinates must be in
Bohr. Unsupported spin, ECP, triples, frozen-core, auxiliary or unit conventions
are rejected instead of silently ignored. Numerical solving uses the strict
endpoint options, not tolerance suggestions stored with reference metadata.

```python
from tools.vibeqc_cc import CCSDGradientOptions, complete_gradient_validation
from tools.vibeqc_posthf.sources import NativeSource

with NativeSource(
    atoms=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
    basis="sto-3g",
) as source:
    result = complete_gradient_validation(source, options=CCSDGradientOptions())
    gradient = result.gradient  # dE/dR, Eh/bohr
    forces = result.forces      # -dE/dR, Eh/bohr
```

The source is borrowed and remains open. The convenience endpoint owns/closes
its temporary provider; its returned arrays are detached and immutable. For an
already qualified CC/Lambda state, `BoundCCSDGradient(response, provider)`
reuses `BoundCCSDResponse`, binds the native source and exposes checked MO/AO
weights and `gradient()`. The provider/source must stay current and open.

## Mathematical chain

Let `h` and `g` be raw full-MO one-electron and chemists'-notation two-electron
integrals. With orbital transform `U` and reference occupation matrix `P`, the
primal TensorIR specifies

```text
h(U) = U.T h U
g(U) = U^4 g                       (one U per orbital index)
P = diag(2 occupied, 0 virtual)
F(U) = h(U) + J[g(U),P] - K[g(U),P]/2
E_HF,electronic(U) = <P, h(U)+F(U)>/2
```

All ten Fock/integral input fields of the existing RCCSD equations are slices
of these common tensors. Their reverse derivative therefore includes the
normal-ordering chain and accumulates overlapping/permuted ERI fields once.
It does not add guessed spin or packed-RDM factors. The HF electronic energy
is another output of the same primal and receives a unit cotangent exactly
once. Nuclear repulsion remains separate.

The qualified Lambda result supplies the dense-Frobenius correlation weights
for `E_corr + <lambda,R>`. Differentiating the raw Hamiltonian/orbital map with
those weights gives the orbital gradient. The shared rotation convention is

```text
K[i,a] = x[i,a], K[a,i] = -x[i,a]
G = partial L / partial U at U=I
partial L_corr / partial x = G_ov - G_vo.T
dF_ov(exp(K))/dx = -A
A.T z = -partial L_corr/partial x
L_total = E_HF + E_corr + <lambda,R> - <z,F_ov>
```

`A` is the existing `RHFResponseOperator` using native shell-streamed J/K.
`checked_transpose_solve` and the shared GMRES own the numerical solve. The
separately generated Fock JVP supplies a tiny explicit **orbital** matrix for
curvature, action parity and independent final-residual checks. It is not a
T2-by-T2 or full CC Jacobian. A negative/near-zero orbital curvature is rejected,
including when a small RHS would otherwise make the solve appear trivial.

Occupied-occupied and virtual-virtual directions are redundant for the
amplitude-relaxed full CCSD Lagrangian. Their antisymmetric derivatives are
checked directly. They are not divided by potentially degenerate same-space
orbital gaps. After adding Z and HF, the full `G-G.T` must pass stationarity.

For a metric perturbation, symmetric orbital transport is `dU=-dS_MO/2`. The
same generated graph constructs the overlap multiplier

```text
W_S = -(G+G.T)/4
```

This includes correlated and orbital terms; it is not merely the HF energy-
weighted density. The generated `h/g` cotangents and `W_S` are back-transformed
to AO space with staged TensorIR contractions. The endpoint contracts

```text
dE/dR = <W_h, dh/dR> + <W_g, dg/dR> + <W_S, dS/dR> + dE_nuc/dR
```

using the existing native analytic derivative interface. It does not evaluate
nuclear energy differences in the gradient runtime or project away net force.
No physical RDM export convention is inferred from these derivative weights.

The new `orbital` TensorIR index-space kind explicitly represents the complete
MO population, retaining occupied/virtual subranges. It is distinct from AO,
occupied and virtual spaces even when their numerical extents coincide.

## Acceptance and failure behavior

SCF, CC, Lambda and Z have separate physical residual gates. State/reference,
Hamiltonian and generation identities, generated output contracts, finite FP64
values, source lifetime and current provider ownership are checked at their
boundaries. Raw h/g-derived fields must reproduce the original CC inputs and
HF energy. The total generated weights must equal the HF, correlation and Z
component sum; full orbital stationarity is independently required.

`CCSDGradientResult` reports energies, immutable gradient/forces, all four
residuals, orbital stationarity, smallest orbital curvature, exact identities,
stage timing and resource boundaries. `physical_components` separates HF,
correlation, orbital-response and nuclear contributions;
`integral_components` separates hcore, ERI, overlap and nuclear contractions.
Both must sum to the same unprojected complete gradient. A failed gate raises;
no successful partial gradient is returned.

## Resource boundary

The endpoint admits a conservative simultaneous logical-numeric reservation
before its new full-MO integral read. It includes the bound response reservation,
retained generated outputs and independent checks, orbital matrix/GMRES work,
AO weight transforms and returned native derivative storage. The initial
native-derivative output gate is also checked before fresh HF for obviously
insufficient budgets. The provider and shared solver retain their own limits.

The original HF allocation, provider cache/internals, dense native derivative
evaluator scratch, Python graph/object storage and opaque NumPy/BLAS workspaces
remain separate or excluded. The logical reservation is not a total native or
process-RSS guarantee. Raising it does not expand the <=12-AO qualification.
The streamed J/K backend reads whole local shells to avoid redundant partial
shell work; this does not convert the dense gradient chain into a tiled one.

## Reproduce scientific validation

```bash
PYTHONPATH=python:. VIBEQC_LIBRARY=$PWD/build-cpu/libvibeqc.so \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_cc_complete_gradient.py
```

Retained independent PySCF 2.14.0 reference JSONs cover H2, changed-geometry H2,
H2O, NH3, CH4 and H2 with an added Cartesian/spherical d shell. They include
explicit input/settings, source and content identities. Regeneration is an
optional external-oracle task, separate from the runtime:

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/generate_ccsd_gradient_references.py --output /tmp/new-cc-oracles
```

The generator requires exactly PySCF 2.14.0. It tightens only numerical CPHF
controls, not the independent equations, and records the resulting difference
from default CPHF controls without assuming that those settings cause any
observed disagreement.

Tests compare complete analytic gradients with these references at <=1e-6
Eh/bohr and check three-step nuclear differences with fresh HF and CC at every
displaced geometry. Separate tests exercise the generated raw h/g/U chain,
AO duality, shared/generated/independent orbital matrices, finite rotations,
translation/rotation covariance, changed geometry, immutable results and
rejection of stale, failed, nonfinite and under-budget requests. Negative tests
remove overlap or orbital response and require a visible error. Public force
capabilities and (T) gradient support remain outside this validation endpoint.

See the [complete-gradient decision](../.agents/notes/implemented/numerics/2026-09-19-complete-cpu-ccsd-gradient.md)
for the dense validation boundary and migration conditions.
