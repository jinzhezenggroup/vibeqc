# Analytic HF Hessians (issue #180, slice A)

This document is the second-derivative dependency graph requested by step 1 of
issue #180: it maps every term of the RHF energy to the Hessian contribution it
produces, and records which layer supplies that term. It is deliberately
written before any assembly code exists, so the term list and the sign
conventions are fixed in one place instead of being reverse-engineered from
three different providers later.

## Current implementation status

The native CPU tools path now consumes a VibeQC RHF snapshot and generated
first/second integral derivatives, with native J/K response. PySCF is confined
to external comparison oracles. The qualified small-system domain and remaining
#180 production/GPU/HVP work are described below; a tiny analytic Hessian does
not close the whole Hessian/HVP roadmap or expose a public Calculator API.

## Scope of this slice

Slice A targets a **tiny dense analytic RHF Hessian** with component checks, on
systems small enough for the existing reference exporter (`nbf <= 12`). It is
CPU-only.

Explicitly outside this slice, and left fail-closed rather than approximated:

- **DFT** (slice C) — needs the complete LDA/GGA nuclear gradients from #163 and
  #161's derivative kernels;
- **HVP and bounded full-Hessian execution** (slice B);
- **DF, ECP, range-separated and meta-GGA Hessians** — each needs its own
  complete second-derivative/response chain and is *not* inherited from energy
  or first-force support;
- **CUDA execution** and any performance claim.

## What the providers do and do not supply

The two upstream layers are needed here, and both stop short of a molecular
Hessian. Keeping that boundary explicit is the point of this section.

**#178 — second integral derivatives**
(`python/vibeqc_compiler/integral/second_derivatives.py`,
`docs/second_integral_derivatives.md`)

Supplies S/T/V and four-center Coulomb second derivative **integral primitives**
with a fixed external weight, in `raw_hessian`, `weighted_hessian` and
`weighted_hvp` forms. Its own scope note states that external weights,
directions and primitive parameters are fixed, that the provider is unscreened,
and that **electronic response and molecular Hessian assembly are excluded**.

The caller therefore owns, and must fold itself:

- the density and energy-weighted-density values that become the fixed weights;
- every orbit/multiplicity factor — a per-tile weight is not restricted to a
  triangular domain and the provider applies no HF density formula;
- all signs and prefactors (the output name never changes a sign);
- basis-representation conversion for spherical inputs.

**#179 — shared orbital response**
(`tools/vibeqc_response/`, `docs/response.md`)

Supplies the matrix-free CPHF operator (`RHFResponseOperator`), a true-residual
Krylov solver with multi-RHS strategies, and two independent oracles
(`explicit_rhf_response_matrix`, `finite_rotation_jvp`). It constructs **no
right-hand side**: `docs/response.md` assigns nuclear-perturbation RHS
construction to the caller. This work reuses that operator and does not add a
second response solver.

The shared response layer now also has an explicit direct-CUDA J/K adapter,
`CudaDirectJKBackend`; see [response backend boundaries](response.md#backend-boundary).
It does not change this document's CPU-only molecular Hessian scope. Its
AO/MO transforms and Krylov solve remain host-orchestrated, and nuclear
directional RHS, complete HVP assembly and device-resident execution are
separate #180 integration gates.


**#141 / #144 — first derivatives**, used to build the response RHS:

- one-electron raw tensors, `build_one_electron_derivative_ir(..., weighted=False)`
  yields a `RawBlock` with layout `(center, xyz, *tensor_indices)`;
- four-center ERI first derivatives exist only as a **weighted contraction**
  (`build_weighted_eri_ir`), not as a raw tensor.

## Term map

Writing the closed-shell RHF energy as

```text
E = E_nuc + Tr[P h] + ½ Tr[P G(P)]
G(P)_μν = Σ_λσ P_λσ [ (μν|λσ) - ½ (μλ|νσ) ]
P = 2 C_occ C_occᵀ
W = 2 Σ_i ε_i C_μi C_νi      (energy-weighted density)
```

the Hessian `H_{R R'} = ∂²E/∂R∂R'` splits into a **skeleton** part, taken with
the density and the orbital coefficients held fixed, and a **relaxation** part
carried by the orbital response.

### The two-electron weight, derived rather than inspected

Expanding the two-electron energy once gives

```text
E_2 = ½ Σ_{μνλσ} P_μν P_λσ (μν|λσ)  −  ¼ Σ_{μνλσ} P_μν P_λσ (μλ|νσ).
```

Renaming the dummy indices in the exchange sum so that its ``(ac|bd)`` becomes
``(μν|λσ)`` turns ``P_ab P_cd`` into ``P_μλ P_νσ``, which puts both terms over
the same integral:

```text
E_2 = Σ_{μνλσ} [ ½ P_μν P_λσ − ¼ P_μλ P_νσ ] (μν|λσ)  =  Σ_{μνλσ} W2_μνλσ (μν|λσ).
```

``W2`` is a **four-index** integral weight and is a different object from the
two-index energy-weighted density ``W`` defined above; the two are deliberately
given distinct names so a reader never has to infer which one appears in a
formula. The same distinction applies to the weight slot in the table below,
which is written out rather than abbreviated.

**The outer ``½`` is already inside ``W2``.** Writing the skeleton as
``½ Σ W2 ∂²(μν|λσ)`` on top of this ``W2`` would halve both the Coulomb and the
exchange contribution — the row below therefore carries no further factor. The
folding is implemented as ``two_electron_weight`` in
:mod:`tools.vibeqc_hessian.weights` and checked against a direct
``½ Tr[P G(P)]`` evaluation, so the factor is verified where it is consumed
rather than only asserted here.

| # | Term | Form | Supplier |
|---|---|---|---|
| 1 | Nuclear repulsion | `∂²E_nuc/∂R∂R'` | caller, closed form |
| 2 | One-electron skeleton | `Tr[P ∂²h/∂R∂R']` | #178 `build_one_electron_second_ir`, families `overlap`/`kinetic`/`nuclear_attraction`, `weighted_hessian`, provider weight = `P` |
| 3 | Overlap (Pulay) skeleton | `-Tr[W ∂²S/∂R∂R']` | same provider, `weighted_hessian`, provider weight = `-W` (negated **energy-weighted density**) |
| 4 | Two-electron skeleton | `Σ_{μνλσ} W2_μνλσ ∂²(μν|λσ)/∂R∂R'` with `W2_μνλσ = ½ P_μν P_λσ - ¼ P_μλ P_νσ` (no further factor; see the derivation above) | #178 `build_eri_second_ir`, `weighted_hessian`, provider weight = `W2` |
| 5 | Nuclear-perturbation RHS | `b[i,a]` includes the frozen-P Fock derivative, overlap metric connection, and its induced density/Fock response (defined below) | caller: #141 raw `∂h/∂R` and `∂S/∂R`, #144 weighted ERI first derivatives, and the shared J/K backend |
| 6 | Orbital response | solve `A u = -b` | #179 `RHFResponseOperator` + `solve_many` |
| 7 | Relaxation contribution | `u` combined with first derivatives of h, S, and the ERIs | caller |

Terms 2–4 are the "skeleton": they use the *second* derivatives of the
integrals with the density frozen, and they are exactly the shape the #178
provider emits. Terms 5–7 are the "relaxation": they exist because the
coefficients depend on the geometry, and they are what `docs/response.md` hands
to its callers.

**Component separation is a deliverable, not a debugging aid.** Terms 1–4 and
terms 5–7 are accumulated and reported separately, so a missing contribution
shows up as an isolated component error instead of a plausible-looking total.

### The moving AO metric in the nuclear RHS

Nuclear displacements change the AO overlap. The frozen-density derivative
`Cᵀ [h^R + G^R(P)] C` alone is therefore not the CPHF right-hand side.
One explicit convention compatible with #179 is the symmetric metric gauge:
write `C^R = C[-½ S_R + X]`, where `S_R = Cᵀ (∂S/∂R) C`,
`X[a,i] = x[i,a]`, and `X[i,a] = -x[i,a]`. With the MO occupation matrix
`D = diag(2_occ, 0_virt)`, the known metric density response and RHS are

```text
P_metric^R = -½ C (S_R D + D S_R) Cᵀ
b[i,a] = { Cᵀ [h^R + G^R(P) + G(P_metric^R)] C }_ai
         - ½ (ε_a + ε_i) (S_R)_ai
A x = -b
```

Here `G^R(P)` differentiates the integrals at fixed AO density, whereas
`G(P_metric^R)` applies the ordinary J/K map to the known density connection.
The response operator supplies only the remaining rotation-induced density
response. The relaxation assembly must also retain the known metric terms;
neither the overlap second derivative in term 3 nor the rotation solution alone
replaces them. A2 must check this complete RHS and metric contribution against
displaced references before claiming an analytic Hessian.

## Conventions to pin down, and how

Three layers meet here with independently chosen conventions, which is the
highest-risk part of this work:

- **#178** carries `output_sign` on the consumer and `sign` / `prefactor` on the
  weight descriptor. The output name alone never changes a sign.
- **#179** stores vectors in occupied-major/virtual-minor order `x[i,a]`.
  The linear solver receives `-b` for the `A x = -b` convention above; the
  sign is applied once at the caller boundary.
- **W** above is the energy-weighted density in the same doubled-occupation
  convention as `P = 2 C_occ C_occᵀ`.

Two further conventions are fixed by position, not by symbol, and must be
carried through assembly explicitly:

- **Hessian axes.** #178 emits `(center_row, xyz_row, center_column, xyz_column)`
  in requested mathematical-center order, *not* physical-atom order. The
  center→atom mapping (which also handles several mathematical centers sharing
  one atom) is applied through the caller's chain rule, and translation recovery
  is already performed inside the generated kernel.
- **Sign of a reported force.** Native forces are `-dE/dR`. The numerical oracle
  below compares against the *gradient*, so the conversion happens once, at a
  named boundary.

These are pinned by component-wise finite-difference checks rather than by
overall numerical agreement, because a wrong sign or factor in one term can
otherwise cancel against another.

## Nuclear response RHS

The next A2 boundary is ``build_rhf_nuclear_rhs``. It consumes the MO forms
of the frozen Fock derivative, overlap derivative, and the Fock response to
the known metric density connection, then returns ``b[i, a]`` in the
occupied-major/virtual-minor layout required by #179. The metric Fock input is
mandatory, and the caller passes ``-b.reshape(-1)`` to solve ``A x = -b``.
``metric_density_response_mo`` exposes the corresponding
``-1/2 (S_R D + D S_R)`` connection with closed-shell occupations.

This keeps the orbital-response solve and AO integral derivative construction
outside the helper; omitting either the metric density/Fock term or the
energy-gap overlap term is rejected by the component tests rather than hidden
inside a default.

## Frozen skeleton assembly

The A2 assembly boundary is now represented by
``tools.vibeqc_hessian.assemble_frozen_skeleton``. It accepts the nuclear,
one-electron, overlap/Pulay, and folded two-electron second-derivative
components in the canonical ``(atom, xyz, atom, xyz)`` layout, validates that
they are finite and shape-compatible, and returns each component alongside
their raw sum. The overlap component must already carry its negative Pulay
sign, and the two-electron component must already contain the density-folded
``1/2`` and ``1/4`` factors; the assembler applies no hidden prefactors.

The result reports ``includes_response: false``. Orbital-response RHSs and
relaxation terms remain a separate implementation boundary for the complete
analytic Hessian and are not substituted with zero arrays.

## Numerical oracle

Step 2 supplies a finite-difference-of-analytic-gradient Hessian: central
differences of the *analytic energy gradient* over nuclear coordinates, at
several step sizes, with no best-step selection. It is clearly labelled a
numerical Hessian and is an oracle and early utility — it does not constitute
analytic Hessian support.

It is independent of the assembly in step 4 in the useful direction: it depends
only on the analytic first derivatives, so an error shared between the skeleton
and relaxation assembly cannot hide from it.

## Failure behaviour

Unsupported requests fail explicitly rather than substituting a lower-level
result. A full-Hessian request that cannot be expressed within the provider's
tile bounds is rejected; an unimplemented DF/ECP/range-separated/meta-GGA
Hessian is reported as unsupported rather than silently answered with an
HF or energy-only quantity.

## Independent semi-numerical reference

`tools.vibeqc_hessian.reference` supplies a tiny CPU oracle with an independent
dense CPHF solve and finite-difference first/second integral derivatives. It
requires optional PySCF, all-electron closed-shell RHF, Cartesian AOs, at most
18 AOs and four atoms, and a nonzero occupied/virtual gap. Invalid steps,
unsupported molecules, unconverged SCF references and failed response solves
raise errors. It is imported explicitly; importing the Hessian weight/RHS
helpers does not require PySCF.

This reference does **not** complete slice-A step 4 or qualify the analytic
provider chain. That integration must consume #178 generated second-integral
blocks and #179's shared response operator/solver. The dense reference stays
independent so it can test that future implementation. The occupied CPHF block
is fixed by the metric gauge, and its induced density contributes to the virtual
response. `_first_order_mo1_e1_vir_only` implements the equivalent reduced
(nvir, nocc) solve: the known occupied metric response is eliminated into the
right-hand side as `b_v - F_vo b_o`. Independent dense full/reduced regressions
compare orbital response, occupied-energy response and the assembled Hessian
for H2, water and the multi-virtual d-shell fixture. These tests establish the
need to include the occupied metric contribution, not a need to iterate the
redundant full response space.

`System.derive()` differences fresh-molecule integrals. `hessian_components()`
returns nuclear, core, overlap/Pulay, two-electron and relaxation contributions.
With energy-weighted density `W = 2 sum_i eps_i C_i C_i^T`, the full-coordinate
Pulay skeleton is `-Tr[W S_RR]`; the complete coordinate derivative already
includes both AO slots. `hessian_total()` evaluates both atom orders without
symmetrizing the result. Its layout is `(atom, atom, xyz, xyz)` (PySCF convention),
and its units are Eh/Bohr². Transpose to `(atom, xyz, atom, xyz)` before using the
existing skeleton/diagnostic helpers.

Reproduce the reference gates with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python tools/hessian_examples.py --case h2,water --output /tmp/hessian-reference.json
```

The driver retains the full basis/geometry and source hashes, compares against
PySCF's analytic Hessian (5e-6 absolute tolerance), total-energy differences
(5e-4), and all three analytic-gradient difference steps (2e-4 each). It also
gates raw symmetry, translation, component sums and the omitted-relaxation
negative case; any failed gate produces a nonzero exit status. H2 uses STO-3G;
water uses a **custom 12-AO O(s,p,d) + H/STO-3G stress basis**, with five occupied
and seven virtual orbitals. The tests additionally cover genuine 7-AO water
STO-3G. No complete-method performance or generated-provider claim is made.

See the [reference-boundary rationale](../.agents/notes/implemented/numerics/2026-09-17-hessian-reference-boundary.md).

## Bounded native CPU analytic RHF integration

`NativeRHFState` and `analytic_hessian` provide a native-input tools integration
for **at most 12 Cartesian AOs and four atoms**, with a closed-shell, all-electron,
conventional unscreened RHF reference. The supported integral primitives cover
s/p/d/f; end-to-end default tests qualify H2 and STO-3G water, while the 12-AO
custom d-shell case is an explicitly requested slow test. This is not a public
production-size Hessian, CUDA Hessian, DFT/DF/ECP/UHF Hessian, or molecular HVP
capability. Those remain separate #180 acceptance items.

### State and derivative ownership

The calculation side requires no PySCF installation and imports no Hessian
reference oracle. The chain is:

1. `NativeSource` owns the native geometry/basis and integral source.
   `export_rhf` runs the existing native CPU RHF solver and exports a checked
   immutable `ReferenceSnapshot`. Its existing small-system bridge canonicalizes
   the final Fock with NumPy on the CPU and records the measured physical/density
   residuals; it is not a GPU-resident or generated SCF implementation.
2. `NativeRHFState` binds that same snapshot to its live source. Geometry, basis,
   representation, Hamiltonian, electron count, dimension, and source lifetime
   are checked. Hessian helpers do not rerun SCF or manufacture convergence data.
3. `first_order.generated_first_order` obtains S/T/V and ERI first derivatives
   from the existing compiler DAGs. The explicit CPU first-component adapter
   emits bounded Cartesian component subsets and streams primitive contractions
   through the native runtime template. It introduces no new integral recurrence.
4. ERI first derivatives are immediately contracted with the fixed reference
   density into the frozen-Fock perturbation. Nuclear-attraction operator-center
   motion and all basis-center motions are accumulated onto physical atoms.
   No molecular `3N * NAO^4` first-derivative tensor is retained.
5. `build_rhf_nuclear_rhs` constructs the symmetric-gauge RHS, including the
   known metric-density Fock term. #179 `RHFResponseOperator` / true-residual
   GMRES uses `NativeJKBackend`, not the dense AO response oracle.
6. The explicit second-derivative skeleton uses #178 generated providers.
   Two-electron energy weights are folded per shell quartet rather than stored
   as a molecular four-index tensor. Nuclear repulsion is closed-form.
   Relaxation evaluates every ordered atom/axis pair independently; raw symmetry
   is checked without copying one triangle onto the other.

The known occupied response is `U_ij = -S_ij/2`; the virtual response is
`U_ai = x_ia.T - S_ai/2`. Exact elimination of the known occupied block is
algebraically equivalent to the full redundant reference solve. It is the
occupied metric contribution, not redundant iteration, that must be retained.

### Usage and resource boundaries

```python
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_hessian import NativeRHFState, analytic_hessian

with NativeSource([(1, (0, 0, 0)), (1, (0, 0, 1.4))], basis="sto-3g") as source:
    state = NativeRHFState.from_source(source, tolerance=1e-12)
    components = analytic_hessian(state)
    hessian = components["total"]  # (atom, atom, xyz, xyz), Eh / Bohr**2
```

The caller owns `source` lifetime and must keep it open while evaluating a
Hessian. State-bound first-order matrices are cached as immutable arrays;
a changed geometry requires a new source/state. Compiler artifacts are cached
under `.artifacts` by default; pass `cache=...` to `from_source` to choose a
separate writable location. A C++ compiler is required for the generated kernels.

First-component records and component outputs have an explicit numeric budget.
The complete tools integration still retains all coordinate H1/S1 and response
vectors, the full molecular Hessian, and the existing tiny native SCF workspace.
It does not claim #180's global memory-budgeted production assembly or a
matrix-free molecular HVP. Python orchestration and cold compilation can be
expensive; no performance advantage is asserted.

An optional supplied `relax` tensor must be finite, real and exactly
`(natoms, natoms, 3, 3)`. It is a diagnostic component override, not evidence
that a native electronic-response solve occurred. Unsupported state domains and
closed/mismatched sources fail before derivative-provider execution.

### Verification

`tests/python/test_hessian_analytic.py` checks the following separately:

- A fresh-process test blocks imports of PySCF and the semi-numerical Hessian
  reference, and forbids the dense native derivative and dense AO response
  oracles while running the complete native H2 chain.
- Generated frozen-Fock/overlap perturbations are compared against an independent
  native derivative oracle used only on the assertion side.
- The final Hessian and individual components are compared with optional external
  PySCF analytic and finite-difference references at the same exact basis records.
- Three-step directional differences of VibeQC analytic forces independently
  test the total Hessian; raw symmetry and per-axis translation identities are
  checked before any presentation operation.
- Wrong relaxation tensors, out-of-domain sizes, unrelated references, repeated
  SCF attempts, and closed sources are explicitly tested. The earlier reduced
  CPHF regression still verifies orbital, occupied-energy and Hessian equivalence.

`tests/python/test_first_derivatives_native.py` separately checks generated
primitive components, Cartesian normalization and coincident-center scatter,
metadata/resource rejection, late-chunk failure isolation, and native output
publication. PySCF-dependent comparisons may skip when the optional oracle is
not installed; the no-oracle native test must not skip for that reason.

See the [native first-order source decision](../.agents/notes/implemented/numerics/2026-09-19-hessian-native-first-order-sources.md)
for the superseded PySCF-backed integration design and its replacement.

## Directional nuclear RHS and density response

`directional_rhf_response(state, direction, jk_backend="cpu" | "cuda")`
implements one nuclear perturbation without retaining every coordinate's
H1/S1 matrix. The input has shape `(atom, xyz)` and its magnitude is preserved.
The first-integral adapter contracts each shell's mathematical-center gradient
with the corresponding physical direction, including repeated atom slots and
the independent nuclear-attraction center. It accumulates only two `(AO, AO)`
matrices: the frozen-density Fock derivative and overlap derivative.

The same `solve_rhf_nuclear_perturbation` helper now serves this direction and
the existing complete-coordinate CPU Hessian assembly. It includes the known
metric-density Fock response, solves the nonredundant CPHF problem once, and
retains the occupied metric response and full occupied-energy response block.
The returned response includes occupied coefficient derivatives, the complete
AO density derivative and the energy-weighted-density derivative. In
particular, the occupied-energy response cannot be replaced with only its
diagonal or omitted from the latter.

```python
from tools.vibeqc_hessian import NativeRHFState, directional_rhf_response
from tools.vibeqc_posthf.sources import NativeSource

with NativeSource([(1, (0, 0, 0)), (1, (0.1, 0.2, 1.4))]) as source:
    state = NativeRHFState.from_source(source)
    result = directional_rhf_response(
        state, [[0, 0, 0], [0.1, 0.2, 0.3]], jk_backend="cuda"
    )
    dP = result.response.density_derivative
    dW = result.response.energy_weighted_density_derivative
```

This remains a **small-system tools integration** under `NativeRHFState`'s
12-Cartesian-AO/four-atom, all-electron conventional RHF boundary. It does not
remove that size limit, expose a Calculator derivative API or produce `Hv`.
First-integral execution defaults to the existing generated CPU path.
`first_backend="cuda"` plus an explicit `first_compiler` instead runs generated
S/T/V/four-center derivatives, direction contraction, density weighting and
AO-matrix accumulation on CUDA. Direction and density are uploaded once to a
shared accumulator; only the final H1(v)/S1(v) matrices are downloaded.
`jk_backend="cuda"` independently sends all metric/CPHF/final-response J/K
actions to the exact unscreened CUDA provider, without a CPU or DF fallback.
AO/MO transformations and Krylov work still remain host-side, so selecting both
CUDA providers is not a complete GPU-resident response/HVP claim. No performance
promotion or global peak-memory bound follows from those selections. The
[directional CUDA provider](first_directional_derivatives.md) documents its
compiler, ownership, memory and numerical boundaries.
Diagnostics report actual residency, reference/operator identity, the single
RHS/solve, input-matrix storage, solver controls and residuals. The CUDA plan's
retained-allocation budget and the solver workspace budget remain separate.

Direction arrays and published numeric results are detached and immutable.
Invalid directions, wrong state/operator identity, closed sources, failed
provider work and nonconverged or workspace-limited solves do not publish a
partial result. No reference-engine derivative or fresh SCF solve occurs inside
the directional consumer; displaced SCF solves are used only in its independent
finite-difference tests.

`tests/python/test_hessian_directional.py` checks generated inputs against the
independent native integral derivative oracle, three-step finite differences
of frozen Fock/overlap and reconverged density/energy-weighted density, occupied
metric identities, translation/linearity, omitted-metric negatives and failed
call recovery. Device qualification additionally runs
`tests/python/test_hessian_directional_cuda.py` with
`VIBEQC_RESPONSE_CUDA_TEST=1` inside a finite Slurm allocation. It forbids CPU
J/K and all-coordinate/dense-input fallbacks and checks the CUDA-assisted
response against independently reconverged density differences.

The next HVP assembly consumes these directional density/energy-weighted-density
responses and the existing second-integral directional providers. It must still
supply every explicit, relaxation, overlap/Pulay and nuclear term; neither this
RHS slice nor its CUDA J/K calls alone complete the molecular Hessian/HVP.
See the [directional response decision](../.agents/notes/implemented/numerics/2026-09-19-directional-rhf-nuclear-response.md).


For explicit CUDA first-source qualification, add these arguments to the
`directional_rhf_response` example above:

```python
from pathlib import Path
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

# Select the actual installed toolkit and target; compilation does not probe GPUs.
compiler = CudaCompilerAdapter(Path("/path/to/nvcc"), cuda_target_info("sm_120"))
# Within the live source/state scope:
result = directional_rhf_response(
    state, [[0, 0, 0], [0.1, 0.2, 0.3]],
    first_backend="cuda", first_compiler=compiler, jk_backend="cuda",
)
```

`tests/python/test_hessian_first_cuda.py` checks the generated device sources
against independent native first-integral derivatives and three-step displaced
native Fock/overlap/density/energy-weighted-density differences. It forbids the
CPU first-derivative interpreter, dense derivative inputs and CPU J/K on the
CUDA execution side. `tests/python/test_first_directional_cuda.py` independently
checks a selected f-shell contraction, repeated centers, signed weights, runtime
compatibility, invalid/partial records, nonfinite arithmetic and failed-call
recovery. Run these opt-in tests under a finite GPU allocation with
`VIBEQC_RESPONSE_CUDA_TEST=1` and the selected `nvcc` on `PATH`.
