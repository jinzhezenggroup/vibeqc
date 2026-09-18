# Analytic HF Hessians (issue #180, slice A)

This document is the second-derivative dependency graph requested by step 1 of
issue #180: it maps every term of the RHF energy to the Hessian contribution it
produces, and records which layer supplies that term. It is deliberately
written before any assembly code exists, so the term list and the sign
conventions are fixed in one place instead of being reverse-engineered from
three different providers later.

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
response; a correctly constructed reduced occupied/virtual solve is equivalent.

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
