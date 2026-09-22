# Decision: Make GeometryIR/PairIR the compiler owner of two-body D3(BJ)

Status: implemented
Date: 2026-09-20

## Problem

The production D3(BJ) correction introduced under #492 has a correct bounded CPU/CUDA
runtime, but its scientific equation lives in the handwritten
`src/dft/dispersion/d3_bj.hpp` evaluator. Keeping a second handwritten
energy/CN/gradient definition while the compiler grows a reusable geometry/pair layer
would make future D3 optimization, D4/gCP reuse, and generated derivatives vulnerable
to scientific drift.

The compiler therefore needs one auditable two-body D3(BJ) equation expressed through
GeometryIR/PairIR/TensorIR without changing the already-qualified native production
runtime before the shared pair layer can represent dynamic/ragged production work.

## Decision

`python/vibeqc_compiler/geometry/d3.py` is the compiler owner of the nonperiodic,
real-FP64, two-body D3(BJ) equation for atomic numbers 1 through 86 with `s9=0`.
It consumes a structural D3 specification rather than importing MethodIR, preserving
the `geometry -> {geometry,tensor,common}` ownership boundary.

The compiler reuses the pinned xTBloom-derived source assets rather than carrying a
second table copy. The accepted identities are:

- `upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1_d3.json`:
  `9ff932ea598f690c1fb599a67762060ba1907102d5ec132164f2a7e8886cd22e`;
- `upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json#elements[].covalent_radius_bohr`:
  derived-table SHA-256 `92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111`.

For a fixed geometry state, `D3PairTopology` records the canonical `i < j` union
of every pair needed by either coordination-number response or pair energy, plus an
explicit CN-active bit and energy region (`inner`, `switch`, or `off`) per pair.
These decisions are part of the compiler identity. Coordinate replay is allowed only
while rebuilding the pair state produces the same identity; crossing a CN cutoff,
pair cutoff, or smooth-switch region boundary is rejected as stale state.

The numerical conventions match the native two-body evaluator:

- CN uses
  `1 / (1 + exp(-16 * (Rcov_ij / r_ij - 1)))`;
- a finite CN cutoff includes the boundary, matching
  `r^2 <= cn_cutoff^2`;
- reference weights are proportional to
  `exp(-4 * (CN_ref - CN)^2)`;
- packed C6 lookup preserves the native lower/higher-element orientation and then
  interpolates C6 from both atoms' normalized reference weights;
- `R_R = 3 * r4r2_i * r4r2_j` and
  `R_D = a1 * sqrt(R_R) + a2`;
- the BJ pair energy is
  `-C6 * cutoff * (s6 / (r^6 + R_D^6) + s8 * R_R / (r^8 + R_D^8))`;
- with a hard pair cutoff (`pair_switch_width == 0`), the cutoff boundary is
  included, matching the native pair-loop `r^2 <= pair_cutoff^2`;
- with a positive switch width, the factor is one for
  `r <= pair_cutoff - width`, the quintic
  `x^3 * (10 + x * (-15 + 6*x))` for
  `pair_cutoff - width < r < pair_cutoff`, and zero at/above the cutoff.

The TensorIR energy graph is the derivative source of truth. Cartesian
`dE/dR` is generated with the shared TensorIR VJP; there is no second handwritten
D3 gradient expression in the compiler.

The native evaluator has a defensive fallback when all normalized reference weights
become non-finite after extreme CN underflow. The current TensorIR graph has no
branch/select primitive for that fallback. The compiler therefore fails closed when
the reference-weight normalization is non-finite or zero instead of silently changing
the scientific rule. The native path remains the production oracle for that domain.

The existing `d3_bj.hpp` runtime remains the production CPU/CUDA execution owner for
ragged batches and dynamic changed-geometry replay. This change establishes compiler
ownership of the scientific equation; it does not claim that the native production
runtime already executes generated PairIR.

## Rejected alternatives

- Copy the native analytic gradient into compiler code. Rejected because energy and
  derivative would become independently maintained scientific formulas.
- Import `method.D3Spec` from the geometry layer. Rejected because that reverses the
  compiler ownership boundary and couples reusable geometry lowering to MethodIR.
- Generate or check in a second expanded C6 table for the compiler. Rejected because
  duplicate scientific data would create an unnecessary provenance/drift boundary.
- Delete or bypass the native production evaluator immediately. Rejected because the
  current shared PairIR is fixed-topology while the public D3 owner supports ragged
  systems and changed-geometry replay; production promotion needs an explicit
  dynamic/ragged pair execution contract.
- Approximate the native reference-weight fallback with an unversioned rule. Rejected
  because a numerically different fallback must not be hidden inside a supposedly
  identical D3(BJ) identity.

## Invariants

- Coordinates are bohr, energy is Hartree, and the generated vector is gradient
  `dE/dR`, not force.
- `s9 != 0`, zero damping, unsupported table identities, and elements outside
  H--Rn remain rejected by this compiler lowering.
- Table/radius SHA-256 values and the D3 specification identity are part of the
  scientific contract.
- Pair ordering is canonical and deterministic; C6 packed-table orientation must
  remain identical to the native lookup.
- CN cutoff and pair/switch-region membership are execution-state identity, not
  untracked runtime branches.
- Energy and coordinate gradient must continue to originate from one TensorIR graph.
- The native production path remains an independent oracle until a separately
  qualified dynamic/ragged generated path replaces it.

## Evidence

`tests/python/test_d3_geometry_ir.py` is part of the explicit
`python (compiler-heavy)` CI shard. It checks all nine independent
simple-dftd3 1.4.0 fixtures in `tests/data/d3_bj_reference.json` for energy and the
generated Cartesian gradient, checks gradient translation invariance, validates the
GFN1 smooth-switch region against a centered finite difference, rejects stale
pair/switch topology, and lowers both primal and generated VJP through the shared CUDA
TensorIR planner/emitter.

The same PR also passes compiler type/structure/pre-commit gates, CPU GCC/Clang
build/tests, CUDA 12.9 compile qualification, CuMetal CUDA qualification, and Python
wheel construction. These are correctness/build gates, not a pair-parallel performance
claim.

## Consequences

The D3 mathematical definition is now available to the common compiler stack and its
automatic differentiation machinery, providing a reusable pairwise pattern for later
geometry corrections. The initial graph is specialized to a fixed pair state and can
materialize pair/reference constants proportional to that state, so it is a
correctness/ownership slice rather than the final large-system scheduling strategy.

The native and compiler equations intentionally coexist for now: native code supplies
the production ragged runtime/oracle, while compiler code supplies the generated
scientific DAG and derivative. Their duplication is temporary and has an explicit
retirement condition instead of becoming two untracked production definitions.

## Revisit when

Revisit this decision when the shared geometry layer gains dynamic/ragged PairIR
execution, runtime neighbor-list rebuild semantics, or an equivalent bounded generated
pair schedule. At that point, qualify the generated path against the native oracle and
retire handwritten production D3 scientific arithmetic if the public ragged/replay and
resource contracts are preserved.

Also revisit the normalization boundary if TensorIR gains a stable conditional/select
or another explicitly versioned mechanism that can reproduce the native extreme-CN
fallback without non-finite intermediate arithmetic. ATM, zero damping, D4, gCP,
periodicity, and pair-parallel performance require separate identities and acceptance
evidence rather than silently extending this two-body D3(BJ) contract.

## References

- #492
- PR #627
- `docs/dft_d3.md`
- `docs/geometry_pair_ir.md`
- `src/dft/dispersion/d3_bj.hpp`
- `tests/python/test_d3_geometry_ir.py`
- `tests/data/d3_bj_reference.json`
- `.agents/notes/implemented/architecture/2026-09-19-d3-xtbloom-baseline.md`
- `.agents/notes/implemented/architecture/2026-09-19-d3-production-runtime.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
