# Qualified Libxc SCAN-family Maple import

Issue: #742
Parent: #739

## Decision

Extend the existing pinned Libxc Maple importer only with syntax and intrinsics
required by the vendored SCAN/rSCAN/r2SCAN sources. The importer remains a
fail-closed source compiler, not a general Maple interpreter.

The importer semantic identity is bumped from `libxc-maple-graph/v3` to
`libxc-maple-graph/v4`. External Libxc scalar parameters are canonicalized and
included in the transitive identity so different parameter sets cannot reuse
the same generated artifact identity.

## Qualified semantics

The v4 importer adds:

- strict `<`, `<=`, `>`, and `>=` comparisons lowered through lazy
  `Graph.select_le`;
- `my_piecewise3` and `my_piecewise5` without evaluating inactive branches;
- bounded integer `add(body, i=N..M)` reductions, capped at 33 terms;
- the source-evidenced `m_min`, `m_max`, `m_abs`, `t_total`, `n_total`, and
  `opz_pow_n` helpers;
- an explicitly gated duplicate-include mode for the pinned r2SCAN include
  graph, while duplicate includes remain rejected by default;
- caller-owned scalar bindings for C-side parameter structs, with binding
  values included in source identity;
- only the exact two-argument
  `eval(diff(f(x1,x2), xi), [x1=..., x2=...])` form used by r2SCAN
  correlation. It is lowered with the existing `Graph.differentiate` and a
  bounded graph substitution, rather than by embedding a symbolic runtime.

The SCAN `scan_gx` expression has an explicit source-qualified `x -> 0+`
continuation. This represents the analytic limit of the pinned expression and
prevents its inactive reciprocal-square-root branch from being evaluated at
exactly zero same-spin gradient.

## Feature adapters

Correlation is qualified directly from the pinned `scan_f` / `r2scan_f`
source functions and their transitive PBE/PW dependencies.

For exchange, the pinned Maple source owns the SCAN/r2SCAN enhancement
function, while the test adapter assembles the two spin-channel exchange
energy densities directly in the canonical VibeQC feature coordinates. This
preserves exact spin separability in the scalar DAG. Reconstructing the same
energy as `n * epsilon(rs,zeta)` is algebraically equivalent but can leave
floating cancellation residue in cross-spin Hessian entries at extreme
polarized boundary points where the exact value is zero.

This phase does not cut production XC expressions over to imported artifacts
and does not promote any public capability. Artifact/profile ownership and
production cutover remain downstream phases of #739.

## Invariants

- No runtime Libxc, Maple, PySCF, or GPU dependency is introduced.
- Pinned `.mpl` and parameter-owner `.c` hashes remain provenance inputs.
- Unsupported Maple control flow, reductions, `eval`, and `diff` forms fail
  closed.
- Existing Graph AD owns first and second derivatives.
- Existing lazy `select_le` owns branch-local evaluation and differentiation.
- Duplicate includes require an explicit importer opt-in.

## Evidence

Focused SCAN-family importer tests cover source/parameter identity, strict
piecewise boundary semantics, inactive singular branches, SCAN/r2SCAN
exchange and correlation, both spin layouts, typical and boundary fixtures,
energy, every first feature derivative, the packed feature Hessian, and
Scalar C/CUDA emitter lowering.

The combined regression command covering the Maple importer, parser/comment
gates, SCAN-family tests, and general XC expression fixtures completed with
`215 passed, 1 skipped`. Ruff passed on the changed Python files,
`git diff --check` passed, and `tools/check_compiler_structure.py` reported
`258 compiler modules; 0 dependency errors`.

Agent: ChatGPT
Model: GPT-5.6 Sol
