# Decision: Stage a fail-closed Libxc Maple-to-Graph importer

Status: implemented
Date: 2026-09-21

## Problem

VibeQC pins the exact Libxc 7.0.0 Maple sources but currently translates audited
semilocal XC expressions manually into the compiler scalar DAG. That preserves a
clear numerical contract, but adding more Libxc functionals repeats formula
translation work and creates another opportunity for transcription drift.

The compiler already owns symbolic differentiation and C/CUDA emission, so a
second symbolic algebra or a generated-C parser would duplicate existing
ownership rather than removing that manual step.

## Decision

Add an experimental, fail-closed Maple front end that lowers a deliberately small
Libxc syntax subset directly into the existing `integral.expr.Graph`. The first
qualified slice is the standard `gga_x_pbe.mpl` branch
(`gga_x_pbe_params`) on the existing `libxc-7.0.0/interior-v1` domain.
The front end parses assignments, arrow functions, scalar arithmetic and named
calls. Libxc's shared `gga_exchange` helper is represented as a compiler
intrinsic whose interior-domain form follows the pinned `util.mpl` definition.
Density/zeta screening is not silently reimplemented: the existing VibeQC domain
contract owns admission of this slice.

The imported DAG is ordinary `Graph` data. Existing symbolic differentiation,
algebra passes and scalar C/CUDA emitters remain the only downstream algebra and
code-generation implementation.

This PoC does not replace `xc/expressions.py`, change a public functional,
promote a method, or alter production XC execution.

## Rejected alternatives

- **Use Python `eval` or execute Maple-like source.** Rejected because the
  importer must accept a bounded grammar and fail closed on unsupported syntax.
- **Introduce SymPy as a second XC algebra.** Rejected for this slice because
  VibeQC already has a differentiated scalar DAG and code emitters; another
  algebra would add identity and lowering boundaries before demonstrating need.
- **Parse Libxc generated C.** Rejected because it loses the high-level
  mathematical structure and makes derivative/code-generation ownership less
  direct.
- **Switch production PBE to the importer immediately.** Rejected until includes,
  shared helpers, piecewise semantics and broader independent qualification are
  implemented.
## Invariants

- Unsupported Maple directives and syntax fail explicitly.
- Importing source never loads Libxc, PySCF, the VibeQC runtime or a GPU.
- Imported expressions lower into the canonical `Graph`; no parallel
  differentiation or emitter implementation is introduced.
- Source provenance remains pinned to the checked-in Libxc 7.0.0 bytes.
- The `interior-v1` screening/domain boundary is not weakened by the importer.

## Evidence

`tests/python/test_libxc_maple_import.py` verifies the pinned source SHA-256,
standard PBE parameter branch, fail-closed unsupported directives, and direct use
of the existing `ScalarCEmitter`.

For three polarized interior-domain feature points, imported PBE exchange energy
density and all first derivatives with respect to
`rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb` match the existing audited
`GGA_X_PBE` DAG. The exploratory maximum absolute difference was
`5.551115123125783e-16`; the committed test uses tighter deterministic
`allclose` gates.

Reproduction:

```bash
PYTHONPATH=python python -m pytest tests/python/test_libxc_maple_import.py -q
PYTHONPATH=python python tools/check_compiler_structure.py
```

Observed on the implementation snapshot: 6 tests passed; 254 compiler modules,
0 dependency errors.
## Consequences

The next Libxc functional can extend a small front-end surface instead of copying
an entire formula into Python, while all derivative and backend generation
continues through VibeQC's compiler. The current utility intrinsic is intentionally
narrow; it is not evidence that arbitrary Libxc Maple is already supported.

## Revisit when

Extend the importer only with a pinned functional that exercises the missing
construct. PBE correlation is the natural next gate because it requires
`$include`, PW correlation helpers and stable exponential/logarithmic structure.
Piecewise/screening support should preserve lazy branch semantics before SCAN-like
sources are admitted.

A production source-of-truth switch should require component-level energy and
derivative equivalence, source/identity integration, and the existing independent
Libxc/high-precision validation gates.

## References

- `external/libxc-7.0.0/gga_x_pbe.mpl`
- `external/libxc-7.0.0/util.mpl`
- `python/vibeqc_compiler/xc/expressions.py`
- `docs/xc_expressions.md`

Agent: ChatGPT
Model: GPT-5.6 Sol


## Independent parser review (2026-09-21)

Two malformed-input cases violated the declared fail-closed grammar. Empty
parameter slots were discarded during splitting, and repeated parameter names
later overwrote an argument during dictionary binding. Conditional frames also
could not distinguish a terminal `$else` from an earlier matched branch, so a
second `$else` or a later `$elif` was accepted silently.

The repair preserves split slots and rejects empty/repeated parameter names.
Each conditional frame now tracks whether its terminal else occurred. Invalid
continuations after else raise MapleImportError independently of branch activity;
valid nested parent/child selection remains unchanged. No PBE scalar equation,
source-hash rule, screening contract or emitter is changed.

Independent review reproduced eight failures on the original parser, then passed
all eight rejection cases plus six valid nested-selection cases after repair.
The combined importer, parser and existing XC expression selection passed all
125 tests without skips. Ruff check/format passed, and the compiler structure
check covered 255 modules with zero dependency errors. This does not qualify
arbitrary Maple, includes, piecewise screening or a production XC source switch.
The PR remains an experimental Draft pending its final integration and review.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Review correction: lexical comments and intrinsic binding

Statement whitespace normalization previously flattened a Maple line comment and
its following source line before Python AST parsing. The accepted expression
`x # comment` followed by `+ 1` consequently lowered to `x`, silently changing the
formula. Comment removal now precedes statement normalization and preserves line
boundaries, nested block comments and unescaped-backslash comment continuation.
Quoted source is preserved for the existing grammar to accept or reject; unmatched
block delimiters fail explicitly. The source hash still covers the original bytes.

Compiler-intrinsic definition names are now reserved. Previously an assignment to
`X2S` was accepted but ignored in favor of the hard-coded intrinsic. Until rebinding
is separately qualified, explicit rejection is safer than accepting another
scientific definition and executing the built-in one. Existing local parameter
binding and the pinned PBE branch are unchanged.

Fifteen regression cases failed before repair; the combined comments, parser,
importer and existing XC expression selection passes all 141 cases afterward.
A separate 128-point spin-polarized oracle test compiles the actual ScalarCEmitter
output into a shared library and compares both that executable and the interpreted
imported DAG against PySCF/Libxc 7.0.0 PBE exchange energy and every first feature
partial. It passes without a GPU or production runtime. Compiler structure checks
cover 255 modules with zero dependency errors. These are bounded importer checks,
not a production XC source switch or arbitrary Maple qualification.

Comment semantics: Maplesoft Maple Help, `comment` (single-line and nested
multi-line comments). Existing numerical thresholds are unchanged.

Agent: ChatGPT
Model: GPT-6 Astra Pro
