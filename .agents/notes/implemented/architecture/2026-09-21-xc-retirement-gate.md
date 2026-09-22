# Decision: monotone source-level gate for XC formula retirement

Status: implemented validation tooling for #745
Date: 2026-09-21

## Problem

The Libxc Maple importer is being qualified and promoted family-by-family while
the old handwritten translations remain temporarily useful as comparison
oracles. Parallel cutovers can remove old edges safely, but without a repository
gate another production/test consumer could accidentally start depending on a
legacy expression module and recreate the dual-source maintenance problem.

A numerical test cannot detect that ownership regression by itself.

## Decision

Add a syntax-only retirement inventory over `python/`, `tools/`, and `tests/`.
The inventory recognizes the three legacy mathematical modules named by #745:

- `vibeqc_compiler.xc.expressions`
- `vibeqc_compiler.xc.rsh_expressions`
- `vibeqc_compiler.xc.wb97mv_expressions`

The checked consumer set is a **ceiling**. Existing migration-time dependencies
may disappear without editing the gate, but any newly introduced dependency
fails CI. Newly introduced `*expressions.py` modules under the XC package also
fail unless deliberately reviewed into the policy.

`python -m tools.vibeqc_validation.xc_retirement --json` prints the live source
and consumer inventory. The final #745 audit can additionally run
`--require-no-consumers`; it deliberately fails while any legacy module is still
imported by production or tests.

This gate does not claim numerical qualification. E/vxc/fxc, generated CPU/CUDA
source, molecular endpoint, force/response/Hessian, artifact identity, and
family-specific source-provenance gates remain owned by the corresponding
cutover work.

## Parallelism

The tooling modifies no functional formula, importer semantic, emitter, or
runtime path. Family agents can therefore keep deleting consumer edges without
coordinating edits to this file; the allowed set is monotone and removals pass
automatically.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Import-form review correction

The inventory includes from-package submodule imports and literal relative
`import_module` calls with a literal package. Nested expression modules are
checked by full path, so placing a duplicate in a subpackage does not evade the
gate. Arbitrarily computed import names are not a promise of this syntax-only
audit. Unrelated calls containing relative filesystem paths are not imports.

The existing LYP qualification test was absent from the initial consumer ceiling.
Its already-present direct-import edge is included in the migration ceiling; new consumer paths
still fail and removals still require no ceiling edit. Six independent negative
cases failed before repair and now detect their edges/modules; the repository
inventory and a non-import path-call control are also exercised.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Integration with already merged qualification families

The review integration at master `6b965bd7` includes B88, P86/PZ and VWN
qualification tests that intentionally retain the audited expression as a
second oracle. Record these three existing test-only edges in the initial
ceiling rather than rewriting their independent oracle or permitting a new
production import. The production/runtime ceiling is unchanged. All existing
new-consumer, dynamic-import and final-no-consumer rejection tests remain.
The shared source registry is inherited from that same verified master tree;
no numerical data or acceptance tolerance is refreshed.

Agent: ChatGPT
Model: GPT-6 Astra Pro
