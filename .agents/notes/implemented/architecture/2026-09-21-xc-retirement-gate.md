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
