# Decision: pinned source catalogs precede method parameter code generation

Status: implemented
Date: 2026-09-20

## Problem

The expanded D3/D4 catalog is imported from pinned upstream tables. Treating
`method_parameters.json` as the editable authority would let the next sync
silently overwrite a local scientific edit. The earlier method-parameter note
records the previous ownership model; this decision supersedes that source layer.

## Decision

`tools/parameters/upstream/` holds exact, unmodified source snapshots.
`dispersion_parameter_sources.json` binds their repository, revision, path,
license and SHA-256. `method_parameter_overrides.json` owns deliberate local
choices: two-body data identity, D4 runtime defaults/profile overrides, local
GFN2 parameters and gCP parameters. Update those inputs, not generated records.

`tools/sync_dispersion_parameters.py` verifies snapshots and deterministically
resolves upstream defaults, variants, decoded method keys and local overrides
into `python/vibeqc_compiler/method/method_parameters.json`. This JSON is now a
committed generated intermediate. `tools/generate_method_parameters.py` remains
the single typed lowering into committed Python constants and CMake-generated
C++/CUDA constexpr accessors. Production does not parse any input table.

## Invariants and rejected alternatives

Do not edit vendored bytes to fit the parser or change a pinned digest merely
to accept different data. Do not hand-edit either generated view. A new
upstream revision requires an explicit manifest update and independent review.
Do not equate catalog availability with signed-damping, ATM or named public
DFT capability; the existing runtime admission remains in force.

## Evidence and regeneration

The freshness tests protect both synchronization stages. Independent review
compared all 157 D3 and 118 D4 source records with stdlib TOML decoding and
verified the exact upstream Git blobs. Four quoted-key counterexamples caught
and protect the SKALA lookup repair; numeric parameters were unchanged.

```sh
python tools/sync_dispersion_parameters.py
python tools/generate_method_parameters.py --python-output python/vibeqc_compiler/method/_generated_parameters.py
PYTHONPATH=python:. python -m pytest tests/python/test_dispersion_parameter_sync.py tests/python/test_dispersion_quoted_method_keys.py tests/python/test_method_parameter_codegen.py -q
```

## References and revisit conditions

Supersedes only the editable-source layer in
[the earlier codegen decision](2026-09-20-method-parameter-codegen.md).
Refs #676, #492, #493. Revisit the input model when adding independently
qualified damping families; do not introduce a second runtime parameter path.

Agent: ChatGPT
Model: GPT-6 Astra Pro
