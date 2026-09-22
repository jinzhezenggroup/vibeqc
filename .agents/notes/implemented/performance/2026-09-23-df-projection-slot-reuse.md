# Decision: retain adjacent streamed DF projections in compiler traversal

Status: implemented; full 96-atom energy-plus-force endpoint remains open
Date: 2026-09-23

## Problem

Issue #1078 remains open after reducing occupied K from seven raw tensor passes
to 1.5 at 768 AOs, 3712 auxiliaries and rank 160. A bounded 60-second run with
the original 13,685,173,124-byte value allowance still expires during SCF, before
forces. It completes 68 raw generations taking 51.664325 seconds. One uncached
J takes 18.0173 seconds; one uncached occupied K takes 17.35904 seconds. These
are intrusive progress-trace times, not clean endpoint benchmarks.

The actual tile is 579 auxiliaries, and the compiler chooses two blocks of 384
AO rows. The executor loads row blocks 0, 384, then 0 again, although the first
projection could remain live in its existing slot. The four charged buffers
already suffice. The same unnecessary adjacent-row regeneration occurs for
larger block counts.

## Decision

The compiler emits the visit order and two-slot identity in addition to the
capacity/work policy. Native callbacks bind projection/GEMM execution to those
slots. Triangular traversal alternates slots for new output rows, contracts the
diagonal, then visits columns in descending order. The first off-diagonal uses
the preceding output row retained in the other slot. Earlier prefixes overwrite
that other slot only after its last use; the current output row stays live for
the next outer visit. Full-matrix traversal retains its existing order.

For n rows, b balanced blocks and width h, raw generation changes from
`n + h*b*(b-1)/2` to `n + h*(b-1)*max(0,b-2)/2`. One and two blocks both need
one pass; ties prefer fewer matrix products, then the smallest balanced width.
At the original shape, K generates 2,189,426,688 values instead of 3,284,140,032.
J remains two passes, so one eligible uncached J+K build needs three raw tensor
passes instead of 3.5. Seed retries, cache hits and final dense K must still be
counted separately from reported SCF iterations.

## Invariants and rejected alternatives

No additional allocation, coefficient cache across calls, response-budget loan,
precision change, density acceptance relaxation or force-response lease is
introduced. Both retained projections remain disjoint from raw input and metric
scratch. All commands use the same stream, so overwritten projections are dead
before reuse, including graph replay with changed coefficients. Private metric
eigendirection projections are never published as symmetric-C response factors.

Changing only the work estimate would undercount real execution. A native-only
pointer swap would leave compiler admission disconnected from execution. The
emitted visitor and independent lifetime/census tests keep both consistent.
Retaining the full raw tensor exceeds the original value budget; this change
uses only existing bounded storage.

## Evidence

- Host tests execute the emitted C++ visitor over small divisible/nondivisible
  shapes and independently verify slot identities, complete pair coverage,
  capacity, exact source census and immediate callback failure propagation.
  Every legal row width is enumerated to qualify the work/width choice.
- The exact 768/3712/rank-160 fixture reports two blocks, 768 generated rows
  and unchanged dense fallback work of seven tensor passes.
- Native independent physical-integral tests cover four/five AO rows, ranks
  zero/one/two, triangular/full traversal, both coefficient layouts, truncated
  metric fallback and captured replay after changing coefficients.
- Slurm 11418: composed v12 completes independent 12-atom cold/warm/changed-
  geometry energy-only and energy-plus-force gates. Maximum errors are
  4.67e-12 Eh and 1.12e-11 Eh/Bohr; all original thresholds remain unchanged.
  Slurm 11420 passes all 13 independently linked fixed-density J/K and selector
  cases, including the added two-block fixture and exact cache/source counters.
  The first attempt lacked the saved library's SONAME symlink; correcting that
  local snapshot layout required no production change.
- Slurm 11422: the expanded native occupied-DF/response suite compiled from
  this branch and linked against composed v12 passes. Slurm 11423 runs that
  executable under compute-sanitizer memcheck and reports zero errors. Both
  cover ragged projection slots and changed-coefficient captured replay.
- Slurm 11417: clean 24-atom / 192-AO / 928-auxiliary GPU energy endpoints with
  a 256 MiB value allowance, comparing saved v11 and v12. Cold execution falls
  from 21.515763 to 19.541194 seconds; two warm samples change from
  4.645121/4.640905 to 4.364438/4.374767 seconds. Both use 15 cold and two warm
  iterations. Every measured GPU4PySCF pair passes, with maximum errors
  1.137e-12 and 6.822e-13 Eh respectively. Reference warm branches take one
  iteration, so this is not an iteration-matched cross-engine speed claim.
  Cold energies are not serialized by that comparator; the independent
  all-phase numerical evidence is the 12-atom test above.
- Slurm 11415: the same bounded 96-atom diagnostic still stops at 60 seconds.
  The first uncached K falls from 17.359040 to 11.810260 seconds and has eight
  completed raw generations instead of twelve. The seed iteration falls from
  35.437733 to 29.863104 seconds; J remains about 18 seconds. The capture ends
  in further source generation, before convergence/forces. It does not qualify
  the original complete endpoint.

The retained baseline diagnosis uses integration v11, library SHA256
`9d0dda6deeb41dc83f9a7bac11f83b4573af7fc0a1cce2f2439b3f7765093aad`,
source archive SHA256
`23022825bfbc4b8b827b6ac1c75dcfe6b79e90f9f13be753cc4b8be4398befdc`.
Slurm 11410 artifacts are `hf96-streamed-steps-v11*` in the preserved integration
worktree. Its energy-only resident control (Slurm 11408) changes the resource
regime and cannot establish completion of the original energy-plus-force case.

## Revisit when

Raw-source work still dominates after this bounded reuse. Further improvement
requires source-generation or J/consumer fusion evidence, or a separately
audited live-set strategy. Complete 96-atom energy-plus-force qualification and
README publication remain paused; this pass reduction does not close #1078.

This supersedes the triangular source-work count in
[the streamed value decision](2026-09-23-streamed-df-value-exchange.md).

The v12 composed measurements change only `df_exchange_schedule.py` and
`df_occupied_exchange.cpp` relative to v11. Candidate library SHA256:
`b7751872e207d9dc45668b56b093a4e8035abca75e3c18eb70ec9b6b517fabe7`;
source archive SHA256:
`4789eb19fd4031b6838b9231238ecdebc56d1689b55cac9d1ce6ee5fda1f177e`.
Artifacts include `hf96-streamed-steps-v12*`, `hfdf24-projection-v{11,12}*`
and `integration-source-v12.{json,patch,tar.gz}` in the preserved integration.
These are composed measurements, not a standalone current-master library.

## Standalone branch qualification

The Release sm_120 build based on master `e9fa40b1` completes. Slurm 11435
passes the expanded native occupied-DF/response suite and all 22 selected
Python checks (39.08 seconds): independent fixed-density J/K, source/cache
census, selector, cold/warm/changed-geometry energy and force endpoints, and
host scheduling/admission policies. This uses the standalone branch library,
not the composed v12 owner. Tested source is `b96e95eb`; its complete CI passes.
The subsequent commit only adds this evidence and the sanitizer record.

Standalone library SHA256:
`7baea1cee603b00fd48ab70081e75b5361a937665c83baec90bbd5962cb939d8`.
Local artifacts under `.artifacts/df-projection-reuse/` include `build.log`,
`gpu-gates.{sh,log}`, the `pytest/` references/traces, and
`standalone-library.sha256`. The final repair is ready for review; the bounded
96-atom timeout above still leaves #1078 open.
