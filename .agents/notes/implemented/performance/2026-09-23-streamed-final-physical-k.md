# Decision: qualify exact streamed final physical occupied K independently

Status: implemented (automatic streamed final occupied-K promotion)
Date: 2026-09-23

## Problem

The constrained 768-AO / 3712-auxiliary HF-DF case now admits a one-pass
occupied K in each value/SCF update, but final physical RHF F[D] still rejects
streamed storage before inspecting its exact final frame. It then executes a
seven-pass dense K even when the current final factor and charged occupied
projection buffers are available. This work is distinct from SCF iteration,
the two-pass J output and force response, and must remain visible in endpoint
qualification.

## Decision

Allow unset/`VIBEQC_DF_FINAL_EXCHANGE=auto` and explicit `occupied` to request
streamed final K for singleton generated-source RHF with a full-rank metric,
charged occupied buffers and an admitted compiler projected-exchange schedule.
Preserve the existing final-state token, supplied/retained density, device
generation, solver-info, work and capacity checks before contracting final C.
When any qualification fails, retain the bounded dense fallback; explicit
`dense` remains the comparison/debug override.

The same current canonical final C and RHF weight two feed the existing
occupied-K executor; J retains its independent streamed contraction. No new
allocation, SCF convergence rule, metric cutoff or force-response formula is
introduced. Streamed projections remain private eigendirection factors: do
not publish a `final_projection_token` for force-response consumers.

\n## Promotion update\n\nAfter the corrected-final path landed, the exact source tree passed the host lifetime gate and the targeted NVIDIA cold/warm/changed CUDA correctness suite. The retained 192-AO matched measurements showed occupied final K faster than the dense automatic baseline, while later composed changed-geometry evidence removed the earlier correction cliff. Automatic selection is therefore promoted to the same qualified streamed occupied path; the safety gates and dense fallback remain unchanged. Historical `auto` versus `occupied` measurements below describe the pre-promotion policy and are retained as negative/qualification evidence.\n
## Rejected alternatives

- Widening the resident final-projection lease to all streamed plans would
  mistake private metric-eigendirection panels for reusable symmetric-C
  factors, and could let force response read overwritten scratch.
- Borrowing response budget to hold a full B tensor changes the original
  bounded value/response contract and exceeds the charged value allowance.
- Enabling `auto` before independently qualified final-state/replay and
  complete endpoint gates would promote an unmeasured performance claim.

## Invariants and evidence gates

The resident, packed, UHF, batch, truncated-metric, unsupported-source and
explicit-dense routes keep their existing fallbacks. Failed/stale tokens must
not issue a final occupied contraction. Numerical gates require independent
PySCF/GPU4PySCF energy and analytic forces for cold, warm and changed
geometry, as well as native exact-density/generation rejection. Trace final
physical J/K separately from SCF and force response. Benchmark clean
same-source, same-budget, same-iteration endpoints on and off the opt-in;
sample peak memory separately and preserve the previous fallback.

At 768/3712, the compiler schedule predicts one K raw tensor rather than
the final dense K's seven: a reduction of 13,136,560,128 generated values
for one completed final physical F[D]. If composed with the separately
qualified shared-source candidate (#1139), the previous complete 96-atom
58-tensor work count would become 52 tensors *if* the final state is admitted
and no other phase changes. This is a source-work prediction, not a measured
endpoint latency or a claim that the independent branches have identical
SCF histories.

The `cuda-dev-fast` diagnostic library passes four Slurm GPU tests for
explicit occupied versus automatic final selection, energy-only and
energy-plus-force, over cold, warm and changed water-tetramer geometry with
independent PySCF energy/force gates (job in
`/tmp/qc-1117-final-fast-gpu-gate.log`). For a separately traced 192-AO /
928-auxiliary water octamer at a 384-MiB DF allowance,
`/tmp/qc-1117-final-fast-192-{auto,occupied}.jsonl` records three accepted
retained final J/K calls only under `occupied`. The corresponding three dense
`ri_k` builds under `auto` each request 171,048,960 raw source evaluations
(five tensor equivalents, 25 tile productions), while each occupied
`ri_k_occupied` build requests 34,209,792 (one equivalent, six tiles).
Both arms pass the independent GPU4PySCF energy gate within 4.55e-13 Eh,
but warm SCF iterations differ (two versus three); the diagnostic timings
are intrusive fast-compile measurements and establish **no** endpoint speedup.
The raw JSON results are `/tmp/qc-1117-final-fast-192-{auto,occupied}.json`.

The separately built Release sm_120 library (SHA-256
`1259ff8fca10517815add2ffdbb62aa7e29e5aa5017191b58671b1ce361ba15d`)
passes the same four GPU tests (`/tmp/qc-1117-final-release-gpu-and-force.log`).
With `VIBEQC_DF_JK_SHARED_SOURCE=0`, four ABBA-style separate-process,
untraced 192/928 energy-plus-analytic-force endpoint pairs, the same 384-MiB
allowance and the same GPU4PySCF basis snapshots/policies, all take 17 cold
and three warm native SCF iterations per arm. `auto`/`occupied` cold medians
are 18.862/16.372 s (13.20% less); warm medians are 8.422/5.931 s
(29.58% less). All converge, with a maximum independent force error below
1.44e-10 Eh/Bohr and energy error below 4.55e-13 Eh. The raw evidence is
`/tmp/qc-1117-final-release-192-p[1-4]-{auto,occupied}.json`, with full
per-process provenance, and the log is
`/tmp/qc-1117-final-release-192-repeats.log`. Both arms report the same
167,936,645-byte value-plan peak under the same 70-auxiliary tile, not a
sampled whole-endpoint memory high-water mark.

Separate Release diagnostic traces in
`/tmp/qc-1117-final-release-192-{auto,occupied}.jsonl` confirm three accepted
retained final builds on the opt-in and none on `auto`. Each dense final K
requests 478,937,088 generated values / 196 tiles versus 45,613,056 values
/ 20 tiles for occupied final K; the projected K visits three row blocks,
so one occupied traversal still regenerates 256 of 192 AO rows. Both traced
arms reconstruct the force-response final projection instead of borrowing
the value-path factor. These trace timings are intrusive and are excluded
from the clean medians. This qualifies a local 192/928 performance change;
the combined 96-atom endpoint and changed-geometry 96-atom gates remain open.

Composition validation used PR #1139 at `73ddbcfd` plus this PR at
`1e186e24` in an uncommitted validation branch, with the two test policy axes
combined (`git diff --cached --binary` SHA-256
`9c0f7dcc036e31665d1d2a18e22bc0d9c648557da7109f524216692d725633ce`).
The composed Release sm_120 library SHA-256 is
`02b61ca6d8de0c00b6c0d97a90c6d9dd9e23ac40ac1a0b7a81c3468ea91cd6e2`.
Eight Slurm GPU policy/geometry/force test combinations pass
(`/tmp/qc-1117-composed-gpu-gate.log`). Under the original
21,421,977,600-byte DF allowance, shared source and explicit final occupied
K, job 11495 completes the 768-AO / 3712-auxiliary 96-atom cold energy and
analytic force in 626.142 s, with 23 iterations. Against the prior
independent GPU4PySCF oracle, errors are 5.23e-11 Eh and 1.50e-10 Eh/Bohr.
This is a clean single observation from
`/tmp/qc-1117-composed-96-clean.log` and
`/tmp/qc-1117-composition-endpoint.py` (script SHA-256
`3e54073a5c89bfc4446f3d2a2f33e0b32ce9ef291ed524a429d42d867095fd74`).
The earlier shared-only 690.670-s run had the same 23 iterations and budget;
one observation per binary does not establish a repeatable speedup.

In job 11496, a separate same-library prepared batch takes 625.698 s cold
and 290.205 s unchanged-geometry warm (23 and seven native iterations);
both pass the saved independent reference energy/force gates within
5.23e-11 Eh / 1.51e-10 Eh/Bohr. The changed-geometry execution **does not
return** before the entire three-phase job's explicit 2,050-s timeout:
at least approximately 1,134 s remained after cold and warm. No
changed-geometry energy, force, iteration count or independent reference
was obtained. Keep this failure visible; do not claim 96-atom replay/rebuild
closure or infer that the final-K change caused it. Raw evidence is
`/tmp/qc-1117-composed-96-replay.log` and
`/tmp/qc-1117-composition-replay.py` (script SHA-256
`468c6411aa89aa5ee71d33b56e01b51d6b77f08f67757b471fb8dc6ca63f4611`).
At that point the predicted 58-to-52 total raw-tensor work count was
unverified by a full composed trace; neither the incomplete changed geometry
nor the 192-AO endpoint alone supported automatic promotion at that stage.

A bounded 192-AO composed control in Slurm 11497 (`/tmp/qc-1117-composed-192-rebuild.log`)
uses the same 384-MiB value allowance, shared source, 0.025-Bohr displacement
and independent changed-geometry GPU4PySCF reference. Under `auto` then
`occupied`, cold/warm/changed force endpoints take 18.776/8.376/30.116 s
and 16.362/5.890/27.630 s, respectively. Each arm follows 17/3/12 native
iterations; changed-geometry errors are below 1.26e-11 Eh and
1.12e-10 Eh/Bohr. This small control does not erase the 96-atom
changed-geometry timeout, and one sample per arm is not a generalized
performance claim.

The independent changed-geometry GPU4PySCF oracle completes in 56.991 s
under Slurm 11498 with energy -2431.233695541083 Eh and finite analytic
96-by-3 forces (`/tmp/qc-1117-changed96-reference.log`, script SHA-256
`887fc75ffa07b4a93a06814562a58f7dbbbc262e44e8c6e9f51d1f41377195d5`).
Slurm 11499 then executes the *same composed Release library, source,
geometry, budget, and policies* with diagnostic source and changed-phase
progress tracing. Cold, warm and changed energy-plus-force calls complete in
626.047 / 290.256 / **1207.797 s**, at 23 / 7 / 13 native SCF iterations.
All pass the independent energy and force gates: changed errors are
5.91e-12 Eh and 1.26e-10 Eh/Bohr. This explains why the earlier job's
approximately 1,134-s remaining changed-phase allowance was insufficient;
it does **not** satisfy a 900-s changed-phase performance target. The raw
journal is `/tmp/qc-1117-composed-96-diagnostic.log`, script SHA-256
`7150f51ab27d6c530d43dd8e5d11714a93f8f7e44951cba9ea9b798a9e0f1544`.

The job 11499 cold trace confirms the earlier 58-to-52 prediction: 24
shared SCF J/K source traversals plus 24 J output passes, final physical J
two raw passes, final occupied K one, and force response one, totaling **52
raw-tensor equivalents / 113,850,187,776 values**. Changed geometry
instead consumes **83 equivalents / 181,722,415,104 values**: 14 shared
SCF K+J pairs, four two-pass final J evaluations, one occupied final K,
three dense seven-pass final K corrections after `stale_final_state_token`,
and a 25-pass dense force response. The changed compact SCF finishes in
272.640 s; finalization then takes 931.548 s, including 617.334 s for
force response (110.300 s in the cold trace). Its force response requests
437,885,337,600 raw bytes versus 17,515,413,504 bytes cold. The exact
retained SCF token must still reject corrected densities; `auto` force
response currently rejects SCF factors on corrected generations, so the
full-rank streamed response falls back to bounded repeated raw loading.
Progress and source journals are
`/tmp/qc-1117-96-11499-{cold,changed}.jsonl` and
`/tmp/qc-1117-96-11499-changed-progress.jsonl`. These diagnostic traces
are **not** clean timing or a high-water GPU memory measurement. The next
candidate must independently certify any corrected-density factor before
reuse, retain a bounded dense fallback, and preserve the final-state and
response provenance fences.

## Revisit when

Endpoint parity, complete timing or memory evidence rejects the admitted
streamed final path, or a compiler-owned final J/K source-sharing schedule
can improve it further without changing response ownership.

## References

- #1117, #1078, #682, PR #1139
- `.agents/notes/implemented/performance/2026-09-23-streamed-df-value-exchange.md`
