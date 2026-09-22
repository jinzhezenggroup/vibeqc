# Decision: cooperative CUDA density-feature reduction

Status: implemented; full 96-atom SCF remains unqualified
Date: 2026-09-23

## Problem

After #1106's dense contractions, a bounded PBE water96 diagnosis still spends
3.876547 s forming density features across two unconverged SCF iterations.
There are 768 AOs, 2,654,208 grid points and 20,736 feature launches. The old
schedule assigns one thread per point: default 256-point tiles launch only
two blocks, and neighboring lanes read different AO rows with 768-double
stride. This leaves little parallel work in flight on RTX 5090.

The same v10 binary at tile=1024 reduces the feature sum to 0.995671 s but
quadruples tile storage and changes the resource plan. The default two-step
execution is 25.085391 s; the larger-tile probe is 19.915654 s. Both are Nsight
diagnostics, not converged benchmarks. See #1102 and the matrix schedule note.

## Decision

The compiler emits scalar and cooperative instances of the same feature
consumer. For nAO >= 32, each warp owns one spin/point; lanes traverse adjacent
AO components and combine five FP64 accumulators with a warp tree. The body
continues to call `vibeqc_grid_policy::add_features`, the existing mathematical
owner. Smaller AO spaces retain their scalar summation order. Native code only
binds the launch for physical density and signed response; it owns no schedule.

The point loop is uniform within a warp, with whole-warp admission and zero
initial partials for missing AO lanes. All participating lanes execute every
shuffle. Grid-stride traversal retains finite launch bounds. There is no new
global/shared storage, library handle, screening, tile policy, precision,
feature layout, or consumer-lifetime change. Only reduction order changes.

## Gates and rejected alternatives

The expanded native suite retains small 17/23-AO scalar cases and adds 36/49-AO
cooperative cases, including partial AO/point groups, spherical/Cartesian f
shells, all semilocal families/spins, variational differences, signed LDA/PBE
responses, capture and resource canaries. It is stacked on the #1108 stable
spin-coordinate fix; both same-input and complete independently computed
CPU/GPU minority-potential gates remain active.

Do not substitute a kernel-only claim for complete endpoint timing. Compare
the same 96-atom two-step work and retain the 90-second diagnostic watchdog,
then qualify complete smaller native/GPU4PySCF energy endpoints. Source and
binary snapshots are retained before performance runs. Full96 convergence
and README publication remain separate outstanding gates.

Increasing the default point tile is deliberately deferred: it needs complete
endpoint and resource-policy qualification and does not address strided AO
loads. Reimplementing rho/gradient/tau algebra inside the schedule would split
the scientific owner and is unnecessary.

## Measured qualification

Slurm 11405 passes the full expanded native GPU suite. Twenty selected host
schedule/factor tests and compiler/CUDA ownership checks pass.

Slurm 11406 repeats the exact two-step water96 work with the default tile=256:
preparation 4.083406953 s, execution 21.549902662 s; feature kernel sum
0.166425316 s across the same 20,736 launches, versus 3.876546773 s at v10.
Density/potential/point work counts remain unchanged. The complete diagnostic
execution decreases from 25.085390903 s; this remains an unconverged profile.

Slurm 11407: PBE24 direct and DF complete cold/priming/two-warm comparisons
pass all four unchanged GPU4PySCF energy pairs. Direct: preparation
0.393872153 s, cold 20.033694157 s, warm 2.531739437/2.541388164 s, maximum
error 1.23919e-11 Eh. DF: preparation 0.596533106 s, cold 10.512064624 s,
warm 1.367667102/1.385316522 s, maximum error 4.16094e-11 Eh. Both cold runs
use 16 iterations. Prior v10 direct/DF cold times were 21.9398/12.3651 s.
Preparation is additional to cold execution; reference warm iteration branches
differ, so no matched-iteration reference speedup is claimed.

Preserved integration HEAD is 60592ea9 plus the archived overlay; only the
feature schedule differs from v10 for these PBE measurements. v11 library
SHA-256: `9d0dda6deeb41dc83f9a7bac11f83b4573af7fc0a1cce2f2439b3f7765093aad`.
Source archive SHA-256:
`23022825bfbc4b8b827b6ac1c75dcfe6b79e90f9f13be753cc4b8be4398befdc`.
Artifacts are `integration-source-v11.*`, `ks96-two-steps-v11*`, and
`pbe24-*-features-v11.*` in the retained integration workspace. Exact standalone
branch native qualification is separate from that composed endpoint evidence.

## Broader complete endpoint qualification

Slurm 11419 runs complete matched-grid PBE48 direct/DF energy endpoints with
an unchanged 120-second guard per endpoint. Both pass all four numerical pairs.
Direct: preparation 0.967611386 s, cold 77.368161802 s (22 iterations), warm
7.030856312/7.041812285 s (two iterations), maximum error 8.89031e-11 Eh.
DF: preparation 3.173285206 s, cold 49.135108257 s (21 iterations), warm
7.022172632/7.031696032 s (three iterations), maximum error 3.88809e-11 Eh.

The direct v10 comparison had 21 cold and three warm iterations, so its earlier
83.5460/11.9246/11.9700-second timings do not establish a matched-work speedup.
DF retains the same 21/three iteration branches as v10 (58.9862 cold and
8.4282/8.4315 warm seconds). Both reference warm branches differ from native;
no iteration-matched cross-engine claim is made. Precision, grid and SCF
acceptance settings are unchanged.

These measurements explicitly load the saved v11 library, not the later HF-DF
projection-reuse overlay. They share the source/binary hashes above. Artifacts
are `pbe48-{direct,df}-features-v11.{json,log,progress.jsonl}` as applicable and
`pbe48-features-v11.log` in the retained integration directory. Native branch
validation and these composed complete 24/48 endpoints support review of the
feature schedule. Full 96 convergence and the complete #1102 acceptance remain
open; the bounded 96 diagnostic must not be promoted into README results.
