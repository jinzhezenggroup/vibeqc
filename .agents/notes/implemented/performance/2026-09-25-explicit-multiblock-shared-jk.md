# Decision: expose multiblock shared J/K only as an explicit experiment

Status: implemented
Date: 2026-09-25

## Problem

The streamed DF shared J/K path had only been admitted for schedules with at most two projected AO-row blocks. That bound prevented direct measurement of whether regenerating each physical AO row once and sharing it across J/K remains useful when the projected schedule requires more blocks.

## Decision

Keep the default, unset, and `auto` policy limited to the previously bounded at-most-two-block domain. When `VIBEQC_DF_JK_SHARED_SOURCE=1` is explicitly requested, admit any compiler-valid triangular projected schedule that passes the same execution-side capacity and identity checks.

Retain execution-side admission rather than trusting the SCF selector alone. Record projected row-block and regenerated-row counts so complete endpoint measurements can distinguish actual multiblock execution from fallback.

This is an explicit diagnostic/performance experiment. It does not change memory budgets, numerical tolerances, final-Fock ownership, force-response ownership, or the automatic policy.

## Invariants

- Unset, `0`, and `auto` do not gain the >2-block route.
- Every AO row is regenerated exactly once for an admitted triangular projected traversal.
- Capacity, full-rank, coefficient, density, and source-identity checks remain mandatory at execution.
- Numerical correctness remains independently gated; a timing difference cannot relax energy/convergence criteria.
- Mixed endpoint results do not imply an equal-work kernel speedup or justify automatic promotion.

## Evidence

Host schedule/admission tests execute the generated visitor and independently census slots/coverage, including a three-block case and fallback controls.

On the exact implementation head `1af282521a20a767b3f8174e0fce757216b33566`, RTX 5090 / CUDA 12.9 source-matched validation used a 192-AO, 928-auxiliary water-octamer case with a 256 MiB budget. Explicit `1` entered three projected row blocks while `auto` and `0` fell back; the independent PySCF energy error was about 6e-12 Eh.

Repeated complete Release endpoints reported:

- `auto`: cold 15.296/15.120 s at 17 iterations; warm 5.053/5.049 s at 3 iterations.
- explicit `1`: cold 16.799/16.780 s at 16 iterations; warm 3.923/3.922 s at 2 iterations.

The two strategies converged within 9.1e-13 Eh. Because iteration counts differ, the warm wall-time difference is not an equal-work kernel speedup. The cold endpoint is slower. These results qualify the explicit experiment and its cost; they do not support promoting it into `auto`.

## Consequences

The repository retains a bounded, opt-in way to collect >2-block profitability evidence without changing production defaults. This adds a diagnostic branch and counters that should be removed if future evidence shows no continuing value.

## Revisit when

Revisit after representative source-matched cases establish a stable equal-work or complete-endpoint benefit across geometries/bases, or when a new schedule makes multiblock sharing automatic without increasing total work.

## References

#1078, #1117, #682, PR #1295.

Agent: ChatGPT
Model: GPT-5.6 Sol
