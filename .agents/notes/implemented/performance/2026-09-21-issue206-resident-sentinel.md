# Decision: Make resident DF response admission auditable

Status: implemented
Date: 2026-09-21

## Problem

The #206 endpoint matrix can retain a good wall-time sample even when a future
policy regression silently selects a bounded host-panel path. That path can
repeat an `N^2` raw slice upload for each auxiliary panel and recreate the
performance cliff previously seen in the resident DF campaign. Numerical tests
alone do not expose this admission error.

## Decision

The component-trace path now classifies the executed response route and exposes
an opt-in strict `resident` sentinel. The sentinel validates host-gather absence,
one-full-tensor upload bounds, streamed raw-pass absence, auxiliary panel
consistency, transformed-tile ceilings, and caller-declared H2D/D2H ceilings.
The default `auto` mode records these observations without converting an
exploratory streamed or fallback run into a passing resident result.

## Rejected alternatives

- Inferring residency from environment variables was rejected because controls
  describe a request and may be rejected or re-planned by capacity policy.
- Declaring resident performance from endpoint timing alone was rejected because
  repeated transfers can be hidden by overlapping CUDA work and host timing.
- Applying the strict gate to every force probe was rejected because the legacy
  compatibility and constrained-memory routes are intentional controls.

## Invariants

- Resident rows have zero `raw_panel_host_gather_elements`.
- Resident rows do not regenerate raw values in streamed response panels.
- Host raw values may be uploaded at most once as one full `[N^2, Naux]`
  tensor; source-backed resident rows upload no host raw tensor.
- The observed auxiliary panel count lies between the tile-implied lower bound
  `ceil(Naux/tile)` and the shape-only upper bound `Naux`; shell-aligned
  consumers may legitimately add panels below the nominal tile size.
- Any total H2D/D2H or transformed-tile acceptance limit is supplied explicitly
  by the benchmark row; no generic budget is guessed by the validator.
- Route labels are derived from completed trace counters and are retained with
  every component-traced response row.

## Evidence

The hardware-free protocol coverage is in
`tests/python/test_issue206_resident_sentinel.py`. The live integrations are
`benchmarks/issue206_df_force_probe.py` and
`benchmarks/issue308_response_timeline.py`.

The response timeline also validates exact three-center derivative work for
both dense AO pairs (`Naux*N^2`) and symmetric packed pairs
(`Naux*N*(N+1)/2`). This matters at the 768-AO endpoint, whose qualified
occupied response reports packed work rather than the dense tensor shape. The
hardware-free regression is in
`tests/python/test_issue308_response_timeline.py`; Slurm-backed 384 and 768
resident traces passed with the selected dense/packed labels and zero raw
panel slices.

## Consequences

Future #206 resident campaigns fail before publication when a fast-path
admission change reintroduces repeated raw transfer or unbounded panel work.
The check is intrusive only when component tracing is already requested; clean
timing endpoints remain unchanged.

## Revisit when

The response trace publishes a stable native policy enum and complete transfer
owners. At that point the Python classification can consume the enum directly,
while retaining the current counter cross-check as a compatibility guard.

## References

- Issue #206: matched DF completion gates and resident-path regression audit.
- `docs/df_response_timeline.md`
- `benchmarks/df_component_ledger.py`
