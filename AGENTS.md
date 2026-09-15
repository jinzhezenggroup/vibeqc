# VibeQC agent instructions

These repository-local rules complement the user-level agent instructions.
They apply to performance, CUDA, compiler, response, DFT, and post-HF work.

## Performance engineering

Before optimizing a kernel, establish whether the endpoint is doing the right
amount of work. A memory-bounded plan is not necessarily work-bounded.

For performance-sensitive changes:

- Measure complete endpoint time and semantic work counts. At minimum inspect
  relevant tile/panel counts, source evaluations, contraction/GEMM counts, and
  host/device transfer bytes. Do not select an optimization from kernel time
  alone.
- Audit nested tiling for work amplification. If an expensive quantity is
  invariant to an outer consumer tile, panel, or block, do not recompute it
  inside that outer loop without measured justification.
- Prefer source-driven reuse: generate an expensive integral/intermediate once
  and feed as many consumers as its lifetime and memory policy safely allow.
- Reuse already-accounted resident state when identity, lifetime, stream
  ordering, and resource ownership are valid. Do not force recomputation merely
  to preserve an artificial private scratch budget.
- Keep an explicit bounded fallback when resident storage is unavailable.
  Resident reuse must not silently turn a bounded path into an unbounded one.
- Treat memory and work as separate planner objectives. Record work
  amplification across budget/tile choices; a lower-memory schedule that
  repeats expensive source or cubic work may be the wrong default.
- Keep oracle/reference/compatibility preparation out of production CUDA hot
  paths unless the production consumer actually needs it.
- For derivative/response code, contract generated derivatives with final
  weights as early as practical instead of materializing large derivative
  tensors or round-tripping them through the host.
- Test at a larger size before promoting a performance policy. A schedule that
  wins at 192--384 AOs may cross a tiling/work-amplification cliff at 768 AOs.
- Separate cold, same-geometry warm, changed-geometry, batch, and constrained-
  memory evidence. Do not use one regime as a proxy for another.

See `docs/performance_engineering.md` for rationale, evidence requirements, and
the #373 DF-response case study that motivated these rules.
