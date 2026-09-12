# Generated DF forward residency

A generated integral source owns immutable geometry, basis transforms and
metric policy. Its presence does not imply streamed J/K execution. Full AO-pair
and auxiliary tiles select a retained transformed tensor; partial tiles select
the bounded streamed route. Both implement the same DF provider.

For a positive DF sub-budget, the native planner first tries full tiles and
checks the complete value-plan setup/contraction/SCF allowance. Half the supplied
DF sub-budget remains reserved by the caller for response staging. If the full
allowance fails, the existing deterministic tile-shrinking policy applies.
Zero-budget compatibility selection remains unchanged. The common global HF
ledger also charges the source-specific persistent storage and bounds actual
allocations; a DF sub-budget alone is not a whole-HF peak-memory measurement.

Generated resident setup writes raw public-AO values into existing K scratch
once per system, applies the retained inverse metric factor with cuBLAS, and
keeps the transformed tensor across fixed-geometry J/K builds. It allocates
neither a host full tensor nor an extra raw device tensor. The independent
host-tensor resident implementation retains its own setup upload allowance.

The private shape-query v2 adds an explicit generated-source flag. The original
v1 query retains host-tensor accounting. Source-generated response may coexist
with resident forward data, so a global resource candidate's `recomputed` mode
does not imply every forward tile is regenerated. Per-bucket tile decisions
and executed metric diagnostics expose actual forward storage.

Prepared plans compare geometry, basis and metric policy before reuse, and
sources have no mutation API. Physical source changes create a new plan and
tensor; changing a density alone reuses the same fixed-geometry tensor. Metric
eigenvectors, eigenvalues, rank validity and inverse factors remain available
to the existing complete spectral derivative, including discarded directions.

With `VIBEQC_DF_TRACE`, `resident_three_center_materialization` records raw and
transformed logical production plus retained bytes. Later resident J/K records
contain no tile generation. Capture counters describe construction and must
not be interpreted as replay execution counts.

This is the resident slice of #282. Streamed traversal optimization, complete
96/192-AO batch evidence and occupied-factor RI-K are separate work.
