# Generated DF forward residency

A generated integral source owns immutable geometry, basis transforms and
metric policy. Its presence does not imply streamed J/K execution. Full AO-pair
and auxiliary tiles select a retained transformed tensor; partial tiles select
the bounded streamed route. Both implement the same DF provider.

For a positive DF sub-budget, the native planner first tries full tiles and
checks the complete value-plan setup/contraction/SCF allowance. Energy-only
execution uses the whole supplied allowance; force execution reserves half for
its sequential DF and one-electron response bridges. Cached plans record the
value allowance and rebuild when a property change alters it. If the full
allowance fails, the existing deterministic tile-shrinking policy applies.
Zero-budget selection retains host-raw residency. The common global HF
ledger also charges the source-specific persistent storage and bounds actual
allocations; a DF sub-budget alone is not a whole-HF peak-memory measurement.

Generated resident setup writes raw public-AO values into existing K scratch
once per system, applies the retained inverse metric factor with cuBLAS, and
keeps the transformed tensor across fixed-geometry J/K builds. It allocates
neither a host full tensor nor an extra raw device tensor. The independent
host-tensor resident implementation retains its own setup upload allowance.

Source-backed replay traverses compact public-AO expansions from normalized
molecule metadata. Each expansion carries
only its actual Cartesian indices and coefficients, in Cartesian summation
order; batch items own separate maps. Cartesian AOs use one-term maps. No
generated integral or derivative formula changes with this representation.
Unbudgeted host-raw preparation retains its separate Cartesian integral
exporter and public-basis host transform. Source timing must not be attributed
to this separate path.

The [sparse-source decision](../.agents/notes/implemented/performance/2026-09-15-sparse-df-source.md)
distinguishes the source traversal speedup from the separate resident exporter
that bypasses it, with full-generation counters and qualification.

The private shape-query v2 adds an explicit generated-source flag. The original
v1 query retains host-tensor accounting. Source-generated response may coexist
with resident forward data, so a global resource candidate's `recomputed` mode
does not imply every forward tile is regenerated. Per-bucket tile decisions
and executed metric diagnostics expose actual forward storage.

Both generated storage modes first execute CUDA SCF, including graph capture
for streamed integral generation. The drivers retain the existing CPU numerical
recovery if the device solve fails; source ownership does not select CPU DIIS.

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
