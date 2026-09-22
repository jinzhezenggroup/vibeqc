# Bounded generated DF J/K panels

The generated streamed provider uses the same four physical tile buffers as
the existing value plan. No full host tensor, additional device allocation,
stream, or synchronization is introduced by the traversal.

For raw three-center values `A[pair,P]` and the retained symmetric inverse
metric factor `X`, Coulomb uses `J = A X X^T A^T D`. A first raw pass accumulates
auxiliary charges and applies `X` to small vectors; a second vector transform
and raw pass produce J. This avoids transformed three-center production for J.
Two raw passes remain because retaining the entire raw tensor would exceed
the resolved capacity. The full metric factor, including discarded directions,
is unchanged, so the existing metric-rank policy still defines the operator.

For K, the existing capacity is rebalanced into full AO panels with a smaller
auxiliary width whenever one AO matrix fits. Each transformed panel then feeds
both exchange GEMMs before eviction. If even one matrix cannot fit, the bounded
row traversal consumes its own diagonal column first, retaining that one cache
hit; other column panels still repeat across row blocks. Disjoint output blocks
keep their ascending auxiliary accumulation order despite the changed visitation
order. Dense nonsymmetric input densities retain their established orientation.

Panels wider than four auxiliary directions stage raw blocks and apply the
metric factor with GEMM. Skinny panels retain the fused recurrence/transform
kernel: an initial 192-AO/32-MiB experiment with raw staging produced 36,864
small raw launches per K build and was cancelled. The cutoff is a provisional
schedule heuristic for this measured domain, rather than a universal optimum.
The physical source, recurrence implementation and derivative equations remain
the existing generated provider's responsibility.

Opt-in traces retain logical raw/transformed tile identities, byte counts,
`transformed_tile_cache_hits`, `fused_metric_panel_productions`, and J/K timings.
The RI-J charge and output passes count two productions of each raw tile. Full
AO K panels count one transformed production and one cache hit per panel.
Graph-construction records are not executed-call or graph-replay measurements.

Fixed-density 192-AO evidence includes a 32-MiB allowance below the 56,623,104
bytes required by a full transformed tensor. J/K and the complete two-electron
gradient agree with the resident reference, but K remains expensive (about
86 seconds per call). The corresponding ten-minute full-SCF attempt timed out;
it is not a passing total-energy/force benchmark. Complete endpoint results
and exact library identities belong in the accompanying evidence archive.

One-electron preparation now uses matrix-only topology packing, omitting direct
ERI resident tasks. Its preflight and the common HF planner reserve quadratic
pair metadata rather than a fictitious quartet array. Direct-SCF packing keeps
its existing defaults. This makes preparation capacity follow the data actually
needed by the DF exporter without treating its sub-budget as a whole-HF peak.
