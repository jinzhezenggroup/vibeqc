# Bounded split occupied Gram experiment (#412)

Stage A qualifies `split4` for complete endpoint testing; it does not qualify an
automatic production policy. The four real 768/768/rank-160 fixed-U inputs show
12.4% net Gram savings against SYRK plus mirror. Stage B holds merged #415's
derivative mapping fixed and measures the incremental Gram change. Independent
derivative and Gram savings must not be added.

`stage-a-summary.json` retains extrema and the four target ratios.
`experiment-v1/` retains all twelve fixtures, six candidates, seven interleaved
samples each, separate component measurements, numerical gates, true work and
resource reservations. `inputs.json` binds binary inputs and executables to
hashes; large U/K binaries remain local. Real capture used the standalone #404
baseline, source `796ce85a607dfde85bf6bafd20575d7c538fdad1`, before its promoted
derivative mapping. Capture timing is intrusive and is not performance evidence.
The first capture attempt failed graph capture; the second skips graph
construction and captures successful eager molecular builds.

Original capture/trial sources are retained with `.txt` appended to preserve
their exact measured bytes under repository formatting hooks. Restore the
original names to compile/reproduce them; `stage-a-retention.json` records both
names and hashes. Scripts preserve historical workstation paths as provenance;
adapt those paths and regenerate fixtures for a new campaign. Execute every GPU
command inside finite `srun --partition=main --gres=gpu:5090:1` allocations,
preserving scheduler device visibility. The original trial used Slurm job 9817.
`stage-a-sanitizers.json` records error summaries and hashes of the local raw
logs. Its memcheck did not request full leak checking.

Full-GEMM partials compute both triangles: roughly twice SYRK's leading FLOPs.
Stage A reserves capacity for the largest candidate and separately reports
candidate-specific partial storage. Stage B borrows only existing charged
`exchange_intermediate`, preserves raw A/final U and adds no owned allocation.
Its existing reducer starts at zero, adds four partials, then accumulates into
zero K; the source-level addition count differs from Stage A's custom reducer.

`endpoint-protocol.json` was declared before endpoint measurement. It requires
seven interleaved paired samples, >=1% complete force saving at 768 and a paired
bootstrap upper ratio bound below one, unchanged independent scientific/work
gates, and four-size energy/force regressions. Automatic selection is unchanged
pending that evidence. #206 owns the fresh matched stock GPU4PySCF comparison.

The integrated candidate is frozen by `candidate-build.json` and
`candidate-source.patch` against merged source `25e8efe`. Before endpoint
timing, the native density-fitting suite, two CPU policy tests and 68 Python
GPU tests pass. Native memcheck with full leak checking and initcheck report
zero errors. `validation.json` binds jobs 9823/9824, observed small-shape split
and fallback counts, and raw-log hashes. An initial UHF trace assertion failed
because an already captured occupied body emits no new per-product records;
its energy/force checks had passed. `validation-harness-failure.json` preserves
that failure and the corrected executed-provenance check; the UHF follow-up
passed without a native or scientific-gate change.
