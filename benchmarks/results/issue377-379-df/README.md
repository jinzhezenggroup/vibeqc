# Occupied DF execution and compact source qualification (#377–379)

This bundle records current-master comparisons on NVIDIA GeForce RTX 5090
(sm_120), production Release CUDA 12.9.1, spherical def2-SVP RHF water clusters,
Naux=NAO and ranks 20/40/80/160. All real-device work uses finite Slurm jobs on
`main` with `--gres=gpu:5090:1`. Energy/force gates remain 1e-9 Ha and 1e-8
Ha/Bohr; the metric relative threshold remains 1e-10. No approximation changes.

Final independent-engine and component qualification is in progress in the
initial draft of this bundle; it must be complete before merging the PR.

## Interpretation

- `master-768-exchange.json` contains five interleaved dense/occupied K endpoint
  pairs against current master f23e041, with one frozen post-cold warm density.
- `qualified-768-response.json` contains the corresponding five response-space
  pairs with K held occupied. `rejected-small-panel-768.json` preserves the
  earlier slower schedule. `rejected-resident-export-768.json` preserves the
  later cold-setup regression; its resident-export integration is excluded
  from the final code. Its warm response measurements remain valid evidence.
- `96/192/384-exchange.json` and `192/384-response.json` retain every clean
  sample, priming cost, iteration branch, energy and force. The 384 response
  comparison changes both algebra and borrowed resident storage: it is a
  combined-route crossover, not an isolated low-rank GEMM speedup. Automatic
  selection remains limited to the qualified 768/768 RHF rank-160 domain.
- `*-raw-source.json` records complete generation, including readback, at
  192/384/768 AOs. Counter arithmetic follows the exact launched ranges and
  mapping. Coefficient reads are logical lane operations, not measured DRAM
  transactions. Both old and compact traversal counts are reported for the
  same nonzero expansion and primitive work. No intrusive atomics affect timing.
- `*-fixed-k.json` retains fixed-density calls including upload and readback;
  `*-fixed-k-components.json` separates actual stream timings from capture.
- `profiles/` contains intrusive component/work/transfer/scratch and final-state
  diagnostics, never clean performance samples. Capture counts describe graph
  construction; executed iteration provenance and fixed-K traces are separate.
- `gpu4pyscf/` contains fresh matched complete energy+force comparisons using
  each engine's frozen post-cold seed. Basis metadata and scientific settings
  are retained with the unchanged independent gates.

The source probe is not the unbudgeted resident Cartesian exporter. Its
approximately 103.9-to-24.3-second improvement must not be attributed to that
separate preparation path. The rejected integration demonstrates this boundary.
Whole-plan capacity, borrowed scratch and new response allocations are reported
separately; capacity is not a measured global GPU peak.

## Reproduction and identity

Build with `cmake --preset cuda-release-sm120` using CUDA 12.9.1 and production
compiler settings, then build `vibeqc`, `vibeqc_df_source_probe`,
`vibeqc_df_occupied_probe`, `vibeqc_df_value_probe`, and
`vibeqc_df_occupied_response_tests`. The retained native libraries used
`-lineinfo`. Use a fresh artifact directory for every run.

Run `python -m benchmarks.df_policy_endpoint --aos 768 --repeats 5 --output ...`
with `PYTHONPATH=python:.` and `VIBEQC_LIBRARY` pinned to the desired library.
Change `--control VIBEQC_DF_RESPONSE_SPACE` with `VIBEQC_DF_EXCHANGE=occupied`
for response comparisons; `--trace --journal --repeats 1` is a separate
intrusive pass. If a library directory contains `source.patch`, the runner
retains that frozen source rather than a later dirty checkout. An explicit
`--source-patch` is also supported. Every command must run within finite `srun`.

`vibeqc_df_source_probe inputs/768.txt 3 report.jsonl raw.bin` generates the
full raw source. The inputs contain unnormalized physical shell primitives;
the native molecule owner applies its usual normalization. For fixed K,
append row-major converged occupied coefficients to the same physical input
before invoking `vibeqc_df_occupied_probe INPUT 5 0 0 arrays.bin`. The retained
input metadata identifies the independent converged checkpoint and exact
orthogonality check. Large raw/K arrays, logs, binaries and detailed traces
remain outside Git; `local-artifacts.json` pins their paths, sizes and SHA-256.

The source patches reconstruct cited native versions from master f23e041.
Compact endpoint files omit only repeated basis metadata, which is retained
once, and record the full original JSON hash. Historical provisional probe
results with an incorrect lane label are excluded from final causal counts.
