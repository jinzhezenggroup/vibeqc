# Final-state and combined host-work ablations at 96 AOs

> **Checkout retention (2026-09-21):** `endpoints.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue206-final-state-96/endpoints.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue206-final-state-96/endpoints.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

The current #206 runner at `90e4f53` compares verified retention with actual
forced ordinary-device rebuilding, and separately compares the combined
lazy/cache/device/verified-state path against eager/rebuilt reference setup
plus forced reference finalization. Both sides use the same strict physical
state gates. The combined baseline is a diagnostic control on this binary,
not the historical single-rebuild finalizer or an external engine.

Slurm job 9500 used one RTX 5090 on `main`, with a finite 20-minute limit.
Eight CLI invocations separate traced and clean energy/full-force runs.
Each has five interleaved pairs for cold, unchanged, output-selected and
changed-geometry workloads. Seeds are frozen and every SCF iteration/retry
branch matches. No compilation or substantial CPU work ran during clean timing.
The source/library hashes, all endpoint JSON, 160 raw host traces and commands
are retained. The initial launcher path failure (job 9499) is retained separately;
it ran no benchmark and contributes no sample.

| Comparison | Warm baseline | Warm candidate | Ratio |
| --- | ---: | ---: | ---: |
| combined-host-energy | 232.228 ms | 10.976 ms | 21.16x |
| combined-host-forces | 489.057 ms | 267.511 ms | 1.83x |
| final-state-energy | 19.445 ms | 10.948 ms | 1.78x |
| final-state-forces | 275.436 ms | 265.779 ms | 1.04x |

These medians describe unchanged-geometry spherical RHF water-tetramer replay,
batch one, 96 AOs, resident DF, FP64. All four warm effects pass the declared
2%/noise check. The isolated final-state ablation has no significant cold or
changed-geometry improvement: those paths still require correction. Its complete
force saving is modest because other force work remains. All workloads and
raw timings, including the weak effects and initial slow samples, remain in the
manifest/archive. No ratio is multiplied across earlier binaries or ablations.

The traced runs require actual current-F validation even when the candidate
needs no final solve. Forced samples execute correction; setup providers,
per-spin leaves, W and force-response counts match their declared choices.
Independent scientific force/state correctness is qualified by #320; these
same-library comparisons establish work attribution and endpoint savings.
83 protocol/ownership Python checks pass (18 CUDA-only skips), and hooks pass.

Larger AOs, batch four, constrained source execution, independent host-iterative
Fock migration and matched GPU4PySCF/#206 acceptance remain open. The manifest
pins every member hash; the compressed archive was restored and compared
byte-for-byte before publication.
