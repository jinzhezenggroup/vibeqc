# #309 prepared overlap cache and separate preparation ablations

> **Checkout retention (2026-09-21):** `raw-evidence.zip` was moved out of the normal checkout. Exact bytes remain in Git revision `d8f64a93fe0dfebd889fd0ba1fadbd5ad7d840e5` and are checksum-bound by [the checkout-trim manifest](../retention-2026-09-21/migration.json). Restore locally with:
>
> ```bash
> python tools/restore_retained_evidence.py benchmarks/results/issue309-overlap-cache/raw-evidence.zip \
>   --manifest benchmarks/results/retention-2026-09-21/migration.json \
>   --output .artifacts/issue309-overlap-cache/raw-evidence.zip
> ```
> Restored archives belong under ignored `.artifacts/`; do not recommit them.

For commands below that previously unpacked the checkout directly, pass the
restored archive explicitly:

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-overlap-cache \
  --archive .artifacts/issue309-overlap-cache/raw-evidence.zip \
  --output .artifacts/issue309-overlap-cache-unpacked
```

The [larger-domain follow-up](../issue309-overlap-cache-larger/README.md) adds
five-pair 96-AO/batch-4 and 192-AO/batch-1 comparisons using the same validated
native scientific source and library. This record preserves the first domain.

One prepared owner retains the original FP64 symmetric X with exact S and
coordinates. Ordered basis/representation/device belong to that immutable
owner. Per-source caches survive output/provider replans, with separate
single, batch and independent-Fock coverage. Geometry changes and singular
failures replace only the affected X. The existing host lifetime ledger
charges S/X/coordinates (147,744 numeric bytes for this 96-AO item); the
existing device dX allocation is unchanged.

## First four-way endpoint comparison

RHF spherical def2-SVP, 96 AO, batch 1, the same orbital auxiliary basis,
a 1 GiB DF allowance, and Slurm's RTX 5090. Each row uses five interleaved
pairs on one clean binary with a fixed post-cold density. The baseline restores
both eager core guesses and overlap decomposition. The three candidates
isolate lazy guesses, cached X and their combination.

| Candidate | Clean warm endpoint | Baseline (s) | Candidate (s) | Baseline / candidate |
| --- | --- | ---: | ---: | ---: |
| lazy-core | energy | 0.235034 | 0.161263 | 1.457 |
| lazy-core | force | 0.794219 | 0.714123 | 1.112 |
| overlap-cache | energy | 0.235280 | 0.160786 | 1.463 |
| overlap-cache | force | 0.782382 | 0.712008 | 1.099 |
| combined | energy | 0.235160 | 0.086978 | 2.704 |
| combined | force | 0.789288 | 0.635148 | 1.243 |

Cold construction/destruction and every changed-geometry call are also in the
archive and summary. Original geometry is restored before each changed sample.
All iteration/retry branches match, and energies/full forces pass the existing
1e-9 Eh / 1e-8 Eh/Bohr replay gates against separately prepared cold endpoints.
These replay gates complement independent native/Fock scientific tests; they
are not a GPU4PySCF parity claim. Cold work has no meaningful improvement and
this record does not claim all-workload performance promotion.

Separate traced pairs verify actual overlap/core leaves: warm baseline 1/1,
lazy-only 1/0, cache-only 0/1, combined 0/0. Every selection still has one
reference final-Fock solve per RHF item. Required device provider integration
and verified final-state reuse remain #310/#311. Profiled timing is excluded
from the table. Larger/batch-4/constrained and other spin/representation timing
domains, plus the remaining #206/#308 acceptance matrix, remain open.

## Validation and provenance

25 CPU native, 52 protocol/resource Python checks (one CUDA-library-only skip),
103 GPU Python checks and 3 native CUDA DF/Fock tests passed. The latter include
budget/metric-policy replans and recovery; a pre-existing infeasible-tile
status classification was repaired without weakening its OOM assertion.
Four cache cases passed memcheck with zero errors and zero leaked bytes.
Every GPU test and measurement used a finite Slurm allocation. Pre-commit passes.

The archive retains every original result collection and JSONL trace with
member hashes, exact source commit and library identity. The compiled native
scientific source hash was checked against the measured tree. All archive
members were restored and compared byte for byte before publication.

```bash
python -m tools.unpack_evidence benchmarks/results/issue309-overlap-cache \
  --archive .artifacts/issue309-overlap-cache/raw-evidence.zip \
  --output /tmp/issue309-cache-evidence
```

Rebuild the source recorded by the manifest with CUDA 12.9.1, Release,
architecture 120 and AOT shells disabled, following the P0 recipe. Repeat each
of `lazy-core`, `overlap-cache` and `combined` with the existing #206 runner:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/issue206_df_matrix.py --run --host-workloads \
  --preparation-ablation combined --case water-tetramer-def2-svp-spherical \
  --batch 1 --library build/cuda/libvibeqc.so --memory-budget-bytes 1073741824 \
  --energy-only --repeats 5 --output-dir /tmp/issue309-combined-energy
```

Omit `--energy-only` for complete forces. Use a separate fresh output directory
and `--host-trace-dir /tmp/issue309-combined-traces` for actual-call verification.
