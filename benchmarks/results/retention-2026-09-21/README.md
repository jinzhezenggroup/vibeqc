# 2026-09-21 benchmark checkout trim

This retention pass removes **24** historical benchmark records and legacy
archives from the normal checkout while preserving their exact bytes in existing
Git history. It does not rewrite history, change numerical/performance acceptance
criteria, remove test reference inputs, create a Release, or upload artifacts
elsewhere.

The removed records total **8,235,992 bytes**. The final set excludes every
member bound by a checked-in benchmark publication manifest and every permanent
archive directly consumed by repository tests (including the RCCSD/XC evidence
archive fixtures). CUDA-ownership and density-source/candidate publications,
incremental-low-rank publications, permanent reference data, and #418 root audits
remain in the checkout.

Restore one original file into ignored local storage:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/issue310-setup-eigen/raw-evidence.zip \
  --manifest benchmarks/results/retention-2026-09-21/migration.json
```

For an archive that has an adjacent scientific member manifest, restore the ZIP
under ignored `.artifacts/`, then run `tools.unpack_evidence` against the retained
family directory with `--archive PATH_TO_RESTORED_ZIP`. Do not copy the restored
archive back into the tracked checkout.

Restore the complete removed set:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-2026-09-21/migration.json \
  --output .artifacts/retention-2026-09-21
```

Every restored blob is checked against its recorded byte count and SHA-256 before
being written. Missing historical Git objects fail closed and are never fetched
implicitly.

The aggregate `benchmarks/results/` checkout budget is tightened from 128 MiB
to **96 MiB** so future small files cannot silently rebuild the removed volume.

Agent: ChatGPT
Model: GPT-5.6 Sol
