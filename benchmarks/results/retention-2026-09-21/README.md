# 2026-09-21 benchmark checkout trim

This retention pass removes **15 legacy evidence archives** from the normal
checkout while preserving their exact bytes in existing Git history. It does not
rewrite history, change numerical/performance acceptance criteria, remove live
summarizer/index inputs, create a Release, or upload artifacts elsewhere.

The removed archives total **4,840,512 bytes**. The final set excludes
every member bound by a checked-in benchmark publication manifest, every
permanent archive directly consumed by repository tests, the generated-DF
promotion reports consumed by `tools/summarize_df_promotion.py`, and f-shell
reports named by `f-shell-135/index.json`.

Restore one original archive into ignored local storage:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/issue310-setup-eigen/raw-evidence.zip \
  --manifest benchmarks/results/retention-2026-09-21/migration.json \
  --output .artifacts/issue310-setup-eigen/raw-evidence.zip
```

For an archive with an adjacent scientific member manifest, run
`tools.unpack_evidence` against the retained family directory and pass the
restored ZIP via `--archive`. Do not copy restored archives back into the tracked
checkout.

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
