# 2026-09-21 benchmark checkout trim

This retention pass removes 12 bulky historical benchmark records from the normal
checkout while preserving their exact bytes in existing Git history. It does not
rewrite history, change numerical/performance acceptance criteria, remove test
reference inputs, create a Release, or upload artifacts elsewhere.

The removed records total **5,621,295 bytes**. They were selected only from
campaigns with retained human-readable summaries and no repository test consumer
of the removed path. Current CUDA-ownership and density-candidate publication
fixtures, permanent test/reference data, and #418 root audits remain in the
checkout.

Restore one original file into ignored local storage:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/density-sources-gpu/evidence.json \
  --manifest benchmarks/results/retention-2026-09-21/migration.json
```

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
back to **96 MiB** so future small files cannot silently rebuild the removed
volume.

Agent: ChatGPT
Model: GPT-5.6 Sol
