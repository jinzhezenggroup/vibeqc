# 2026-09-21 benchmark checkout trim

This retention pass removes 9 bulky historical benchmark records from the normal
checkout while preserving their exact bytes in existing Git history. It does not
rewrite history, change numerical/performance acceptance criteria, remove test
reference inputs, create a Release, or upload artifacts elsewhere.

The removed records total **3,395,480 bytes**. The final set excludes any member
still bound by a checked-in publication manifest. Permanent test/reference data,
CUDA-ownership and density-source/candidate publication fixtures, incremental
low-rank publication fixtures, and #418 root audits remain in the checkout.

Restore one original file into ignored local storage:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/f-shell-135/fpps.json \
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
