# 2026-09-25 benchmark raw-sample checkout trim

This retention pass removes **12 bulky historical raw-sample/tuning files**
(**3,974,035 bytes; 3.79 MiB**) from the normal checkout while preserving
their exact bytes in existing Git history at `49a2e664aaeef0deeb3860aa7c0912c8d078068a`.
It does not rewrite Git history, create a GitHub Release, upload an external
archive, change a numerical tolerance, or alter production/runtime code.

The current checkout keeps the compact scientific records needed to understand
the decisions:

- `incremental-low-rank`: numerical validation envelopes, median timing summaries,
  source identity, reproduction commands and self-contained compact publication
  manifests remain; only the two complete raw sample ledgers move to history.
- `tensor-cuda-146`: equation identities, the validation envelope, provider
  allocation audit and accepted/rejected selection summary remain; the six full
  tuning ledgers move to history.
- `weighted-eri-144`: the endpoint hash index, independent numerical gates,
  provenance, resource records and decision summary remain; the four complete
  endpoint sample matrices move to history.

Repository search found no test or production-code consumer of any of the 12
moved paths. Exact path, byte count and SHA-256 identities are recorded in
[`migration.json`](migration.json).

Restore one file into ignored local storage:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/incremental-low-rank/cpu/samples.json \
  --manifest benchmarks/results/retention-2026-09-25/migration.json \
  --output .artifacts/incremental-low-rank-cpu-samples.json
```

Restore the complete removed set:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-2026-09-25/migration.json \
  --output .artifacts/retention-2026-09-25
```

The helper validates every recorded size and SHA-256 and does not fetch missing
history implicitly. The aggregate `benchmarks/results/` checkout budget is
tightened from **96 MiB to 93 MiB**, preserving roughly the previous growth
headroom instead of allowing the removed volume to return unnoticed.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Bulk campaign archival pass

A second pass removes **758 additional historical campaign members**
(**27,343,545 bytes; 26.08 MiB**) while retaining
the small files required by production manifests, tests, user/developer docs,
and compact decision summaries. These bulk members are recorded in
[`bulk.manifest.json`](bulk.manifest.json) using the exact Git blob object ID,
original path, size, and source revision.

The restoration helper accepts this Git-object snapshot directly and verifies
the reconstructed blob identity before writing. This permits binary NPZ/NPY/GZ
and large groups of small records to leave the checkout without an external
archive or Git-history rewrite.

After this pass the aggregate benchmark-results budget is **64 MiB**.
