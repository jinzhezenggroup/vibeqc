# Recover complete DF campaign snapshots from existing Git history

Integrating #486 and #494 raised tracked benchmark results above the unchanged
96 MiB aggregate limit. This follow-up keeps compact summaries, validation,
source patches, manifests, and every record referenced by the production DF
policy. Only 29 large raw JSON records move out of the current checkout.
No numerical sample, failed attempt, threshold or historical verdict is changed.

`snapshot.manifest.json` records every original member of both campaigns at
existing ancestor `5a7fdeb2689553c0a304dad3338ba184d850ef60`, including the
members retained in checkout. Each original path, byte count and SHA-256 is
recorded. `checkout: git-history` identifies the 29 removed paths. Historical
README hashes describe that pinned snapshot, not later documentation edits.

```sh
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-df-integration/snapshot.manifest.json \
  --output .artifacts/df-campaign-snapshot
```

Use the restored original directory tree when inspecting full raw records or
replaying historical commands. The helper verifies every member, refuses an
existing destination, blocks implicit Git fetching and does not execute scripts.
A shallow or partial clone must obtain the recorded history explicitly first.
No Release, tag, upload, external archive or history rewrite is involved.

The full source tree was checked against GitHub before compaction. Recovery was
executed for every member using an unpushed local Git commit of that identical
tree, followed by independent byte-count/SHA-256 rechecks. The published manifest
uses the real ancestor above, not the local test-only commit.

Agent: ChatGPT
Model: GPT-6 Astra Pro
