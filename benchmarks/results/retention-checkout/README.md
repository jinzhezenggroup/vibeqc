# Historical benchmark checkout snapshot

Complete original snapshots of 18 historical DF campaigns remain in existing
Git commit `e215b30685f8a36ef0cc5772c9166837527f64a2`, an ancestor of this
cleanup. **No GitHub Release, new tag, release asset, external archive or
publishing workflow is required or authorized.** Nothing is uploaded by the
restoration helper; Git history is not rewritten.

Only 197 bulky reports/archives (40,562,398 bytes; 38.68 MiB) are removed from
the current checkout. [`snapshot.manifest.json`](snapshot.manifest.json)
inventories all 917 original members (55,852,909 bytes), including summaries,
source patches, manifests and failed/incomplete runs, by original path, size
and SHA-256. `checkout: git-history` identifies the removed subset; `retained`
means the path remains in the checkout, not that future edits must match this
old snapshot. The manifest is a Git snapshot identity, not a ZIP manifest.

Concise summaries, original timing/statistical samples, reproduction scripts,
input arrays, test-consumed bundles and production-referenced evidence remain
in the current tree. Existing ancestor objects are the recovery source, not a
policy to commit new bulky runs. This reduces checkout size, not full-clone
historical size, and does not requalify any scientific claim.

## Restore the complete snapshot

With the recorded Git objects already present, this command is fully offline:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-checkout/snapshot.manifest.json \
  --output .artifacts/retention-snapshot
```

The destination must be new. Every original path, byte count and SHA-256 is
validated in temporary local storage **before** the destination is created.
Unsafe/duplicate/conflicting paths, missing Git objects or corrupt bytes fail
without a partial destination. Memory is bounded by the largest member.
The helper restores files as data; it never executes historical scripts.
The full original layout is `benchmarks/results/<campaign>/` under the output.
Use this complete snapshot to inspect original campaign manifests, which are
not inventories of today's reduced checkout. Measured revisions and dirty
source patches retain their original scientific meaning.

## Restore one file

```bash
python tools/restore_retained_evidence.py \
  --manifest benchmarks/results/retention-checkout/snapshot.manifest.json \
  benchmarks/results/issues388-391-df/profiles/384.json
```

Output defaults to `.artifacts/retention-restore/` plus the original path.
`--output NEW_FILE` selects a new filename; existing files are never overwritten.
The older size-limit migration and evidence-archive manifests remain supported.

## Shallow, partial and source-archive checkouts

Normal full clones already contain the source history. In a shallow/partial
clone missing it, the helper fails with an explicit instruction instead of
fetching implicitly. The user may separately retrieve the existing commit:

```bash
git fetch origin e215b30685f8a36ef0cc5772c9166837527f64a2
```

Then repeat restoration. Missing promisor blobs must also be obtained explicitly
outside the offline helper. A downloaded source archive without `.git` needs a
Git clone for historical recovery; no Release is a fallback. Builds and normal
tests neither fetch history nor require the complete historical snapshot.

Recovery depends on retaining existing source history. Administrators can
rewrite history or remove objects; hashes establish byte identity, not an
absolute availability guarantee. A future off-repository backup requires
separate, explicit authorization, not an implicit extension of cleanup work.

## Audited checkout removals

| Campaign | Files removed from checkout | MiB removed |
| --- | ---: | ---: |
| `issue206-aot-and-stock` | 6 | 0.99 |
| `issue206-current-df` | 13 | 2.81 |
| `issue206-stable-response` | 4 | 1.24 |
| `issue308-shell-schedules` | 23 | 4.30 |
| `issue309-overlap-cache-larger` | 1 | 0.68 |
| `issue309-preparation-192b4` | 1 | 0.98 |
| `issue311-force-state` | 1 | 0.66 |
| `issue382-packed-df` | 7 | 0.69 |
| `issue385-df-signatures` | 22 | 2.37 |
| `issue392-393-rejected` | 10 | 2.18 |
| `issue394-000-rys` | 7 | 0.88 |
| `issue395-df-work` | 14 | 4.06 |
| `issue399-density-seed` | 9 | 0.84 |
| `issue404-combined-rys` | 6 | 0.59 |
| `issue408-device-validation` | 3 | 0.26 |
| `issue409-packed-values` | 38 | 8.28 |
| `issue412-split-gram` | 5 | 0.52 |
| `issues388-391-df` | 27 | 6.34 |

The existing 1 MiB per-file guard remains. A 96 MiB aggregate
`benchmarks/results/` budget prevents unchecked growth by many individually
small reports; permanent test fixtures are outside that aggregate budget.

Agent: ChatGPT
Model: GPT-6 Astra Pro
