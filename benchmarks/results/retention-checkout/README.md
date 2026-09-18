# Historical benchmark checkout archive

The [upstream data release](https://github.com/jinzhezenggroup/vibeqc/releases/tag/benchmark-evidence-e215b306)
retains complete original snapshots of 18 historical DF campaigns from
`e215b30685f8a36ef0cc5772c9166837527f64a2`. This is storage cleanup, not a
software release or numerical requalification.

Only 197 bulky reports/archives (40,562,398 bytes; 38.68 MiB) were removed from
the current checkout. The 10,518,603-byte ZIP contains all 917 original campaign
members, including summaries, source patches, manifests and failed/incomplete
runs. Every member's original path, byte count and SHA-256 is in
[`raw-evidence.manifest.json`](raw-evidence.manifest.json).
`checkout: external` identifies the removed subset; `retained` means the path
remains source controlled, not that future edits must match this old snapshot.

Concise root summaries, original timing/statistical summaries, source patches,
reproduction scripts and input arrays remain in Git. Test-consumed bundles,
previous storage-migration fixtures and production-referenced evidence were
left alone. This does not rewrite Git history or reduce full-clone history.

## Restore the complete snapshot

Download explicitly; normal tests and builds do not access this release:

```bash
mkdir -p .artifacts/retention-download
gh release download benchmark-evidence-e215b306 \
  --repo jinzhezenggroup/vibeqc \
  --pattern benchmark-evidence-e215b306.zip \
  --dir .artifacts/retention-download
python tools/unpack_evidence.py benchmarks/results/retention-checkout \
  --archive .artifacts/retention-download/benchmark-evidence-e215b306.zip \
  --output .artifacts/retention-snapshot
```

The output must be new. The existing unpacker checks the complete archive hash,
member list, safe relative paths, member sizes and hashes **before** creating
the output. It never executes historical scripts. The restored tree contains
`benchmarks/results/<campaign>/` with the original relative layout. Use this
complete snapshot to check the original campaign manifests; they are not
inventories of the reduced current checkout. Recorded measured revisions and
dirty-source patches retain their original meaning.

## Restore one file from existing Git history

No network request is made by this command:

```bash
python tools/restore_retained_evidence.py \
  --manifest benchmarks/results/retention-checkout/raw-evidence.manifest.json \
  benchmarks/results/issues388-391-df/profiles/384.json
```

Output defaults to `.artifacts/retention-restore/` plus the original path.
A shallow clone missing the source revision needs an explicit
`git fetch origin e215b30685f8a36ef0cc5772c9166837527f64a2` first. A full clone
normally already contains the ancestor objects. This second recovery route
does not depend on release assets.

The upstream data release has no automatic expiry; it is not an expiring
Actions artifact. Do not replace the asset in place. Publish any future
replacement with a new content identity. Repository administrators can still
remove Git objects or release assets, so checksums establish identity rather
than an absolute availability guarantee.

## Audited moved payloads

| Campaign | Files moved | MiB moved |
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

Every original member was restored and hash/size checked before upload and
again after downloading the uploaded asset. The existing 1 MiB per-file guard
remains; a 96 MiB aggregate `benchmarks/results/` budget now prevents unlimited
growth by many individually small reports. Permanent test fixtures are outside
that aggregate budget.

Agent: ChatGPT
Model: GPT-6 Astra Pro
