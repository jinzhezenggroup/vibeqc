# Scientific evidence retention

VibeQC keeps the independent fixtures and numerical/performance evidence
required by [the validation protocol](validation.md) (#138). Storage policy
does not change tolerances, reference independence, statistics or promotion
gates. Three classes have different lifetimes:

| Class | Examples | Retention |
| --- | --- | --- |
| Scientific references | `tests/reference_data/`, `tests/data/`, audited `external/` sources, generator inputs | Versioned permanently; tests work without an artifact service |
| Accepted benchmark evidence | Reproduction commands, source/equation/schedule identities, hardware/software versions, comparison samples, errors/residuals, memory, decisions and negative results | Compact reviewed records in `benchmarks/results/` |
| Transient runs | Retries, stdout/stderr, test XML, temporary checkpoints, profiler databases, scheduler receipts, compiler products | `.artifacts/`, build directories or external artifacts; excluded from Git by default |

"Accepted evidence" describes reviewed storage. It may document a rejected or
inconclusive candidate; it does not imply production promotion. Scientific
solver trajectories in NPZ files differ from temporary restart checkpoints.
Fixture location and explicit justification take precedence over suffixes.
Audited Libxc source snapshots remain versioned and unchanged.

## Run, inspect, publish

Benchmark writers use `.artifacts/benchmarks/` by default or require an explicit
output location. Existing `--output` arguments remain supported. Use separate
directories for repeated experiments; a working output is not a publication:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/validate_xc.py --tier endpoint \
  --functionals PBE --output .artifacts/benchmarks/xc-run-1
```

Keep Slurm's device visibility. Record actual source identity and dirty state
at measurement time; the publishing checkout may be newer. Retain a source
reconstruction patch for dirty runs. Never infer missing historical provenance
from today's machine or manufacture a passing gate.

The publisher consumes the existing `vibeqc.validation` envelope. Its additional
`vibeqc.benchmark-publication.v1` storage manifest selects files for review:

- `source`: measured full `revision` and boolean `dirty`;
- `reproduction.command`: argument-vector form of the stable runner command;
- `files`: relative `path` and `role` (`evidence`, `summary`, `samples`, `input`,
  `reproduction`, or `source-patch`); exactly one `evidence` file is required;
- `decision`: `status` (`accepted`, `rejected`, `inconclusive`), `scope`
  (`numerical` or `performance`), and a concrete `reason`;
- `archives`: optional HTTPS `uri`, `sha256`, `bytes`, `retention`/expiry, and
  `required_for_reproduction: false` for each large debugging artifact.

The envelope retains mathematical identities, input hash/settings, measured
hardware/software/backend, raw timings, numerical errors/tolerances, residuals,
resources and stage outcomes. Inapplicable identity fields need explicit
reasons. The publisher validates existing scientific gates, computes file
sizes/checksums, rejects run debris and escaping paths, and creates a new
directory without overwriting evidence:

```bash
python tools/evidence.py publish \
  --run-directory .artifacts/benchmarks/xc-run-1/PBE-polarized-fused \
  --specification .artifacts/benchmarks/xc-publication-spec.json \
  --destination benchmarks/results/xc-reviewed-example
```

Prepare the specification from the measured run after reviewing its result.
The command writes local files only: it does not upload or authorize a GitHub
Release, tag, asset or publishing workflow. It does not infer a winner, rewrite
samples, stage Git changes or modify production selectors. Performance acceptance still requires #138's
comparison, numerical, endpoint, compilation and resource gates. Historical
schemas retain their original meaning; migration does not retroactively
certify incomplete provenance. The publisher adds storage metadata, not a
second numerical schema or a different promotion policy.

## Inventory and review guard

```bash
python tools/evidence.py inventory --output .artifacts/tracked-inventory.json
python tools/evidence.py check
```

The inventory lists each tracked path, byte count, SHA-256 and class, with
counts/bytes for references, accepted evidence, transients, generated products,
source and unknown files. It reads **Git index bytes**, so partially staged
files are checked exactly as they will be committed. Use `--revision SHA` on
`inventory` for a historical tree. Stage new files before checking them.

Pre-commit and CI reject tracked `attempts/`, routine logs/XML, checkpoint
suffixes, compressed profiler databases and generated binary/object products.
There is no global XML/JSON ban. The original
`check-added-large-files --maxkb=1024 --enforce-all` hook enforces a **hard 1 MiB
limit on every file it checks**, in both local pre-commit and all-file CI. The
index retention checker independently enforces the same cap. Entries in
`benchmarks/evidence-policy.json` can justify classification of small scientific
fixtures, but cannot waive or raise this size limit.

Keep compact summaries, raw comparison samples and reproduction commands in
Git. Group large JSON lists by workload or observable into readable companion
files; `tools.vibeqc_validation.record.load_record` reconstructs the original
record after checking each part's SHA-256 and size. No numerical values or
sample ordering are changed. Permanent array inputs can be stored as named NPY
files, preserving the original array bytes and numeric identity checks.

The same retention checker also enforces the optional
`benchmark_results_max_bytes` policy field across **all** indexed files under
`benchmarks/results/`, including manifests and summaries. Its current budget is
128 MiB. Many individually sub-limit files cannot bypass this aggregate guard;
classification exceptions cannot waive it. Permanent fixtures under
`tests/reference_data/`, `tests/data/` and audited external sources are not
counted. Changing the budget is an explicit policy review, not an automatic
response to another benchmark dump.

Full logs, retries and profiler traces belong in `.artifacts/` or external
storage. Do not compress them, rename them or split binary archives into chunks
to bypass the limit. New benchmark archive paths are also ignored by default.

## Restoring historical oversized archives

The [size-limit migration](../benchmarks/results/retention-size-limit/migration.json)
records the path, original revision, bytes and SHA-256 of removed archives. The
current tree is smaller; existing Git history has **not** been rewritten and a
full clone still contains those historical objects. Restoring one for inspection
does not download or execute anything:

```bash
python tools/restore_retained_evidence.py \
  benchmarks/results/issue284-occupied-exchange/raw-evidence.zip
```

The default output is `.artifacts/retention-restore/` followed by the original
path. It must not already exist. A shallow clone may need to fetch the revision
recorded in the manifest. For a bundle with `raw-evidence.manifest.json`, pass
that restored archive to `tools/unpack_evidence.py --archive PATH --output NEW_DIR`.
Small test-consumed archives remain in the current tree. Historical numerical
claims retain their original identities and limitations; the migration itself
is not a new scientific qualification.

## Historical report snapshots

The [checkout snapshot](../benchmarks/results/retention-checkout/README.md)
uses **existing ancestor Git objects**, not a Release or external archive.
Its `vibeqc.git-snapshot.v1` manifest pins the original full commit SHA and
inventories every original path, size and SHA-256, plus total files and bytes.
Concise summaries and test/production consumers remain in the current tree.
Historical Markdown links point to pinned Git revisions rather than missing
checkout files. Restore the complete snapshot before inspecting its original
campaign manifests or explicitly running historical reproduction scripts:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/retention-checkout/snapshot.manifest.json \
  --output .artifacts/retention-snapshot
```

All members are verified before creating a new destination. Invalid identities,
unsafe/duplicate/conflicting paths, missing objects or checksum/size mismatches
leave no partial destination; existing files/directories are never overwritten.
Omit `--all` and provide an original repository-relative path to restore one
member. The legacy default migration and evidence-archive manifests still work.

The helper does not fetch, including promisor-object lazy fetches, upload or
execute anything. Missing history requires an explicit user-controlled Git
fetch; source archives without `.git` need a Git clone. Full-clone history must
remain available for recovery; cleanup does not rewrite it or reduce its size.
No new release, tag or external storage is necessary. Normal tests/builds remain
offline and do not depend on fetching these historical snapshots.

## External artifacts and expiry

CI uploads `.artifacts/` as debugging output with 14-day retention when present.
Full stdout, test XML, profiler databases and retries may go there. Compute
SHA-256 before upload; reference the immutable Actions run/artifact ID, checksum
and expiry from the publication when it helps explain a decision. Longer-lived
external storage requires separate, explicit user authorization, with checksums
and retention policy recorded. A cleanup or benchmark task does not authorize
creating GitHub Releases, release tags/assets or dispatching publishing workflows;
do not substitute a fork or another host to work around that boundary.

Expired archives must not break tests or reproduction. Permanent reference
inputs, mathematical definitions, accepted raw samples/statistics and stable
reproduction entrypoints stay in Git. A URL without a checksum is not evidence
identity. Availability of debugging archives is distinct from numerical truth.

## Baseline cleanup (#238)

The [migration audit](../benchmarks/results/retention-238/migration.json)
records original hashes/bytes, extracted JSON measurements, compiler resource
diagnostics, test totals/failures/skips and retained profiler CSV exports.
It removes 161 transient files (3,482,611 bytes) from baseline
`ce3b6c89a418985d5b8a979e5adba5812b5a6145`, retaining a 382,561-byte audit.
Class totals are stored beside it. Reproduce the full per-file reports with:

```bash
python tools/evidence.py inventory \
  --revision ce3b6c89a418985d5b8a979e5adba5812b5a6145 \
  --output .artifacts/inventory-before.json
python tools/evidence.py audit-migration \
  --revision ce3b6c89a418985d5b8a979e5adba5812b5a6145 \
  --output .artifacts/migration.json
```

Original debris remains recoverable from existing baseline Git objects, with
`git:REVISION:PATH` identities in the audit. Use `git show REVISION:PATH` and
verify SHA-256 to inspect an old full trace. This does not rewrite history or
store new transient runs there. Profiling conclusions retain CSV/JSON exports;
stdout timing/memory measurements are extracted without resampling. Scientific
arrays, compact replay archives and measured-source patches retain their exact
bytes and pinned reasons.
