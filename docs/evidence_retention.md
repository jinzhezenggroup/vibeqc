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
The command does not infer a winner, rewrite samples, stage Git changes or
modify production selectors. Performance acceptance still requires #138's
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
There is no global XML/JSON ban. Tracked files above **1 MiB** require explicit
size review, including references/source. Exceptions in
`benchmarks/evidence-policy.json` name exact paths/SHA-256, responsible
subsystems and scientific/storage reasons. Changed bytes or missing files make
exceptions stale. Useful large references may remain; large logs need summaries
and archives. Do not hide logs in ZIPs or rename suffixes to bypass review.
Existing compact replay archives have explicit, hash-pinned justifications.

## External artifacts and expiry

CI uploads `.artifacts/` as debugging output with 14-day retention when present.
Full stdout, test XML, profiler databases and retries may go there. Compute
SHA-256 before upload; reference the immutable Actions run/artifact ID, checksum
and expiry from the publication when it helps explain a decision. Release
assets or another immutable HTTPS store can provide longer retention, with
their checksum and retention policy recorded explicitly.

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
