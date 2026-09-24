# DFT-MP-v1 acceptance contract

Issue #1185 owns the frozen inputs and acceptance tooling. Issue #1191 is the
product index; only a reviewed #1190 PASS on one upstream-merged production
revision can complete that index. The checked-in manifest does not report a
scientific result. Every capability starts `not-run` until real evidence exists.

## Immutable input and row inventory

`tools/dft_mp_v1/manifest.json` is the versioned contract. Its SHA-256 covers
the complete canonical JSON except the `contract_sha256` field. Each case points
to checked-in Bohr coordinates and a fixed changed-geometry input with separate
LF-normalized text hashes, matching Git blobs across Windows and Linux. Raw
runtime logs and artifacts retain exact byte hashes. The 8/16/32-water scaling
cases reuse the exact WATER27-derived
VibeQC series: 24/48/96 atoms and 192/384/768 real-spherical def2-SVP AOs.
AO counts were expanded from the pinned bundled basis pack; receipts must report
the same actual count. The 16/32 clusters are translated S4 octamer arrays,
not independently optimized clusters. The short peptide is neutral
Ace-Gly-Gly-NMe, fixed by a canonical SMILES and one RDKit 2026.03.4
ETKDGv3/MMFF94s conformer with seed 1185. Caffeine and benzene use the same
documented generation route. Input JSON records charge, multiplicity, exact
atoms, generator/source, redistribution license, and attribution. WATER27 data
are [CC BY 4.0](https://github.com/grimme-lab/GMTKN55); the repository's
source snapshot is pinned to `6f6a3789`.

The 115 required rows distinguish FP64 energy, FP64 energy plus analytic
forces, actual mixed execution with strict final correctness, and promoted
net benefit. PBE, r2SCAN and PBE0 each have RKS/UKS correctness rows and five
prefrozen RKS performance rows (water8/16/32, caffeine, peptide). B3LYP is a
versioned FP64 hybrid reuse gate; LDA is a regression sentinel. Direct J/K and
real-spherical def2-SVP are the target identity. DF, another GPU, another
basis or TF32/FP16/BF16 need separate contracts; they cannot fill these rows.
Two optional B3LYP/O2 FP64 adversarial rows remain explicit. Their unknown,
unsupported, failed or not-run results do not substitute for or block any
required row; a fabricated optional `pass` is still rejected.

The exact quadrature is a **shared explicit GridSpec**, frozen from
`GridPolicy("tight").resolve("pbe", derivative_order=1)` at contract creation.
It is not the default grid for each method: r2SCAN's public default retains
the GridSpec v1 route, and global hybrids require an explicit grid. All result
receipts bind the case-specific point/weight identity from this explicit spec;
an independent finer-grid convergence check remains a separate gate. Changing
the spec requires a versioned contract amendment before measurement.
The public direct-ERI screening tolerance is frozen at `1e-12`; strict and
mixed receipts must report the same numeric control and SCF targets.

Changing a case, source, mathematical model, basis, grid, mandatory row or
acceptance gate requires a new contract version and hash. Record the reason,
affected gates and migration mapping in the issue and PR. A change may replace
an infeasible implementation detail with an equivalent reviewed contract, but
cannot silently weaken the numerical gates or the single-revision #1190 gate.

## Reproduce and collect

From the repository root, with `PYTHONPATH=python:.` (PowerShell:
`$env:PYTHONPATH='python;.'`):

```text
python -m tools.dft_mp_v1.validate
python -m pytest tests/python/test_dft_mp_v1_contract.py -q
python -m tools.dft_mp_v1.run --plan /absolute/campaign-plan.json --out /absolute/output
python -m tools.dft_mp_v1.validate /absolute/output/receipt.json --final
```

`generate_inputs.py` is a read-only audit by default. RDKit-derived geometry
reconstruction is qualified only for the platform, Python ABI and exact RDKit
wheel recorded in `generator_provenance.json`; the version string alone is not
portable evidence. An unqualified environment fails before generating or
writing. A deliberate new-version candidate uses `--candidate-output` with a
separate empty tree; the v1 input directory is never overwritten. The tool
serializes every candidate before comparison or output. `freeze_contract.py`
rebuilds the manifest from checked-in inputs and the bundled basis. These are
audit tools, not benchmark-time geometry generators. The runner plan contains
`adapter_command` as an argv array, `adapter_command_file_index`, bounded
`timeout_seconds` and a `campaign` object matching `receipt.schema.json`. It
checks and hashes the adapter,
library, artifact and build record before starting. The adapter receives
`--manifest`, `--row`, `--input` and `--progress` and writes one
`result.schema.json` object
or an explicit non-pass status to stdout. It must call the installed public
energy/analytic-force endpoint and independent oracle; the runner is an
orchestrator, not a replacement scientific implementation. Each attempt writes
raw stdout/stderr, a progress journal and an atomic receipt update. Rerunning
the same output resumes only `not-run` rows; retries of failed/timed-out rows
need a distinct campaign directory so negative evidence stays visible.
The adapter appends per-sample observations to its `--progress` JSONL path;
the runner hashes stdout, stderr and progress for every attempted row. It
writes a `running` receipt and journal event before launch. If the runner dies
before finalizing a row, restart refuses to launch more work because the old
process tree may still be alive, even when no sample was written. The operator
must confirm it has ended and start a new campaign; raw files remain available.
A timeout terminates the adapter process tree; if cleanup cannot be confirmed,
the campaign stops.

For a promoted row, the adapter's progress JSONL is the `attempt_ledger` bound
to the runner's captured progress hash. It contains one record per attempted
route, in boundary/pair/order sequence: `attempt_id` (contiguous from zero),
`boundary` (`cold`, `warm`, `changed_geometry`), `pair_index`, `route`
(`strict` or `mixed`), `status`, `conditions_sha256`, and `total_ms` on success.
The validator compares every ledger record with the reported timing pairs;
failed, timed-out, missing and dropped attempts prevent a promoted PASS.

The semantic validator verifies file hashes, every mandatory row, exact
input/basis/grid/method/source/library/artifact identity, attained state,
physical residual, independent oracle and grid/finite-difference evidence,
native precision provenance reconciled with a Fock-event ledger, actual FP32
operator work, strict refinement and raw complete-endpoint timing. It rejects
unknown, unsupported, not-run, failed and timed-out rows as product passes.
For promotion, each core method separately needs five interleaved unprofiled
pairs per case and boundary, at least 1.20x geometric-mean speedup on cold and
changed-geometry complete E+force, and a stratified paired-bootstrap 95%
lower bound above 1. Warm replay is retained separately and is also subject
to the per-case gate against unexplained latency regression above 5%. Every case is kept,
including losses and timeouts. The declared 20 GiB simultaneous memory budget
is part of this first RTX 5090/sm_120 contract; a different allocation must be
versioned before measurement.

`--final` additionally requires the #1190 reviewed raw receipt. It resolves
`refs/heads/master` from the explicit official URL
`https://github.com/jinzhezenggroup/vibeqc.git`, fetches that immutable OID
without updating local remotes or `FETCH_HEAD`, and checks that the exact
source commit is its ancestor and contains this exact manifest blob. A fork's
arbitrary `origin/master` cannot satisfy this gate.
The build record must bind that source to the installed binary and AOT artifact.
An adapter can still lie in JSON or a build record can be forged; human review
and independent reproduction of retained raw evidence remain mandatory. A
schema-only or mocked runner test is never scientific acceptance.
The installed-production adapter and runnable final campaign plan are owned
by #1190, consuming the public endpoints from #1186/#1187/#1189. This package
provides the frozen orchestration and validation contract they must satisfy.
