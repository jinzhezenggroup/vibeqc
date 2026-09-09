# RCCSD B acceptance, 2026-09-08

Parent: A commit 5059736e3644f273cdd2347803f4a9adccaed7de. Refs #148.
Complete opposite-spin physical R2, shared intermediate inventory and
expanded/shared/optimized equivalence are implemented. C remains separate.

- qz independent worktree `projects/vibeqc-148-a`, Python 3.11.16/NumPy 2.2.6,
  PySCF 2.14.0, CPU only, single BLAS/OpenMP thread; existing A native build.
- `python -m pytest tests/python/test_cc_equations.py
  tests/python/test_cc_references.py tests/python/test_cc_provenance.py
  tests/python/test_cc_doubles.py tests/python/test_cc_doubles_references.py -q`:
  **41 passed in 21.19 s** before the additional provenance regression.
- Final `python -m pytest tests/python/test_cc_doubles_references.py -q`:
  **5 passed in 2.95 s** after the provenance repair. The routine log has been
  consolidated into this result; numerical records and provenance are retained.
- `python -m tools.validate_ccsd --output build/cc-b-final`: all five independent
  records passed. The archived JSON files contain per-form/per-intermediate
  errors, actual source/fixture/upstream hashes and dirty status.
- Ruff check passed. Equations unchanged after the complete 41-test run.
- Reference generation used `python -m tools.generate_cc_references --full`
  twice with `--compare`; identical cases hash:
  `e036dbe8acd1e94399733542192ba66692ae9e8dd54256640cee8289cde7059e`.
- Final uploaded eight-file source tar SHA-256:
  `fd2c598be044424907b5c57f22e21b5c03f2fe0cdf4a7424fde4d3e0dceb84b0`;
  independently read back before extraction/execution. All Inspire commands
  used `--no-env-file --account qz` and explicit remote worktree.

## Archived raw records

Detailed JSON results and execution logs are stored in
[`raw-evidence.zip`](raw-evidence.zip). The
[manifest](raw-evidence.manifest.json) lists every member's size and SHA-256,
plus the archive hash and the Git commit from which the original bytes were
copied. That commit identifies the storage migration input, not a new
scientific run. Existing source snapshots and the acceptance summary below
retain the original experiment identity, tolerances and limitations.

From the repository root, verify without extracting, or restore into a **new**
directory (Python standard library only):

```bash
python -m tools.unpack_evidence benchmarks/results/rccsd-148-b
python -m tools.unpack_evidence benchmarks/results/rccsd-148-b \
  --output build/rccsd-148-b-history
```

Files listed in the manifest are relative to the restored directory. Small
provenance records remain beside this README. Restoration checks every hash before writing and refuses
an existing output directory. Historical scripts are records, not commands to
execute. Test fixtures remain directly available under `tests/reference_data/`.

## Independent review and repair

`review_math` (not an implementer) reviewed the fixed eight-file B snapshot,
checked every source hash, and found no mathematical blockers. Its read-only
11-test recheck passed. Nine additional doubled-amplitude cases spanning
(3,1), (1,3), (3,2), three seeds each, checked all three forms and projector
norms; maximum E/R1/R2 discrepancy was 1.55e-15.

`review_architecture` found one P2: the B evidence runner omitted fixture and
TensorIR execution-stack identity. The runner now records reference file,
case and compared-output hashes, pinned upstream metadata, TensorIR source
hashes and dirty state. A new regression checks these fields. No equations
changed. Following the desktop restart, a new uninvolved read-only reviewer
`b_provenance_recheck` independently confirmed the P2 fix and found no new
blockers. It did not claim to rerun remote calculations.

No T2 acceptance relies solely on two versions of the same inventory: tiny
fermionic determinant projections and pinned external full-update residuals
also pass. PySCF update denominators have both virtual level shifts and are
reconstructed as D2*(update-t2). GPU/(T)/Lambda/gradients are outside #148.

Portable final A/B/C reproduction commands, including reference regeneration,
are in [the C archive](../rccsd-148-c/README.md#reproduction). Historical machine
paths above identify the original experiment, not prerequisites for replay.
